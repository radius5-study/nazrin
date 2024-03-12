import io
import json
import os
import time
import traceback
import urllib.parse
import requests
from concurrent.futures import Future, ProcessPoolExecutor
from typing import *

import PIL.Image
import pillow_avif
import toml
import tqdm
import yaml
from fsspec.implementations.local import LocalFileSystem

from . import constants, filter
from .config.scrape import ScrapeConfig, ScrapeSubset
from .types import DanbooruPost, ImageMeta


def createURL(path="/"):
    return (
        constants.DANBOORU_URL
        + path
        + f"?login={constants.DANBOORU_USER_ID}&api_key={constants.DANBOORU_API_KEY}"
    )


def get_post_meta(id: str):
    url = createURL(f"/posts/{id}.json")
    try:
        res = requests.get(url)
        text = res.content.decode()
    except Exception:
        return None

    return json.loads(text)


def apply_parent_config(subset: ScrapeSubset, parent_config: ScrapeConfig):
    for key, value in parent_config.dict().items():
        if key not in ScrapeSubset.__fields__:
            continue
        d = subset.dict()
        if key not in d or d[key] is None:
            setattr(subset, key, value)
    return subset


def run(
    raw_config: Dict[str, Any],
    tasks: List[DanbooruPost],
):
    config = ScrapeSubset.parse_obj(raw_config)
    if config.fs == "local":
        fs = LocalFileSystem()

    os.makedirs(config.outpath, exist_ok=True)

    i = 0
    for task in tasks:
        if task["is_banned"]:
            continue
        if task["file_ext"] not in constants.IMAGE_EXTS:
            continue
        if "file_url" not in task:
            continue
        url = (
            task["file_url"]
            if not config.save_original_size or not hasattr(task, "large_file_url")
            else task["large_file_url"]
        )

        if config.filter is not None and not filter.check(config.filter, task):
            continue

        filename = f"{task['id']}.{config.convert if config.convert is not None else task['file_ext']}"
        if os.path.exists(os.path.join(config.outpath, filename)):
            break

        if not url.startswith("http"):
            url = f"{constants.DANBOORU_URL}/{url}"

        try_count = 0
        meta = ImageMeta(
            tags=task["tag_string_general"].split(" "),
            tags_character=task["tag_string_character"].split(" "),
            tags_copyright=task["tag_string_copyright"],
            tags_artist=task["tag_string_artist"].split(" "),
            tags_meta=task["tag_string_meta"].split(" "),
            rating=task["rating"],
            score=task["score"],
            up_score=task["up_score"],
            down_score=task["down_score"],
            file_ext=task["file_ext"],
            file_url=url,
            filename=filename,
        )
        while True:
            try:
                r = requests.get(url, headers={"User-Agent": "ddpn08"})
                if config.convert is not None:
                    buf = io.BytesIO(r.content)
                    img = PIL.Image.open(buf)
                    data = io.BytesIO()
                    img.save(data, config.convert)
                    data = data.getvalue()
                else:
                    data = r.content
                break
            except Exception as e:
                traceback.print_exc()
                if try_count > 5:
                    print(f"Failed to fetch {url}")
                    break
                try_count += 1

        with fs.open(os.path.join(config.outpath, filename), "wb") as f:
            f.write(data)
        with fs.open(os.path.join(config.outpath, f"{task['id']}.json"), "w") as f:
            f.write(json.dumps(meta, indent=4))
        i += 1

    return i


def process_subset(config: ScrapeConfig, subset: ScrapeSubset, tqdm_idx: int = 0):
    subset = apply_parent_config(subset, config)
    iter = 1
    limit = subset.limit or 1000
    progress_bar = tqdm.tqdm(total=limit, position=tqdm_idx, leave=False)
    while limit > 0:
        url = createURL("/posts.json")
        url = f"{url}&tags={urllib.parse.quote(subset.tags)}&page={iter}&limit={subset.task_size}"
        iter += 1
        try_count = 0
        while True:
            try:
                res = requests.get(url)
                data: List[DanbooruPost] = json.loads(res.content.decode())
                break
            except Exception:
                try_count += 1
                if try_count > 3:
                    print(f"Failed to fetch {url}")
                    break
                time.sleep(1)

        if len(data) < 1:
            break
        if len(data) > limit:
            data = data[:limit]

        if config.multi_worker_mode == "task" and config.max_workers > 1:
            with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
                for i in range(config.max_workers):
                    task = data[i :: config.max_workers]

                    def on_done(feature: Future):
                        nonlocal limit
                        num = feature.result()
                        progress_bar.update(num)
                        limit -= num

                    ps = executor.submit(run, subset.dict(), task)
                    ps.add_done_callback(on_done)
        else:
            for task in data:
                run(subset.dict(), [task])
                progress_bar.update(1)


def process_subsets(config: ScrapeConfig):
    if config.multi_worker_mode == "task":
        for subset in config.subsets:
            process_subset(config, subset)
    elif config.multi_worker_mode == "subset":
        if config.max_workers == 1:
            for subset in config.subsets:
                process_subset(config, subset)
        else:
            with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
                for idx, subset in enumerate(config.subsets):
                    executor.submit(process_subset, config, subset, idx)


def scrape(config_path: str):
    if os.path.splitext(config_path)[1] == ".json":
        raw = json.load(open(config_path))
    elif os.path.splitext(config_path)[1] == ".yaml":
        raw = yaml.load(open(config_path), Loader=yaml.SafeLoader)
    elif os.path.splitext(config_path)[1] == ".toml":
        raw = toml.load(open(config_path))

    if "subsets" not in raw:
        raw["subsets"] = [{}]

    for k, v in raw.items():
        if k == "subsets":
            continue
        if k not in ScrapeConfig.__fields__:
            continue
        if v is None:
            continue
        for subset in raw["subsets"]:
            if k not in subset:
                subset[k] = v

    config = ScrapeConfig.parse_obj(raw)
    process_subsets(config)
