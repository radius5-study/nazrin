import io
import json
import os
import time
import traceback
import urllib.parse
import urllib.request
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
        + f"?login={ constants.DANBOORU_USER_ID}&api_key={ constants.DANBOORU_API_KEY}"
    )


def get_post_meta(id: str):
    url = createURL(f"/posts/{id}.json")
    try:
        with urllib.request.urlopen(url) as r:
            buf: bytes = r.read()
            text = buf.decode("utf-8")
    except urllib.error.URLError as e:
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

        if not filter.check(config.filter, task):
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
                req = urllib.request.Request(url=url, headers={"User-Agent": "ddpn08"})
                with urllib.request.urlopen(req) as r:
                    if config.convert is not None:
                        buf = io.BytesIO(r.read())
                        img = PIL.Image.open(buf)
                        data = io.BytesIO()
                        img.save(data, config.convert)
                        data = data.getvalue()
                    else:
                        data = r.read()
                    with fs.open(os.path.join(config.outpath, filename), "wb") as f:
                        f.write(data)
                    with fs.open(
                        os.path.join(config.outpath, f"{task['id']}.json"), "w"
                    ) as f:
                        f.write(json.dumps(meta, indent=4))
                    i += 1
                    break
            except Exception as e:
                traceback.print_exc()
                if try_count > 5:
                    print(f"Failed to fetch {url}")
                    break
                try_count += 1

    return i


def _scrape(config: ScrapeConfig):
    for subset in config.subsets:
        subset = apply_parent_config(subset, config)
        iter = 1
        limit = subset.limit or 1000
        progress_bar = tqdm.tqdm(total=limit)
        while limit > 0:
            url = createURL("/posts.json")
            url += f"&tags={urllib.parse.quote(subset.tags)}&page={iter}&limit={subset.task_size}"
            iter += 1
            try_count = 0
            while True:
                try:
                    with urllib.request.urlopen(url) as res:
                        data: List[DanbooruPost] = json.loads(res.read().decode())
                        if len(data) < 1:
                            break
                        if len(data) > limit:
                            data = data[:limit]
                        with ProcessPoolExecutor(
                            max_workers=config.max_workers
                        ) as executor:
                            for i in range(config.max_workers):
                                task = data[i :: config.max_workers]

                                def on_done(feature: Future):
                                    nonlocal limit
                                    num = feature.result()
                                    progress_bar.update(num)
                                    limit -= num

                                ps = executor.submit(run, subset.dict(), task)
                                ps.add_done_callback(on_done)
                        break
                except Exception as e:
                    try_count += 1
                    if try_count > 3:
                        print(f"Failed to fetch {url}")
                        break
                    time.sleep(1)


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
    _scrape(config)
