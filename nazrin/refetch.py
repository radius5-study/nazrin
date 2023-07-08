import json
import os
from glob import glob

from tqdm import tqdm

from nazrin.types import ImageMeta

from .scraper import get_post_meta


def refetch(dir: str, recursive: bool = True):
    glob_str = f"{dir}/**/*" if recursive else f"{dir}/*"

    for file in tqdm(glob(glob_str, recursive=recursive)):
        basename = os.path.splitext(os.path.basename(file))[0]
        if not basename.isnumeric():
            continue
        metafile_path = os.path.join(os.path.dirname(file), f"{basename}.json")
        if os.path.exists(metafile_path):
            continue

        post = get_post_meta(basename)
        if post is None:
            return
        meta = ImageMeta(
            tags=post["tag_string_general"].split(" "),
            tags_character=post["tag_string_character"].split(" "),
            tags_copyright=post["tag_string_copyright"],
            tags_artist=post["tag_string_artist"].split(" "),
            tags_meta=post["tag_string_meta"].split(" "),
            rating=post["rating"],
            score=post["score"],
            up_score=post["up_score"],
            down_score=post["down_score"],
            file_ext=post["file_ext"],
            file_url=post["file_url"],
            filename=os.path.basename(file),
        )
        with open(metafile_path, "w") as f:
            f.write(json.dumps(meta, indent=4))
