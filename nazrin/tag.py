import glob
import json
import os
from typing import *

import tqdm

from nazrin.config.tag import TagConfig
from nazrin.types import ImageMeta


def get_quality(map: Dict[str, str], score: int):
    for key, value in map.items():
        max, min = key.split(",")
        max = float(max) if max != "~" else float("inf")
        min = float(min) if min != "~" else float("-inf")
        if min <= score <= max:
            return value


def get_rating(rating: str):
    return "nsfw" if rating != "g" else ""


def tag(config_path: str):
    config = TagConfig.parse_auto(config_path)
    glob_str = os.path.join(
        config.directory, "**/*.json" if config.recursive else "*.json"
    )

    for path in tqdm.tqdm(glob.glob(glob_str, recursive=config.recursive)):
        with open(path) as f:
            post: ImageMeta = json.load(f)

        use_wd14tag = config.use_wd14tag and "wd14_tags" in post

        if use_wd14tag:
            tags = post["wd14_tags"]
        else:
            tags = post["tags"]

        if len(config.prefix) > 0:
            tags = config.prefix + [tag for tag in tags if tag not in config.prefix]
        if len(config.suffix) > 0:
            tags = [tag for tag in tags if tag not in config.suffix] + config.suffix

        if config.tag_rating and "rating" in post:
            tags = [get_rating(post["rating"])] + tags

        if config.tag_meta and "tags_meta" in post:
            tags = post["tags_meta"] + tags
        if config.tag_artist and "tags_artist" in post:
            tags = post["tags_artist"] + tags
        if not use_wd14tag:
            if config.tag_character and "tags_character" in post:
                tags = post["tags_character"] + tags

        if "score" in post and config.quality_tag:
            quality_tag = get_quality(config.score_map, post["score"])
            if quality_tag is not None:
                tags = [quality_tag] + tags

        if config.remove_underline:
            tags = [tag.replace("_", " ") for tag in tags if tag.strip()]

        if "caption" in post and post["caption"] is not None:
            if config.where_to_insert_caption == "top":
                tags = [post['caption']] + tags
            else:
                tags = tags + [post['caption']]

        tag_string = ", ".join(tags)
        with open(path.replace(".json", ".txt"), "w") as f:
            f.write(tag_string)
