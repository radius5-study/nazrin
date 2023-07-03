from nazrin.config.filter import FilterConfig, TagFilter
from nazrin.types import DanbooruPost


def tags(config: TagFilter, post: DanbooruPost):
    if config.must is not None:
        must = config.must if isinstance(config.must, list) else config.must.split(" ")
        for tag in must:
            if tag.strip() and tag not in post["tag_string"]:
                return False
    if config.deny is not None:
        deny = config.deny if isinstance(config.deny, list) else config.deny.split(" ")
        for tag in deny:
            if tag.strip() and tag in post["tag_string"]:
                return False
    return True


def check(config: FilterConfig, post: DanbooruPost):
    result = True
    if config.tags is not None:
        result = tags(config.tags, post)
    return result
