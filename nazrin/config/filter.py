from typing import *

from nazrin.config.base import BaseConfig


class TagFilter(BaseConfig):
    must: Optional[Union[List[str], str]] = None
    deny: Optional[Union[List[str], str]] = None


class FilterConfig(BaseConfig):
    tags: Optional[TagFilter] = None
