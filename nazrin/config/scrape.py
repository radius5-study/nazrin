from typing import *

from .base import BaseConfig
from .filter import FilterConfig


class ScrapeSubset(BaseConfig):
    fs: Literal["local"] = "local"
    outpath: str = "data"
    limit: int = 100
    task_size: int = 100
    tags: str = "1girl"
    save_original_size: bool = False
    convert: Optional[str] = None
    filter: Optional[FilterConfig] = None


class ScrapeConfig(ScrapeSubset):
    max_workers: int = 1
    multi_worker_mode: Literal["task", "subset"] = "task"
    subsets: List[ScrapeSubset] = []
