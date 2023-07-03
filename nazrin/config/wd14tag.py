from typing import *

from .base import BaseConfig


class WD14TagConfig(BaseConfig):
    directory: str = "data"
    recursive: bool = False
    remove_underline: bool = True
    score_threshold: float = 0.35
    model_repo: str = "SmilingWolf/wd-v1-4-vit-tagger"
    batch_size: int = 1
    max_workers: int = 4
