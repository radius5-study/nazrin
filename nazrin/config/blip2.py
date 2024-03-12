from typing import *

from .base import BaseConfig


class Blip2Config(BaseConfig):
    directory: str = "data"
    recursive: bool = False
    model_type: str = "pretrain_flant5xl"
    caption_type: str = "Beam Search"
    length_penalty: float = 1.0
    repetition_penalty: float = 1.5
    temperature: float = 1.0
