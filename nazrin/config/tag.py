from typing import *

from .base import BaseConfig


class TagConfig(BaseConfig):
    directory: str = "data"
    recursive: bool = False
    remove_underline: bool = True
    tag_character: bool = True
    tag_copyright: bool = True
    tag_artist: bool = True
    tag_meta: bool = False
    tag_rating: bool = True
    score_map: Dict[str, int] = {
        "~,800": "masterpiece",
        "800,200": "best quality",
        "200,100": "high quality",
        "100,50": "normal quality",
        "50,10": "low quality",
        "10,~": "worst quality",
    }
    use_wd14tag: bool = True
