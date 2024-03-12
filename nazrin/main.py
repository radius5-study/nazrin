from fire import Fire

from .refetch import refetch
from .scraper import scrape
from .tag import tag
from .clear import clear
from .blip2 import blip2


def wd14tag(config_file: str):
    from . import wd14tagger

    wd14tagger.tag(config_file)


def cli():
    Fire(
        {
            "scrape": scrape,
            "tag": tag,
            "wd14tag": wd14tag,
            "blip2": blip2,
            "clear": clear,
            "refetch": refetch,
        }
    )


if __name__ == "__main__":
    cli()
