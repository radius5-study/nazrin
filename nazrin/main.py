from fire import Fire

from .scraper import scrape
from .tag import tag


def wd14tag(config_file: str):
    from . import wd14tagger

    wd14tagger.tag(config_file)


def cli():
    Fire({"scrape": scrape, "tag": tag, "wd14tag": wd14tag})


if __name__ == "__main__":
    cli()
