from fire import Fire

from .scraper import scrape
from .tag import tag
from .wd14tagger import tag as wd14tag


def cli():
    Fire({"scrape": scrape, "tag": tag, "wd14tag": wd14tag})


if __name__ == "__main__":
    cli()
