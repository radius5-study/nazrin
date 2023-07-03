import os

from dotenv import load_dotenv

load_dotenv()

DANBOORU_API_KEY = os.environ.get("DANBOORU_API_KEY", "")
DANBOORU_USER_ID = os.environ.get("DANBOORU_USER_ID", "")
DANBOORU_URL = "https://danbooru.donmai.us"

IMAGE_EXTS = ["jpg", "jpeg", "png", "webp", "tiff"]
