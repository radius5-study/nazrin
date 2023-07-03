from datetime import datetime
from typing import *


class DanbooruPost(TypedDict):
    id: int
    created_at: datetime
    uploader_id: int
    score: int
    source: str
    md5: Optional[str]
    last_comment_bumped_at: Optional[datetime]
    rating: str
    image_width: int
    image_height: int
    tag_string: str
    fav_count: int
    file_ext: str
    last_noted_at: Optional[datetime]
    parent_id: Optional[int]
    has_children: bool
    approver_id: Optional[int]
    tag_count_general: int
    tag_count_artist: int
    tag_count_character: int
    tag_count_copyright: int
    file_size: int
    up_score: int
    down_score: int
    is_pending: bool
    is_flagged: bool
    is_deleted: bool
    tag_count: int
    updated_at: datetime
    is_banned: bool
    pixiv_id: Optional[int]
    last_commented_at: Optional[datetime]
    has_active_children: bool
    bit_flags: int
    tag_count_meta: int
    has_large: bool
    has_visible_children: bool
    tag_string_general: str
    tag_string_character: str
    tag_string_copyright: str
    tag_string_artist: str
    tag_string_meta: str
    file_url: Optional[str]
    large_file_url: Optional[str]
    preview_file_url: Optional[str]


class ImageMeta(TypedDict):
    tags: List[str]
    tags_character: List[str]
    tags_copyright: List[str]
    tags_artist: List[str]
    tags_meta: List[str]
    rating: Literal["g", "s", "q", "e"]
    score: int
    up_score: int
    down_score: int
    file_ext: str
    file_url: str
    filename: str
    wd14tags: Optional[List[str]]
