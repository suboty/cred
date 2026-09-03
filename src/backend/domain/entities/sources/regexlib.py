from datetime import datetime

from pydantic import BaseModel


__all__ = [
    'RegexLib', 'RegexLibCreate', 'RegexLibUpdate', 'RegexLibFilterSchema'
]


class _RegexLibBase(BaseModel):
    source_id: int
    title: str
    pattern: str
    matching_text: str
    non_matching_text: str
    description: str
    is_dirty: int
    author_name: str
    rating: int
    source_date_modified: datetime


class RegexLib(_RegexLibBase):
    created_at: datetime
    updated_at: datetime


class RegexLibCreate(_RegexLibBase):
    ...


class RegexLibUpdate(_RegexLibBase):
    id: int


class RegexLibFilterSchema(BaseModel):
    rating: int
    is_dirty: int
    key_words_in_title: str
    key_words_in_description: str
