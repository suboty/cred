from datetime import datetime

from pydantic import BaseModel, ConfigDict

__all__ = [
    'RegexLib', 'RegexLibCreate', 'RegexLibUpdate', 'RegexLibFilterSchema'
]


class _RegexLibBase(BaseModel):
    source_id: int
    title: str | None
    pattern: str | None
    matching_text: str | None
    non_matching_text: str | None
    description: str | None
    is_dirty: int
    author_name: str | None
    rating: int
    source_date_modified: datetime


class RegexLib(_RegexLibBase):
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(frozen=True)


class RegexLibCreate(_RegexLibBase):
    ...


class RegexLibUpdate(_RegexLibBase):
    id: int


class RegexLibFilterSchema(BaseModel):
    rating: int = None
    is_dirty: int = None
    key_words_in_title: str = None
    key_words_in_description: str = None
