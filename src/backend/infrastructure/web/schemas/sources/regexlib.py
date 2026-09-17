from datetime import datetime

from pydantic import BaseModel


__all__ = [
    'RegexLibResponse', 'RegexLibCreateRequest',
    'RegexLibUpdateRequest',
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


class RegexLibResponse(_RegexLibBase):
    created_at: datetime
    updated_at: datetime


class RegexLibCreateRequest(_RegexLibBase):
    ...


class RegexLibUpdateRequest(_RegexLibBase):
    id: int
