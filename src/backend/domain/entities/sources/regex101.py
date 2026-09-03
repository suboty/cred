from datetime import datetime

from pydantic import BaseModel


__all__ = [
    'Regex101', 'Regex101Create', 'Regex101Update', 'Regex101FilterSchema'
]


class _Regex101Base(BaseModel):
    regex: str
    flags: str | None
    delimiter: str
    dialect: str
    title: str
    description: str | None


class Regex101(_Regex101Base):
    created_at: datetime
    updated_at: datetime


class Regex101Create(_Regex101Base):
    ...


class Regex101Update(_Regex101Base):
    id: int


class Regex101FilterSchema(BaseModel):
    flags: str
    delimiter: str
    dialect: str
    key_words_in_title: str
    key_words_in_description: str
