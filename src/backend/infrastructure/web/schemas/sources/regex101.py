from datetime import datetime

from pydantic import BaseModel


__all__ = [
    'Regex101Response', 'Regex101CreateRequest',
    'Regex101UpdateRequest',
]


class _Regex101Base(BaseModel):
    permalink: str
    regex: str
    flags: str | None
    delimiter: str
    dialect: str
    title: str
    description: str | None


class Regex101Response(_Regex101Base):
    created_at: datetime
    updated_at: datetime


class Regex101CreateRequest(_Regex101Base):
    ...


class Regex101UpdateRequest(_Regex101Base):
    id: int
