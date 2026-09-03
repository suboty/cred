from enum import Enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel


__all__ = [
    'Sources', 'Regex', 'RegexCreate', 'RegexUpdate', 'RegexesFilterSchema'
]


class Sources(str, Enum):
    regexlib = "regexlib"
    regex101 = "regex101"


class _RegexBase(BaseModel):
    regex: str
    source: Sources
    params: dict[Any, Any]
    metadata: dict[Any, Any]


class Regex(_RegexBase):
    created_at: datetime
    updated_at: datetime


class RegexCreate(_RegexBase):
    ...


class RegexUpdate(_RegexBase):
    id: int


class RegexesFilterSchema(BaseModel):
    source: str
