from enum import Enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel


__all__ = [
    'Sources', 'RegexResponse', 'RegexCreateRequest',
    'RegexUpdateRequest',
]


class Sources(str, Enum):
    regexlib = "regexlib"
    regex101 = "regex101"


class _RegexBase(BaseModel):
    regex: str
    source: Sources
    params: dict[Any, Any]
    regex_metadata: dict[Any, Any]


class RegexResponse(_RegexBase):
    created_at: datetime
    updated_at: datetime


class RegexCreateRequest(_RegexBase):
    ...


class RegexUpdateRequest(_RegexBase):
    id: int
