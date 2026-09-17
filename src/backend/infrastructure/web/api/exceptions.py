from typing import Dict

from fastapi import status
from fastapi.exceptions import HTTPException


class FormFieldValidationException(HTTPException):
    fields = None

    def __init__(self, fields: Dict, *args, **kwargs):
        kwargs["status_code"] = status.HTTP_422_UNPROCESSABLE_ENTITY
        super().__init__(*args, **kwargs)
        self.fields = fields
