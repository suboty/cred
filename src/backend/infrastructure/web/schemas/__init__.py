from pydantic import BaseModel, Field


class SuccessResponse(BaseModel):
    success: bool = Field(True, description="Operation`s status")
    message: str = Field("Request is success", description="Message")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error Message")
