from typing import Dict

from pydantic import BaseModel, Field


class HealthComponentDataTransferObject(BaseModel):
    status: bool = Field(...)
    message: str = Field(...)


class LivenessResponseDataTransferObject(BaseModel):
    status: str = Field(...)


class ReadinessResponseDataTransferObject(BaseModel):
    status: str = Field(...)
    checks: Dict[str, HealthComponentDataTransferObject] = Field(default_factory=dict)
