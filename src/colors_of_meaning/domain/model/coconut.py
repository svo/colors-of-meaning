from pydantic import UUID4, BaseModel, Field
from typing import Optional


class Coconut(BaseModel):
    id: Optional[UUID4] = Field(default=None, alias="id")
