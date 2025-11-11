from pydantic import UUID4, BaseModel, Field
from typing import Optional, Any


class CoconutApiRequestDataTransferObject(BaseModel):
    id: Optional[UUID4] = Field(default=None)

    @classmethod
    def from_domain_model(cls, domain_model: Any) -> "CoconutApiRequestDataTransferObject":
        return cls(id=getattr(domain_model, "id", None))


class CoconutApiResponseDataTransferObject(BaseModel):
    id: UUID4 = Field(...)

    @classmethod
    def from_domain_model(cls, domain_model: Any) -> "CoconutApiResponseDataTransferObject":
        id_value = getattr(domain_model, "id", None)

        if id_value is None:
            raise ValueError("Domain model id cannot be None")

        return cls(id=id_value)
