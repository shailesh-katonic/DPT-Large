from pydantic import BaseModel


class PredictSchema(BaseModel):
    data: str
