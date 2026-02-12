from pydantic import BaseModel

class CreateSessionSchema(BaseModel):
    class_name: str
    subject: str
    period: int
