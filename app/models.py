# app/models.py
from pydantic import BaseModel

class ContextRequest(BaseModel):
    question: str
    content: str
