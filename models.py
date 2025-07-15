from pydantic import BaseModel

class ChatRequest(BaseModel):
    prompt: str
    companyid: int

class ChatResponse(BaseModel):
    output: str
