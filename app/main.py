from fastapi import FastAPI

from .schemas import ChatRequest, ChatResponse, HealthResponse
from . import dialogue

app = FastAPI(title="SHL Assessment Recommendation Agent")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    messages = [m.model_dump() for m in req.messages]
    result = dialogue.handle_chat(messages)
    return ChatResponse(**result)
