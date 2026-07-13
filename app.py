from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph_agent import LangGraphRAGAgent
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["LANGSMITH_TRACING"]="true"

app = FastAPI(
    title="Agent Assistant",
    description="Langgraph Agent API",
    version="1.0.0",
)
agent=LangGraphRAGAgent()

class ChatRequest(BaseModel):
    question: str
    thread_id: str="default-thread"

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
def home():
    return {
        "status":"Running",
        "message":"Welcome to Agentic Assistant API"
    }

@app.post("/chat",
          response_model=ChatResponse)
def chat(request:ChatRequest):
    try:
        answer=agent.run(
            question=request.question,
            thread_id=request.thread_id,
        )
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error:{str(e)}")
    