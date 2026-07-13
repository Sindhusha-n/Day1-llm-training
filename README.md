# LangChain, LangGraph & Agentic AI Systems

## Overview

This project contains the code developed during LangChain, LangGraph and Agentic AI Systems training.

It covers:

- Basic LLM interaction
- Conversational memory
- RAG
- LangGraph workflow
- Multi-agent system
- FastAPI integration

## Files

- 'hello_agent.py' - Basic LLM example
- 'conversational_chain.py' - Conversation with memory
- 'rag_agent.py' - RAG agent with retrieval and calculator tool
- 'app.py' - FastAPI application
- 'guardrails.py' - Input validation
- 'sample_document.txt' - Sample knowledge document

## Requirements

Install the required packages:

'''bash
pip install -r requirements.yxy
'''

## Run the API

'''bash
uvicorn app:app --reload
'''

## Sample request

'''json
{
    "question": What are the benifits of AI?",
    "thread_id": "demo-thread"
}
'''