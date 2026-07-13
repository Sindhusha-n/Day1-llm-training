from fastapi import HTTPException
MAX_QUESTION_LENGTH = 1000

def validate_question(question:str)->str:
    """Validate the user question before passing to the LangGraph agent."""
    if not question:
        raise HTTPException(status_code=400, detail="Question can't be empty.")
    
    question=question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question can't be empty.")
    
    if len(question)>MAX_QUESTION_LENGTH:
        raise HTTPException(status_code=400, detail=f"Question length exceeds {MAX_QUESTION_LENGTH} characters.")
    
    blocked_words=["rm -rf", "shutdown", "delete database", "drop database", "hack", "exploit", "malware"]

    lower_question=question.lower()

    for word in blocked_words:
        if word in lower_question:
            raise HTTPException(status_code=400, detail="request contains restricted content.")
        
        return question 