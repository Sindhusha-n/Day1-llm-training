import os
from typing import List
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is not set in the environment variables.")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.9,
)

basic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer clearly."),
        ("human", "{question}"),
    ]
)

basic_chain = basic_prompt | llm | StrOutputParser()

store = {} # For storing chat history


def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


history_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that remembers prior conversation context."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

memory_chain = history_prompt | llm | StrOutputParser()

conversational_chain = RunnableWithMessageHistory(
    memory_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
    output_messages_key=None,
)

sample_path = "sample_document.txt"

with open(sample_path, "r", encoding="utf-8") as f:
    sample_text = f.read()

#Splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
documents = text_splitter.create_documents([sample_text])

# A simple document chain that uses the chunks to answer questions.
retrieval_prompt = ChatPromptTemplate.from_template(
    """
You are given the following document chunks:
{context}

Answer the user's question using the document content.
Question: {question}
"""
)

retrieval_chain = (
    {
        "context": lambda x: "\n\n".join([doc.page_content for doc in documents]),
        "question": lambda x: x["question"],
    }
    | retrieval_prompt
    | llm
    | StrOutputParser()
)

class DocumentSummary(BaseModel):
    """Structured summary of a document."""

    title: str = Field(description="A short title for the document")
    summary: str = Field(description="A concise summary of the document")
    key_points: List[str] = Field(description="Three important bullet points")

structured_chain = llm.with_structured_output(DocumentSummary)

# Example
def run_basic_example():
    """Run a simple single-turn LCEL chain."""
    result = basic_chain.invoke({"question": "What is AI?"})
    print("Basic chain output:")
    print(result)
    print()


def run_conversational_example():
    """Run a multi-turn chat using memory."""
    session_id = "demo-session"
    first = conversational_chain.invoke(
        {"question": "My name is Alex. Remember that."},
        config={"configurable": {"session_id": session_id}},
    )
    second = conversational_chain.invoke(
        {"question": "What is my name?"},
        config={"configurable": {"session_id": session_id}},
    )
    print("Conversational chain output:")
    print(first)
    print(second)
    print()


def run_retrieval_example():
    """Answer a question using the sample document."""
    result = retrieval_chain.invoke({"question": "What are the benefits of AI in healthcare and education?"})
    print("Retrieval chain output:")
    print(result)
    print()


def run_structured_output_example():
    """Return a Pydantic-structured response."""
    result = structured_chain.invoke(
        "Summarize the following document in a structured format: " + sample_text
    )
    print("Structured output:")
    print(result)
    print()


if __name__ == "__main__":
    run_basic_example()
    run_conversational_example()
    run_retrieval_example()
    run_structured_output_example()
