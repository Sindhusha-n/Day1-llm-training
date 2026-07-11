import os
import re
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r"`langchain-community` is being sunset.*",
    category=DeprecationWarning,
)

from dotenv import load_dotenv
from langchain.agents import create_agent

try:
    from langchain.agents import AgentExecutor
except ImportError:  # Compatibility fallback for newer LangChain versions
    class AgentExecutor:
        def __init__(self, agent, verbose: bool = False):
            self.agent = agent
            self.verbose = verbose

        def invoke(self, inputs):
            return self.agent.invoke(inputs)


def create_tool_calling_agent(model, tools, system_prompt):
    return create_agent(model=model, tools=tools, system_prompt=system_prompt)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

load_dotenv()

DOCUMENT_FOLDER = Path("documents")
DEFAULT_DOCUMENT = Path("sample_document.txt")
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30


def load_documents(folder: Path | None = None):
    """Load all text files from a folder, or fall back to the sample document."""
    target = folder or DOCUMENT_FOLDER
    if target.exists() and target.is_dir():
        files = sorted(target.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No .txt files found in {target}")
        docs = []
        for file_path in files:
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs.extend(loader.load())
        return docs

    if DEFAULT_DOCUMENT.exists():
        loader = TextLoader(str(DEFAULT_DOCUMENT), encoding="utf-8")
        return loader.load()

    raise FileNotFoundError("No document source found")


def build_vector_store(documents):
    """Split documents into chunks and create a FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding=embeddings)
    return vector_store, chunks


def get_llm() -> ChatGroq:
    """Create the Groq-backed chat model."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in the environment variables.")
    return ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0.2)


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely."""
    expression = expression.strip()
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        return "Error: only basic arithmetic expressions are supported"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception as exc:
        return f"Error: {exc}"
    return str(result)


class RAGAgent:
    def __init__(self, folder: Path | None = None):
        self.documents = load_documents(folder)
        self.vector_store, self.chunks = build_vector_store(self.documents)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        self.llm = get_llm()
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant. Use the retrieved document context when relevant.
If the user asks a math question, use the calculator tool.

Context:
{context}

Question: {question}
"""
        )
        self.rag_tool = self._build_rag_tool()
        self.agent = self._build_agent()

    def _build_rag_tool(self):
        @tool
        def retrieve_context(question: str) -> str:
            """Retrieve the most relevant document passages for a user question."""
            docs = self.retriever.invoke(question)
            return "\n\n".join(doc.page_content for doc in docs)

        return retrieve_context

    def _build_agent(self):
        tools = [self.rag_tool, calculator]
        agent = create_tool_calling_agent(
            model=self.llm,
            tools=tools,
            system_prompt=(
                "You are a helpful assistant. Use the retrieval tool whenever the user asks about the document content. "
                "Use the calculator tool for arithmetic. Combine the retrieved context and tool results into a concise answer."
            ),
        )
        return AgentExecutor(agent=agent, verbose=True)

    def _extract_final_answer(self, result) -> str:
        if isinstance(result, dict):
            messages = result.get("messages", [])
            for message in reversed(messages):
                content = getattr(message, "content", None)
                if isinstance(content, str) and content.strip():
                    tool_calls = getattr(message, "tool_calls", None)
                    if tool_calls:
                        continue
                    return content
            return result.get("output") or result.get("result") or str(result)
        return str(result)

    def run(self, question: str) -> str:
        result = self.agent.invoke({"messages": [{"role": "user", "content": question}]})
        return self._extract_final_answer(result)


if __name__ == "__main__":
    agent = RAGAgent()
    examples = [
        "What are the benefits of AI in healthcare and education?",
        "What is 12 * 7?",
        "Using the document, explain the benefits of AI in healthcare and also calculate 12 * 7.",
    ]
    for question in examples:
        print(f"\nQuestion: {question}")
        print("Answer:", agent.run(question))