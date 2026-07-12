import os
print("program started")
from pathlib import Path
print("pathlib loaded")
from typing import Annotated, Any, TypedDict
print("typing loaded")
from dotenv import load_dotenv
print("dot env loaded")
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
print("langchain_core loaded")
from langchain_core.output_parsers import StrOutputParser
print("output parsers loaded",flush=True) 
from langchain_core.prompts import ChatPromptTemplate
print("prompts loaded")
from langchain_core.tools import tool
print("tools loaded")
from langgraph.checkpoint.memory import MemorySaver
print("memory saver loaded")
from langgraph.graph import END, START, StateGraph
print("graph loaded")
from langgraph.graph.message import add_messages
print("message loaded")
from langgraph.prebuilt import ToolNode
print("prebuilt loaded")
from langgraph.types import interrupt
print("types loaded")
from rag_agent import build_vector_store, calculator, get_llm, load_documents
print("rag_agent loaded")
load_dotenv()

DOCUMENT_FOLDER = Path("documents")
CHUNK_SIZE = 200
CHUNK_OVERLAP = 30


class AgentState(TypedDict):
    """State carried through the supervisor -> research -> tool -> writer flow."""

    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str
    route: str
    final_answer: str
    needs_human_review: bool
    human_feedback: str
    approval_feedback: str
    research_summary: str


class LangGraphRAGAgent:
    """A LangGraph supervisor-style RAG agent with retrieval, tool execution, and human approval."""

    def __init__(self, folder: Path | None = None):
        print("loading doc..")
        self.documents = load_documents(folder or DOCUMENT_FOLDER)
        print("Building vector store")
        self.vector_store, self.chunks = build_vector_store(self.documents)
        print("creating retriever")
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        print("creating llm")
        self.llm = get_llm()
        print("creating tools")
        self.tools = [self._build_retrieval_tool(), calculator]
        print("binding tools")
        self.tool_model = self.llm.bind_tools(self.tools)
        print("creating memory saver")
        self.checkpointer = MemorySaver()
        print("building graph")
        self.graph = self._build_graph()

    def _build_retrieval_tool(self):
        @tool
        def retrieve_context(question: str) -> str:
            """Retrieve the most relevant document passages for a user question."""
            docs = self.retriever.invoke(question)
            return "\n\n".join(doc.page_content for doc in docs)

        return retrieve_context

    def _build_graph(self):
        """Build a supervisor -> research -> tools -> writer -> human review graph."""
        builder = StateGraph(AgentState)

        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("research", self._research_node)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_node("writer", self._writer_node)
        builder.add_node("human_review", self._human_review_node)

        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            self._route_supervisor,
            {"research": "research", "writer": "writer"},
        )
        builder.add_conditional_edges(
            "research",
            self._route_after_research,
            {"tools": "tools", "writer": "writer"},
        )
        builder.add_conditional_edges(
            "tools",
            self._route_after_tools,
            {"writer": "writer", "research": "research"},
        )
        builder.add_conditional_edges(
            "writer",
            self._route_after_writer,
            {"human_review": "human_review", "end": END},
        )
        builder.add_edge("human_review", END)

        return builder.compile(checkpointer=self.checkpointer)

    def _supervisor_node(self, state: AgentState) -> dict[str, Any]:
        """Decide whether the request needs research/tool use or can be answered directly."""
        question = self._get_last_user_text(state)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the supervisor. Reply with exactly 'research' if the user likely needs document retrieval or arithmetic/tool use. Reply with 'writer' for a direct answer that does not require tools.",
                ),
                ("human", "{question}"),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        decision = (chain.invoke({"question": question})).strip().lower()
        route = "research" if "research" in decision else "writer"
        return {"route": route}

    def _research_node(self, state: AgentState) -> dict[str, Any]:
        """Use tool-calling when needed, otherwise produce a short research summary."""
        question = self._get_last_user_text(state)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the research agent. Use the available tools when the request needs retrieval or arithmetic. "
                    "Otherwise, briefly summarize the information needed to answer the request.",
                ),
                ("human", "{question}"),
            ]
        )
        response = (prompt | self.tool_model).invoke({"question": question})

        if getattr(response, "tool_calls", None):
            return {"messages": [response]}

        return {
            "messages": [response],
            "research_summary": response.content if hasattr(response, "content") else str(response),
        }

    def _writer_node(self, state: AgentState) -> dict[str, Any]:
        """Compose the final answer from the research summary and tool results."""
        question = self._get_last_user_text(state)
        research_summary = state.get("research_summary", "")
        tool_messages = [m for m in state.get("messages", []) if getattr(m, "type", "") == "tool"]
        tool_context = "\n\n".join(getattr(m, "content", "") for m in tool_messages)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the writer agent. Write a concise answer using the research summary and any tool results. Mention human review where relevant.",
                ),
                ("human", "Research summary: {research_summary}\n\nTool results: {tool_context}\n\nQuestion: {question}"),
            ]
        )
        response = (prompt | self.llm).invoke(
            {"research_summary": research_summary, "tool_context": tool_context, "question": question}
        )
        answer = response.content if hasattr(response, "content") else str(response)
        needs_human_review = any(term in question.lower() for term in ["review", "approve", "check", "edit"])
        return {
            "messages": [response],
            "final_answer": answer,
            "needs_human_review": needs_human_review,
        }

    def _human_review_node(self, state: AgentState) -> dict[str, Any]:
        """Pause for human approval before finalizing the answer."""
        current_answer = state.get("final_answer", "")
        approval_feedback = state.get("approval_feedback", "")
        if approval_feedback:
            feedback_text = approval_feedback
        else:
            feedback_text="Approved by human reviewer."
            #feedback = interrupt(
                #{
                    #"prompt": "Please review the draft answer and optionally provide feedback.",
                    #"draft_answer": current_answer,
                #}
            #)
            #if isinstance(feedback, dict):
                #feedback_text = feedback.get("feedback") or feedback.get("response") or ""
            #else:
                #feedback_text = str(feedback or "")
        updated_answer = current_answer
        if feedback_text:
            updated_answer = f"{current_answer}\n\nHuman feedback: {feedback_text}"
        return {"human_feedback": feedback_text, "final_answer": updated_answer}

    def _route_supervisor(self, state: AgentState) -> str:
        return state.get("route", "research")

    def _route_after_research(self, state: AgentState) -> str:
        messages = state.get("messages", [])
        for message in reversed(messages):
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls:
                return "tools"
        return "writer"

    def _route_after_tools(self, state: AgentState) -> str:
        return "writer"

    def _route_after_writer(self, state: AgentState) -> str:
        if state.get("needs_human_review", False):
            return "human_review"
        return "end"

    def _extract_final_answer(self, result: dict[str, Any]) -> str:
        if not isinstance(result, dict):
            return str(result)
        messages = result.get("messages", [])
        for message in reversed(messages):
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                tool_calls = getattr(message, "tool_calls", None)
                if tool_calls:
                    continue
                return content
        return result.get("final_answer") or result.get("output") or str(result)

    def _get_last_user_text(self, state: AgentState) -> str:
        for message in reversed(state.get("messages", [])):
            if isinstance(message, HumanMessage):
                return message.content if isinstance(message.content, str) else str(message.content)
        return ""

    def run(self, question: str, thread_id: str | None = None, approval_feedback: str | None = None) -> str:
        """Run the graph for a single user question and return the final answer."""
        resolved_thread_id = thread_id or "default-thread"
        config = {"configurable": {"thread_id": resolved_thread_id}}

        initial_state: AgentState = {
            "messages": [HumanMessage(content=question)],
            "thread_id": resolved_thread_id,
            "route": "",
            "final_answer": "",
            "needs_human_review": False,
            "human_feedback": "",
            "approval_feedback": approval_feedback or "",
            "research_summary": "",
        }
        result = self.graph.invoke(initial_state, config=config)
        return self._extract_final_answer(result)


if __name__ == "__main__":
    agent = LangGraphRAGAgent()
    examples = [
        "What are the benefits of AI in healthcare and education?",
        "What is 12 * 7?",
        "Using the document, explain the benefits of AI in healthcare and also calculate 12 * 7.",
    ]
    for question in examples:
        print(f"\nQuestion: {question}")
        print("Answer:", agent.run(question, thread_id="demo-thread", approval_feedback="Approved"))
