"""
Graph orchestration using LangGraph.

This module sets up the global vector store and LLM, builds a StateGraph with
conditional routing, and provides a function to handle a user query by invoking the graph.
"""
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from .agents import (
    WorkflowState,
    planning_agent,
    retrieval_agent,
    tool_execution_agent,
    synthesis_agent,
    set_vector_store,
    set_llm,
)
from .rag import initialize_vector_store, create_local_llm

# Build the graph on import to avoid repeated compilation

def initialise_system(pdf_path: str, faiss_directory: str = "./faiss_store") -> None:
    """
    Initialise the vector store and LLM and register them with the agents.
    Must be called before using handle_user_query.
    """
    vs = initialize_vector_store(pdf_path=pdf_path, faiss_directory=faiss_directory)
    llm = create_local_llm(model_id="google/flan-t5-base", max_new_tokens=220)
    set_vector_store(vs)
    set_llm(llm)


def build_graph() -> StateGraph:
    """Construct the LangGraph for the multi-agent workflow."""
    graph = StateGraph(WorkflowState)
    graph.add_node("planner", planning_agent)
    graph.add_node("rag", retrieval_agent)
    graph.add_node("tool", tool_execution_agent)
    graph.add_node("synth", synthesis_agent)
    graph.set_entry_point("planner")
    # Function to route to rag or tool based on state.operation
    def route_agent(state: WorkflowState) -> str:
        return "tool" if state.get("operation") == "tool" else "rag"
    graph.add_conditional_edges(
        "planner",
        route_agent,
        {"rag": "rag", "tool": "tool"},
    )
    graph.add_edge("rag", "synth")
    graph.add_edge("tool", "synth")
    graph.add_edge("synth", END)
    return graph

# Compile graph once
_graph = None


def get_app_graph() -> StateGraph:
    global _graph
    if _graph is None:
        g = build_graph()
        _graph = g.compile()
    return _graph


def handle_user_query(user_query: str) -> Dict[str, Any]:
    """
    Invoke the compiled graph on a user query and return a dict with answer and metadata.
    The system must be initialised via `initialise_system()` before calling this.
    """
    graph = get_app_graph()
    init_state: WorkflowState = {
        "user_query": user_query,
        "operation": "",
        "plan": "",
        "react_steps": [],
        "retrieved_context": "",
        "citations": [],
        "tool_name": "",
        "tool_input": {},
        "tool_result": {},
        "final_answer": "",
        "error": "",
    }
    out = graph.invoke(init_state)
    return {
        "query": user_query,
        "operation": out.get("operation"),
        "plan": out.get("plan"),
        "answer": out.get("final_answer"),
        "citations": out.get("citations", []),
        "react_steps": out.get("react_steps", []),
        "tool_name": out.get("tool_name", ""),
        "tool_result": out.get("tool_result", {}),
    }
