import logging
import os
import sqlite3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.config import CHAT_MODEL
from src.agent.state import AgentState
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools.offline_retriever import get_offline_context
from src.agent.tools.online_search import get_online_context

logger = logging.getLogger(__name__)

def research_node(state: AgentState):
    """Retrieves context based on the current mode."""
    last_message = state["messages"][-1]
    query = last_message.content
    
    if state["mode"] == "offline":
        logger.info(f"--- RESEARCHING OFFLINE: {query} ---")
        context = get_offline_context(query)
    else:
        logger.info(f"--- RESEARCHING ONLINE: {query} ---")
        context = get_online_context(query)
        
    return {"context": context}

def generate_node(state: AgentState):
    """Generates an answer based on the retrieved context."""
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL)
    
    context = state["context"]
    messages = state["messages"]
    
    formatted_prompt = SYSTEM_PROMPT.format(context=context)
    
    response = llm.invoke([SystemMessage(content=formatted_prompt)] + list(messages))
    
    # Handle list of content blocks
    content = response.content
    if isinstance(content, list):
        text_blocks = [block["text"] for block in content if isinstance(block, dict) and "text" in block]
        content = "\n".join(text_blocks)
    
    return {"messages": [response], "answer": content}

from langgraph.checkpoint.memory import MemorySaver

def create_graph():
    """Defines and compiles the LangGraph."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("research", research_node)
    workflow.add_node("generate", generate_node)
    
    workflow.set_entry_point("research")
    workflow.add_edge("research", "generate")
    workflow.add_edge("generate", END)
    
    # Add SQLite-based persistence for conversation history
    db_path = os.path.join("data", "checkpoints.sqlite")
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    return workflow.compile(checkpointer=memory)
