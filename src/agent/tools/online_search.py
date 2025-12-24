import os
import logging
from langchain_tavily import TavilySearch
from src.config import ONLINE_K

logger = logging.getLogger(__name__)

def get_online_context(query: str) -> str:
    """Retrieves context from the internet using Tavily Search."""
    if not os.environ.get("TAVILY_API_KEY"):
        error_msg = "TAVILY_API_KEY not found in environment. Online search disabled."
        logger.warning(error_msg)
        return error_msg
    
    search_tool = TavilySearch(max_results=ONLINE_K)
    results = search_tool.invoke({"query": query})
    
    # Format the results for context
    return str(results)
