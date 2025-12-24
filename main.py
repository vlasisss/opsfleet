import os
import argparse
import logging
import warnings
from dotenv import load_dotenv

# Suppress third-party warnings from Tavily and other libraries
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_tavily")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
from langchain_core.messages import HumanMessage

from src.agent.graph import create_graph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="LangGraph Helper Agent")
    parser.add_argument("query", type=str, nargs="?", help="The question to ask the agent")
    parser.add_argument("--mode", type=str, choices=["offline", "online"], 
                        default=os.getenv("AGENT_MODE", "offline"),
                        help="Operating mode: offline or online")
    
    args = parser.parse_args()
    
    if not args.query:
        print("Please provide a question. Example: python main.py 'How do I add persistence?'")
        return

    mode = args.mode
    logger.info(f"--- MODE: {mode.upper()} ---")
    
    # Check for API keys
    if not os.getenv("GOOGLE_API_KEY"):
        logger.error("Error: GOOGLE_API_KEY not found in environment.")
        return
    
    if mode == "online" and not os.getenv("TAVILY_API_KEY"):
        logger.warning("Warning: TAVILY_API_KEY not found. Online mode may fail.")

    # Create and run the graph
    app = create_graph()
    
    inputs = {
        "messages": [HumanMessage(content=args.query)],
        "mode": mode
    }
    
    config = {"configurable": {"thread_id": "1"}}
    
    for output in app.stream(inputs, config=config):
        # We don't need to print every step unless debugging
        pass
        
    # Get the final answer from the last state
    final_state = app.get_state(config).values
    logger.info("\nANSWER:")
    logger.info(final_state.get("answer", "No answer generated."))

if __name__ == "__main__":
    main()
