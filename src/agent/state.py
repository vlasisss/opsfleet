from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # The messages in the conversation
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # The current mode: 'offline' or 'online'
    mode: str
    # Context retrieved from tools
    context: str
    # Final answer
    answer: str
