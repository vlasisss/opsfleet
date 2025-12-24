from langchain_core.messages import HumanMessage
from src.agent.graph import research_node, generate_node
from src.agent.state import AgentState

def test_research_node_offline(mocker):
    """Verify that research_node calls the offline retriever in offline mode."""
    mock_offline = mocker.patch("src.agent.graph.get_offline_context", return_value="offline context")
    
    state: AgentState = {
        "messages": [HumanMessage(content="test query")],
        "mode": "offline",
        "context": "",
        "answer": ""
    }
    
    result = research_node(state)
    assert result["context"] == "offline context"
    mock_offline.assert_called_once_with("test query")

def test_research_node_online(mocker):
    """Verify that research_node calls the online search in online mode."""
    mock_online = mocker.patch("src.agent.graph.get_online_context", return_value="online context")
    
    state: AgentState = {
        "messages": [HumanMessage(content="test query")],
        "mode": "online",
        "context": "",
        "answer": ""
    }
    
    result = research_node(state)
    assert result["context"] == "online context"
    mock_online.assert_called_once_with("test query")

def test_generate_node(mocker):
    """Test response generation with a mocked LLM."""
    mock_llm = mocker.patch("src.agent.graph.ChatGoogleGenerativeAI")
    mock_instance = mock_llm.return_value
    mock_instance.invoke.return_value.content = "mocked answer"
    
    state: AgentState = {
        "messages": [HumanMessage(content="test query")],
        "mode": "offline",
        "context": "sample context",
        "answer": ""
    }
    
    result = generate_node(state)
    assert result["answer"] == "mocked answer"
    assert len(result["messages"]) == 1
