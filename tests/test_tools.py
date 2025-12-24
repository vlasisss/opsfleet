import pytest
from src.agent.tools.offline_retriever import get_offline_context
from src.agent.tools.online_search import get_online_context

def test_offline_retriever_missing_db(mocker):
    """Verify error message when ChromaDB doesn't exist."""
    mocker.patch("os.path.exists", return_value=False)
    result = get_offline_context("test query")
    assert "not found" in result.lower()

def test_offline_retriever_success(mocker):
    """Test successful retrieval with mocked Chroma."""
    mocker.patch("os.path.exists", return_value=True)
    mock_chroma = mocker.patch("src.agent.tools.offline_retriever.Chroma")
    mock_instance = mock_chroma.return_value
    mock_doc = mocker.Mock()
    mock_doc.page_content = "retrieved info"
    mock_instance.similarity_search.return_value = [mock_doc]
    
    result = get_offline_context("test query")
    assert "retrieved info" in result

def test_online_search_missing_key(mocker):
    """Test graceful handling of missing Tavily key."""
    mocker.patch("os.environ.get", return_value=None)
    result = get_online_context("test query")
    assert "disabled" in result.lower()
