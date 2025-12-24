import pytest
import os

@pytest.fixture(autouse=True)
def mock_env():
    """Ensure tests run with consistent environment variables."""
    # Set dummy keys so models don't crash on init during tests
    os.environ["GOOGLE_API_KEY"] = "test-google-key"
    os.environ["TAVILY_API_KEY"] = "test-tavily-key"
    os.environ["AGENT_MODE"] = "offline"
