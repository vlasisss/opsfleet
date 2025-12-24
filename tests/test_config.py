from src import config

def test_config_defaults():
    """Verify that default configuration constants are present."""
    assert hasattr(config, "EMBEDDING_MODEL")
    assert hasattr(config, "CHAT_MODEL")
    assert "gemini" in config.CHAT_MODEL.lower()
    assert config.BATCH_SIZE > 0
    assert config.CHUNK_SIZE > config.CHUNK_OVERLAP
