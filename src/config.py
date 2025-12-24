import os

# --- Models ---
# The latest Gemini embedding model (released July 2025)
EMBEDDING_MODEL = "models/gemini-embedding-001"
# The latest Gemini chat model (released Dec 17, 2025)
CHAT_MODEL = "gemini-3-flash-preview"

# --- Retrieval Parameters ---
# Number of chunks to retrieve for offline RAG
OFFLINE_K = 5
# Number of results to retrieve for online search
ONLINE_K = 5

# --- Ingestion Parameters ---
CHUNK_SIZE = 8000
CHUNK_OVERLAP = 800
BATCH_SIZE = 10
INGEST_WAIT_TIME = 60  # seconds between batches to stay under TPM limits

# --- Storage ---
DATA_DIR = "data"
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")

# --- URLs ---
LANGGRAPH_URLS = [
    "https://langchain-ai.github.io/langgraph/llms-full.txt"
]
LANGCHAIN_URLS = [
    "https://docs.langchain.com/llms-full.txt"
]
