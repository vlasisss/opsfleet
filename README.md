# LangGraph Helper Agent

An AI agent designed to help developers work with LangGraph and LangChain by answering practical questions. Supports both **Offline (RAG)** and **Online (Real-time Search)** modes.

## Architecture

The agent is built using **LangGraph V1** with a simple research-and-generate orchestration:

- **State Management**: Uses `AgentState` (TypedDict) to track messages, mode, context, and final answers.
- **Nodes**:
  - `research`: Determines the mode and fetches context using `offline_retriever` (ChromaDB) or `online_search` (Tavily).
  - `generate`: Uses **Gemini 3 Flash** to synthesize an answer from the retrieved context.
- **Persistence**: Employs `SqliteSaver` as a checkpointer to support stateful interactions across container runs. Conversation history is stored in `data/checkpoints.sqlite`.

## Data Portability

By default, the `data/` directory is ignored by Git to keep the repository lightweight. 
- **Recommendation**: Each developer should run the ingestion once (as described in Setup).
- **Shortcut**: If you want to skip the embedding wait, you can manually copy the `data/chroma_db` folder from a colleague. Vector databases are portable as long as the OS and architecture are similar (e.g., within Docker).

## Operating Modes

### Offline Mode (Default)
- **Mechanism**: Uses a local RAG (Retrieval-Augmented Generation) system.
- **Data Source**: Documentation from `llms.txt` resources (LangGraph and LangChain).
- **Data Freshness**: Users can refresh data by re-running the ingestion script.

### Online Mode
- **Mechanism**: Real-time web search integration.
- **Services**: Leveraging [Tavily Search](https://tavily.com/) for high-quality developer-focused results.

## Data Preparation Strategy

1. **Ingestion**: The `src/utils/ingest.py` script downloads the latest documentation directly from official sources.
2. **Processing**: Documents are split using `RecursiveCharacterTextSplitter` into large, optimized chunks (8,000 chars, 800 char overlap) to minimize API calls.
3. **Indexing**: Chunks are embedded using **Gemini gemini-embedding-001** and stored in a local **ChromaDB** instance (`data/chroma_db`).
4. **Rate Limiting**: The ingestion script strictly handles Gemini Free Tier limits (30,000 TPM) using batching and explicit wait intervals.

## Setup Instructions

### Prerequisites
- Docker and Docker Compose installed.
- [Google AI Studio API Key](https://aistudio.google.com/app/apikey) (Free).
- [Tavily API Key](https://tavily.com/) (Free).

### Configuration
1. Clone the repository.
2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
3. Fill in your API keys in the `.env` file.

### Running via Docker (Recommended)

**1. Build Information (First Time or after dependency changes):**
```bash
docker compose build
```

**2. Prepare Documentation (Offline Database):**
```bash
docker compose run --rm ingest
```

**3. Ask a Question (Offline):**
```bash
docker compose run --rm agent "How do I add persistence?"
```

**4. Ask a Question (Online):**
```bash
docker compose run --rm agent "What are the latest LangGraph features?" --mode online
```

### Running Locally (Alternative)
1. Install dependencies: `pip install -r requirements.txt`
2. Run ingestion: `python src/utils/ingest.py`
3. Run agent: `python main.py "How do I handle errors?" --mode offline`

## Automated Testing

The project includes a suite of unit and integration tests using `pytest` and `pytest-mock`.

**Run tests via Docker:**
```bash
docker compose run --rm --entrypoint bash agent -c "PYTHONPATH=. pytest"
```

**Run tests locally:**
```bash
PYTHONPATH=. pytest
```

## Example Questions
- "How do I add persistence to a LangGraph agent?"
- "What's the difference between StateGraph and MessageGraph?"
- "Show me how to implement human-in-the-loop with LangGraph"
- "How do I handle errors and retries in LangGraph nodes?"
- "What are best practices for state management in LangGraph?"

## Version Specifications
- **Python**: 3.11
- **LangGraph**: V1
- **LangChain**: latest (compatible with V1)
- **Model**: Gemini 3 Flash (released Dec 17, 2025)
- **Embeddings**: gemini-embedding-001 (latest, 3072 dimensions)
