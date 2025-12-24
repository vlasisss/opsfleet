import os
import logging
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import EMBEDDING_MODEL, CHROMA_DIR, OFFLINE_K

logger = logging.getLogger(__name__)

def get_offline_context(query: str) -> str:
    """Retrieves context from local documentation using ChromaDB."""
    if not os.path.exists(CHROMA_DIR):
        error_msg = "Offline database not found. Please run ingestion script first."
        logger.error(error_msg)
        return error_msg
    
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    
    docs = vectorstore.similarity_search(query, k=OFFLINE_K)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context
