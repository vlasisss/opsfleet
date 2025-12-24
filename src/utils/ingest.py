import os
import requests
import time
import logging
import warnings

# Suppress third-party warnings from Tavily and other libraries
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_tavily")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
import warnings

# Suppress third-party warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_tavily")
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import (
    LANGGRAPH_URLS, 
    LANGCHAIN_URLS, 
    DATA_DIR, 
    CHROMA_DIR, 
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    BATCH_SIZE, 
    EMBEDDING_MODEL,
    INGEST_WAIT_TIME
)

logger = logging.getLogger(__name__)

def download_docs():
    """Downloads documentation and saves to local data directory."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    docs_content = []
    
    for url in LANGGRAPH_URLS + LANGCHAIN_URLS:
        logger.info(f"Downloading {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            content = response.text
            docs_content.append(Document(page_content=content, metadata={"source": url}))
            
            # Also save file locally for reference/offline backup
            filename = url.split("/")[-1]
            if "langchain-ai.github.io" in url:
                filename = f"langgraph_{filename}"
            else:
                filename = f"langchain_{filename}"
            
            with open(os.path.join(DATA_DIR, filename), "w") as f:
                f.write(content)
                
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            
    return docs_content

def ingest_data():
    """Splits and indexes downloaded documentation."""
    docs = download_docs()
    if not docs:
        logger.warning("No documentation found to ingest.")
        return

    logger.info("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)

    logger.info(f"Indexing {len(splits)} chunks into ChromaDB at {CHROMA_DIR}...")
    
    # We use Google Gemini for embeddings (latest model as of July 2025)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    # Batch the ingestion to avoid rate limits
    for i in range(0, len(splits), BATCH_SIZE):
        batch = splits[i:i + BATCH_SIZE]
        logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(splits)-1)//BATCH_SIZE + 1}...")
        
        while True:
            try:
                if i == 0:
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=CHROMA_DIR
                    )
                else:
                    vectorstore.add_documents(batch)
                break
            except Exception as e:
                if "429" in str(e):
                    logger.warning("Rate limit reached (429) or Tier limit. Waiting 60 seconds...")
                    time.sleep(60)
                else:
                    raise e
        
        # Consistent wait between batches to stay under TPM limits
        time.sleep(INGEST_WAIT_TIME)
    
    logger.info("Ingestion complete!")

if __name__ == "__main__":
    from dotenv import load_dotenv
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    load_dotenv()
    ingest_data()
