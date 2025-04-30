#!/bin/bash
set -e

##############################################################################
# PrivateLLM Q&A Service Installation Script
#
# This script automates the deployment of a document question-answering service
# using Ollama for LLM processing and FastAPI for the web interface.
#
# Features:
# - Installs Ollama server and required models
# - Sets up Python virtual environment with all dependencies
# - Configures systemd services for automatic startup
# - Implements port forwarding from 80 to 8000
# - Includes retry logic for model downloads
#
# Requirements:
# - Ubuntu/Debian-based system
# - Root privileges
# - Internet connectivity
# - GPU (Preferred)
#
# Usage:
#   sudo ./install.sh
##############################################################################

# Configuration
SERVICE_NAME="PrivateLLM"
INSTALL_DIR="/etc/privatellm"
DATA_DIR="/var/lib/privatellm"
VENV_DIR="/opt/privatellm-venv"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
OLLAMA_MODELS=("tinyllama" "phi" "mistral" "gemma:2b" "mistral:7b-instruct-v0.2-q4_K_M")

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
  echo "This script must be run as root. Try 'sudo ./install.sh'"
  exit 1
fi

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y python3-pip python3-venv python3-full libopenblas-dev

# Create and activate virtual environment
echo "Creating Python virtual environment..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# Install Ollama
echo "Installing Ollama server..."
curl -fsSL https://ollama.com/install.sh | sh

# Configure Ollama service
echo "Configuring Ollama systemd service..."
sudo tee /etc/systemd/system/ollama.service <<EOF
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
User=root
ExecStart=/usr/local/bin/ollama serve
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start Ollama with retries
echo "Starting Ollama service..."
systemctl enable ollama
systemctl start ollama

# Wait for Ollama to be fully ready (up to 30 seconds)
echo "Waiting for Ollama to start..."
for i in {1..30}; do
    if curl -s http://localhost:11434 >/dev/null; then
        echo "Ollama is ready!"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo "Error: Ollama failed to start after 30 seconds"
        journalctl -u ollama --no-pager -n 50
        exit 1
    fi
done

# Download models with retries
echo "Downloading Ollama models (this may take a while)..."
for model in "${OLLAMA_MODELS[@]}"; do
    echo "Pulling ${model}..."
    for attempt in {1..3}; do
        if ollama pull "${model}"; then
            break
        else
            echo "Attempt ${attempt} failed, retrying in 5 seconds..."
            sleep 5
            systemctl restart ollama
            sleep 5
        fi
        if [ $attempt -eq 3 ]; then
            echo "Error: Failed to pull model ${model} after 3 attempts"
            exit 1
        fi
    done
done

# Create directories
echo "Creating installation directories..."
mkdir -p "${INSTALL_DIR}" "${DATA_DIR}/documents"
chown -R ollama:ollama "${INSTALL_DIR}" "${DATA_DIR}"

# Create Python service file
echo "Creating Python service file..."
cat > "${INSTALL_DIR}/qa_service.py" << 'EOL'
#!/usr/bin/env python3
import os
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import uvicorn
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- Configuration Loading ---
load_dotenv() # Load environment variables from .env file

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class DocumentInfo(BaseModel):
    id: str
    chunks: int

class QAResponse(BaseModel):
    answer: Any # Langchain result can be complex
    context: str
    documents_used: List[str] = Field(default_factory=list)

class ChunkedQAResult(BaseModel):
    documents: List[str]
    answer: Any # Langchain result can be complex

class ChunkedQAResponse(BaseModel):
    results: List[ChunkedQAResult]
    total_documents: int
    context: str

# --- Core Q&A Service Class ---
class QAService:
    def __init__(self):
        logger.info("Initializing QAService...")
        # Configuration from environment variables
        self.max_documents: int = int(os.getenv("MAX_DOCUMENTS", "10"))
        self.document_storage: str = os.getenv("DOCUMENT_STORAGE", "/var/lib/privatellm/documents")
        self.language_model_name: str = os.getenv("LANGUAGE_MODEL", "tinyllama")
        self.embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.ollama_base_url: str = f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}"
        self.fallback_to_general: bool = os.getenv("FALLBACK_TO_GENERAL_KNOWLEDGE", "true").lower() == "true"

        os.makedirs(self.document_storage, exist_ok=True)
        logger.info(f"Document storage directory: {self.document_storage}")

        # Initialize components
        try:
            logger.info(f"Initializing LLM: {self.language_model_name} at {self.ollama_base_url}")
            self.llm = OllamaLLM(model=self.language_model_name, base_url=self.ollama_base_url)
            # Perform a simple invoke to check connection early
            self.llm.invoke("Respond with 'OK'")
            logger.info("LLM connection successful.")

            logger.info(f"Initializing Embedding Model: {self.embedding_model_name}")
            self.embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            logger.info("Embedding Model loaded.")

        except Exception as e:
            logger.exception(f"FATAL: Failed to initialize LLM or Embedding Model: {e}")
            raise RuntimeError(f"Failed to initialize models: {e}") from e

        # In-memory storage for vector stores
        self.vector_stores: Dict[str, FAISS] = {}
        logger.info(f"QAService initialized. Max documents: {self.max_documents}")

    def _get_doc_path(self, doc_id: str) -> str:
        """Helper to get the full path for a document."""
        return os.path.join(self.document_storage, f"{doc_id}.pdf")

    async def upload_document(self, doc_id: str, file: UploadFile) -> Dict[str, Any]:
        """Processes and stores an uploaded PDF document."""
        logger.info(f"Attempting to upload document with id: {doc_id}")
        if len(self.vector_stores) >= self.max_documents:
            logger.warning(f"Upload failed: Maximum document limit ({self.max_documents}) reached.")
            raise HTTPException(status_code=400, detail=f"Maximum {self.max_documents} documents reached")
        if doc_id in self.vector_stores:
            logger.warning(f"Upload failed: Document ID '{doc_id}' already exists.")
            raise HTTPException(status_code=400, detail=f"Document ID '{doc_id}' already exists")

        doc_path = self._get_doc_path(doc_id)
        try:
            # Save document temporarily
            logger.info(f"Saving uploaded file to: {doc_path}")
            file_content = await file.read()
            with open(doc_path, "wb") as f:
                f.write(file_content)
            logger.info(f"File saved successfully.")

            # Process PDF
            logger.info(f"Loading PDF document: {doc_path}")
            loader = PyPDFLoader(doc_path)
            documents = loader.load()
            if not documents:
                 logger.error(f"Failed to load any content from PDF: {doc_path}")
                 raise ValueError("PDF loaded successfully, but no content found.")
            logger.info(f"Loaded {len(documents)} pages from PDF.")

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            split_docs = splitter.split_documents(documents)
            logger.info(f"Split document into {len(split_docs)} chunks.")
            if not split_docs:
                logger.error(f"Text splitting resulted in zero chunks for document: {doc_id}")
                raise ValueError("Document splitting resulted in zero chunks.")

            # Create vector store
            logger.info(f"Creating FAISS vector store for document: {doc_id}")
            self.vector_stores[doc_id] = FAISS.from_documents(split_docs, embedding=self.embedding_model)
            logger.info(f"Vector store created and added for document ID: {doc_id}")

            return {
                "message": f"Document {doc_id} processed successfully",
                "chunks": len(split_docs),
                "status": "uploaded"
            }

        except Exception as e:
            logger.exception(f"Error processing document {doc_id}: {e}")
            # Clean up saved file if processing fails
            if os.path.exists(doc_path):
                try:
                    os.remove(doc_path)
                    logger.info(f"Cleaned up failed upload file: {doc_path}")
                except OSError as rm_err:
                    logger.error(f"Error cleaning up file {doc_path}: {rm_err}")
            # Remove potentially partially created vector store entry
            if doc_id in self.vector_stores:
                 del self.vector_stores[doc_id]
                 logger.info(f"Removed potentially incomplete vector store entry for: {doc_id}")
            raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
        finally:
            await file.close()


    def list_documents(self) -> List[DocumentInfo]:
        """Lists all currently loaded documents and their chunk counts."""
        logger.info("Listing all documents.")
        return [
            DocumentInfo(id=doc_id, chunks=store.index.ntotal)
            for doc_id, store in self.vector_stores.items()
        ]

    def delete_document(self, doc_id: str) -> Dict[str, str]:
        """Deletes a specific document and its vector store."""
        logger.info(f"Attempting to delete document: {doc_id}")
        if doc_id not in self.vector_stores:
            logger.warning(f"Deletion failed: Document '{doc_id}' not found.")
            raise HTTPException(status_code=404, detail="Document not found")

        doc_path = self._get_doc_path(doc_id)
        try:
            # Remove vector store from memory
            del self.vector_stores[doc_id]
            logger.info(f"Removed vector store for document: {doc_id}")

            # Remove PDF file from storage
            if os.path.exists(doc_path):
                os.remove(doc_path)
                logger.info(f"Removed document file: {doc_path}")
            else:
                logger.warning(f"Document file not found for deletion, but removing store: {doc_path}")

            return {"message": f"Document {doc_id} deleted successfully"}
        except Exception as e:
            logger.exception(f"Error deleting document {doc_id}: {e}")
            # Attempt to restore vector store if file deletion failed? Maybe not needed.
            raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

    def purge_all_documents(self) -> Dict[str, str]:
        """Deletes all documents and their vector stores."""
        logger.warning("Attempting to purge all documents.")
        doc_ids_to_purge = list(self.vector_stores.keys()) # Get keys before modifying dict
        try:
             # Clear all vector stores from memory
            self.vector_stores.clear()
            logger.info("Cleared all vector stores from memory.")

            # Delete all PDF files from storage
            purged_files = 0
            failed_files = 0
            for doc_id in doc_ids_to_purge:
                doc_path = self._get_doc_path(doc_id)
                if os.path.exists(doc_path):
                    try:
                        os.remove(doc_path)
                        purged_files += 1
                    except OSError as e:
                        logger.error(f"Error deleting file during purge {doc_path}: {e}")
                        failed_files += 1
                else:
                     logger.warning(f"File not found during purge: {doc_path}")

            logger.info(f"Purged {purged_files} document files. Failed to delete {failed_files} files.")
            return {"message": f"All {len(doc_ids_to_purge)} document entries purged. {purged_files} files deleted."}
        except Exception as e:
            logger.exception(f"Error purging all documents: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to purge documents: {str(e)}")

    def ask_question_single_doc(self, doc_id: str, question: str) -> QAResponse:
        """Answers a question based on a single specified document."""
        logger.info(f"Asking question on single document '{doc_id}': '{question[:50]}...'")
        if doc_id not in self.vector_stores:
            logger.warning(f"Query failed: Document '{doc_id}' not found.")
            raise HTTPException(status_code=404, detail="Document not found")

        try:
            store = self.vector_stores[doc_id]
            retriever = store.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)
            response = qa_chain.invoke(question)
            logger.info(f"Successfully answered question for doc '{doc_id}'.")
            return QAResponse(
                answer=response,
                context=f"document_{doc_id}",
                documents_used=[doc_id]
            )
        except Exception as e:
            logger.exception(f"Error asking question on document {doc_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to query document: {str(e)}")

    def ask_general_question(self, question: str) -> QAResponse:
        """Answers a general question using the LLM without document context."""
        logger.info(f"Asking general knowledge question: '{question[:50]}...'")
        try:
            # Direct LLM query without document context
            response = self.llm.invoke(question)
            logger.info("Successfully answered general knowledge question.")
            return QAResponse(
                answer=response,
                context="general_knowledge",
                documents_used=[]
            )
        except Exception as e:
            logger.exception(f"Error asking general question: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to query LLM for general question: {str(e)}")

    def _create_merged_store(self, doc_ids: List[str]) -> Optional[FAISS]:
        """Helper to create a temporary merged FAISS store from specified doc_ids."""
        if not doc_ids:
            return None

        combined_store: Optional[FAISS] = None
        logger.info(f"Creating merged store for {len(doc_ids)} documents.")

        for doc_id in doc_ids:
            if doc_id not in self.vector_stores:
                logger.warning(f"Document ID '{doc_id}' not found during merge. Skipping.")
                continue

            current_store = self.vector_stores[doc_id]
            # NOTE: The UUID regeneration logic is crucial for FAISS merge if IDs might clash.
            # FAISS.from_documents inherently creates new unique IDs if none are provided.
            # If the original docs within the store need unique IDs preserved across merges,
            # more complex handling is needed. Here, we assume recreating is acceptable.
            # Direct access to `_dict` is fragile, using `docstore.search` is better if possible,
            # but `from_documents` is the standard way to rebuild/add.

            docs_to_add = list(current_store.docstore._dict.values()) # Get Langchain Document objects
            if not docs_to_add:
                 logger.warning(f"No documents found in store for '{doc_id}' during merge. Skipping.")
                 continue


            # Add documents from the current store to the combined store
            # FAISS handles ID generation internally here.
            if combined_store is None:
                 logger.debug(f"Initializing merged store with {len(docs_to_add)} docs from '{doc_id}'")
                 # Need to ensure ids are provided if we want control, otherwise FAISS makes them.
                 # The original code used UUIDs, let's stick to that for explicit uniqueness
                 ids = [str(uuid.uuid4()) for _ in docs_to_add]
                 combined_store = FAISS.from_documents(
                     documents=docs_to_add,
                     embedding=self.embedding_model,
                     ids=ids # Explicitly provide unique IDs
                 )
            else:
                 logger.debug(f"Merging {len(docs_to_add)} docs from '{doc_id}' into existing merged store.")
                 ids = [str(uuid.uuid4()) for _ in docs_to_add]
                 combined_store.add_documents(documents=docs_to_add, ids=ids) # Add with new unique IDs

        if combined_store:
             logger.info(f"Merged store created with {combined_store.index.ntotal} total chunks.")
        else:
             logger.warning("Merged store creation resulted in an empty store.")
        return combined_store


    def ask_all_documents(self, question: str) -> QAResponse:
        """Answers a question using all available documents or falls back to general knowledge."""
        logger.info(f"Asking question across all documents: '{question[:50]}...'")
        if not self.vector_stores:
            logger.warning("No documents available for 'ask_all'.")
            if self.fallback_to_general:
                logger.info("Falling back to general knowledge query.")
                return self.ask_general_question(question)
            else:
                raise HTTPException(status_code=400, detail="No documents available to query.")

        all_doc_ids = list(self.vector_stores.keys())
        try:
            combined_store = self._create_merged_store(all_doc_ids)
            if combined_store is None or combined_store.index.ntotal == 0:
                 logger.error("Failed to create a valid merged store from all documents.")
                 raise HTTPException(status_code=500, detail="Failed to process documents for combined query.")


            retriever = combined_store.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)
            response = qa_chain.invoke(question)
            logger.info("Successfully answered question using all documents.")

            # Explicitly delete the temporary combined store to free memory
            del combined_store
            logger.debug("Cleaned up temporary merged store for ask_all.")

            return QAResponse(
                answer=response,
                context="document_based_all",
                documents_used=all_doc_ids
            )
        except Exception as e:
            logger.exception(f"Error asking question across all documents: {e}")
            # Ensure cleanup happens even on error if combined_store was created
            if 'combined_store' in locals() and combined_store is not None:
                 del combined_store
                 logger.debug("Cleaned up temporary merged store after error in ask_all.")
            raise HTTPException(status_code=500, detail=f"Failed to query all documents: {str(e)}")

    def ask_all_chunked(self, question: str, chunk_size: int = 3) -> ChunkedQAResponse:
        """Answers a question by querying documents in chunks."""
        logger.info(f"Asking question across documents in chunks of {chunk_size}: '{question[:50]}...'")
        if not self.vector_stores:
            logger.warning("No documents available for 'ask_all_chunked'.")
            if self.fallback_to_general:
                 logger.info("Falling back to general knowledge query (chunked format).")
                 general_answer = self.ask_general_question(question)
                 return ChunkedQAResponse(
                     results=[ChunkedQAResult(documents=[], answer=general_answer.answer)],
                     total_documents=0,
                     context="general_knowledge"
                 )
            else:
                raise HTTPException(status_code=400, detail="No documents available to query.")

        all_doc_ids = list(self.vector_stores.keys())
        results: List[ChunkedQAResult] = []
        if chunk_size <= 0:
            chunk_size = 3 # Default to a sensible chunk size
            logger.warning("Invalid chunk_size <= 0 provided, defaulting to 3.")

        try:
            for i in range(0, len(all_doc_ids), chunk_size):
                chunk_doc_ids = all_doc_ids[i:i + chunk_size]
                logger.info(f"Processing chunk {i//chunk_size + 1}: Documents {chunk_doc_ids}")

                # Create a merged store *only* for this chunk
                combined_store = self._create_merged_store(chunk_doc_ids)

                if combined_store is None or combined_store.index.ntotal == 0:
                    logger.warning(f"Skipping chunk {chunk_doc_ids} as merged store is empty.")
                    results.append(ChunkedQAResult(
                        documents=chunk_doc_ids,
                        answer={"result": "Skipped - No content found or error merging chunk."} # Or similar indicator
                    ))
                    continue # Move to the next chunk

                try:
                    retriever = combined_store.as_retriever()
                    qa_chain = RetrievalQA.from_chain_type(llm=self.llm, retriever=retriever)
                    response = qa_chain.invoke(question)
                    results.append(ChunkedQAResult(
                        documents=chunk_doc_ids,
                        answer=response
                    ))
                    logger.info(f"Successfully processed chunk {i//chunk_size + 1}.")
                except Exception as chunk_e:
                     logger.exception(f"Error processing chunk {chunk_doc_ids}: {chunk_e}")
                     results.append(ChunkedQAResult(
                         documents=chunk_doc_ids,
                         answer={"result": f"Error processing chunk: {str(chunk_e)}"}
                     ))
                finally:
                     # Clean up the store for this chunk immediately
                     if combined_store:
                         del combined_store
                         logger.debug(f"Cleaned up temporary merged store for chunk {i//chunk_size + 1}.")

            logger.info("Finished processing all chunks for ask_all_chunked.")
            return ChunkedQAResponse(
                results=results,
                total_documents=len(all_doc_ids),
                context="document_based_chunked"
            )
        except Exception as e:
            logger.exception(f"Error during chunked processing: {e}")
            # Cleanup might be partial if error occurred mid-loop
            raise HTTPException(status_code=500, detail=f"Failed during chunked query: {str(e)}")


# --- FastAPI Application Setup ---
app = FastAPI(
    title="Private LLM Q&A Service",
    description="API for uploading documents and asking questions using Ollama and Langchain.",
    version="1.0.0"
)

# --- Service Instantiation ---
# Create a single instance of the service for the application lifetime
# Dependency Injection can also be used for more complex scenarios
try:
    qa_service = QAService()
except RuntimeError as e:
     logger.fatal(f"Application startup failed: Could not initialize QAService: {e}", exc_info=True)
     # Exit or prevent FastAPI from starting if critical components failed
     raise SystemExit(f"FATAL: QAService initialization failed: {e}")


# --- API Endpoints ---
# These endpoints now delegate to the qa_service instance

@app.post("/upload", summary="Upload PDF Document", status_code=201)
async def upload_pdf_endpoint(doc_id: str, file: UploadFile = File(...), service: QAService = Depends(lambda: qa_service)):
    """
    Uploads a PDF file, processes it, and creates a vector store.
    Requires a unique `doc_id` for referencing the document later.
    """
    return await service.upload_document(doc_id, file)

@app.get("/documents", response_model=List[DocumentInfo], summary="List Uploaded Documents")
async def list_documents_endpoint(service: QAService = Depends(lambda: qa_service)):
    """
    Retrieves a list of all currently processed documents and the number of text chunks for each.
    """
    return service.list_documents()

@app.delete("/document/{doc_id}", summary="Delete Document")
async def delete_document_endpoint(doc_id: str, service: QAService = Depends(lambda: qa_service)):
    """
    Deletes a specific document by its `doc_id`, removing its vector store and the original file.
    """
    return service.delete_document(doc_id)

@app.delete("/purge_all", summary="Purge All Documents")
async def purge_all_documents_endpoint(service: QAService = Depends(lambda: qa_service)):
    """
    Deletes ALL documents, vector stores, and associated files. Use with caution!
    """
    return service.purge_all_documents()


@app.get("/ask/{doc_id}", response_model=QAResponse, summary="Ask Question on Specific Document")
async def ask_question_endpoint(doc_id: str, question: str, service: QAService = Depends(lambda: qa_service)):
    """
    Asks a question specifically against the content of the document identified by `doc_id`.
    """
    return service.ask_question_single_doc(doc_id, question)

@app.get("/ask_general", response_model=QAResponse, summary="Ask General Knowledge Question")
async def ask_general_question_endpoint(question: str, service: QAService = Depends(lambda: qa_service)):
    """
    Asks a question directly to the LLM without using any uploaded document context.
    """
    return service.ask_general_question(question)

@app.get("/ask_all", response_model=QAResponse, summary="Ask Question Across All Documents")
async def ask_all_documents_endpoint(question: str, service: QAService = Depends(lambda: qa_service)):
    """
    Asks a question against the combined content of ALL uploaded documents.
    May fallback to general knowledge if no documents are present and configured to do so.
    """
    return service.ask_all_documents(question)

@app.get("/ask_all_chunked", response_model=ChunkedQAResponse, summary="Ask Question Across Documents (Chunked)")
async def ask_all_chunked_endpoint(question: str, chunk_size: int = 3, service: QAService = Depends(lambda: qa_service)):
    """
    Asks a question against all documents, but processes them in chunks to manage resources.
    Returns a list of answers, one for each chunk of documents.
    May fallback to general knowledge if no documents are present and configured to do so.
    """
    return service.ask_all_chunked(question, chunk_size)

@app.get("/health", summary="Health Check")
async def health_check():
    """Basic health check endpoint."""
    # Could add checks for Ollama connection here if needed
    return {"status": "ok"}

# --- Main Execution ---
if __name__ == "__main__":
    # Use environment variables for host, port, workers
    host = os.getenv("QA_HOST", "0.0.0.0")
    port = int(os.getenv("QA_PORT", "8000")) # Defaulting to 8000 as per install script logic
    workers = int(os.getenv("QA_WORKERS", "1"))

    logger.info(f"Starting Uvicorn server on {host}:{port} with {workers} worker(s)...")
    uvicorn.run(
        "__main__:app", # Reference the app object in the current file
        host=host,
        port=port,
        workers=workers,
        reload=False # Important for production, set to True for development if needed
    )
EOL

# Create environment file
echo "Creating environment file..."
cat > "${INSTALL_DIR}/.env" << 'EOL'
# Ollama Connection
# OLLAMA_HOST=0.0.0.0 # CAUTION: UNCOMMENT TO OPEN OLLAMA SERVER TO REQUESTS FROM ANY SOURCE!
OLLAMA_PORT=11434
LANGUAGE_MODEL=tinyllama

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Set default behavior when no docs exist
FALLBACK_TO_GENERAL_KNOWLEDGE=true

# Service Configuration - USING PORT 8000 NOW
QA_HOST=0.0.0.0
QA_PORT=8000
QA_WORKERS=1
MAX_DOCUMENTS=10
EOL

# Create requirements file
echo "Creating requirements file with compatible versions..."
cat > "${INSTALL_DIR}/requirements.txt" << 'EOL'
langchain-huggingface==0.1.2
langchain-ollama==0.3.2
langchain-community==0.3.22
faiss-cpu==1.11.0
sentence-transformers==4.1.0
pypdf==5.4.0
hf_xet #==1.0.5
fastapi==0.115.12
uvicorn==0.34.2
python-multipart==0.0.20
python-dotenv==1.1.0
pydantic==2.11.3
numpy==2.2.5
torch==2.7.0 #--index-url https://download.pytorch.org/whl/cpu
transformers==4.51.3
tokenizers==0.21.1
tqdm==4.67.1
EOL

# Install Python dependencies in venv
echo "Installing Python dependencies..."
"${VENV_DIR}/bin/pip" install --no-cache-dir -r "${INSTALL_DIR}/requirements.txt"

# Update service file to use venv
echo "Updating service configuration..."
cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=Private LLM Service
After=network.target ollama.service
Requires=ollama.service

[Service]
User=ollama
Group=ollama
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=${INSTALL_DIR}/.env
ExecStart=${VENV_DIR}/bin/python ${INSTALL_DIR}/qa_service.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
echo "Starting services..."
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl start "${SERVICE_NAME}"

# ADD PORT FORWARDING FROM 80 TO 8000
echo "Setting up port forwarding from 80 to 8000..."
apt-get install -y iptables-persistent
iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8000
iptables-save > /etc/iptables/rules.v4

# Verify installation
echo "Checking services..."
systemctl status ollama --no-pager
systemctl status "${SERVICE_NAME}" --no-pager

echo -e "\n\033[1;32mInstallation complete!\033[0m"
echo -e "Service is running on:"
echo -e "  - Direct access: \033[1;34mhttp://$(hostname -I | awk '{print $1}'):8000\033[0m"
echo -e "  - Port 80 forwarded: \033[1;34mhttp://$(hostname -I | awk '{print $1}')\033[0m"
echo -e "\n\033[1;33mAPI Endpoints:\033[0m"
echo "  POST   /upload?doc_id={id}       - Upload a PDF document"
echo "  GET    /documents               - List all documents"
echo "  DELETE /document/{id}           - Remove a document"
echo "  GET    /ask/{id}?question={q}   - Query a specific document"
echo "  GET    /ask_all?question={q}    - Query all documents at once"
echo "  GET    /ask_all_chunked?question={q}&chunk_size={n} - Query in chunks"

echo -e "\n\033[1;33mExample usage:\033[0m"
echo "  curl -X POST -F \"file=@document.pdf\" \"http://localhost/upload?doc_id=doc1\""
echo "  curl \"http://localhost/documents\""
echo "  curl \"http://localhost/ask/doc1?question=summary\""
echo "  curl -X DELETE \"http://localhost/document/doc1\""
