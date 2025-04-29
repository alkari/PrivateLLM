#!/bin/bash
set -e

##############################################################################
# Ollama PDF QA Service Installation Script
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
#
# Usage:
#   sudo ./install.sh
##############################################################################

# Configuration
SERVICE_NAME="ollama-qa"
INSTALL_DIR="/etc/ollama-pdf-qa"
DATA_DIR="/var/lib/ollama-qa"
VENV_DIR="/opt/ollama-qa-venv"
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
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from tempfile import NamedTemporaryFile
import uvicorn
from typing import Dict, List
from pydantic import BaseModel

app = FastAPI()

# Configuration
MAX_DOCUMENTS = 10
DOCUMENT_STORAGE = "/var/lib/ollama-qa/documents"
os.makedirs(DOCUMENT_STORAGE, exist_ok=True)

# Initialize components
llm = OllamaLLM(
    model=os.getenv("LANGUAGE_MODEL", "tinyllama"),
    base_url=f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}"
)

embedding_model = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
)

vector_stores: Dict[str, FAISS] = {}

class DocumentInfo(BaseModel):
    id: str
    chunks: int

@app.post("/upload")
async def upload_pdf(doc_id: str, file: UploadFile = File(...)):
    if len(vector_stores) >= MAX_DOCUMENTS:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_DOCUMENTS} documents reached")
    if doc_id in vector_stores:
        raise HTTPException(status_code=400, detail="Document ID already exists")
    
    try:
        # Save document
        doc_path = os.path.join(DOCUMENT_STORAGE, f"{doc_id}.pdf")
        with open(doc_path, "wb") as f:
            f.write(await file.read())
        
        # Process PDF
        loader = PyPDFLoader(doc_path)
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(documents)
        
        # Create vector store
        vector_stores[doc_id] = FAISS.from_documents(split_docs, embedding=embedding_model)
        
        return {
            "message": f"Document {doc_id} processed successfully",
            "chunks": len(split_docs),
            "status": "uploaded"
        }
    
    except Exception as e:
        if os.path.exists(doc_path):
            os.remove(doc_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    return [
        DocumentInfo(id=doc_id, chunks=vector_stores[doc_id].index.ntotal)
        for doc_id in vector_stores
    ]

@app.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    if doc_id not in vector_stores:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Remove vector store
        del vector_stores[doc_id]
        
        # Remove PDF file
        doc_path = os.path.join(DOCUMENT_STORAGE, f"{doc_id}.pdf")
        if os.path.exists(doc_path):
            os.remove(doc_path)
        
        return {"message": f"Document {doc_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/purge_all")
async def purge_all_documents():
    try:
        # Delete all PDF files
        for doc_id in list(vector_stores.keys()):
            doc_path = os.path.join(DOCUMENT_STORAGE, f"{doc_id}.pdf")
            if os.path.exists(doc_path):
                os.remove(doc_path)
        # Clear all vector stores
        vector_stores.clear()
        return {"message": "All documents purged"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask/{doc_id}")
async def ask_question(doc_id: str, question: str):
    if doc_id not in vector_stores:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        retriever = vector_stores[doc_id].as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        response = qa_chain.invoke(question)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask_general")
async def ask_general_question(question: str):
    """Answer general questions using Ollama without requiring documents"""
    try:
        # Direct LLM query without document context
        response = llm.invoke(question)
        return {
            "answer": response,
            "context": "general_knowledge",
            "documents_used": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask_all")
async def ask_all_documents(question: str):
    """Answer questions using either uploaded documents or general knowledge"""
    try:
        if not vector_stores:
            # Fall back to general knowledge if no docs available
            if os.getenv("FALLBACK_TO_GENERAL_KNOWLEDGE", "true").lower() == "true":
                return await ask_general_question(question)
            raise HTTPException(status_code=400, detail="No documents available")
           
        # Original document-based QA logic
        combined_store = None
        for store in vector_stores.values():
            if combined_store is None:
                combined_store = store
            else:
                combined_store.merge_from(store)
        
        retriever = combined_store.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        response = qa_chain.invoke(question)
        
        return {
            "answer": response,
            "context": "document_based",
            "documents_used": list(vector_stores.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask_all_chunked")
async def ask_all_chunked(question: str, chunk_size: int = 3):
    if not vector_stores:
        if os.getenv("FALLBACK_TO_GENERAL_KNOWLEDGE", "true").lower() == "true":
            # Return general answer in the same chunked format for consistency
            general_answer = await ask_general_question(question)
            return {
                "results": [{
                    "documents": [],
                    "answer": general_answer["answer"]
                }],
                "total_documents": 0,
                "context": "general_knowledge"
            }
        raise HTTPException(status_code=400, detail="No documents available")
    
    try:
        # Process in chunks to control memory usage
        all_docs = list(vector_stores.keys())
        results = []
        
        for i in range(0, len(all_docs), chunk_size):
            chunk = all_docs[i:i + chunk_size]
            combined_store = None
            
            for doc_id in chunk:
                if combined_store is None:
                    combined_store = vector_stores[doc_id]
                else:
                    combined_store.merge_from(vector_stores[doc_id])
            
            retriever = combined_store.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            results.append({
                "documents": chunk,
                "answer": qa_chain.invoke(question)
            })
        
        return {
            "results": results,
            "total_documents": len(all_docs),
            "context": "document_based"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("QA_HOST", "0.0.0.0"),
        port=int(os.getenv("QA_PORT", "80")),
        workers=int(os.getenv("QA_WORKERS", "1"))
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
Description=Ollama PDF QA Service
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
