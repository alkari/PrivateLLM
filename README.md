# PrivateLLM
Host a Private LLM of choice, upload documents and query with simple REST API calls.


Provides a document question-answering system that uses Ollama for LLM processing and FastAPI for API access. Upload PDFs and get answers to your questions through a simple REST API.

## Features

- 📄 **PDF Processing**: Upload and extract text from PDF documents
- ❓ **Question Answering**: Get accurate answers from uploaded documents
- 🔍 **Multi-Document Search**: Query across all uploaded documents at once
- ⚡ **Efficient Chunking**: Process large documents in manageable chunks
- 🔄 **Persistent Storage**: Maintains document vectors between restarts

## Prerequisites

- Ubuntu/Debian-based Linux system
- Root access (for installation)
- Minimum hardware:
  - 4GB RAM (8GB recommended)
  - 2 CPU cores
  - 10GB free disk space (more for larger document collections)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/alkari/PrivateLLM.git 
   cd PrivateLLM```

2. Run the installation script:
   ```sudo ./install.sh```

The installation will:

- Set up Ollama server
- Download specified language models
- Create a Python virtual environment
- Configure systemd services
- Set up port forwarding (80 → 8000)

## Configuration
Customize the service by editing these files:

1. Environment variables (/etc/ollama-pdf-qa/.env):
```
# Ollama Connection
OLLAMA_PORT=11434
LANGUAGE_MODEL=tinyllama  # Change to 'mistral' or other supported models

# Service Configuration
QA_PORT=8000              # Internal service port
MAX_DOCUMENTS=10          # Maximum documents to process```

2. Available models (edit install.sh):
```OLLAMA_MODELS=("phi" "mistral" "gemma:2b")  # Add/remove models as needed```

# API Usage

Endpoints
Method	Endpoint	Description
POST	/upload?doc_id={id}	Upload a PDF document
GET	/documents	List all uploaded documents
DELETE	/document/{id}	Remove a specific document
GET	/ask/{id}?question={q}	Ask a question about a document
GET	/ask_all?question={q}	Ask across all documents
GET	/ask_all_chunked?question={q}&chunk_size={n}	Query in memory-efficient chunks

# Example Usage

1. Upload a document:
```curl -X POST -F "file=@research.pdf" "http://[HOST]/upload?doc_id=paper1"```

2. Ask a question:
```curl "http://[HOST]/ask/paper1?question=What%20is%20the%20main%20conclusion?"```

3. List documents:
```curl "http://[HOST]/documents"```

# Customization
1. Changing Models
- Edit the OLLAMA_MODELS array in install.sh
- Re-run the installation or manually pull new models:
```
ollama pull llama3```

2. Adjusting Performance
- Chunk Size: Modify in qa_service.py:
```splitter = RecursiveCharacterTextSplitter(chunk_size=500)  # Adjust as needed```

- Memory Limits: Edit service file (/etc/systemd/system/ollama-qa.service):
```MemoryMax=4G  # Add memory limit if needed```

# Maintenance
Common Commands
- Start/Stop service:
```sudo systemctl restart ollama-qa```

- View logs:
```journalctl -u ollama-qa -f```

- Update models:
```ollama pull mistral```

- Uninstalling
Stop and disable services:
```sudo systemctl stop ollama-qa ollama
sudo systemctl disable ollama-qa ollama```

- Remove files:
```sudo rm -rf /etc/ollama-pdf-qa /opt/ollama-qa-venv /var/lib/ollama-qa```

# Troubleshooting
Error: Connection refused

- Verify Ollama is running: systemctl status ollama
- Check ports: ss -tulnp | grep -E '80|8000|11434'

Error: Model download failed

- Retry manually: ollama pull model-name
- Check disk space: df -h

# Performance Issues

- Reduce chunk size in qa_service.py
- Limit concurrent requests by adjusting QA_WORKERS in .env
