# RAG-based Legal Document Retrieval System

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system for retrieving and processing German legal documents, specifically those related to the BGB (Bürgerliches Gesetzbuch). The system integrates FAISS for efficient document retrieval, SentenceTransformers for embedding generation, and an LLM (Ollama Chat) for query optimization and reranking.

## Features
- **Legal Query Optimization**: Enhances user queries by incorporating relevant legal terminology and BGB references.
- **Semantic Search**: Uses FAISS to retrieve relevant documents based on embeddings.
- **BGB Reference Matching**: Extracts and prioritizes documents containing explicit BGB references.
- **LLM Reranking**: Uses a legal-specific language model to improve retrieval accuracy.
- **Multilingual Support**: Detects and responds in the user's language.

## Installation
### Requirements
- Python 3.8+
- Required Python libraries:
  ```bash
  pip install -r requirements.txt
  ```

## File Structure
```
RAG/
|-- data/
|   |-- embeddings.json
|   |-- extracted_metadata.json
|   |-- cleaned_bgb/
|
|-- retrival.py
|-- database.py
|-- clean.py
|-- parse.py
|-- rag.py
|-- README.md
|-- requirements.txt
```

### Core Modules
- `rag.py`: Implements the main RAG pipeline integrating retrieval and generation.
- `retrival.py`: Handles document retrieval, query optimization, and reranking.
- `database.py`: Loads and processes document embeddings.
- `clean.py`: Prepares and cleans raw text data for embedding.
- `parse.py`: Extracts metadata and legal references.

## Usage
### Running the RAG System
```bash
python rag.py
```
This will prompt a sample legal query and return a relevant response.

### Example Query
```python
from rag import Rag

rag = Rag()
user_query = "What happens in case of unjustified subletting?"
response = rag.response(user_query)
print(response)
```

## Retrieval Pipeline
1. **Optimize Query**: The retriever refines the user's legal query.
2. **Retrieve Documents**: A combination of BGB reference matching and FAISS-based semantic search retrieves candidate documents.
3. **Rerank Results**: The LLM assigns relevance scores based on legal criteria.
4. **Generate Response**: The top documents are used to generate a final response in the user's language.

## Contribution
Feel free to contribute by submitting issues or pull requests to improve retrieval accuracy and legal query optimization.

## License
This project is licensed under the MIT License.

