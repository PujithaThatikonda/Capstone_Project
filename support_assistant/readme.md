# Module 3 — AI Support Assistant

## 1. Project Overview

This project implements an AI-powered Support Assistant using Retrieval-Augmented Generation (RAG).

The system retrieves relevant information from a document knowledge base and generates structured answers for user queries.

Workflow:

Documents → Embeddings → Vector Database →
Similarity Search → Context Retrieval →
Answer Generation → API Response

The assistant is deployed using FastAPI and can run completely offline.

---

## 2. Technologies Used

The project uses:

- Python
- FastAPI
- Pydantic
- ChromaDB
- Sentence Transformers
- LangGraph
- Uvicorn

Install dependencies:

pip install fastapi uvicorn chromadb sentence-transformers langgraph pydantic

---

## 3. Project Architecture

User Query
     │
     ▼
FastAPI Endpoint
     │
     ▼
LangGraph Workflow
     │
     ▼
Embedding Model
     │
     ▼
ChromaDB Retrieval
     │
     ▼
Relevant Documents
     │
     ▼
Answer Generation
     │
     ▼
Structured Response

---

## 4. Knowledge Base

The assistant uses text documents stored inside:

support_assistant/docs/

Example documents:

- refund_policy.txt
- delivery_policy.txt
- support_faq.txt

These documents form the knowledge base.

---

## 5. Document Ingestion

File:

ingest.py

Purpose:

Convert documents into vector embeddings and store them in ChromaDB.

Steps:

1. Read documents from docs folder
2. Generate embeddings
3. Store embeddings
4. Save metadata

Command:

python support_assistant/ingest.py

Output:

Documents indexed successfully

---

## 6. Embedding Model

Model Used:

sentence-transformers/all-MiniLM-L6-v2

Purpose:

Convert text into numerical vectors.

Why this model?

- Lightweight
- Fast
- High-quality semantic embeddings
- Open-source

Example:

"Refund request"

↓

[0.12, -0.53, 0.89, ...]

These vectors capture semantic meaning.

---

## 7. Vector Database

Technology:

ChromaDB

Purpose:

Store document embeddings.

Stored Information:

- Document text
- Metadata
- Embedding vectors

Benefits:

- Fast retrieval
- Similarity search
- Persistent storage

Database Location:

support_assistant/chroma_db/

---

## 8. Query Processing Workflow

When a user asks a question:

Example:

"What is the refund policy?"

The workflow performs:

Step 1

Convert query into embedding.

Step 2

Search ChromaDB.

Step 3

Retrieve top matching documents.

Step 4

Generate answer using retrieved context.

Step 5

Return structured response.

---

## 9. LangGraph Workflow

File:

graph.py

LangGraph orchestrates the retrieval process.

Responsibilities:

- Accept query
- Generate embeddings
- Retrieve documents
- Construct response
- Return confidence score

Output Structure:

{
  "answer": "...",
  "sources": [...],
  "confidence": 0.92
}

---

## 10. FastAPI Service

File:

main.py

The API exposes endpoints for querying the assistant.

---

### Home Endpoint

GET /

Response:

{
  "message":
  "Zepto Support Assistant Running"
}

---

### Ask Endpoint

POST /ask

Request:

{
  "query":
  "What is the refund policy?"
}

Response:

{
  "answer":
  "...",
  "sources":
  ["refund_policy.txt"],
  "confidence":
  0.92
}

---

## 11. API Testing

Run the server:

uvicorn support_assistant.main:app --reload

Open:

http://127.0.0.1:8000/docs

Swagger UI allows interactive testing of all endpoints.

---

## 12. Project Workflow

Documents
     │
     ▼
Ingestion
     │
     ▼
Embeddings
     │
     ▼
ChromaDB Storage
     │
     ▼
User Query
     │
     ▼
Embedding Generation
     │
     ▼
Similarity Search
     │
     ▼
Context Retrieval
     │
     ▼
Answer Generation
     │
     ▼
API Response

---

## 13. Project Structure

support_assistant/
│
├── docs/
│   ├── delivery_policy.txt
│   ├── refund_policy.txt
│   └── support_faq.txt
│
├── chroma_db/
│
├── ingest.py
├── graph.py
├── models.py
├── main.py
└── README.md

---

## 14. Running the Project

Step 1

Activate Virtual Environment

venv\Scripts\activate

Step 2

Install Dependencies

pip install fastapi uvicorn chromadb sentence-transformers langgraph pydantic

Step 3

Index Documents

python support_assistant/ingest.py

Step 4

Start API

uvicorn support_assistant.main:app --reload

Step 5

Open Swagger Documentation

http://127.0.0.1:8000/docs

---

## 15. Features Implemented

✔ Document Ingestion

✔ Semantic Embeddings

✔ ChromaDB Vector Storage

✔ Similarity Search

✔ Context Retrieval

✔ FastAPI Integration

✔ LangGraph Workflow

✔ Confidence Scoring

✔ Source Attribution

✔ API Documentation

---

## 16. Learning Outcomes

Through this project I learned:

- Retrieval-Augmented Generation (RAG)
- Embedding Models
- Vector Databases
- Semantic Search
- LangGraph Workflows
- FastAPI Development
- API Deployment
- Knowledge Base Construction

---

## 17. Final Results

Successfully implemented:

✔ Document Indexing

✔ Embedding Generation

✔ ChromaDB Storage

✔ Semantic Retrieval

✔ FastAPI API Service

✔ LangGraph Workflow

✔ Question Answering System

✔ Structured API Responses

This completes the AI Support Assistant module and demonstrates a complete Retrieval-Augmented Generation (RAG) application.