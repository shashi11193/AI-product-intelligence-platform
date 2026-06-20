🧠 AI Product Intelligence Platform (Local RAG System)

🚀 Overview

This project is a local Retrieval-Augmented Generation (RAG) system that enables intelligent question answering over enterprise-style documents using a combination of:

Vector search (ChromaDB)
HuggingFace embeddings
Local LLM (Ollama - Phi3)
Observability layer (latency + query logs)
Source attribution and evaluation metrics


🏗️ Architecture

<img width="1024" height="843" alt="image" src="https://github.com/user-attachments/assets/f5031e17-143b-45d6-918e-4d5890fa3729" />


Documents → Chunking → Embeddings → Vector DB (Chroma)
                ↓
          Retrieval (Top-K)
                ↓
          Prompt Augmentation
                ↓
        Local LLM (Phi-3 via Ollama)
                ↓
     Answer + Sources + Evaluation
                ↓
   Logging + Observability Layer


⚙️ Features

🔍 RAG Pipeline

Semantic search using embeddings
Context-aware LLM responses
Retrieval of top-k relevant chunks

📊 Observability

End-to-end latency tracking
Query logging to CSV
Chunk retrieval metrics

📚 Source Attribution

Chunk-level grounding of answers
Traceability of responses

🧪 Evaluation Layer

Faithfulness scoring (context overlap)
Basic hallucination detection


📁 Data Flow

Raw Documents
   ↓
Chunking (LangChain)
   ↓
Embeddings (MiniLM)
   ↓
Chroma Vector Store
   ↓
Semantic Retrieval
   ↓
LLM Response (Phi-3)
   ↓
Evaluation + Logging

## Project Highlights

- Built a fully local Retrieval-Augmented Generation (RAG) system using Ollama and Phi-3 Mini
- Implemented semantic retrieval with ChromaDB and HuggingFace embeddings
- Added observability through latency tracking and query logging
- Implemented source attribution for explainable responses
- Added evaluation capabilities including faithfulness scoring and hallucination detection
- Designed using production-inspired AI architecture principles

  
🧠 Example Query

Question:
What is SEV1 response time?

Answer:
Immediate response (< 30 minutes)

Evaluation:
Faithfulness Score: 0.86
Hallucination: None detected


📈 Logs Example:

timestamp,question,answer,latency_sec,chunks_retrieved
2026-06-18,What is SEV1 response time?,Immediate response,12.3,3


🛠️ Tech Stack

Python
LangChain
ChromaDB
HuggingFace Transformers
Ollama (Phi-3 Mini)
CSV-based observability


📌 Key Learning Outcomes

Built end-to-end RAG system from scratch
Implemented vector database retrieval pipeline
Added observability and evaluation layer
Understood real-world LLM system architecture


⚠️ Limitations

CPU-based inference (slower response time)
Simple heuristic evaluation (not full RAGAS)
Local-only deployment


🚀 Future Improvements

Add reranking model
Add dbt-style data pipeline layer
Replace heuristics with RAGAS evaluation
Deploy as API (FastAPI)
