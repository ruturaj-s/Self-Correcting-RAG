# 🤖 Self-Correcting RAG System

An advanced Retrieval-Augmented Generation (RAG) application that answers questions from uploaded PDF documents using semantic retrieval, reranking, answer verification, confidence scoring, and self-correction.

The system is designed to reduce hallucinations by verifying whether an answer is actually supported by the uploaded knowledge base.

---

## 🚀 Project Overview

Traditional LLM applications can generate convincing answers even when the required information is not available in their knowledge source.

This project addresses that problem by implementing a **Self-Correcting RAG Pipeline**.

The system:

1. Accepts PDF documents from the user
2. Extracts and processes document content
3. Splits documents into searchable chunks
4. Generates vector embeddings
5. Performs semantic vector search
6. Reranks retrieved chunks
7. Generates an answer using an LLM
8. Verifies whether the answer is supported by the retrieved evidence
9. Calculates answer confidence
10. Performs self-correction when required
11. Detects unsupported questions
12. Maintains chat and evaluation history
13. Provides RAG performance analytics

---

## ✨ Key Features

### 📄 Document Processing
- Upload PDF documents
- Multi-document knowledge base
- Automatic document processing
- Text extraction and chunking
- Document management

### 🔎 Retrieval Pipeline
- Query rewriting
- Vector similarity search
- Semantic retrieval
- Chunk reranking
- Context-aware answer generation

### 🧠 Self-Correction
- Answer verification
- Unsupported-question detection
- Automatic self-correction
- Confidence scoring
- Hallucination reduction

### 📊 Evaluation & Analytics
- Accuracy measurement
- Response-time tracking
- Retrieval statistics
- Reranking statistics
- Verification accuracy
- Average attempts
- Self-correction rate
- Supported vs unsupported question analysis
- Question-wise evaluation
- Historical evaluation runs
- CSV export

### 💬 User Experience
- Interactive chat interface
- Source evidence display
- Answer confidence indicator
- Knowledge-base management
- Pipeline status indicators

---

## 🏗️ RAG Architecture

```text
                  ┌──────────────────────┐
                  │    Upload PDF        │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Document Ingestion   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │      Chunking        │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │     Embeddings       │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    Vector Search     │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │      Reranking       │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    LLM Generation    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Answer Verification  │
                  └──────────┬───────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                 Supported        Unsupported
                    │                 │
                    ▼                 ▼
              Final Answer       Self-Correction /
                                  Safe Response