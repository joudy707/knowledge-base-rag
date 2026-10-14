# Knowledge Base RAG API

A learning project focused on building a Retrieval-Augmented Generation (RAG) API using FastAPI and Python.

The goal is to understand and implement the core components of a RAG pipeline step by step, with an emphasis on retrieval quality, modular architecture, and evaluation.

## Project Status

**Work in Progress — Learning Project**

This project is being developed incrementally as part of a hands-on learning roadmap. Features listed under Planned Features are not necessarily implemented yet.

## Goals

- Build a modular RAG backend using FastAPI.
- Parse documents and prepare text for retrieval.
- Explore chunking strategies and metadata management.
- Implement semantic search using embeddings.
- Combine semantic search with keyword-based retrieval.
- Experiment with Reciprocal Rank Fusion (RRF) and reranking.
- Evaluate retrieval quality and answer relevance.

## Planned Features

- PDF and TXT document ingestion
- Chunking and metadata extraction
- Embeddings and vector search
- BM25 keyword search
- Hybrid search and Reciprocal Rank Fusion
- Cross-encoder reranking
- API endpoints for document ingestion and querying
- Answers with source references
- Retrieval evaluation and automated tests

## Tech Stack

- Python
- FastAPI
- Uvicorn
- PyMuPDF
- Pytest

Additional libraries will be introduced as the project evolves.

## Current Progress

- Initial FastAPI application setup
- Basic text chunking implementation
- Environment setup for continued development
- Implemented an embedding module for converting text into numerical vectors.
- Implemented an initial in-memory vector search using cosine similarity.

Document parsing, improved chunking, metadata handling, retrieval, and evaluation are part of the ongoing development roadmap. Update this section as each feature is implemented and tested.

## Learning Roadmap

The project is being developed in stages:

1. Document parsing and chunking
2. Embeddings and vector search
3. Keyword search and hybrid retrieval
4. Rank fusion and reranking
5. API integration and answer generation
6. Evaluation, testing, and documentation

## Running Locally

Create and activate a Python virtual environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the API from the project root:

```bash
python -m uvicorn app.main:app --reload
```

When the server is running, open `http://127.0.0.1:8000/docs` to explore the API documentation.

## Learning Purpose

This project is intended to demonstrate the development process and practical exploration of RAG components. Planned features should not be considered complete until they have been implemented and tested.