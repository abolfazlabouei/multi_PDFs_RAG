# Multi PDFs RAG

## Overview
Multi PDFs RAG is a **Retrieval-Augmented Generation (RAG)** application that allows users to upload multiple PDF documents and ask questions about their content. Built with Streamlit, it provides an interactive interface for querying PDFs using advanced natural language processing.

## What is RAG?
**Retrieval-Augmented Generation (RAG)** combines document retrieval and text generation. It retrieves relevant document chunks using embeddings stored in a vector database (FAISS) and generates precise answers with a large language model, leveraging external knowledge for better responses.

## Models Used
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (22.7M parameters) for creating semantic embeddings of PDF text.
- **Language Model**: `Mistral-7B-Instruct` (7.3B parameters) via Ollama for generating context-aware answers.
- **Vector Store**: FAISS for efficient similarity search and document retrieval.

## Features
- Upload multiple PDFs through a Streamlit interface.
- Extract and process text from PDFs using PyPDF2.
- Split text into chunks for efficient retrieval.
- Maintain conversation history for contextual responses.
- Run on CPU with optimized, quantized models.

## Requirements
- Python 3.10
- Ollama with `mistral:7b-instruct` model installed
- Dependencies listed in `requirements.txt`

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/abolfazlabouei/multi_PDFs_RAG.git
   cd multi_PDFs_RAG