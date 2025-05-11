# RAG-based Query Answering System

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** model for academic query answering. The system is designed to answer questions related to various research papers, including papers on **BERT**, **GPT**, and **Attention is All You Need**. It classifies queries into categories (e.g., definition, calculation, or general questions) and retrieves relevant information from the documents. Additionally, for **definition queries**, the system uses an external dictionary API to fetch concise definitions.

## Features

- **RAG Pipeline**: Uses the retrieval-augmented generation technique to answer research paper queries.
- **Query Classification**: Classifies queries into three categories:
  1. **Definition Query**: Looks up definitions for specific terms using a [dictionary API](https://dictionaryapi.dev/).
  2. **Calculation Query**: Identifies calculation-related queries and processes them using basic mathematical evaluation.
  3. **General Query**: Routes general questions to the RAG pipeline to retrieve answers from research papers.
- **Streamlit UI**: Provides a simple and interactive user interface for users to ask questions.
- **Papers Referenced**: 
  - **BERT**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
  - **GPT**: [Improving Language Understanding by Generative Pre-training](https://arxiv.org/abs/1801.06146)
  - **Attention is All You Need**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  
## How to use

1. ```git clone https://github.com/vdhkcheems/inflera-assignment.git```
2. ```cd inflera-assignment```
3. ```python -m venv .venv```
4. ```source .venv/bin/activate``` or however it is done on your system
5. ```pip install -r requirements.txt```
6. Add your GEMINI API key to a .env file in project root, ```GEMINI_API_KEY=your_api_key_here```
7. For CLI run ```python query_rag.py```
8. For streamlit app run ```streamlit run app.py```