# RAG-Based Application with LLaMA 3.1 and Ollama

This repository contains a Retrieval-Augmented Generation (RAG) application built using LLaMA 3.1 and Ollama. The application processes PDFs, creates a FAISS index for efficient document retrieval, and uses LLaMA 3.1 to generate answers based on the retrieved information.

## Features

- Converts PDF documents into text format.
- Creates a FAISS index from the text files.
- Uses the LLaMA 3.1 model via Ollama to answer questions based on retrieved documents.
- Allows querying in real-time through the terminal.

## Prerequisites

- Python 3.11 or later
- Ollama installed on your machine
- Required Python packages (see Installation section)

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/GPT-Laboratory/RAG-LLM-Development-Guidebook-from-PDFs/tree/main/RAGUsingLlama3.1
    cd your-repo-folder
    ```

2. Install the required Python packages:

    ```bash
    pip install pymupdf langchain-huggingface faiss-cpu
    pip install ollama sentence-transformers sentencepiece
    pip install langchain-community
    ```

3. Install Ollama from [official Ollama website](https://ollama.com/).

4. Pull the LLaMA 3.1 model using Ollama:

    ```bash
    ollama pull llama3.1
    ```

## PDF to Text Conversion

To convert your PDFs into text files, run the `pdf_to_text.py` script. Make sure your PDF files are placed in the `Data/` folder.

Usage:

```bash
python pdf_to_text.py
```

This will convert all PDF files in the `Data/` folder and store the text files in the `DataTxt/` folder.

## Creating the FAISS Index

Run the `txt_to_index.py` script to generate the FAISS index from the text files in the `DataTxt/` folder.

Usage:

```bash
python txt_to_index.py
```

This will create a FAISS index and save it in the `DataIndex/` folder.

## Running the LLaMA 3.1 Model with Ollama

To use the LLaMA 3.1 model for question answering, you need to run Ollama with the LLaMA model. You can either directly run:

```bash
ollama run llama3.1
```

Or create a `.bat` file for easier usage in VS Code:

1. Create a `ollama.bat` file in the root folder with the following content:

    ```bash
    @echo off
    "C:\path\to\Ollama\ollama.exe" %*
    ```

2. Replace the path with your Ollama installation path.

3. Now you can run the model with:

    ```bash
    .\ollama.bat run llama3.1
    ```

## Running the RAG System

The main script for running the RAG-based question-answering system is `main.py`. This script retrieves relevant documents using the FAISS index and generates answers using LLaMA 3.1.

Usage:

```bash
python main.py
```

The system will prompt you to ask a question. You can type your questions in the terminal and get answers based on the documents in the FAISS index. To exit, type `exit`.

## Common Issues and Best Practices

- **Incompatible Embeddings**: Ensure that the same embeddings model is used when creating and loading the FAISS index.
- **Model Version Issues**: Make sure you're using the correct version of LLaMA (e.g., `llama3.1`) as specified.
- **Prompt Crafting**: Use precise and context-specific prompts to get better responses from the model.
- **Memory Leaks**: Monitor system resources during extended use to prevent memory issues.
- **Reuse Models**: Avoid reloading or re-initializing the model unnecessarily to improve performance.
