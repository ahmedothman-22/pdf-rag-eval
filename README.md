# PDF RAG & Evaluation 

A production-ready Retrieval-Augmented Generation (RAG) application built with **LangChain**, **Cohere**, **ChromaDB**, and **Streamlit**.

Unlike standard RAG prototypes, this project includes a **dedicated retrieval evaluation module** to measure performance using Ground Truth data and standard information retrieval metrics (Precision, Recall, F1-Score).

<img width="1522" height="782" alt="Screenshot 2026-09-11 014608" src="https://github.com/user-attachments/assets/f9b9167f-22d9-4ec4-8d60-6deb32747e4e" />

---

## Project Architecture

```text
pdf-rag-eval/
├── app.py
├── src/
│   ├── config.py
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── rag_chain.py
│   └── evaluation.py
├── notebooks/
│   └── PDF_RAG_Retrieval_Evaluation.ipynb
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md

```
---

## Key Features

* **Modular RAG Pipeline:** Clean separation of concerns (Ingestion, Chunking, Retrieval, Generation).
* **Multi-PDF Processing:** Upload and query across multiple documents simultaneously.
* **Advanced Configs:** Customizable chunk size, chunk overlap, and Top-k retrieval directly from the UI.
* **Source Tracking:** Transparent citations with exact source file and page number references.
* **Retrieval Evaluation:** Built-in testing suite to measure Precision, Recall, and F1-Score using Ground Truth JSON data.
* **Interactive UI:** A seamless, user-friendly Streamlit interface.

---

## Getting Started

### Prerequisites

* Python 3.9 or higher
* A valid [Cohere API Key](https://dashboard.cohere.com/api-keys)

### 1. Clone the Repository

```bash
git clone https://github.com/ahmedothman-22/pdf-rag-eval.git
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory based on the provided example:

```bash
cp .env.example .env

```

Add your Cohere API key to the `.env` file:

```text
COHERE_API_KEY=your_actual_cohere_api_key_here

```

### 5. Run the Application

```bash
streamlit run app.py

```

---


## Tech Stack Details

**Core AI & Orchestration**

* **LLM & Embeddings:** Cohere (`command-a-03-2025`, `embed-v4.0`)
* **Orchestration:** LangChain (LCEL)
* **Vector Database:** ChromaDB

**Frontend & Data Processing**

* **UI Framework:** Streamlit
* **Data Handling:** PyPDF, Pandas, Matplotlib

<div align="left">
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/Cohere-3959A4?style=for-the-badge&logo=cohere&logoColor=white" alt="Cohere" />
  <img src="https://img.shields.io/badge/ChromaDB-FF4B4B?style=for-the-badge" alt="ChromaDB" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit" />
</div>



