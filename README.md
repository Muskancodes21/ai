# Presales Enterprise Assistant (GenAI RAG)

**A Privacy-First, Local RAG Solution for Efficient Proposal & RFP Management**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![LangChain](https://img.shields.io/badge/LangChain-Framework-green) ![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange) ![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-yellow)

---

## 📌 Project Overview

The **Presales Enterprise Assistant** is a digital solution designed to streamline the workflow of Presales teams. Traditionally, teams spend excessive time manually navigating SharePoint repositories to find past proposals, RFPs, and technical documents.

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that ingests sensitive multi-format documents and allows users to query them using natural language. It runs entirely locally to ensure data security.

### 🚀 Key Business Value

- **Efficiency:** Drastically reduces time-to-insight for finding specific clauses or technical specs in historical data
- **Security:** Utilizing local LLMs (DeepSeek via Ollama) ensures sensitive RFP data never leaves the local infrastructure
- **Versatility:** Handles a wide variety of enterprise document formats automatically

---

## ⚙️ Technical Architecture

The solution uses a local RAG architecture to transform raw data into a searchable knowledge base.

### 1. Data Ingestion & Processing

**Multi-Document Loading:** The system supports a wide range of file formats commonly used in business including PDFs, Word Docs (.docx), PowerPoint (.pptx), Excel (.xlsx), and CSVs. It utilizes a custom `MultiDocumentLoader` class that routes files to specific loaders (e.g., `PyPDFLoader`, `Docx2txtLoader`, `UnstructuredPowerPointLoader`).

**Chunking:** Documents are split into manageable chunks using `RecursiveCharacterTextSplitter` with a chunk size of 1000 and overlap of 150 to maintain context windows.

### 2. Embeddings & Vector Storage

**Embeddings:** Uses `HuggingFaceEmbeddings` with the `sentence-transformers/all-MiniLM-L6-v2` model, which runs efficiently on CPU.

**Vector Store:** Chunks are indexed using **FAISS** (Facebook AI Similarity Search) and saved locally as `multi_format_faiss_index` for fast retrieval.

### 3. Retrieval & Generation (RAG)

**Model:** Powered by **Ollama** running the `deepseek-r1:7b` model locally for complete data privacy.

**Context Injection:** Retrieves the top 3 most relevant chunks (`k=3`) based on the user query.

**Prompt Engineering:** A custom prompt enforces strict context adherence ("Answer ONLY using the context provided") to prevent hallucinations.

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- Model pulled: `ollama pull deepseek-r1:7b`

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/presales-assistant.git
cd presales-assistant
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies include:**
- `langchain-community` - Framework for building LLM applications
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Embedding generation
- `pypdf` - PDF processing
- `python-docx` - Word document processing
- `python-pptx` - PowerPoint processing

### 3. Configuration

Ensure your raw documents are placed in the `dataset` folder. You may need to update the `folder_path` variable in `ingest.py` to point to your data directory.

---

## 💻 Usage

### Step 1: Ingest Data (Build the Brain)

Run the ingestion process to read your documents and create the FAISS vector index:

```bash
python ingest.py
```

This command loads documents, splits them into chunks, and saves the `multi_format_faiss_index` locally.

### Step 2: Chat with Your Data

Once the index is built, launch the assistant to start querying:

```bash
python app.py
```

This launches the CLI loop where you can ask questions based on the ingested data.

---

## 📸 Features

### Smart Retrieval with Citations

The system doesn't just answer; it tells you where it found the answer.

**Example:**
- **Input:** "What were the security compliance requirements in the Q3 proposal?"
- **Output:** Generates a comprehensive summary and lists the source filenames used for the answer

### Response Cleaning

Includes a post-processing layer to strip chain-of-thought artifacts (e.g., "Let me think...", "Sure, here is the answer") ensuring a clean, professional output suitable for direct use in business communications.

---

## 🔮 Future Roadmap

- **SharePoint API Integration:** Automate the ingestion directly from live SharePoint folders rather than local directories
- **Streamlit UI:** Migrate the current CLI backend into a full web interface for non-technical users
- **Hybrid Search:** Implement keyword search alongside semantic search for better precision on acronyms and exact terms

---

## 👨‍💻 Author

**Muskan Verma**  
Final Year Student @ BITS Pilani | Intern @ C5i

- **LinkedIn:** https://www.linkedin.com/in/muskan-verma-37377b278/
- **Email:** Muskan21234@gmail.com

---


## Acknowledgments

This project was developed as part of an internship at C5i to address real-world challenges faced by Presales teams in managing large document repositories.

