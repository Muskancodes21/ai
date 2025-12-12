# Presales Enterprise Assistant (GenAI RAG)

**A Privacy-First, Local RAG Solution for Efficient Proposal & RFP Management**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![LangChain](https://img.shields.io/badge/LangChain-Framework-green) ![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange) ![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-yellow)

## 📌 Project Overview

The **Presales Enterprise Assistant** is a digital solution designed to streamline the workflow of Presales teams. Traditionally, teams spend excessive time manually navigating SharePoint repositories to find past proposals, RFPs, and technical documents.

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that ingests sensitive multi-format documents and allows users to query them using natural language. It runs entirely locally to ensure data security.

### 🚀 Key Business Value
* **Efficiency:** Drastically reduces time-to-insight for finding specific clauses or technical specs in historical data.
* **Security:** Utilizing local LLMs (DeepSeek via Ollama) ensures sensitive RFP data never leaves the local infrastructure.
* **Versatility:** Handles a wide variety of enterprise document formats automatically.

---

## ⚙️ Technical Architecture

The solution uses a local RAG architecture to transform raw data into a searchable knowledge base.



### 1. Data Ingestion & Processing
* **Multi-Document Loading:** The system supports a wide range of file formats commonly used in business:
    * [cite_start]**PDFs, Word Docs (.docx), PowerPoint (.pptx), Excel (.xlsx), and CSVs** [cite: 125-131].
    * [cite_start]Utilizes a custom `MultiDocumentLoader` class that routes files to specific loaders (e.g., `PyPDFLoader`, `Docx2txtLoader`, `UnstructuredPowerPointLoader`) [cite: 122, 145-192].
* [cite_start]**Chunking:** Documents are split into manageable chunks using `RecursiveCharacterTextSplitter` (Size: 1000, Overlap: 150) to maintain context windows[cite: 216].

### 2. Embeddings & Vector Storage
* **Embeddings:** Uses `HuggingFaceEmbeddings` with the `sentence-transformers/all-MiniLM-L6-v2` model. [cite_start]This runs efficiently on CPU [cite: 219-222].
* [cite_start]**Vector Store:** Chunks are indexed using **FAISS** (Facebook AI Similarity Search) and saved locally for fast retrieval[cite: 225].

### 3. Retrieval & Generation (RAG)
* [cite_start]**Model:** Powered by **Ollama** running the `deepseek-r1:7b` model locally[cite: 9, 32].
* [cite_start]**Context Injection:** Retrieves the top 3 most relevant chunks (`k=3`) based on the user query[cite: 7, 67].
* [cite_start]**Prompt Engineering:** A custom prompt enforces strict context adherence ("Answer ONLY using the context provided") to prevent hallucinations[cite: 41].

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.8+
* [Ollama](https://ollama.ai/) installed and running.
* Model pulled: `ollama pull deepseek-r1:7b`

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/presales-assistant.git](https://github.com/yourusername/presales-assistant.git)
cd presales-assistant

### 2. Install Dependencies
Bash

pip install -r requirements.txt

Key dependencies include langchain-community, faiss-cpu, sentence-transformers, pypdf, python-docx, and python-pptx .

3. Configuration
Ensure your raw documents are placed in the dataset folder. You may need to update the folder_path variable in ingest.py to point to your data directory.

💻 Usage
Step 1: Ingest Data (Build the Brain)
Run the ingestion process to read your documents and create the FAISS vector index.

Bash

# Ensure the main() function calls loader.load_documents() and create_vector_store()
python ingest.py 

This loads documents, splits them, and saves the multi_format_faiss_index locally .

Step 2: Chat with Your Data
Once the index is built, launch the assistant to start querying.

Bash

python app.py

Launches the CLI loop to answer questions based on the ingested data .

📸 Features
Smart Retrieval with Citations
The system doesn't just answer; it tells you where it found the answer.

Input: "What were the security compliance requirements in the Q3 proposal?"


Output: Generates a summary and lists the source filenames used for the answer.


Response Cleaning
Includes a post-processing layer to strip chain-of-thought artifacts (e.g., "Let me think...", "Sure, here is the answer") ensuring a clean, professional output .

🔮 Future Roadmap
SharePoint API Integration: Automate the ingestion directly from live SharePoint folders rather than local directories.


Streamlit UI: Migrate the current CLI backend into a full web interface for non-technical users.

Hybrid Search: Implement keyword search alongside semantic search for better precision on acronyms.

👨‍💻 Author
[Your Name] Final Year Student @ BITS Pilani | Intern @ C5i

LinkedIn: [Your Profile Link]

Email: [Your Email]
