# 🧠 Secure Local PDF Chat

A secure, private, and 100% offline **Retrieval-Augmented Generation (RAG)** application for interacting with your PDF documents via chat-based queries. Built with LangChain, Streamlit, and Ollama, this project ensures your sensitive documents never leave your local machine.

---

## ✨ Features

- 🔒 **100% Offline & Private:** All text extraction, embedding, vector database indexation, and LLM reasoning happen locally. No API keys or internet connection required.
- 🎨 **Premium UI/UX:** A modern, glassmorphic dark-themed web interface built with Streamlit, complete with real-time response streaming, visual status updates, and interactive chat history.
- 🤖 **Local LLM Integration:** Powered by local models running on [Ollama](https://ollama.com) (default: `llama3.2` for generation and `nomic-embed-text` for embeddings).
- 🔍 **Advanced Multi-Query Retrieval:** Generates multiple perspectives of your questions using LangChain's `MultiQueryRetriever` to fetch context with higher accuracy, bypassing the limitations of standard distance-based similarity searches.
- 📊 **Dynamic Document Metrics:** Real-time feedback on document processing, displaying the file name, page count, and total text chunks created.

---

## 🛠️ Tech Stack

- **Frontend Interface:** [Streamlit](https://streamlit.io/)
- **Orchestration Framework:** [LangChain](https://www.langchain.com/)
- **Local Model Provider:** [Ollama](https://ollama.com/)
- **LLM Model:** `llama3.2`
- **Embedding Model:** `nomic-embed-text`
- **Vector Database:** [ChromaDB](https://www.trychroma.com/)
- **PDF Parser:** `PyPDFLoader`

---

## 🚀 Getting Started

### Prerequisites

1. Install **Python 3.10+** on your machine.
2. Install **Ollama** by downloading it from [Ollama's Official Website](https://ollama.com).
3. Open your terminal and pull the required models:
   ```bash
   # Pull the LLM model
   ollama pull llama3.2

   # Pull the Embedding model
   ollama pull nomic-embed-text
   ```

### Installation

1. Clone this repository (or use the extracted workspace):
   ```bash
   git clone https://github.com/divyaparadkar/Chat-with-PDF-using-Local-LLM-.git
   cd Chat-with-PDF-using-Local-LLM-
   ```

2. Install the required Python packages:
   ```bash
   pip install streamlit langchain-ollama langchain-chroma langchain-community pypdf
   ```

---

## 💻 Running the Application

### 1. Run the Web Interface (Streamlit)
To start the interactive web application, run:
```bash
streamlit run app.py
```
Open the URL (typically `http://localhost:8501`) in your browser, upload your PDF document using the sidebar control panel, and begin chatting!

### 2. Run the CLI Test script
If you prefer running a command-line test against a static PDF file (`Divya Paradkar.pdf`):
```bash
python main.py
```

### 3. Notebook Walkthrough
To inspect the RAG pipeline step-by-step, view the Jupyter notebook:
```bash
jupyter notebook main.ipynb
```

---

## 📂 File Structure

```
├── app.py           # Streamlit web-based UI application
├── main.py          # CLI prototype script for testing the RAG pipeline
├── main.ipynb       # Step-by-step Jupyter Notebook walkthrough
├── .gitignore       # Git ignore rules for database indices and temp files
└── README.md        # Project documentation
```

---

## 🔒 Security & Privacy

This application was designed with strict privacy constraints. Unlike third-party LLM wrappers (such as OpenAI or Anthropic integrations), your PDF text chunks are converted to vector representations and queried locally on your machine. **No telemetry, document contents, or query histories are sent to any cloud providers.**
