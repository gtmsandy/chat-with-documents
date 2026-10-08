# 📚 Chat With Documents

A **Retrieval-Augmented Generation (RAG)** application that lets users chat with their own documents and web content using Large Language Models.

Built with **Python, Streamlit, LangChain, Hugging Face embeddings, and ChromaDB**, the application retrieves relevant information from uploaded sources and uses it as context when generating responses.

![App Screenshot](app.png)

---

## 🚀 Features

- 📄 Upload **PDF, DOCX, TXT, and Markdown** documents
- 🌐 Load and query content from **URLs**
- 🔎 Semantic document retrieval using vector embeddings
- 🧠 **Retrieval-Augmented Generation (RAG)**
- 💬 History-aware conversational retrieval
- 🔁 Context-aware follow-up questions using previous conversation history
- 📌 Source and page citations for RAG responses
- 🛡️ Strict document-grounded answers with insufficient-context fallback
- 🤖 Support for **OpenAI, Groq, and Anthropic** LLM providers
- ⚡ Streaming AI responses
- 💾 ChromaDB vector storage
- 🧩 Local Hugging Face embeddings
- 🔄 RAG can be enabled or disabled from the interface
- 🚫 Prevents duplicate sources within the active session
- 🧹 Filters empty document chunks before embedding
- 🔐 API keys are supplied at runtime rather than stored in the repository
- 🔀 Provider-specific LLM routing
- ✅ Automated regression tests for citations, conversation history, contextualized retrieval, and provider routing

---

## 🧠 How It Works

```text
Document / URL
      ↓
Content Loading
      ↓
Text Chunking
      ↓
Hugging Face Embeddings
      ↓
Chroma Vector Database
      ↓
History-Aware Retriever
      ↓
Relevant Document Context
      ↓
Selected LLM
      ↓
Grounded Response + Source Citations
```

The application uses the **`all-mpnet-base-v2`** embedding model to convert document chunks into vector representations.

When a question is asked, Chroma retrieves semantically relevant chunks from the indexed documents.

For conversational follow-up questions, LangChain's history-aware retrieval flow uses previous user and assistant messages to transform context-dependent questions into standalone retrieval queries.

For example:

```text
User:
Which Binary Search pattern contains Koko Eating Bananas?

Assistant:
Allocation / Capacity Problems (Binary Search on Answer)

User:
What are the other problems in that same pattern?
```

The application uses the previous conversation to understand what **"that same pattern"** refers to before searching the vector database.

The retrieved document content is then passed to the selected LLM as context.

When RAG is enabled, responses are instructed to rely on retrieved document evidence. If the available context does not contain enough information, the application returns an insufficient-information response instead of inventing unsupported facts.

RAG responses also display the relevant source filename and page number when that metadata is available.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application |
| Streamlit | Web interface |
| LangChain | RAG and LLM orchestration |
| ChromaDB | Vector database |
| Hugging Face | Local text embeddings |
| `all-mpnet-base-v2` | Embedding model |
| OpenAI | LLM provider |
| Groq | LLM provider |
| Anthropic | LLM provider |
| `unittest` | Automated regression testing |

---

## 📂 Project Structure

```text
chat-with-documents/
│
├── streamlit_app.py      # Streamlit UI and application flow
├── rag_utils.py          # Loading, chunking, embeddings and RAG logic
├── test_rag_utils.py     # Automated regression tests
├── requirements.txt      # Direct runtime dependencies
├── app.png               # Application screenshot
├── LICENSE               # MIT license and attribution
├── .gitignore
└── README.md
```

The application may also create runtime directories such as:

```text
source_files/             # Temporary uploaded files
chroma_db_*/              # Local Chroma vector stores
```

These directories contain generated runtime data and are excluded from Git.

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/gtmsandy/chat-with-documents.git
cd chat-with-documents
```

### 2. Create a virtual environment

**Python 3.11 is recommended and the project has been tested with Python 3.11.9.**

#### Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks virtual-environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Linux/macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --no-cache-dir -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run streamlit_app.py
```

Streamlit will normally start the application at:

```text
http://localhost:8501
```

---

## 🔑 LLM Configuration

Select an LLM provider from the application sidebar:

- **OpenAI**
- **Groq**
- **Anthropic**

Provide the API key for the selected provider through the application.

API keys are entered at runtime and are not intended to be stored in the repository.

The application routes requests based on the provider selected in the UI:

```text
Groq
  ↓
ChatGroq

OpenAI
  ↓
ChatOpenAI

Anthropic
  ↓
ChatAnthropic
```

For Groq, the application supports:

```text
openai/gpt-oss-20b
```

The `openai/` prefix is part of the Groq model identifier. Provider routing is determined by the selected provider and not by the model-name prefix.

After configuring the provider, upload documents or provide a URL and initialize the knowledge base.

---

## 💬 Example Usage

Upload a document and ask questions such as:

```text
Summarize this document.

What are the main conclusions?

Explain this section in simpler terms.

What does the document say about this topic?

Compare the important points mentioned in the uploaded sources.
```

The application also supports conversational follow-up questions.

Example:

```text
Which Binary Search pattern contains Koko Eating Bananas?

What are the other problems in that same pattern?
```

The history-aware retriever uses previous conversation turns to resolve references before performing semantic retrieval.

With RAG enabled, relevant document evidence is retrieved before the response is generated.

---

## 🔄 RAG vs Standard Chat

### RAG Enabled

```text
Conversation History
        +
Current Question
        ↓
Query Contextualization
        ↓
Retrieve Relevant Chunks
        ↓
Document Context
        ↓
Selected LLM
        ↓
Grounded Answer + Sources
```

Best when asking questions about uploaded documents.

When the retrieved context does not contain enough information, the application is instructed to respond with an insufficient-information message rather than guessing.

Example:

```text
The uploaded documents do not contain enough information to answer this.
```

### RAG Disabled

```text
Question
   ↓
Selected LLM
   ↓
Answer
```

Works as a regular LLM conversation without document retrieval or document-grounding restrictions.

---

## ✅ Current Status

The core RAG workflow is functional.

Implemented and verified functionality includes:

- Document loading for PDF files
- Document loading for DOCX files
- Document loading for TXT files
- Document loading for Markdown files
- URL ingestion
- Text chunking
- Empty-chunk filtering
- Local embedding generation using `all-mpnet-base-v2`
- ChromaDB indexing
- Semantic retrieval
- Multi-document retrieval
- Duplicate-source protection in the active session
- History-aware conversational retrieval
- Follow-up query contextualization
- Strict document-grounded responses
- Insufficient-context refusal behavior
- Source and page citations
- OpenAI provider support
- Groq provider support
- Anthropic provider support
- Provider-specific LLM routing
- Streaming responses
- RAG / standard-chat switching
- Reproducible Python 3.11 dependency setup
- Automated regression tests

The application has also been manually tested with real document ingestion and conversational retrieval queries.

A verified follow-up retrieval flow successfully handled:

```text
Which Binary Search pattern contains Koko Eating Bananas?
```

followed by:

```text
What are the other problems in that same pattern?
```

The application correctly used the previous assistant response to resolve the follow-up query and retrieved the intended document section.

Unsupported questions were also tested to verify that the application refuses to invent information that is not available in the uploaded documents.

---

## 🗺️ Planned Improvements

Future improvements are focused on production hardening and retrieval quality:

- 🗂️ Better Chroma database lifecycle and explicit document cleanup
- 🔍 Retrieval evaluation and relevance measurement
- 🎯 Retrieval tuning and optional reranking
- 🔐 Stronger URL validation and SSRF protection
- 📊 Structured application logging and error monitoring
- 👥 Multi-user isolation and authentication
- 📦 Deployment and containerization
- 🧪 Broader integration and end-to-end testing
- ⚙️ Improved application configuration management
- 📈 Retrieval-quality benchmarking

---

## 👨‍💻 Maintained & Extended By

**Sandeep Harijan** — [@gtmsandy](https://github.com/gtmsandy)

This project builds upon an existing open-source implementation and has been further developed with improvements to:

- conversational RAG
- history-aware retrieval
- strict document grounding
- source and page citations
- LLM provider routing
- reproducible dependency management
- automated regression testing
- application reliability and maintainability

The original MIT license and attribution are retained in the repository.

---