# InnerGlow — Agentic AI Mental Wellbeing Chatbot

InnerGlow is an AI-powered mental wellbeing chatbot designed to provide supportive, educational conversations for university students, with a focus on the Sri Lankan university context.

The project demonstrates agentic AI principles through intelligent request routing, tool-assisted knowledge retrieval, and reflection-based response improvement.

**Disclaimer:** InnerGlow is an educational prototype. It does not diagnose mental health conditions, provide professional therapy, or replace qualified mental health services.

## Project Objectives

- Provide empathetic, accessible wellbeing conversations for university students.
- Demonstrate agentic AI design patterns using multiple specialized agents.
- Integrate Retrieval-Augmented Generation (RAG) for knowledge-grounded responses.
- Support multilingual interactions, including English, Sinhala, and Singlish.
- Provide a simple conversational interface using Streamlit.
- Apply safety-oriented handling for crisis-related messages.

## Agentic AI Architecture

InnerGlow implements three primary agentic AI patterns:

### 1. Routing Pattern

A routing agent analyzes incoming student messages and selects an appropriate processing path based on the user's request.

### 2. Tool Use Pattern

The chatbot can use external tools and knowledge sources, including:

- A ChromaDB-based RAG retriever for mental wellbeing educational resources.
- A web search tool for relevant external information when required.

### 3. Reflection Pattern

A reflection agent reviews draft responses for safety, clarity, empathy, and relevance before producing the final response.

The reflection process is designed to preserve relevant information shared earlier in the conversation.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| User interface | Streamlit |
| Language model API | Groq |
| Router model | Qwen model via Groq |
| Support model | GPT-OSS model via Groq |
| Vector database | ChromaDB |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| PDF extraction | PyMuPDF |
| RAG framework | Custom Python retrieval pipeline |
| Version control | Git and GitHub |

Model identifiers can be configured through environment variables.

## Retrieval-Augmented Generation (RAG)

InnerGlow uses a local knowledge base of mental wellbeing educational documents.

The RAG pipeline:

1. Extracts text from PDF documents.
2. Splits extracted text into overlapping chunks.
3. Generates multilingual vector embeddings.
4. Stores embeddings and source metadata in ChromaDB.
5. Retrieves relevant knowledge to support chatbot responses.

### Current Knowledge Base

- 5 PDF source documents.
- 265 text chunks in the previously built index.
- An additional JSON file containing educational CBT starter content.

The current PDF indexing script processes the PDF sources. The JSON starter content is not included in that PDF indexing pipeline.

## Project Structure

```text
new_backend/
├── chatbot.py
├── streamlit_app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── emotion/
│   └── detector.py
├── rag/
│   ├── build_index.py
│   ├── retriever.py
│   ├── evaluate_rag.py
│   └── evaluation_results.txt
├── knowledge/
│   ├── cbt_knowledge.json
│   └── pdfs/
├── chroma_db/
└── README.md
```

## Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/LahiruLuci/innerGlow_Backend.git
cd innerGlow_Backend
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_ROUTER_MODEL=qwen/qwen3.8-27b
GROQ_SUPPORT_MODEL=openai/gpt-oss-20b
```

Never commit your real API key or `.env` file to GitHub.

### 5. Build the RAG index

```bash
python -m rag.build_index
```

This rebuilds the ChromaDB collection from the PDFs in `knowledge/pdfs`.

### 6. Run the chatbot

```bash
streamlit run streamlit_app.py
```

Open the local Streamlit URL shown in the terminal.

## RAG Evaluation

The project includes a retrieval evaluation script:

```bash
python -m rag.evaluate_rag
```

Evaluation results are saved in:

```text
rag/evaluation_results.txt
```

The initial evaluation used five test queries. Three were assessed as relevant and two as mixed. These are preliminary qualitative results, not a comprehensive benchmark.

## Conversation Memory

InnerGlow uses Streamlit session state to maintain conversation history during an active chat session.

Recent user and assistant turns are passed to the response-generation pipeline. This enables contextual follow-up questions, such as recalling a name previously shared by the student.

Conversation memory has been tested locally in both the Python terminal and the Streamlit interface.

## Safety and Limitations

InnerGlow includes safety-oriented response handling for explicitly recognized crisis-related messages.

However:

- The chatbot is not a replacement for professional mental healthcare.
- Automated crisis detection may miss indirect or ambiguous language.
- AI-generated responses can contain inaccuracies.
- Retrieved educational content should not be treated as personalized medical advice.
- The current knowledge base has limited topic coverage.
- Conversation memory is session-based, not permanent user memory.

Further safety testing and evaluation are necessary before relying on the system in real-world mental health situations.

## Deployment

The project is intended for deployment using Streamlit Community Cloud.

Deployment requires:

- A public GitHub repository.
- The `streamlit_app.py` entry point.
- A compatible `requirements.txt` file.
- A Groq API key configured through Streamlit's secrets management.

**Deployment status:** Pending verification.

## Academic Context

InnerGlow was developed as an Agentic AI academic project to demonstrate multi-agent workflows, retrieval-augmented generation, tool use, reflection, and responsible AI design.

## Future Improvements

- Expand the mental wellbeing knowledge base.
- Improve Sinhala and Singlish evaluation.
- Add more comprehensive safety testing.
- Evaluate response quality across a larger set of student scenarios.
- Improve retrieval relevance and source transparency.
- Add privacy-conscious memory management.
