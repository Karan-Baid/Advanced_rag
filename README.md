# Advanced RAG Application

An advanced Retrieval-Augmented Generation (RAG) system that combines multiple state-of-the-art techniques for superior document retrieval and question answering.

## Features

### 🔍 Advanced Retrieval Pipeline
- **Multi-Query Generation**: Generates 3 diverse query variations for better recall
- **Hybrid Search**: Combines BM25 keyword search (30%) with vector semantic search (70%)
- **Reciprocal Rank Fusion**: Intelligently fuses results from multiple queries
- **Cohere Reranking**: Final relevance scoring for top-5 most relevant documents

### 📄 Multi-Format Document Support
- **PDF**: Including tables and embedded images
- **Word Documents**: DOCX and DOC formats
- **Text Files**: Plain text documents
- **Images**: PNG, JPG, JPEG (with OCR capability)

### 📊 Rich Content Extraction
- **HTML Tables**: Preserves table structure for better LLM understanding
- **Base64 Images**: Embeds image data for potential multimodal processing
- **Fast Processing**: Uses Unstructured library's fast strategy for efficient extraction

### 💬 Conversational Interface
- **Chat History**: Maintains context across multiple questions
- **Session Management**: Separate sessions for different conversations
- **Source Citations**: View the exact document chunks used to generate answers

## Tech Stack

- **LLM**: Groq (Llama 3.1 8B Instant)
- **Embeddings**: Cohere (embed-english-v3.0)
- **Vector Store**: Chroma
- **Document Processing**: Unstructured
- **Reranking**: Cohere Rerank v3.0
- **UI**: Streamlit
- **Package Manager**: UV

## Installation

### Prerequisites
- Python 3.11+
- [UV package manager](https://github.com/astral-sh/uv)

### Setup Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd <project-directory>
```

2. **Install dependencies**
```bash
uv sync
```

3. **Configure environment variables**

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
```

Get your API keys:
- [Groq API Key](https://console.groq.com/)
- [Cohere API Key](https://dashboard.cohere.com/)

## Running the Application

### Local Development
```bash
uv run streamlit run main.py
```

The app will open at `http://localhost:8501`

### Usage
1. Upload one or more documents (PDF, DOCX, TXT, or images)
2. Wait for processing to complete
3. Ask questions about your documents
4. View sources to see which document chunks were used

## Configuration

### Processing Strategy

**Current**: `strategy="fast"` (default)
- Fast processing using pdfminer
- Works with all document types
- Extracts tables as HTML and images as base64
- No system dependencies required

**Upgrade to High-Resolution**: `strategy="hi_res"`

For better OCR and layout detection, you can upgrade to high-resolution processing:

1. **Install Tesseract OCR** (system-level package):
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

2. **Update the code** in `main.py` (line 84):
```python
strategy="hi_res",  # Change from "fast" to "hi_res"
```

**Benefits of hi_res:**
- Advanced OCR for scanned documents
- Better table structure detection
- Improved layout analysis
- Higher accuracy on complex PDFs

**Trade-offs:**
- Slower processing
- Requires Tesseract system installation
- Higher resource usage

### Retrieval Parameters

You can adjust these in `main.py`:

**Hybrid Search Weights** (line 101):
```python
weights=[0.3, 0.7]  # [BM25, Vector]
```

**Documents per Retriever** (lines 89, 92):
```python
search_kwargs={"k": 5}  # Vector retriever
bm25_retriever.k = 5    # BM25 retriever
```

**Query Variations** (line 118):
```python
# Generate 3 query variations (configurable)
```

**Reranking** (line 132):
```python
top_n=5  # Final number of documents for context
```

**RRF Parameter** (line 130):
```python
k=60  # Position decay rate (standard value)
```

## Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Deploy on [Streamlit Cloud](https://streamlit.io/cloud)
3. Add your API keys in Streamlit Cloud secrets

**For hi_res strategy on Streamlit Cloud**, create `packages.txt`:
```txt
tesseract-ocr
```

This tells Streamlit Cloud to install Tesseract automatically.

## Project Structure

```
.
├── main.py              # Main application file
├── pyproject.toml       # Python dependencies (managed by UV)
├── .env                 # API keys (not in git)
├── .gitignore          # Git ignore file
└── README.md           # This file
```

## How It Works

```
User Question
    ↓
Generate 3 Query Variations (LLM)
    ↓
Hybrid Search per Query (BM25 + Vector)
    ↓
Reciprocal Rank Fusion (combine all results)
    ↓
Cohere Reranking (top 20 → top 5)
    ↓
Answer Generation (LLM with context)
```

## Advanced Features

### Reciprocal Rank Fusion (RRF)
Combines results from multiple retrievers by scoring documents based on their rank positions. Documents appearing in top positions across multiple queries get higher scores.

### Hybrid Search
- **BM25**: Keyword-based, good for exact matches and technical terms
- **Vector Search**: Semantic understanding, good for conceptual queries
- **Ensemble**: Weighted combination of both approaches

### Multi-Query
Generates variations of your question to improve recall:
- Original: "What is machine learning?"
- Variation 1: "Explain the concept of ML algorithms"
- Variation 2: "How does automated learning work?"

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key for LLM inference |
| `COHERE_API_KEY` | Yes | Cohere API key for embeddings and reranking |

## Troubleshooting

### "Tesseract not found" error
You're using `strategy="hi_res"` without Tesseract installed. Either:
1. Install Tesseract (see Configuration section)
2. Change to `strategy="fast"` in main.py

### "No module named 'unstructured'"
Run `uv sync` to install all dependencies.

### Slow processing
- Use `strategy="fast"` instead of `"hi_res"`
- Reduce query variations from 3 to 2
- Decrease chunk size in RecursiveCharacterTextSplitter

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
