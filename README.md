# RAG API

A simple Retrieval-Augmented Generation (RAG) API built with FastAPI, LangChain, Qdrant, Hugging Face embeddings, and Google Gemini. It allows users to ask questions about a PDF document and receive answers based on its content.

## Features

- **PDF Processing:** Extracts text from PDF documents using PyPDFLoader.
- **Text Chunking:** Splits document text into smaller chunks with overlapping content to preserve context.
- **Vector Embeddings:** Converts text chunks into numerical representations using the `all-MiniLM-L6-v2` model.
- **Vector Storage:** Stores and searches document embeddings using Qdrant.
- **Semantic Search:** Retrieves the three most relevant text chunks for a user's question.
- **AI-Generated Answers:** Uses Google Gemini to generate answers based on the retrieved content.
- **Source References:** Returns relevant document excerpts and page numbers alongside the answer.
- **FastAPI Integration:** Exposes an API endpoint for submitting questions.

## Tech Stack

- **Python** — Core programming language
- **FastAPI** — API development
- **LangChain** — Document processing, embeddings, retrieval, and LLM integration
- **PyPDFLoader** — PDF text extraction
- **Hugging Face Sentence Transformers** — Text embeddings
- **Qdrant** — Vector database
- **Google Gemini** — Answer generation
- **python-dotenv** — Environment variable management

## How It Works

The application has two main phases: indexing and querying.

### 1. Indexing Phase

The indexing process prepares the PDF document for semantic search.

1. Load the PDF file using `PyPDFLoader`.
2. Split the extracted text into chunks of 1,000 characters, with an overlap of 400 characters.
3. Generate embeddings for each chunk using `sentence-transformers/all-MiniLM-L6-v2`.
4. Store the chunks and their embeddings in a local Qdrant collection named `sample_collection_local`.

### 2. Querying Phase

When a user submits a question, the API retrieves relevant information and generates an answer.

1. Accept a question through the `/query/` endpoint.
2. Convert the question into an embedding using the same embedding model used during indexing.
3. Search Qdrant for the three most relevant document chunks.
4. Combine the retrieved chunks into a context containing their page numbers.
5. Send the context and question to Google Gemini.
6. Return the generated answer, original question, and retrieved sources as JSON.

The prompt instructs Gemini to use only the provided context and avoid inventing facts when the document does not contain the answer.

## Project Structure

The following is an example structure based on the code provided:

```text
rag-api/
├── app/
│   ├── routes/
│   │   └── query.py
│   ├── data/
│   │   └── sample.pdf
│   ├── index.py
│   └── main.py
├── .env
├── .gitignore
└── requirements.txt
```

- `app/main.py` — Creates the FastAPI application and registers the query router.
- `app/index.py` — Loads the PDF, splits the text, generates embeddings, and creates the Qdrant vector collection.
- `app/routes/query.py` — Handles questions, retrieves relevant chunks, calls Gemini, and returns the response.
- `app/data/sample.pdf` — The PDF document used as the knowledge source.
- `.env` — Stores environment variables, including the Google API key.
- `requirements.txt` — Lists the Python dependencies.

## Prerequisites

Before running the project, make sure you have:

- Python 3.10 or later
- Docker installed
- A Google Gemini API key

## Getting Started

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd rag-api
```

Replace `<your-repository-url>` with your repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you have not created `requirements.txt` yet, install the packages used by the application:

```bash
pip install fastapi "uvicorn[standard]" langchain-community langchain-text-splitters langchain-qdrant langchain-huggingface langchain-google-genai sentence-transformers pypdf python-dotenv
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
```

Replace `your_google_api_key` with your actual API key.

You can obtain a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

Keep your `.env` file private and do not commit it to version control.

### 5. Start Qdrant

The application connects to Qdrant at `http://localhost:6333`. Start a local Qdrant instance using Docker:

```bash
docker compose up
```

Verify that Qdrant is running by opening:

http://localhost:6333/dashboard

### 6. Add Your PDF

Place the PDF document inside:

```text
app/data/sample.pdf
```

If you use a different filename or location, update the PDF path in `app/index.py`.

### 7. Run the Indexing Process

Run the indexing script from the project root:

```bash
python -m app.index
```

This loads the PDF, creates text chunks and embeddings, and stores them in Qdrant.

After successful indexing, the terminal should display:

```text
Vector store created and embedded sucessfully.
```

Run the indexing process again whenever you want to index a new document or rebuild the collection. The current implementation does not include an incremental update or duplicate-prevention mechanism.

### 8. Start the FastAPI Server

Run the API from the project root:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

http://127.0.0.1:8000

Open the interactive API documentation at:

http://127.0.0.1:8000/docs

## API Usage

### Ask a Question

**Endpoint:** `POST /query/`

**Request Body:**

```json
{
  "question": "What is this document about?"
}
```

**Example Response:**

```json
{
  "question": "What is this document about?",
  "answer": "Based on the provided document, ...",
  "sources": [
    {
      "page": 0,
      "content": "A relevant excerpt from the document..."
    },
    {
      "page": 1,
      "content": "Another relevant excerpt from the document..."
    }
  ]
}
```

The answer and source excerpts depend on the PDF and the question submitted. The page numbers returned by PyPDFLoader are zero-based, so page `0` represents the first page.

You can test the endpoint using the Swagger UI at `/docs`, Postman, or any HTTP client.

## Embedding Model

This project uses the Hugging Face model:

`sentence-transformers/all-MiniLM-L6-v2`

It converts text into embeddings that can be compared to find semantically similar content. The model runs locally after its required files have been downloaded, so you do not need a paid OpenAI embedding API for this implementation.

The same embedding model must be used during indexing and querying to keep the vector representations compatible.

## Future Scope of Improvement

- Support uploading and indexing multiple PDF documents.
- Add document metadata and filtering to improve retrieval.
- Add conversation history for follow-up questions.
- Add authentication
- Support configurable chunk sizes and retrieval counts.
- Deploy the API and Qdrant using suitable hosting services.
- Add a frontend interface so users can upload documents and ask questions without using the API documentation.
