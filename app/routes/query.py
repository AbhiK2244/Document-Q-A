import os

from dotenv import load_dotenv
from fastapi import APIRouter
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

router = APIRouter(
    prefix="/query",
    tags=["query"],
)


class QueryRequest(BaseModel):
    question: str


# Use the SAME embedding model used during indexing.
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to the existing Qdrant collection.
vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    collection_name="sample_collection_local",
    url="http://localhost:6333",
)

# Gemini generates the final answer.
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0, # this ensures deterministic output, which is important for RAG applications. higher temperature values introduce randomness and creativity, which can lead to inconsistent answers. lower temperature values make the model more focused and deterministic, which is crucial for applications where accuracy and reliability are paramount.
)


@router.post("/")
async def query(request: QueryRequest):
    question = request.question

    # Retrieve the three most relevant chunks.
    search_results = vector_db.similarity_search(
        query=question,
        k=3,
    )

    if not search_results:
        return {
            "question": question,
            "answer": "I don't know. The document contains no relevant information.",
            "sources": [],
        }

    # Combine retrieved chunks into context.
    context = "\n\n".join(
        [
            f"[Page {result.metadata.get('page', 'unknown')}] "
            f"{result.page_content}"
            for result in search_results
        ]
    )

    system_prompt = f"""
You are a helpful assistant that answers questions
using only the provided context.

If the context does not contain the answer, say:
"I don't know based on the provided document."

Do not invent facts.
Include the relevant page number in your answer.

Context:
{context}
"""

    response = llm.invoke(
        [
            ("system", system_prompt),
            ("human", question),
        ]
    )

    return {
        "question": question,
        "answer": response.content,
        "sources": [
            {
                "page": result.metadata.get("page", "unknown"),
                "content": result.page_content,
            }
            for result in search_results
        ],
    }