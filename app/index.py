# in this index.py all the steps in indexing phase will be done

# from dotenv import load_dotenv

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings


# does this automatically load openai api key from .env file and set it to embedding model? Yes, the load_dotenv() function loads environment variables from a .env file into the environment. This allows you to access the OPENAI_API_KEY variable in your code without hardcoding it. The OpenAIEmbeddings class will automatically use the OPENAI_API_KEY from the environment when creating an instance of the embedding model.
# load_dotenv()

# Path(__file__) gives the path of the current file (index.py). __file__ is a special variable that holds the path of the current file.
# Path(__file__).parent gives the path of the parent directory (app). /data/sample.pdf is the relative path to the sample.pdf file from the parent directory.
pdf_path = Path(__file__).parent / "data" / "sample.pdf"

# loading the PDF file
# Load the PDF file using PyPDFLoader. loader is an instance of PyPDFLoader that takes the path of the PDF file as an argument. The load() method reads the PDF file and returns a list of documents (docs) that can be used for further processing. This list of documents contains the text extracted from the PDF file, which can be split into smaller chunks for indexing or searching.
loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

# chunking
# Create an instance of RecursiveCharacterTextSplitter with the desired chunk size and overlap. The chunk_size parameter specifies the maximum number of characters in each chunk, and the chunk_overlap parameter specifies the number of characters that should overlap between consecutive chunks. This is useful for ensuring that important context is not lost when splitting the text into smaller pieces.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)
# Split the loaded documents into smaller chunks using the text_splitter instance. The split_documents() method takes the list of documents (docs) as an argument and returns a list of smaller chunks that can be used for further processing, such as indexing or searching.
chunks = text_splitter.split_documents(documents=docs)



#Embedding the chunks using OpenAI Embeddings
# embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Creating a vector store 
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    collection_name="sample_collection_local",
    url="http://localhost:6333"
)

print("Vector store created and embedded sucessfully.")