import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from functools import lru_cache

DB_PATH = "./chroma_db"

@lru_cache(maxsize=1)
def load_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_db():
    """Creates a vector database from PDF files."""
    loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = load_embeddings_model()
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory=DB_PATH)
    vector_db.persist()
    return vector_db

def load_or_create_vector_db():
    """Loads existing vector DB or creates a new one if not found."""
    if not os.path.exists(DB_PATH):
        return create_vector_db()
    embeddings = load_embeddings_model()
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)