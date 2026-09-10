from langchain_chroma import Chroma
import uuid

def build_vectorstore(chunks, embeddings):
    collection_name = f"rag_collection_{uuid.uuid4().hex[:8]}"
    return Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name=collection_name)
