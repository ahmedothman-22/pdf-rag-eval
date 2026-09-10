from langchain_cohere import CohereEmbeddings
from src.config import Config

def get_embeddings():
    return CohereEmbeddings(model=Config.EMBEDDING_MODEL, cohere_api_key=Config.COHERE_API_KEY)
