import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    EMBEDDING_MODEL = "embed-v4.0"
    CHAT_MODEL = "command-a-03-2025"
    TEMP_UPLOAD_DIR = "temp_uploads"
    
    @classmethod
    def validate(cls):
        if not cls.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY environment variable is not set.")
