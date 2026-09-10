import os
import shutil
from src.config import Config

def save_uploaded_files(uploaded_files):
    if not os.path.exists(Config.TEMP_UPLOAD_DIR):
        os.makedirs(Config.TEMP_UPLOAD_DIR)
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(Config.TEMP_UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    return file_paths

def cleanup_temp_files():
    if os.path.exists(Config.TEMP_UPLOAD_DIR):
        shutil.rmtree(Config.TEMP_UPLOAD_DIR)
