from langchain_community.document_loaders import PyPDFLoader

def load_documents(file_paths):
    all_documents = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_documents.extend(docs)
    return all_documents
