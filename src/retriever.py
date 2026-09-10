def get_retriever(vectorstore, k):
    return vectorstore.as_retriever(search_kwargs={"k": k})
