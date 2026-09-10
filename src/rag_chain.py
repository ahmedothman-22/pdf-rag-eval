from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import Config

RAG_SYSTEM_PROMPT = """You are a helpful assistant.
Answer the user's question using ONLY the provided context.
Do not use outside knowledge.
Do not guess.
Do not fabricate information.
If the answer cannot be found in the provided context,
say exactly:
"I don't know."

Context:
{context}

Question:
{question}"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(retriever):
    llm = ChatCohere(model=Config.CHAT_MODEL, temperature=0, cohere_api_key=Config.COHERE_API_KEY)
    prompt = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)
    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return chain
