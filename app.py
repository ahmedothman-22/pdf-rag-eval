import streamlit as st
import json
from src.config import Config
from src.utils import save_uploaded_files, cleanup_temp_files
from src.ingestion import load_documents
from src.chunking import split_documents
from src.embeddings import get_embeddings
from src.vectorstore import build_vectorstore
from src.retriever import get_retriever
from src.rag_chain import build_rag_chain
from src.evaluation import run_evaluation_suite

st.set_page_config(page_title="PDF RAG & Evaluation", layout="wide")

st.title("RAG System with Evaluation")
st.markdown("Upload your PDFs, configure settings, ask questions, and evaluate retrieval performance.")

try:
    Config.validate()
except ValueError as e:
    st.error(f"Configuration Error: {e}. Please check your .env file.")
    st.stop()

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "top_k" not in st.session_state:
    st.session_state.top_k = 5

with st.sidebar:
    st.header("Data Ingestion")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    with st.expander("Advanced Settings", expanded=False):
        st.subheader("Chunking")
        chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=800, step=100, format="%d")
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=100, step=50, format="%d")
        
        st.subheader("Retrieval")
        top_k = st.slider("Top-k Retrieval", min_value=1, max_value=10, value=5)
        st.session_state.top_k = top_k

    if st.button("Build Vector Store", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            with st.spinner("Processing documents (Loading, Chunking, Embedding)..."):
                try:
                    temp_paths = save_uploaded_files(uploaded_files)
                    docs = load_documents(temp_paths)
                    chunks = split_documents(docs, chunk_size, chunk_overlap)
                    st.session_state.vectorstore = build_vectorstore(chunks, get_embeddings())
                    
                    st.session_state.chat_history = [] 
                    st.toast("Vector store built successfully!")
                except Exception as e:
                    st.error(f"Error processing files: {e}")
                finally:
                    cleanup_temp_files()
    
    st.divider()
    
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

tab1, tab2 = st.tabs(["Chat", "Evaluation"])

with tab1:
    is_ready = st.session_state.vectorstore is not None
    
    if not is_ready:
        st.info("Please upload PDFs and click 'Build Vector Store' to start chatting.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("View Sources"):
                    for source in msg["sources"]:
                        st.caption(f"**File:** {source['source']} | **Page:** {source['page']}")

    user_query = st.chat_input("Ask a question about your documents...", disabled=not is_ready)
    
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"): 
            st.markdown(user_query)
        
        retriever = get_retriever(st.session_state.vectorstore, st.session_state.top_k)
        rag_chain = build_rag_chain(retriever)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    retrieved_docs = retriever.invoke(user_query)
                    answer = rag_chain.invoke(user_query)
                    
                    st.markdown(answer)
                    
                    sources = []
                    seen = set()
                    for doc in retrieved_docs:
                        src_name = doc.metadata.get("source", "Unknown").replace("\\", "/").split("/")[-1]
                        page = doc.metadata.get("page", "Unknown")
                        
                        key = (src_name, page)
                        if key not in seen:
                            seen.add(key)
                            sources.append({"source": src_name, "page": page})
                    
                    if sources:
                        with st.expander("View Sources"):
                            for s in sources:
                                st.caption(f"**File:** {s['source']} | **Page:** {s['page']}")
                                
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": answer, 
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

with tab2:
    if st.session_state.vectorstore:
        st.markdown("### Test Retrieval Performance")
        
        default_json = """[
    {"question": "What programming languages are mentioned?", "relevant_pages": [0, 1]}
]"""
        eval_input = st.text_area("Ground Truth JSON", value=default_json, height=150)
        
        if st.button("Run Evaluation", use_container_width=True):
            try:
                eval_data = json.loads(eval_input) 
                
                if not isinstance(eval_data, list):
                    raise ValueError("Input must be a JSON list of objects.")
                    
                with st.spinner("Running evaluation suite..."):
                    retriever = get_retriever(st.session_state.vectorstore, st.session_state.top_k)
                    df = run_evaluation_suite(eval_data, retriever)
                    
                    st.subheader("Aggregate Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Avg Precision", f"{df['Precision'].mean():.3f}")
                    col2.metric("Avg Recall", f"{df['Recall'].mean():.3f}")
                    col3.metric("Avg F1 Score", f"{df['F1'].mean():.3f}")
                    
                    st.subheader("Detailed Results")
                    st.dataframe(df, use_container_width=True)
                    
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON format: {e}. Please ensure you are using double quotes for keys and strings.")
            except Exception as e:
                st.error(f"Evaluation error: {e}")
    else:
        st.info("Please build the vector store from the sidebar first to run evaluations.")