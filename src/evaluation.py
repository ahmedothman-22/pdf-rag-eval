import pandas as pd
import matplotlib.pyplot as plt

def evaluate_retrieval(question, relevant_pages, retriever):
    retrieved_docs = retriever.invoke(question)
    retrieved_pages = [doc.metadata.get("page") for doc in retrieved_docs if doc.metadata.get("page") is not None]
    relevant_set = set(relevant_pages)
    retrieved_unique = set(retrieved_pages)
    tp = len(retrieved_unique & relevant_set)
    fp = len(retrieved_unique - relevant_set)
    fn = len(relevant_set - retrieved_unique)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"question": question, "TP": tp, "FP": fp, "FN": fn, "Precision": precision, "Recall": recall, "F1": f1}

def run_evaluation_suite(evaluation_data, retriever):
    results = [evaluate_retrieval(item["question"], item["relevant_pages"], retriever) for item in evaluation_data]
    return pd.DataFrame(results)
