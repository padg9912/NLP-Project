from retriever import retrieve_with_timestamp, get_vector_db
from langchain_huggingface import HuggingFaceEmbeddings
from reranker import rerank_by_time_and_relevance
from ingest import to_unix_timestamp
import torch
from transformers import BertTokenizer
from bert_classification import predict, BertForSequenceClassification
from llm_classifier import call_gemini
import requests

def call_ollama(model_name, prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data)
    return response.json()["response"]

def synthesize_evidence_with_ollama(claim, date, docs, model_name):
    prompt = f"""
Given the following claim and supporting/contradictory evidence, provide a concise summary and explain whether the evidence supports, refutes, or is insufficient to judge the claim. Then, state your verdict (true/false/unknown) and explain your reasoning step by step.

Claim: {claim}
Date: {date}

Evidence:
"""
    for i, doc in enumerate(docs):
        prompt += f"{i+1}. \"{doc.page_content}\" (Date: {doc.metadata.get('time_stamp')}, Truth: {doc.metadata.get('truthfulness')})\n"
    prompt += """

Your response should include:
- A summary of the evidence.
- A step-by-step explanation.
- A final verdict (true/false/unknown).
"""
    return call_ollama(model_name, prompt)

def synthesize_evidence_with_llm(claim, date, docs):
    prompt = f"""
Given the following claim and supporting/contradictory evidence, provide a concise summary and explain whether the evidence supports, refutes, or is insufficient to judge the claim. Then, state your verdict (true/false/unknown) and explain your reasoning step by step.

Claim: {claim}
Date: {date}

Evidence:
"""
    for i, doc in enumerate(docs):
        prompt += f"{i+1}. \"{doc.page_content}\" (Date: {doc.metadata.get('time_stamp')}, Truth: {doc.metadata.get('truthfulness')})\n"
    prompt += """

Your response should include:
- A summary of the evidence.
- A step-by-step explanation.
- A final verdict (true/false/unknown).
"""
    return call_gemini(prompt, model="2.0-flash-lite")

def full_pipeline(query, query_timestamp, bert_model_path='outputs/bert_classifier.pt', top_k=3, llm_models=None):
    """
    Complete pipeline: retrieval -> reranking -> classification
    """
    # Step 1: Retrieve documents
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = get_vector_db()
    timestamp_unix = to_unix_timestamp(query_timestamp)
    
    retrieved_docs = retrieve_with_timestamp(query, before_time=timestamp_unix, db=db, k=top_k*2)
    print(f"\nRetrieved {len(retrieved_docs)} documents")
    
    # Step 2: Rerank documents
    reranked_docs = rerank_by_time_and_relevance(
        query, 
        timestamp_unix, 
        retrieved_docs, 
        embedding_model, 
        top_k=top_k,
        alpha=0.8, 
        lmbda=0.5
    )
    print(f"\nReranked to {len(reranked_docs)} documents")
    
    # Step 3: Use BERT to classify the claim and provide supporting evidence
    # Load BERT model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3,  # Three labels: false (0), unknown (1), true (2)
        output_attentions=False,
        output_hidden_states=False
    )
    model.load_state_dict(torch.load(bert_model_path, map_location=device))
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Get classification result
    classification_result = predict(query, model, tokenizer, device)
    
    # Print results
    print(f"\n==== Results for query: '{query}' ====")
    print(f"Date: {query_timestamp}")
    print(f"\nVerdict: {classification_result['prediction']}")
    
    # Print confidence scores for all three classes
    print(f"Confidence: True: {classification_result['confidence']['True']:.4f}, " +
          f"False: {classification_result['confidence']['False']:.4f}, " +
          f"Unknown: {classification_result['confidence']['Unknown']:.4f}")
    
    print("\nSupporting evidence:")
    for i, doc in enumerate(reranked_docs):
        print(f"\n[{i+1}] {doc.page_content}")
        print(f"    Date: {doc.metadata.get('time_stamp')}")
        print(f"    Truth: {doc.metadata.get('truthfulness')}")
    
    # Step 3: LLM-based evidence synthesis and explainability (multiple models)
    llm_outputs = {}
    if llm_models is None:
        llm_models = ["llama3", "mistral", "qwen2", "gemma"]
    for model_name in llm_models:
        llm_outputs[model_name] = synthesize_evidence_with_ollama(query, query_timestamp, reranked_docs, model_name)
    
    return {
        'classification': classification_result,
        'supporting_docs': reranked_docs,
        'llm_outputs': llm_outputs
    }

if __name__ == "__main__":
    # Example usage
    query = "Most Americans have committed crimes worthy of prison time"
    query_timestamp = "12/8/2014"
    
    result = full_pipeline(query, query_timestamp)

