from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import util, SentenceTransformer
import pickle
import os
app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("Harsh0910/my_merged_model")
model = AutoModelForCausalLM.from_pretrained("Harsh0910/my_merged_model")
embedding_model = SentenceTransformer(model_name_or_path="all-mpnet-base-v2", device="cuda" if torch.cuda.is_available() else "cpu")

with open("Text-Generation//session_data.pkl", "rb") as session_file:
    session_data = pickle.load(session_file)

embeddings = session_data["embeddings"]
pages_and_chunks = session_data["pages_and_chunks"]

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    answer: str

def retrieve_relevant_resources(query, embeddings):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    device = embeddings.device  
    query_embedding = query_embedding.to(device) 

    dot_scores = util.dot_score(query_embedding, embeddings)[0]

    scores, indices = torch.topk(dot_scores, k=5)
    return scores, indices

def prompt_formatter(query: str, context_items: list[dict]) -> str:
    context = "- " + "\n- ".join([item["sentence_chunk"] for item in context_items])
    base_prompt = f"""GIVE THE ANSWER TO THIS FROM YOUR KNWOLEDGE AND EXCLUDE THE FOLLOWING CONTEXT. THE  query is at the last line:
    {context}
    Query: {query}
    Answer:"""
    return base_prompt

@app.post("/generate", response_model=QueryResponse)
async def generate(query_request: QueryRequest):
    query = query_request.query

    scores, indices = retrieve_relevant_resources(query, embeddings)
    context_items = [pages_and_chunks[i] for i in indices]
    prompt = prompt_formatter(query, context_items)

    input_ids = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**input_ids, max_new_tokens=256, temperature=0.7)
    output_text = tokenizer.decode(outputs[0])

    answer = output_text.replace(prompt, '').strip()
    return QueryResponse(query=query, answer=answer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)