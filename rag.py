import fitz
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ===========================================
# Hugging Face API Key
# ===========================================
# Best practice: Load from environment variables or configure your token safely
HF_TOKEN = os.environ.get("HF_TOKEN", "hf_ryGMsraSDOmqLTkjQIliRyIOmzoTDF8888")

client = InferenceClient(api_key=HF_TOKEN)

# Load Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def read_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

class VectorDB:
    def __init__(self, chunks):
        self.chunks = chunks
        embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
        embeddings = embeddings.astype("float32")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, question, top_k=3):
        query = embedding_model.encode([question], convert_to_numpy=True)
        query = query.astype("float32")
        distance, index = self.index.search(query, top_k)
        
        result = []
        for i in index[0]:
            if i < len(self.chunks):
                result.append(self.chunks[i])
        return result

def ask_llm(context, question):
    prompt = f"""
You are a helpful AI assistant.
Answer ONLY using the provided context.
If the answer is not available in the context, reply:
"I could not find the answer in the document."

Context:
{context}

Question:
{question}
"""
    try:
        completion = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"ERROR : {e}"

# Global reference for initialization in Flask
db_instance = None

def initialize_rag(pdf_path="documents/sampleTest.pdf"):
    global db_instance
    if db_instance is None:
        print("Initializing RAG System...")
        text = read_pdf(pdf_path)
        chunks = split_text(text)
        db_instance = VectorDB(chunks)
        print("RAG System Ready.")
    return db_instance