import fitz  # PyMuPDF
import os
from huggingface_hub import InferenceClient

# Pull the environment variable safely
HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(api_key=HF_TOKEN)

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

class SimpleServerlessDB:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, question, top_k=3):
        """
        Lightweight lookup to find the best matching chunks 
        without requiring PyTorch or compilation.
        """
        words = question.lower().split()
        scored_chunks = []
        for chunk in self.chunks:
            score = sum(1 for word in words if word in chunk.lower())
            scored_chunks.append((score, chunk))
        
        # Sort by the highest match count
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:top_k]]

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

db_instance = None

def initialize_rag(pdf_path):
    global db_instance
    if db_instance is None:
        print("Initializing Lightweight Serverless RAG...")
        text = read_pdf(pdf_path)
        chunks = split_text(text)
        db_instance = SimpleServerlessDB(chunks)
        print("System Ready.")
    return db_instance
