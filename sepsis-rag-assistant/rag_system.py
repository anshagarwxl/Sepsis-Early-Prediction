import pickle, faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

class SepsisRAG:
    def __init__(self, data_dir="data/processed", k=3):
        # Load artifacts
        self.chunks = pickle.load(open(f"{data_dir}/chunks.pkl", "rb"))
        self.sources = pickle.load(open(f"{data_dir}/sources.pkl", "rb"))
        self.index = faiss.read_index(f"{data_dir}/faiss_index")
        
        # Embeddings + OpenAI client
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.k = k

    def query(self, question: str):
        # 1. Embed query
        q_emb = self.model.encode([question])
        
        # 2. Search FAISS
        D, I = self.index.search(q_emb, self.k)
        retrieved_chunks = [self.chunks[i] for i in I[0]]
        retrieved_sources = [self.sources[i] for i in I[0]]
        
        # 3. Prompt LLM
        context = "\n".join(retrieved_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer with clinical guidance:"
        
        resp = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0
        )
        
        answer = resp.choices[0].message.content
        return answer, retrieved_sources
