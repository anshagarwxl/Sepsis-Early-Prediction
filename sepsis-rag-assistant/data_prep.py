from sentence_transformers import SentenceTransformer
import faiss, pickle

# 1. Load docs (could be PDF extracted text)
texts = ["chunk1 text...", "chunk2 text..."]

# 2. Encode with embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

# 3. Store in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save artifacts
pickle.dump(texts, open("data/processed/chunks.pkl", "wb"))
pickle.dump(["source1.pdf", "source2.pdf"], open("data/processed/sources.pkl", "wb"))
faiss.write_index(index, "data/processed/faiss_index")
