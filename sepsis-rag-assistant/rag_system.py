#rag_system
<<<<<<< HEAD
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
=======
"""RAG system implementation for sepsis-rag-assistant.

Loads FAISS index and text chunks produced by data_prep.py, embeds queries
with SentenceTransformer, performs nearest-neighbor search, and returns
an answer string with source references.

Now uses Google Gemini API instead of OpenAI for better performance and cost efficiency.
"""

from __future__ import annotations

import os
import json
import pickle
import logging
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Google Gemini API
try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except Exception:
    _GEMINI_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SepsisRAG:
    """RAG system for sepsis diagnosis and management."""
    
    def __init__(self, 
                 data_dir: str = "data/processed",
                 model_name: str = "all-MiniLM-L6-v2",
                 gemini_api_key: Optional[str] = None,
                 gemini_model: str = "gemini-1.5-flash-latest"):
        """Initialize the RAG system.
        
        Args:
            data_dir: Directory containing processed data files
            model_name: SentenceTransformer model name
            gemini_api_key: Google Gemini API key
            gemini_model: Gemini model name to use
        """
        self.data_dir = data_dir
        self.model_name = model_name
        self.gemini_model = gemini_model
        
        # Configure Gemini if API key provided
        self.gemini_available = False
        if gemini_api_key and _GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_client = genai.GenerativeModel(gemini_model)
                self.gemini_available = True
                logger.info(f"Gemini API configured with model: {gemini_model}")
            except Exception as e:
                logger.warning(f"Failed to configure Gemini API: {e}")
        elif not _GEMINI_AVAILABLE:
            logger.warning("google-generativeai not available. Install with: pip install google-generativeai")
        
        # Load components with eager loading
        self._load_components()
    
    def _load_components(self):
        """Load FAISS index, embeddings model, and data."""
        try:
            # Load SentenceTransformer model
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.encoder = SentenceTransformer(self.model_name)
            
            # Load FAISS index
            index_path = os.path.join(self.data_dir, "faiss_index.index")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index not found at {index_path}. Run data_prep.py first.")
            
            logger.info(f"Loading FAISS index from {index_path}")
            self.index = faiss.read_index(index_path)
            
            # Load text chunks
            chunks_path = os.path.join(self.data_dir, "chunks.pkl")
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            
            # Load source information
            sources_path = os.path.join(self.data_dir, "sources.pkl")
            with open(sources_path, "rb") as f:
                self.sources = pickle.load(f)
            
            # Load metadata
            meta_path = os.path.join(self.data_dir, "meta.json")
            with open(meta_path, "r") as f:
                self.meta = json.load(f)
            
            logger.info(f"RAG system loaded successfully:")
            logger.info(f"  - {len(self.chunks)} text chunks loaded")
            logger.info(f"  - FAISS index dimension: {self.index.d}")
            logger.info(f"  - Embedding model: {self.model_name}")
            logger.info(f"  - Gemini API: {'✓' if self.gemini_available else '✗'}")
            
        except Exception as e:
            logger.error(f"Failed to load RAG components: {e}")
            raise
    
    def _retrieve_chunks(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        """Retrieve top-k relevant chunks for the query.
        
        Args:
            query: User query string
            k: Number of chunks to retrieve
            
        Returns:
            List of (chunk_text, source, similarity_score) tuples
        """
        try:
            # Encode query
            query_embedding = self.encoder.encode([query])
            
            # Search FAISS index
            scores, indices = self.index.search(query_embedding.astype(np.float32), k)
            
            # Collect results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1:  # Valid index
                    chunk_text = self.chunks[idx]
                    source = self.sources[idx]
                    similarity = float(score)
                    results.append((chunk_text, source, similarity))
            
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def _generate_answer(self, query: str, retrieved_chunks: List[Tuple[str, str, float]], 
                        patient_scores: Optional[Dict[str, Any]] = None) -> str:
        """Generate answer using Gemini API or fallback to extractive method.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved (chunk, source, score) tuples
            patient_scores: Optional patient vital signs and scores
            
        Returns:
            Generated answer string
        """
        if not retrieved_chunks:
            return "I couldn't find relevant information to answer your query. Please try rephrasing or asking about sepsis diagnosis, treatment, or management."
        
        # Prepare context from retrieved chunks
        context_parts = []
        for i, (chunk, source, score) in enumerate(retrieved_chunks):
            context_parts.append(f"[{i+1}] {chunk.strip()}")
        
        context = "\n\n".join(context_parts)
        
        # Add patient context if available
        patient_context = ""
        if patient_scores:
            patient_context = f"\n\nPatient Context:\n{self._format_patient_scores(patient_scores)}"
        
        if self.gemini_available:
            return self._generate_with_gemini(query, context, patient_context)
        else:
            return self._generate_extractive(query, retrieved_chunks)
    
    def _generate_with_gemini(self, query: str, context: str, patient_context: str = "") -> str:
        """Generate answer using Gemini API."""
        try:
            prompt = f"""You are a medical AI assistant specializing in sepsis diagnosis and management. 
Answer the user's question based on the provided medical context. Be accurate, concise, and clinical.

Context from medical literature:
{context}
{patient_context}

Question: {query}

Instructions:
- Provide a clear, evidence-based answer
- Reference specific criteria, scores, or guidelines when applicable
- If patient data is provided, relate your answer to their specific situation
- Keep the response focused and clinically relevant
- If you cannot answer based on the context, say so clearly

Answer:"""

            response = self.gemini_client.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._generate_extractive(query, [(context, "medical literature", 1.0)])
    
    def _generate_extractive(self, query: str, retrieved_chunks: List[Tuple[str, str, float]]) -> str:
        """Generate extractive answer by combining relevant chunks."""
        if not retrieved_chunks:
            return "No relevant information found."
        
        # Take the most relevant chunks and combine them
        top_chunks = retrieved_chunks[:3]  # Top 3 most relevant
        
        answer_parts = []
        for chunk, source, score in top_chunks:
            # Clean and truncate chunk if too long
            clean_chunk = chunk.strip()
            if len(clean_chunk) > 200:
                clean_chunk = clean_chunk[:200] + "..."
            answer_parts.append(clean_chunk)
        
        combined_answer = " ".join(answer_parts)
        
        # Add a note about the extractive nature
        return f"Based on the available medical literature: {combined_answer}"
    
    def _format_patient_scores(self, scores: Dict[str, Any]) -> str:
        """Format patient scores for context."""
        parts = []
        for key, value in scores.items():
            if value is not None:
                parts.append(f"{key}: {value}")
        return ", ".join(parts) if parts else "No patient data available"
    
    def query(self, question: str, k: int = 5, patient_scores: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
        """Answer a question using RAG.
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            patient_scores: Optional patient vital signs and scores
            
        Returns:
            Tuple of (answer, list_of_sources)
        """
        try:
            # Retrieve relevant chunks
            retrieved_chunks = self._retrieve_chunks(question, k)
            
            if not retrieved_chunks:
                return ("I couldn't find relevant information to answer your question. "
                       "Please try asking about sepsis diagnosis, treatment, or management.", [])
            
            # Generate answer
            answer = self._generate_answer(question, retrieved_chunks, patient_scores)
            
            # Extract unique sources
            sources = list(set(source for _, source, _ in retrieved_chunks))
            
            return answer, sources
            
        except Exception as e:
            logger.error(f"Error during query processing: {e}")
            return (f"An error occurred while processing your question: {str(e)}", [])

def main():
    """Test the RAG system."""
    try:
        # Initialize RAG system
        api_key = os.getenv("GEMINI_API_KEY")
        rag = SepsisRAG(gemini_api_key=api_key)
        
        # Test queries
        test_queries = [
            "What are the qSOFA criteria for sepsis?",
            "How is sepsis diagnosed?",
            "What are the early signs of sepsis?",
            "What is the treatment for sepsis?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            answer, sources = rag.query(query)
            print(f"Answer: {answer}")
            print(f"Sources: {sources}")
            print("-" * 80)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
>>>>>>> 13f805555b3536fde72e37e09b2bb701e594ca63
