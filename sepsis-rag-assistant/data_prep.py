<<<<<<< HEAD
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
=======
"""Data preparation script for sepsis RAG system.

Processes text documents to create FAISS index for similarity search.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process documents for RAG system."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        
    def load_documents(self, data_dir: str = "data") -> List[Tuple[str, str]]:
        """Load documents from data directory.
        
        Returns:
            List of (text_content, source_path) tuples
        """
        documents = []
        data_path = Path(data_dir)
        
        # Look for text files
        for file_path in data_path.rglob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if content:
                    documents.append((content, str(file_path)))
                    logger.info(f"Loaded: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        # If no txt files found, create sample data
        if not documents:
            logger.warning("No text files found. Creating sample sepsis data...")
            documents = self._create_sample_data()
        
        return documents
    
    def _create_sample_data(self) -> List[Tuple[str, str]]:
        """Create sample sepsis-related data."""
        sample_docs = [
            ("Sepsis is a life-threatening condition that occurs when the body's response to infection damages its own tissues and organs. Early recognition and treatment are critical for patient survival.", "sample_sepsis_definition.txt"),
            
            ("The qSOFA (quick Sequential Organ Failure Assessment) criteria include: respiratory rate ≥22/min, altered mental status (GCS <15), and systolic blood pressure ≤100 mmHg. Two or more criteria suggest possible sepsis.", "sample_qsofa.txt"),
            
            ("SIRS (Systemic Inflammatory Response Syndrome) criteria include: temperature >38°C or <36°C, heart rate >90 bpm, respiratory rate >20/min or PaCO2 <32 mmHg, and white blood cell count >12,000 or <4,000 cells/μL or >10% immature forms.", "sample_sirs.txt"),
            
            ("Early sepsis management includes: obtain blood cultures before antibiotics, measure lactate levels, administer broad-spectrum antibiotics within 1 hour, and provide IV fluid resuscitation (30 mL/kg crystalloid for hypotension or lactate ≥4 mmol/L).", "sample_treatment.txt"),
            
            ("Septic shock is defined as sepsis with persisting hypotension requiring vasopressors to maintain MAP ≥65 mmHg and having a serum lactate level >2 mmol/L despite adequate volume resuscitation.", "sample_shock.txt"),
            
            ("NEWS2 (National Early Warning Score 2) includes: respiratory rate, oxygen saturation, temperature, systolic blood pressure, heart rate, and level of consciousness. Scores ≥7 suggest clinical deterioration.", "sample_news2.txt"),
            
            ("Source control in sepsis may require drainage of infected fluid collections, debridement of infected necrotic tissue, removal of infected devices, and definitive control of infection source.", "sample_source_control.txt"),
            
            ("Antibiotic therapy should be empiric and broad-spectrum initially, then de-escalated based on culture results and clinical response. Duration is typically 7-10 days unless complications occur.", "sample_antibiotics.txt"),
            
            ("Biomarkers for sepsis include procalcitonin (PCT), C-reactive protein (CRP), and lactate. PCT levels >0.5 ng/mL suggest bacterial infection, while lactate >2 mmol/L indicates tissue hypoperfusion.", "sample_biomarkers.txt"),
            
            ("Organ dysfunction in sepsis affects multiple systems: cardiovascular (hypotension, shock), respiratory (ARDS), renal (acute kidney injury), hepatic (elevated bilirubin), hematologic (thrombocytopenia), and neurologic (altered mental status).", "sample_organ_dysfunction.txt")
        ]
        
        # Create data directory if it doesn't exist
        os.makedirs("data/guidelines", exist_ok=True)
        
        # Save sample files
        for content, filename in sample_docs:
            filepath = f"data/guidelines/{filename}"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return sample_docs
    
    def chunk_documents(self, documents: List[Tuple[str, str]], 
                       chunk_size: int = 500, overlap: int = 50) -> List[Tuple[str, str]]:
        """Split documents into chunks.
        
        Args:
            documents: List of (content, source) tuples
            chunk_size: Maximum characters per chunk
            overlap: Character overlap between chunks
            
        Returns:
            List of (chunk_text, source) tuples
        """
        chunks = []
        
        for content, source in documents:
            if len(content) <= chunk_size:
                chunks.append((content, source))
            else:
                # Split into overlapping chunks
                start = 0
                while start < len(content):
                    end = start + chunk_size
                    chunk = content[start:end]
                    
                    # Try to break at word boundary
                    if end < len(content):
                        last_space = chunk.rfind(' ')
                        if last_space > start + chunk_size // 2:
                            chunk = chunk[:last_space]
                            end = start + last_space
                    
                    chunks.append((chunk.strip(), source))
                    start = end - overlap
                    
                    if start >= len(content):
                        break
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def create_embeddings(self, chunks: List[Tuple[str, str]]) -> Tuple[np.ndarray, List[str], List[str]]:
        """Create embeddings for text chunks.
        
        Returns:
            Tuple of (embeddings_array, chunk_texts, sources)
        """
        texts = [chunk for chunk, _ in chunks]
        sources = [source for _, source in chunks]
        
        logger.info(f"Creating embeddings for {len(texts)} chunks...")
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        return embeddings, texts, sources
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index for similarity search."""
        dimension = embeddings.shape[1]
        logger.info(f"Building FAISS index with dimension {dimension}")
        
        # Use IndexFlatIP for cosine similarity
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype(np.float32))
        
        logger.info(f"FAISS index built with {index.ntotal} vectors")
        return index
    
    def save_processed_data(self, index: faiss.Index, chunks: List[str], 
                           sources: List[str], output_dir: str = "data/processed"):
        """Save processed data to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(output_dir, "faiss_index.index")
        faiss.write_index(index, index_path)
        logger.info(f"FAISS index saved to {index_path}")
        
        # Save chunks
        chunks_path = os.path.join(output_dir, "chunks.pkl")
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)
        logger.info(f"Chunks saved to {chunks_path}")
        
        # Save sources
        sources_path = os.path.join(output_dir, "sources.pkl")
        with open(sources_path, "wb") as f:
            pickle.dump(sources, f)
        logger.info(f"Sources saved to {sources_path}")
        
        # Save metadata
        meta = {
            "num_chunks": len(chunks),
            "embedding_dimension": index.d,
            "model_name": self.model_name,
            "index_type": type(index).__name__
        }
        
        meta_path = os.path.join(output_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Metadata saved to {meta_path}")
        
        return output_dir

def main():
    """Main data preparation workflow."""
    try:
        # Initialize processor
        processor = DocumentProcessor()
        
        # Load documents
        documents = processor.load_documents("data")
        if not documents:
            logger.error("No documents found")
            return
        
        # Chunk documents
        chunks = processor.chunk_documents(documents)
        
        # Create embeddings
        embeddings, chunk_texts, sources = processor.create_embeddings(chunks)
        
        # Build FAISS index
        index = processor.build_faiss_index(embeddings)
        
        # Save processed data
        output_dir = processor.save_processed_data(index, chunk_texts, sources)
        
        logger.info(f"Data preparation completed successfully!")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Total chunks: {len(chunk_texts)}")
        logger.info(f"Embedding dimension: {embeddings.shape[1]}")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise

if __name__ == "__main__":
    main()
>>>>>>> 13f805555b3536fde72e37e09b2bb701e594ca63
