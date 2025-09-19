"""RAG system implementation for sepsis-rag-assistant.

Loads FAISS index and text chunks produced by data_prep.py, embeds queries
with SentenceTransformer, performs nearest-neighbor search, and returns
an answer string with source references.

Designed to work offline (pure retrieval). If OPENAI_API_KEY is present, we
optionally summarize with an LLM; otherwise we return a concise extractive
answer composed from top chunks.
"""

from __future__ import annotations

import os
import json
import pickle
import logging
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Optional LLM summarization (only if key is present)
try:
	from openai import OpenAI  # type: ignore
	_OPENAI_AVAILABLE = True
except Exception:
	_OPENAI_AVAILABLE = False

# --- Configuration ---
try:
	from config.settings import GUIDELINES_PATH, PROCESSED_PATH
except Exception:
	GUIDELINES_PATH = "data/guidelines/"
	PROCESSED_PATH = "data/processed/"

MODEL_CACHE_DIR = os.path.join("models", "sentence-transformers")
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SepsisRAG:
	"""Load retrieval artifacts and answer clinical questions with sources."""

	def __init__(self, processed_path: str = PROCESSED_PATH) -> None:
		self.processed_path = processed_path

		# Check required files
		meta_path = os.path.join(self.processed_path, "meta.json")
		faiss_path = os.path.join(self.processed_path, "faiss_index")
		chunks_path = os.path.join(self.processed_path, "chunks.pkl")
		sources_path = os.path.join(self.processed_path, "sources.pkl")

		missing = [p for p in [meta_path, faiss_path, chunks_path, sources_path] if not os.path.exists(p)]
		if missing:
			raise RuntimeError(
				"Processed data missing. Run data_prep.py first. Missing: " + ", ".join(missing)
			)

		# Load metadata
		with open(meta_path, "r", encoding="utf-8") as f:
			self.meta = json.load(f)

		model_name = self.meta.get("model_name", DEFAULT_MODEL_NAME)
		os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
		logging.info("Loading embedding model '%s'...", model_name)
		self.model = SentenceTransformer(model_name, cache_folder=MODEL_CACHE_DIR)

		# Load retrieval artifacts
		self.index = faiss.read_index(faiss_path)
		with open(chunks_path, "rb") as f:
			self.chunks: List[str] = pickle.load(f)
		with open(sources_path, "rb") as f:
			self.sources: List[str] = pickle.load(f)

		if len(self.chunks) != self.index.ntotal:
			logging.warning(
				"Chunks count (%d) does not match index vectors (%d).",
				len(self.chunks), self.index.ntotal,
			)

		# Optional LLM client
		self._client = None
		if _OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
			try:
				self._client = OpenAI()
			except Exception as e:
				logging.warning("OpenAI client init failed: %s", e)

	def _search(self, query: str, top_k: int = 5) -> Tuple[List[int], List[float]]:
		q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
		D, I = self.index.search(q_emb, top_k)
		idxs = I[0].tolist()
		dists = D[0].tolist()
		return idxs, dists

	def _format_context(self, idxs: List[int]) -> str:
		ctx_parts = []
		for i in idxs:
			snippet = self.chunks[i]
			src = self.sources[i] if i < len(self.sources) else "unknown"
			ctx_parts.append(f"[Source: {src}]\n{snippet}")
		return "\n\n".join(ctx_parts)

	def _answer_locally(self, query: str, idxs: List[int]) -> str:
		# Simple extractive approach: return concatenated top snippets with a short preface
		preview = []
		for i in idxs:
			txt = self.chunks[i].strip()
			if len(txt) > 500:
				txt = txt[:500] + "..."
			src = self.sources[i] if i < len(self.sources) else "unknown"
			preview.append(f"• {txt}\n  — {src}")
		if not preview:
			return "No relevant context found. Try rephrasing the question."
		return (
			"Here are relevant guideline excerpts (no LLM summarization active):\n\n"
			+ "\n\n".join(preview)
		)

	def _answer_with_llm(self, query: str, idxs: List[int]) -> str:
		if not self._client:
			return self._answer_locally(query, idxs)
		context = self._format_context(idxs)
		prompt = (
			"You are a clinical assistant. Using only the provided context from\n"
			"evidence-based guidelines, answer the user's question succinctly.\n"
			"If unsure, say so.\n\nContext:\n" + context
		)
		try:
			resp = self._client.chat.completions.create(
				model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
				messages=[
					{"role": "system", "content": "You are a helpful clinical assistant."},
					{"role": "user", "content": prompt},
				],
				temperature=0.2,
				max_tokens=400,
			)
			return resp.choices[0].message.content or self._answer_locally(query, idxs)
		except Exception as e:
			logging.warning("LLM call failed: %s", e)
			return self._answer_locally(query, idxs)

	def query(self, question: str, patient_scores=None, top_k: int = 5):
		"""Return (answer, sources) for a clinical question.

		patient_scores can include risk metrics to optionally tailor the answer
		in the future. For now, we ignore it except for prompt context (LLM path).
		"""
		idxs, _ = self._search(question, top_k=top_k)
		sources = [self.sources[i] if i < len(self.sources) else "unknown" for i in idxs]

		# Prefer LLM summarization if available; otherwise extractive
		answer = self._answer_with_llm(question, idxs) if self._client else self._answer_locally(question, idxs)
		return answer, sources

