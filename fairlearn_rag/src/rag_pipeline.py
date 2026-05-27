"""
rag_pipeline.py
---------------
Core RAG pipeline using FAISS + SentenceTransformers + Qwen2.5-3B-Instruct.
"""

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple


# ─────────────────────────────────────────────
# 1. Embedder + Vector Store
# ─────────────────────────────────────────────

class FAISSDocumentStore:
    """Embed documents and store them in a FAISS index for fast retrieval."""

    def __init__(self, embed_model: str = "all-MiniLM-L6-v2"):
        print(f"[DocumentStore] Loading embedder: {embed_model}")
        self.embedder = SentenceTransformer(embed_model)
        self.index = None
        self.documents: List[str] = []

    def add_documents(self, docs: List[str]) -> None:
        """Embed and index a list of document strings."""
        self.documents = docs
        embeddings = self.embedder.encode(docs, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        print(f"[DocumentStore] Indexed {len(docs)} documents (dim={dim})")

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Return (document, distance) tuples for a query."""
        if self.index is None:
            raise RuntimeError("No documents indexed yet. Call add_documents() first.")
        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, top_k)
        return [(self.documents[i], float(distances[0][j])) for j, i in enumerate(indices[0])]


# ─────────────────────────────────────────────
# 2. Qwen2.5-3B Generator
# ─────────────────────────────────────────────

class QwenGenerator:
    """Wrapper around Qwen2.5-3B-Instruct for RAG answer generation."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        print(f"[Generator] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        print(f"[Generator] Model loaded on: {next(self.model.parameters()).device}")

    def generate(self, query: str, context_chunks: List[str], max_new_tokens: int = 256) -> str:
        """Generate an answer given a query and retrieved context chunks."""
        context = "\n\n".join(f"[Doc {i+1}]: {c}" for i, c in enumerate(context_chunks))
        prompt = (
            "You are a helpful medical assistant. "
            "Answer the question using ONLY the information in the context below. "
            "If the context does not contain enough information, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Extract only the generated answer part
        return full_text.split("Answer:")[-1].strip()


# ─────────────────────────────────────────────
# 3. Full RAG Pipeline
# ─────────────────────────────────────────────

class RAGPipeline:
    """End-to-end RAG: retrieve from FAISS, generate with Qwen."""

    def __init__(self, document_store: FAISSDocumentStore, generator: QwenGenerator, top_k: int = 3):
        self.store = document_store
        self.generator = generator
        self.top_k = top_k

    def run(self, query: str) -> dict:
        """Run the full pipeline and return a result dict."""
        retrieved = self.store.retrieve(query, top_k=self.top_k)
        chunks = [doc for doc, _ in retrieved]
        distances = [dist for _, dist in retrieved]
        answer = self.generator.generate(query, chunks)
        return {
            "query": query,
            "retrieved_chunks": chunks,
            "retrieval_distances": distances,
            "answer": answer,
        }
