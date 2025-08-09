from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from tqdm import tqdm

class ChromaDocumentRetriever:
    def __init__(self, collection_name="support_chunks", persist_directory="chroma_data"):
        self.persist_directory = persist_directory
        self.client = chromadb.Client(Settings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        ))
        self.collection_name = collection_name
        # Use a sentence-transformer for embedding
        # You can switch models as appropriate
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        if self.collection_name in [c.name for c in self.client.list_collections()]:
            return self.client.get_collection(self.collection_name)
        else:
            return self.client.create_collection(self.collection_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts, show_progress_bar=False).tolist()

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Add chunk dicts with text & metadata to ChromaDB, if not already present.
        Each chunk must have a unique ID. Here, we use doc_id + chunk_index.
        """
        ids = [
            f"{chunk['metadata'].get('doc_id','noid')}_{chunk['metadata'].get('chunk_index',0)}"
            for chunk in chunks
        ]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        embeddings = self.embed_texts(texts)
        # Chroma upserts (adds or updates)
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def semantic_search(self, query: str, k: int = 5, filter: Dict = None) -> List[Dict[str, Any]]:
        query_emb = self.embed_texts([query])[0]
        # Chroma's query method uses cosine sim under the hood
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=k,
            where=filter or {},
            include=["documents", "embeddings", "metadatas", "distances"]
        )
        formatted = []
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        for doc, meta, dist in zip(docs, metadatas, distances):
            formatted.append({
                "text": doc,
                "metadata": meta,
                "cosine_distance": dist
            })
        return formatted

    def recall_at_k(self, queries: List[Dict[str, Any]], k: int = 5, answer_keys=['answer']):
        """
        For each query, checks if reference answer is among top-k retrieved chunks.
        queries: list of dicts with 'question' (str) and 'answer' (str or list of strings)
        answer_keys allows to check alternate keys for ground truth.
        """
        hits = 0
        for q in tqdm(queries, desc="Recall@{} eval".format(k)):
            q_text = q['question']
            ref_answers = q[answer_keys[0]] if answer_keys and answer_keys[0] in q else q['answer']
            if isinstance(ref_answers, str):
                ref_answers = [ref_answers]
            results = self.semantic_search(q_text, k=k)
            found = False
            for cand in results:
                for ref in ref_answers:
                    if ref.strip().lower() in cand['text'].strip().lower():
                        found = True
                        break
                if found:
                    break
            if found:
                hits +=1
        recall = hits/len(queries) if queries else 0.0
        print(f"Recall@{k}: {recall:.3f} ({hits}/{len(queries)})")
        return recall
