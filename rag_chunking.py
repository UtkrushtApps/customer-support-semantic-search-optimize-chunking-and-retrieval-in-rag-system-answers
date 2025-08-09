import re
from typing import List, Dict, Any

class SupportDocChunker:
    def __init__(self, chunk_size: int = 200, overlap: int = 40):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def tokenize(self, text: str) -> List[str]:
        """
        Simple whitespace tokenizer that splits text into tokens.
        Could be replaced by a more advanced tokenizer as needed.
        """
        # We'll treat tokens as words for a simple, robust approach
        return text.split()

    def detokenize(self, tokens: List[str]) -> str:
        return ' '.join(tokens)

    def chunk_text(self, text: str) -> List[str]:
        tokens = self.tokenize(text)
        chunks = []
        i = 0
        while i < len(tokens):
            chunk = tokens[i:i + self.chunk_size]
            if chunk:
                chunks.append(self.detokenize(chunk))
            i += self.chunk_size - self.overlap
        return chunks

    def chunk_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits one support document dict into a list of chunk-dicts,
        each including text and metadata.
        """
        text = doc['text']
        chunks = self.chunk_text(text)
        chunk_dicts = []
        for idx, chunk in enumerate(chunks):
            chunk_dicts.append({
                'text': chunk,
                'metadata': {
                    'category': doc.get('category', ''),
                    'priority': doc.get('priority', ''),
                    'date': doc.get('date', ''),
                    'doc_id': doc.get('id', ''),
                    'chunk_index': idx
                }
            })
        return chunk_dicts

    def process_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process all documents and yield all chunk dicts (ready for embedding/storage).
        """
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
