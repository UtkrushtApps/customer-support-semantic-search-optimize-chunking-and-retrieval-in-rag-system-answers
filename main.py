import json
from rag_chunking import SupportDocChunker
from retriever import ChromaDocumentRetriever

# 1. Load documents
# documents = [ {'id':..., 'text':..., 'category':..., 'priority':..., 'date':...}, ...]
with open('support_documents.json', 'r') as f:
    documents = json.load(f)

# 2. Chunk and decorate with metadata
chunker = SupportDocChunker(chunk_size=200, overlap=40)
chunks = chunker.process_documents(documents)
print(f"Total chunks: {len(chunks)}")

# 3. Ingest chunks into Chroma
retriever = ChromaDocumentRetriever(collection_name="support_chunks")
retriever.add_chunks(chunks)
print("Chunks stored in Chroma DB.")

# 4. Example: retrieve relevant answers for a support query
query = "How can I reset my password if I lost access to my email?"
results = retriever.semantic_search(query, k=5)
print("Top-5 retrieved chunks:")
for r in results:
    print(f"[cat:{r['metadata']['category']}] (priority:{r['metadata']['priority']}) {r['text']} (score: {1 - r['cosine_distance']:.3f})")

# 5. (Optional) Evaluate recall@5 for a set of queries
# with open('eval_queries.json', 'r') as f:
#     eval_queries = json.load(f)  # [{'question':..., 'answer':...}, ...]
# retriever.recall_at_k(eval_queries, k=5)
