# Solution Steps

1. 1. Set up a chunking class (SupportDocChunker) with parameters for chunk size (200 tokens) and overlap (40 tokens).

2. 2. Implement tokenization and detokenization functions; here, use simple whitespace splitting, but this can be updated for more sophisticated needs.

3. 3. In the chunking class, write logic to process each document: tokenize text, apply sliding window chunking of 200 tokens with 40-token overlap, and create a metadata dict for each chunk containing category, priority, date, document id, and chunk index.

4. 4. Provide functions to process a batch of documents and return a flat list of chunk dicts (with corresponding metadata attached).

5. 5. Implement a vector DB retriever class (ChromaDocumentRetriever) that manages Chroma collections and sentence-transformer embeddings.

6. 6. In the retriever class, implement a method to add all chunks (with ids and metadata) into the Chroma database, using the selected transformer model for embedding.

7. 7. Add a semantic search function that, given a query string, embeds it, runs a top-k (k=5) nearest neighbor search with cosine similarity via Chroma, and returns the results (including distance, text, and metadata).

8. 8. For evaluation, provide a recall@k function which, for a list of test queries and reference answers, computes the recall of top-k retrieval containing the target passage (recall@5 metric).

9. 9. Write a main script that loads full support documents, runs the chunking pipeline, stores them in Chroma with all metadata, and demonstrates retrieval on a sample query.

10. 10. (Optional) Add a script section to evaluate recall@5 on a test/eval query set, if available.

