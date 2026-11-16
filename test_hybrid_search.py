
from src.ingestion.pdf_loader import load_all_pdfs
from src.chunking.recursive_chunker import create_chunks
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridRetriever, expand_query


def main():

    # 1. Load documents
    documents = load_all_pdfs("data/raw")

    # 2. Create chunks
    chunks = create_chunks(
        documents,
        chunk_size=500,
        chunk_overlap=100
    )

    # 3. Initialize embedding model
    embedding_model = EmbeddingGenerator()

    # 4. Initialize vector store
    vector_store = VectorStore()

    # 5. Initialize hybrid retriever
    retriever = HybridRetriever(
        chunks=chunks,
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    # 6. User query
    query = "How does the fraud detection model handle imbalanced data?"

    print("\n========== QUERY EXPANSION ==========")
    print("Original:", query)
    print("Expanded:", expand_query(query))

    # 7. Hybrid search
    results = retriever.search(
        query=query,
        limit=10,
        candidate_limit=22
    )

    print("\n========== HYBRID SEARCH RESULTS ==========")

    for rank, result in enumerate(results, start=1):

        print(f"\nResult: {rank}")
        print(f"RRF Score: {result['rrf_score']:.6f}")
        print(f"Dense Rank: {result['dense_rank']}")
        print(f"BM25 Rank: {result['bm25_rank']}")
        print(f"Source: {result['metadata']['source']}")
        print(f"Page: {result['metadata']['page']}")
        print(f"Content:\n{result['content']}")

    # 8. Check whether the direct answer was retrieved
    print("\n========== DIRECT ANSWER CHECK ==========")

    found = False

    for rank, result in enumerate(results, start=1):

        if "scale_pos_weight" in result["content"].lower():

            found = True

            print("Correct answer retrieved!")
            print("Hybrid Rank:", rank)
            print("Dense Rank:", result["dense_rank"])
            print("BM25 Rank:", result["bm25_rank"])

    if not found:
        print("Direct answer not found in the top results.")


if __name__ == "__main__":
    main()
