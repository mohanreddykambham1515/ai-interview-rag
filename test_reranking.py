
from src.ingestion.pdf_loader import load_all_pdfs
from src.chunking.recursive_chunker import create_chunks
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridRetriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker


def main():

    # 1. Load documents
    documents = load_all_pdfs("data/raw")

    # 2. Create chunks
    chunks = create_chunks(
        documents,
        chunk_size=500,
        chunk_overlap=100
    )

    # 3. Load embedding model
    embedding_model = EmbeddingGenerator()

    # 4. Connect to Qdrant
    vector_store = VectorStore()

    # 5. Initialize hybrid retrieval
    retriever = HybridRetriever(
        chunks=chunks,
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    query = (
        "How does the fraud detection model "
        "handle imbalanced data?"
    )

    # 6. First-stage hybrid retrieval
    retrieved_results = retriever.search(
        query=query,
        limit=10,
        candidate_limit=22
    )

    print("\n========== BEFORE RERANKING ==========")

    for rank, result in enumerate(retrieved_results, start=1):
        print(f"\nRank: {rank}")
        print("RRF Score:", result["rrf_score"])
        print("Content:", result["content"][:250])

    # 7. Initialize cross-encoder
    reranker = CrossEncoderReranker()

    # 8. Rerank retrieved documents
    reranked_results = reranker.rerank(
        query=query,
        documents=retrieved_results,
        top_k=5
    )

    print("\n========== AFTER RERANKING ==========")

    for rank, result in enumerate(reranked_results, start=1):
        print(f"\nReranked Position: {rank}")
        print("Reranker Score:", result["rerank_score"])
        print("Original Dense Rank:", result["dense_rank"])
        print("Original BM25 Rank:", result["bm25_rank"])
        print("Source:", result["metadata"]["source"])
        print("Content:", result["content"][:350])

    # 9. Check direct answer
    print("\n========== DIRECT ANSWER CHECK ==========")

    found = False

    for rank, result in enumerate(reranked_results, start=1):

        if "scale_pos_weight" in result["content"].lower():
            found = True
            print("Direct answer found!")
            print("Final Reranked Position:", rank)
            print("Reranker Score:", result["rerank_score"])

    if not found:
        print("Direct answer is not in the final top 5.")

    vector_store.close()


if __name__ == "__main__":
    main()
