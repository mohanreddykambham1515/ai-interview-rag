
from src.ingestion.pdf_loader import load_all_pdfs
from src.chunking.recursive_chunker import create_chunks
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridRetriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.compression.contextual_compressor import ContextualCompressor


def main():

    # 1. Load documents
    documents = load_all_pdfs("data/raw")

    # 2. Chunk documents
    chunks = create_chunks(
        documents,
        chunk_size=500,
        chunk_overlap=100
    )

    # 3. Initialize embedding model
    embedding_model = EmbeddingGenerator()

    # 4. Initialize Qdrant
    vector_store = VectorStore()

    # 5. Hybrid retrieval
    retriever = HybridRetriever(
        chunks=chunks,
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    query = (
        "How does the fraud detection model "
        "handle imbalanced data?"
    )

    retrieved_results = retriever.search(
        query=query,
        limit=10,
        candidate_limit=22
    )

    # 6. Reranking
    reranker = CrossEncoderReranker()

    reranked_results = reranker.rerank(
        query=query,
        documents=retrieved_results,
        top_k=5
    )

    print("\n========== BEFORE COMPRESSION ==========")

    for rank, result in enumerate(reranked_results, start=1):
        print(f"\nRank: {rank}")
        print("Content:", result["content"])

    # 7. Contextual compression
    compressor = ContextualCompressor(
        min_sentence_length=20,
        max_sentences=5
    )

    compressed_results = compressor.compress(
        query=query,
        documents=reranked_results
    )

    print("\n========== AFTER COMPRESSION ==========")

    for rank, result in enumerate(compressed_results, start=1):

        print(f"\nRank: {rank}")
        print("Source:", result["metadata"]["source"])
        print("Page:", result["metadata"]["page"])
        print("Original:", result["original_content"])
        print("Compressed:", result["content"])
        print("Compression details:", result["compression"])

    # 8. Calculate overall character reduction
    original_total = sum(
        len(doc["content"])
        for doc in reranked_results
    )

    compressed_total = sum(
        len(doc["content"])
        for doc in compressed_results
    )

    reduction = (
        (1 - compressed_total / original_total) * 100
        if original_total else 0
    )

    print("\n========== COMPRESSION SUMMARY ==========")
    print("Original characters:", original_total)
    print("Compressed characters:", compressed_total)
    print(f"Character reduction: {reduction:.2f}%")

    vector_store.close()


if __name__ == "__main__":
    main()
