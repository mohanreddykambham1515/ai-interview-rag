
from src.ingestion.pdf_loader import load_all_pdfs
from src.chunking.recursive_chunker import create_chunks
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridRetriever


def main():

    documents = load_all_pdfs("data/raw")

    chunks = create_chunks(
        documents,
        chunk_size=500,
        chunk_overlap=100
    )

    embedding_model = EmbeddingGenerator()
    vector_store = VectorStore()

    retriever = HybridRetriever(
        chunks=chunks,
        embedding_model=embedding_model,
        vector_store=vector_store
    )

    query = "How does the fraud detection model handle imbalanced data?"

    filters = {
        "source": "01_Project_Brief.pdf"
    }

    results = retriever.search(
        query=query,
        limit=10,
        candidate_limit=22,
        filters=filters
    )

    print("\n========== METADATA FILTERING ==========")
    print("Query:", query)
    print("Filters:", filters)

    for rank, result in enumerate(results, start=1):

        print(f"\nResult: {rank}")
        print("RRF Score:", result["rrf_score"])
        print("Source:", result["metadata"]["source"])
        print("Page:", result["metadata"]["page"])
        print("Content:", result["content"][:300])

        assert result["metadata"]["source"] == (
            "01_Project_Brief.pdf"
        )

    print("\nMetadata filtering test completed successfully!")


if __name__ == "__main__":
    main()
