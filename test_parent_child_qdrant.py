
from src.ingestion.pdf_loader import load_all_pdfs
from src.parent_child.parent_child_retriever import ParentChildRetriever
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.parent_child.parent_child_store import ParentChildStore


def main():

    # 1. Load PDFs
    documents = load_all_pdfs("data/raw")

    # 2. Create parent-child chunks
    parent_child = ParentChildRetriever(
        parent_chunk_size=1000,
        parent_chunk_overlap=100,
        child_chunk_size=300,
        child_chunk_overlap=50
    )

    children = parent_child.create_parent_child_chunks(
        documents
    )

    # 3. Initialize embeddings
    embedding_model = EmbeddingGenerator()

    # 4. Initialize parent-child store
    store = ParentChildStore(
        embedding_model=embedding_model
    )

    # 5. Preserve complete parent documents
    store.parents = parent_child.parents.copy()
    store.save_parents()

    # 6. Index child vectors
    store.index_children(children)

    # 7. Search
    query = (
        "How does the fraud detection model "
        "handle imbalanced data?"
    )

    child_results = store.search_children(
        query=query,
        limit=5
    )

    print("\n========== CHILD SEARCH RESULTS ==========")

    for rank, result in enumerate(child_results, start=1):

        print(f"\nChild Rank: {rank}")
        print("Score:", result.score)
        print("Parent ID:", result.payload["parent_id"])
        print("Source:", result.payload["source"])
        print("Content:", result.payload["content"])

    # 8. Recover parents
    parent_results = store.retrieve_parents(
        child_results
    )

    print("\n========== PARENT RETRIEVAL RESULTS ==========")

    for rank, parent in enumerate(parent_results, start=1):

        print(f"\nParent Rank: {rank}")
        print("Parent ID:", parent["parent_id"])
        print("Source:", parent["metadata"]["source"])
        print("Page:", parent["metadata"]["page"])
        print("Content:", parent["content"][:500])

    print("\nParent-child Qdrant retrieval completed!")

    store.close()


if __name__ == "__main__":
    main()
