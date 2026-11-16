
from src.ingestion.pdf_loader import load_all_pdfs
from src.parent_child.parent_child_retriever import ParentChildRetriever
from src.parent_child.parent_child_hybrid import ParentChildHybridRetriever
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.reranking.cross_encoder_reranker import CrossEncoderReranker


def main():

    # 1. Load documents
    documents = load_all_pdfs("data/raw")

    # 2. Create parent-child chunks
    chunker = ParentChildRetriever(
        parent_chunk_size=1000,
        parent_chunk_overlap=100,
        child_chunk_size=300,
        child_chunk_overlap=50
    )

    children = chunker.create_parent_child_chunks(documents)

    print("\nParents:", len(chunker.parents))
    print("Children:", len(children))

    # 3. Initialize models
    embedding_model = EmbeddingGenerator()
    reranker = CrossEncoderReranker()

    # 4. Initialize integrated retriever
    retriever = ParentChildHybridRetriever(
        children=children,
        parents=chunker.parents,
        embedding_model=embedding_model,
        reranker=reranker
    )

    # 5. Index children
    retriever.index_children(rebuild=True)

    # 6. Search
    query = (
        "How does the fraud detection model "
        "handle imbalanced data?"
    )

    results = retriever.search(
        query=query,
        candidate_limit=22,
        rerank_top_k=5
    )

    print("\n========== RERANKED CHILD RESULTS ==========")

    for rank, child in enumerate(results["children"], start=1):

        print(f"\nChild Rank: {rank}")
        print("Reranker Score:", child["rerank_score"])
        print("Dense Rank:", child["dense_rank"])
        print("BM25 Rank:", child["bm25_rank"])
        print("Source:", child["metadata"]["source"])
        print("Content:", child["content"])

    print("\n========== RECOVERED PARENTS ==========")

    for rank, parent in enumerate(results["parents"], start=1):

        print(f"\nParent Rank: {rank}")
        print("Source:", parent["metadata"]["source"])
        print("Page:", parent["metadata"]["page"])
        print("Reranker Score:", parent["rerank_score"])
        print("Matched child:", parent["matched_child"])
        print("Parent content:", parent["content"])

    # 7. Direct answer verification
    correct = any(
        "scale_pos_weight" in parent["content"].lower()
        for parent in results["parents"]
    )

    print("\n========== DIRECT ANSWER CHECK ==========")

    if correct:
        print("Success: scale_pos_weight found in recovered parents!")
    else:
        print("Direct answer not found in recovered parents.")

    retriever.close()


if __name__ == "__main__":
    main()
