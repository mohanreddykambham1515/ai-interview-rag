
from src.ingestion.pdf_loader import load_all_pdfs
from src.parent_child.parent_child_retriever import ParentChildRetriever
from src.parent_child.parent_child_hybrid import ParentChildHybridRetriever
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.query_transformation.query_rewriter import QueryRewriter


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

    # 3. Initialize embedding model
    embedding_model = EmbeddingGenerator()

    # 4. Initialize reranker
    reranker = CrossEncoderReranker()

    # 5. Initialize hybrid retriever
    retriever = ParentChildHybridRetriever(
        children=children,
        parents=chunker.parents,
        embedding_model=embedding_model,
        reranker=reranker
    )

    # 6. Index child chunks
    retriever.index_children(rebuild=True)

    # 7. Initialize query rewriter
    query_rewriter = QueryRewriter()

    original_query = "What about the imbalance problem?"

    conversation_context = """
    We are discussing a fraud detection machine learning project.
    The model uses XGBoost to detect fraudulent transactions.
    """

    # 8. Rewrite user query
    rewritten_query = query_rewriter.rewrite_query(
        query=original_query,
        conversation_context=conversation_context
    )

    print("\n========== QUERY TRANSFORMATION ==========")
    print("Original Query:", original_query)
    print("Rewritten Query:", rewritten_query)

    # 9. Retrieve using rewritten query
    results = retriever.search(
        query=rewritten_query,
        candidate_limit=22,
        rerank_top_k=5
    )

    print("\n========== RETRIEVED PARENTS ==========")

    for rank, parent in enumerate(results["parents"], start=1):

        print(f"\nParent Rank: {rank}")
        print("Source:", parent["metadata"]["source"])
        print("Page:", parent["metadata"]["page"])
        print("Content:", parent["content"])

    # 10. Verify answer evidence
    answer_found = any(
        "scale_pos_weight" in parent["content"].lower()
        for parent in results["parents"]
    )

    print("\n========== FINAL RESULT ==========")

    if answer_found:
        print("Success: Relevant answer context retrieved!")
    else:
        print("Relevant answer context not found.")

    retriever.close()


if __name__ == "__main__":
    main()
