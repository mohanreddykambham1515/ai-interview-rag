
from src.ingestion.pdf_loader import load_all_pdfs

from src.parent_child.parent_child_retriever import (
    ParentChildRetriever
)

from src.parent_child.parent_child_hybrid import (
    ParentChildHybridRetriever
)

from src.embeddings.embedding_generator import (
    EmbeddingGenerator
)

from src.reranking.cross_encoder_reranker import (
    CrossEncoderReranker
)

from src.query_transformation.query_rewriter import (
    QueryRewriter
)

from src.llm.answer_generator import (
    AnswerGenerator
)


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

    # 5. Initialize hybrid retrieval
    retriever = ParentChildHybridRetriever(
        children=children,
        parents=chunker.parents,
        embedding_model=embedding_model,
        reranker=reranker
    )

    # 6. Index documents
    retriever.index_children(rebuild=True)

    # 7. Initialize query rewriter
    query_rewriter = QueryRewriter()

    # 8. Initialize answer generator
    answer_generator = AnswerGenerator()

    # 9. Original user question
    original_query = (
        "How does the fraud detection model "
        "handle imbalanced data?"
    )

    conversation_context = """
    We are discussing a fraud detection project.
    The model uses XGBoost.
    """

    # 10. Rewrite query
    rewritten_query = query_rewriter.rewrite_query(
        query=original_query,
        conversation_context=conversation_context
    )

    print("\n========== QUERY REWRITING ==========")
    print("Original:", original_query)
    print("Rewritten:", rewritten_query)

    # 11. Retrieve relevant parent contexts
    results = retriever.search(
        query=rewritten_query,
        candidate_limit=22,
        rerank_top_k=5
    )

    contexts = results["parents"]

    print("\nRetrieved parent contexts:", len(contexts))

    # 12. Generate final answer
    answer = answer_generator.generate_answer(
        question=original_query,
        contexts=contexts
    )

    print("\n========== FINAL AI ANSWER ==========")
    print(answer)

    retriever.close()


if __name__ == "__main__":
    main()
