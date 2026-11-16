
from src.query_transformation.query_rewriter import QueryRewriter


def main():

    rewriter = QueryRewriter()

    original_query = "What about the imbalance problem?"

    conversation_context = """
    We are discussing a fraud detection machine learning project.
    The model uses XGBoost to detect fraudulent transactions.
    """

    rewritten_query = rewriter.rewrite_query(
        query=original_query,
        conversation_context=conversation_context
    )

    print("\n========== QUERY REWRITING ==========")

    print("Original Query:")
    print(original_query)

    print("\nRewritten Query:")
    print(rewritten_query)


if __name__ == "__main__":
    main()
