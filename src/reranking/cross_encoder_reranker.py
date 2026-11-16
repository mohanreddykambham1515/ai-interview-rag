
from fastembed.rerank.cross_encoder import TextCrossEncoder


class CrossEncoderReranker:

    def __init__(
        self,
        model_name="Xenova/ms-marco-MiniLM-L-6-v2"
    ):
        print("Loading cross-encoder reranker...")

        self.model = TextCrossEncoder(
            model_name=model_name
        )

        print("Cross-encoder loaded successfully!")

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5
    ) -> list[dict]:

        if not documents:
            return []

        # Extract document text
        document_texts = [
            document["content"]
            for document in documents
        ]

        # Score each query-document pair
        scores = list(
            self.model.rerank(
                query,
                document_texts
            )
        )

        if len(scores) != len(documents):
            raise ValueError(
                "Reranker returned an unexpected number of scores"
            )

        # Attach reranking scores
        reranked_documents = []

        for document, score in zip(documents, scores):

            result = document.copy()
            result["rerank_score"] = float(score)

            reranked_documents.append(result)

        # Highest relevance score first
        reranked_documents.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return reranked_documents[:top_k]
