
import re
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def expand_query(query: str) -> str:

    query_lower = query.lower()
    expanded_terms = []

    if "imbalanced" in query_lower or "imbalance" in query_lower:
        expanded_terms.extend([
            "class imbalance",
            "class weighting",
            "scale_pos_weight",
            "oversampling",
            "undersampling",
            "sampling"
        ])

    if "fraud" in query_lower:
        expanded_terms.extend([
            "fraud detection",
            "fraud classification"
        ])

    if "rag" in query_lower or "retrieval" in query_lower:
        expanded_terms.extend([
            "dense retrieval",
            "semantic search",
            "keyword search",
            "vector search"
        ])

    return query + " " + " ".join(expanded_terms)


def metadata_matches(chunk_metadata, filters):

    if not filters:
        return True

    for field, expected in filters.items():

        actual = chunk_metadata.get(field)

        if isinstance(expected, list):
            if actual not in expected:
                return False

        elif isinstance(expected, dict):

            if "gte" in expected:
                if actual is None or actual < expected["gte"]:
                    return False

            if "lte" in expected:
                if actual is None or actual > expected["lte"]:
                    return False

        elif actual != expected:
            return False

    return True


class HybridRetriever:

    def __init__(
        self,
        chunks,
        embedding_model,
        vector_store,
        rrf_k=60
    ):

        self.chunks = chunks
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.rrf_k = rrf_k

        self.tokenized_documents = [
            tokenize(chunk["content"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_documents)

    @staticmethod
    def _chunk_key(metadata):

        return (
            metadata["source"],
            metadata["page"],
            metadata["chunk_id"]
        )

    def search(
        self,
        query,
        limit=5,
        candidate_limit=10,
        filters=None
    ):

        # 1. Dense retrieval
        query_vector = (
            self.embedding_model.generate_single_embedding(query)
        )

        dense_results = self.vector_store.search(
            query_vector=query_vector,
            limit=candidate_limit,
            filters=filters
        )

        # 2. Query expansion
        expanded_query = expand_query(query)
        tokenized_query = tokenize(expanded_query)

        # 3. BM25 retrieval with metadata filtering
        bm25_scores = self.bm25.get_scores(tokenized_query)

        eligible_indices = [
            index
            for index, chunk in enumerate(self.chunks)
            if metadata_matches(chunk["metadata"], filters)
        ]

        bm25_indices = [
            index
            for index in eligible_indices
            if bm25_scores[index] > 0
        ]

        bm25_indices.sort(
            key=lambda index: bm25_scores[index],
            reverse=True
        )

        bm25_indices = bm25_indices[:candidate_limit]

        # 4. Reciprocal Rank Fusion
        fused_results = {}

        for rank, result in enumerate(dense_results, start=1):

            payload = result.payload

            metadata = {
                "source": payload["source"],
                "page": payload["page"],
                "chunk_id": payload["chunk_id"]
            }

            key = self._chunk_key(metadata)

            fused_results[key] = {
                "content": payload["content"],
                "metadata": metadata,
                "dense_rank": rank,
                "bm25_rank": None,
                "rrf_score": 1 / (self.rrf_k + rank)
            }

        for rank, index in enumerate(bm25_indices, start=1):

            chunk = self.chunks[index]
            metadata = chunk["metadata"]
            key = self._chunk_key(metadata)

            if key not in fused_results:

                fused_results[key] = {
                    "content": chunk["content"],
                    "metadata": metadata,
                    "dense_rank": None,
                    "bm25_rank": rank,
                    "rrf_score": 0.0
                }

            fused_results[key]["bm25_rank"] = rank

            fused_results[key]["rrf_score"] += (
                1 / (self.rrf_k + rank)
            )

        # 5. Final ranking
        results = sorted(
            fused_results.values(),
            key=lambda item: item["rrf_score"],
            reverse=True
        )

        return results[:limit]
