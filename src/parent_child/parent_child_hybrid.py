
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient, models


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class ParentChildHybridRetriever:

    def __init__(
        self,
        children,
        parents,
        embedding_model,
        reranker,
        collection_name="parent_child_hybrid",
        storage_path="./parent_child_hybrid_data",
        rrf_k=60
    ):

        self.children = children
        self.parents = parents
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.collection_name = collection_name
        self.rrf_k = rrf_k

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.parent_file = self.storage_path / "parents.json"

        self.client = QdrantClient(
            path=str(self.storage_path / "qdrant")
        )

        self.tokenized_documents = [
            tokenize(child["content"])
            for child in children
        ]

        self.bm25 = BM25Okapi(self.tokenized_documents)

    def _build_filter(self, filters=None):

        if not filters:
            return None

        conditions = []

        for field, value in filters.items():

            if isinstance(value, list):
                conditions.append(
                    models.FieldCondition(
                        key=field,
                        match=models.MatchAny(any=value)
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key=field,
                        match=models.MatchValue(value=value)
                    )
                )

        return models.Filter(must=conditions)

    def index_children(self, rebuild=True):

        if not self.children:
            raise ValueError("No child chunks available")

        # Rebuild the dedicated collection to remove stale vectors.
        if rebuild and self.client.collection_exists(
            self.collection_name
        ):
            self.client.delete_collection(self.collection_name)

        if not self.client.collection_exists(self.collection_name):

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                )
            )

        texts = [
            child["content"]
            for child in self.children
        ]

        embeddings = self.embedding_model.generate_embeddings(texts)

        if len(embeddings) != len(self.children):
            raise ValueError("Embedding count does not match child count")

        points = []

        for child, embedding in zip(self.children, embeddings):

            points.append(
                models.PointStruct(
                    id=child["id"],
                    vector=embedding,
                    payload={
                        "content": child["content"],
                        **child["metadata"]
                    }
                )
            )

        # Batch upserts.
        batch_size = 64

        for start in range(0, len(points), batch_size):

            self.client.upsert(
                collection_name=self.collection_name,
                points=points[start:start + batch_size],
                wait=True
            )

        self.parent_file.write_text(
            json.dumps(
                self.parents,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        print(f"Indexed {len(points)} child chunks")

    def hybrid_search(
        self,
        query,
        candidate_limit=20,
        filters=None
    ):

        # 1. Dense retrieval
        query_vector = (
            self.embedding_model.generate_single_embedding(query)
        )

        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=self._build_filter(filters),
            limit=candidate_limit,
            with_payload=True
        ).points

        # 2. BM25 retrieval
        scores = self.bm25.get_scores(tokenize(query))

        eligible_indices = [
            index
            for index, child in enumerate(self.children)
            if all(
                child["metadata"].get(field) == value
                if not isinstance(value, list)
                else child["metadata"].get(field) in value
                for field, value in (filters or {}).items()
            )
        ]

        bm25_indices = [
            index for index in eligible_indices
            if scores[index] > 0
        ]

        bm25_indices.sort(
            key=lambda index: scores[index],
            reverse=True
        )

        bm25_indices = bm25_indices[:candidate_limit]

        # 3. Reciprocal Rank Fusion
        fused = {}

        for rank, result in enumerate(dense_results, start=1):

            payload = result.payload
            child_id = str(result.id)

            fused[child_id] = {
                "id": child_id,
                "content": payload["content"],
                "metadata": {
                    key: payload[key]
                    for key in [
                        "source",
                        "page",
                        "parent_id",
                        "parent_index",
                        "child_index"
                    ]
                },
                "dense_rank": rank,
                "bm25_rank": None,
                "rrf_score": 1 / (self.rrf_k + rank)
            }

        for rank, index in enumerate(bm25_indices, start=1):

            child = self.children[index]
            child_id = child["id"]

            if child_id not in fused:

                fused[child_id] = {
                    "id": child_id,
                    "content": child["content"],
                    "metadata": child["metadata"],
                    "dense_rank": None,
                    "bm25_rank": rank,
                    "rrf_score": 0.0
                }

            fused[child_id]["bm25_rank"] = rank

            fused[child_id]["rrf_score"] += (
                1 / (self.rrf_k + rank)
            )

        candidates = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True
        )

        return candidates[:candidate_limit]

    def search(
        self,
        query,
        candidate_limit=20,
        rerank_top_k=5,
        filters=None
    ):

        # 1. Hybrid retrieval
        candidates = self.hybrid_search(
            query=query,
            candidate_limit=candidate_limit,
            filters=filters
        )

        # 2. Cross-Encoder reranking
        reranked_children = self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=rerank_top_k
        )

        # 3. Parent recovery and deduplication
        unique_parents = {}

        for child in reranked_children:

            parent_id = child["metadata"]["parent_id"]

            if parent_id in unique_parents:
                continue

            parent = self.parents.get(parent_id)

            if parent:

                unique_parents[parent_id] = {
                    "parent_id": parent_id,
                    "content": parent["content"],
                    "metadata": parent["metadata"],
                    "rerank_score": child["rerank_score"],
                    "matched_child": child["content"]
                }

        return {
            "children": reranked_children,
            "parents": list(unique_parents.values())
        }

    def close(self):
        self.client.close()
