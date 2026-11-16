
from qdrant_client import QdrantClient, models


class VectorStore:

    def __init__(
        self,
        collection_name="interview_knowledge",
        vector_size=384
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(path="./qdrant_data")

        collections = self.client.get_collections()

        exists = any(
            collection.name == collection_name
            for collection in collections.collections
        )

        if not exists:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )

    def insert_chunks(self, chunks, embeddings):

        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings count must match")

        points = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            points.append(
                models.PointStruct(
                    id=index,
                    vector=embedding,
                    payload={
                        "content": chunk["content"],
                        **chunk["metadata"]
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )

        print(f"Inserted {len(points)} vectors")

    def build_filter(self, filters=None):

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

            elif isinstance(value, dict):

                if "gte" in value or "lte" in value:
                    conditions.append(
                        models.FieldCondition(
                            key=field,
                            range=models.Range(
                                gte=value.get("gte"),
                                lte=value.get("lte")
                            )
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

    def search(self, query_vector, limit=3, filters=None):

        qdrant_filter = self.build_filter(filters)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=limit,
            with_payload=True
        ).points

        return results

    def close(self):
        self.client.close()
