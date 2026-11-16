
import json
from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient, models


class ParentChildStore:

    def __init__(
        self,
        embedding_model,
        collection_name="parent_child_interview",
        storage_path="./parent_child_data",
        parent_file="./parent_child_data/parents.json",
        vector_size=384
    ):

        self.embedding_model = embedding_model
        self.collection_name = collection_name

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(
            parents=True,
            exist_ok=True
        )

        self.parent_file = Path(parent_file)

        self.client = QdrantClient(
            path=str(self.storage_path / "qdrant")
        )

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

        if self.parent_file.exists():
            self.parents = json.loads(
                self.parent_file.read_text(encoding="utf-8")
            )
        else:
            self.parents = {}

    def save_parents(self):

        self.parent_file.write_text(
            json.dumps(
                self.parents,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

    def index_children(self, children):

        if not children:
            return

        texts = [
            child["content"]
            for child in children
        ]

        embeddings = self.embedding_model.generate_embeddings(
            texts
        )

        points = []

        for child, embedding in zip(children, embeddings):

            metadata = child["metadata"]

            parent_id = metadata["parent_id"]

            self.parents[parent_id] = {
                "content": self.parents.get(
                    parent_id, {}
                ).get("content", ""),
                "metadata": {
                    key: value
                    for key, value in metadata.items()
                    if key != "child_index"
                }
            }

            points.append(
                models.PointStruct(
                    id=str(uuid4()),
                    vector=embedding,
                    payload={
                        "content": child["content"],
                        **metadata
                    }
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )

        self.save_parents()

        print(f"Indexed {len(points)} child chunks")

    def search_children(self, query, limit=5):

        query_vector = (
            self.embedding_model.generate_single_embedding(
                query
            )
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        ).points

        return results

    def retrieve_parents(self, child_results):

        unique_parent_ids = []

        for result in child_results:

            parent_id = result.payload["parent_id"]

            if parent_id not in unique_parent_ids:
                unique_parent_ids.append(parent_id)

        recovered_parents = []

        for parent_id in unique_parent_ids:

            parent = self.parents.get(parent_id)

            if parent:
                recovered_parents.append({
                    "parent_id": parent_id,
                    "content": parent["content"],
                    "metadata": parent["metadata"]
                })

        return recovered_parents

    def close(self):
        self.client.close()
