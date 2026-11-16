
from uuid import uuid5, NAMESPACE_URL

from langchain_text_splitters import RecursiveCharacterTextSplitter


class ParentChildRetriever:

    def __init__(
        self,
        parent_chunk_size=1000,
        parent_chunk_overlap=100,
        child_chunk_size=300,
        child_chunk_overlap=50
    ):

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap
        )

        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap
        )

        self.parents = {}
        self.children = []

    def create_parent_child_chunks(self, documents):

        self.parents = {}
        self.children = []

        for document in documents:

            source = document["metadata"]["source"]
            page = document["metadata"]["page"]

            parent_texts = self.parent_splitter.split_text(
                document["content"]
            )

            for parent_index, parent_text in enumerate(parent_texts):

                parent_id = str(uuid5(
                    NAMESPACE_URL,
                    f"{source}|{page}|{parent_index}|{parent_text}"
                ))

                parent_metadata = {
                    **document["metadata"],
                    "parent_id": parent_id,
                    "parent_index": parent_index
                }

                self.parents[parent_id] = {
                    "content": parent_text,
                    "metadata": parent_metadata
                }

                child_texts = self.child_splitter.split_text(
                    parent_text
                )

                for child_index, child_text in enumerate(child_texts):

                    child_id = str(uuid5(
                        NAMESPACE_URL,
                        f"{parent_id}|{child_index}|{child_text}"
                    ))

                    self.children.append({
                        "id": child_id,
                        "content": child_text,
                        "metadata": {
                            **parent_metadata,
                            "child_index": child_index
                        }
                    })

        return self.children
