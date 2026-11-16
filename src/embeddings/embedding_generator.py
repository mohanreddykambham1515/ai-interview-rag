

from fastembed import TextEmbedding


class EmbeddingGenerator:

    def __init__(self):

        self.model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

    def generate_embeddings(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        embeddings = list(
            self.model.embed(texts)
        )

        return [
            embedding.tolist()
            for embedding in embeddings
        ]

    def generate_single_embedding(
        self,
        text: str
    ) -> list[float]:

        return next(
            self.model.embed([text])
        ).tolist()
