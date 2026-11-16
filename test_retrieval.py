
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.vector_store import VectorStore


# Initialize embedding model
embedding_model = EmbeddingGenerator()

# Initialize Qdrant
vector_store = VectorStore()

# Interview question
query = "How does the fraud detection model handle imbalanced data?"

# Generate query embedding
query_vector = embedding_model.generate_single_embedding(query)

# Retrieve top 3 chunks
results = vector_store.search(
    query_vector=query_vector,
    limit=10
)

# Display results
print("\nInterview Question:", query)

for index, result in enumerate(results, start=1):

    print(f"\n--- Result {index} ---")

    print("Similarity Score:", result.score)
    print("Source:", result.payload["source"])
    print("Page:", result.payload["page"])
    print("Content:", result.payload["content"])
