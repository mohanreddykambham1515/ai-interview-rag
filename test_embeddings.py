
from src.ingestion.pdf_loader import load_all_pdfs
from src.chunking.recursive_chunker import create_chunks
from src.embeddings.embedding_generator import EmbeddingGenerator

# Load documents
documents = load_all_pdfs("data/raw")

# Generate chunks
chunks = create_chunks(
    documents,
    chunk_size=500,
    chunk_overlap=100
)

# Extract chunk text
texts = [
    chunk["content"]
    for chunk in chunks
]

# Initialize embedding model
embedding_model = EmbeddingGenerator()

# Generate embeddings
vectors = embedding_model.generate_embeddings(texts)

print("Total chunks:", len(chunks))
print("Total embeddings:", len(vectors))
print("Vector dimensions:", len(vectors[0]))

print("\nFirst embedding:")
print(vectors[0][:10])
