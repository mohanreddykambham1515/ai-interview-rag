
from src.ingestion.pdf_loader import load_all_pdfs
from src.chunking.recursive_chunker import create_chunks
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.vector_store import VectorStore


# 1. Load PDFs
documents = load_all_pdfs("data/raw")

# 2. Generate chunks
chunks = create_chunks(
    documents,
    chunk_size=500,
    chunk_overlap=100
)

# 3. Generate embeddings
embedding_model = EmbeddingGenerator()

texts = [
    chunk["content"]
    for chunk in chunks
]

embeddings = embedding_model.generate_embeddings(texts)

# 4. Initialize Qdrant
vector_store = VectorStore()

# 5. Insert vectors
vector_store.insert_chunks(
    chunks,
    embeddings
)

print("\nIndexing completed!")
