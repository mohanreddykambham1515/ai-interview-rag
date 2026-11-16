
from src.ingestion.pdf_loader import load_all_pdfs
from src.chunking.recursive_chunker import create_chunks

documents = load_all_pdfs("data/raw")

configs = [
    (200, 40),
    (500, 100),
    (800, 150),
    (1000, 200)
]

for size, overlap in configs:

    chunks = create_chunks(
        documents,
        chunk_size=size,
        chunk_overlap=overlap
    )

    print("\n----------------------------")
    print("Chunk size:", size)
    print("Overlap:", overlap)
    print("Total chunks:", len(chunks))

    avg_size = sum(
        len(c["content"]) for c in chunks
    ) / len(chunks)

    print("Average chunk size:", round(avg_size, 2))
