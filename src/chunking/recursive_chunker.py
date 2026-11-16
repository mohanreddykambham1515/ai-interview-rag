
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[dict]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    all_chunks = []

    for document in documents:

        text = document["content"]

        chunks = splitter.split_text(text)

        for index, chunk in enumerate(chunks):

            all_chunks.append({
                "content": chunk,
                "metadata": {
                    **document["metadata"],
                    "chunk_id": index,
                    "chunk_size": len(chunk)
                }
            })

    return all_chunks
