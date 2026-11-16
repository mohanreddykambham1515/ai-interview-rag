
from src.ingestion.pdf_loader import load_all_pdfs
from src.parent_child.parent_child_retriever import ParentChildRetriever


def main():

    # 1. Load documents
    documents = load_all_pdfs("data/raw")

    # 2. Initialize parent-child retriever
    parent_child = ParentChildRetriever(
        parent_chunk_size=1000,
        parent_chunk_overlap=100,
        child_chunk_size=300,
        child_chunk_overlap=50
    )

    # 3. Create parent and child chunks
    children = parent_child.create_parent_child_chunks(
        documents
    )

    print("\n========== PARENT-CHILD SUMMARY ==========")

    print("Original documents:", len(documents))
    print("Parent chunks:", len(parent_child.parents))
    print("Child chunks:", len(children))

    # 4. Display sample child chunks
    print("\n========== SAMPLE CHILD CHUNKS ==========")

    for index, child in enumerate(children[:5], start=1):

        print(f"\nChild {index}")
        print("Child ID:", child["id"])
        print("Parent ID:", child["metadata"]["parent_id"])
        print("Source:", child["metadata"]["source"])
        print("Content:", child["content"])

    # 5. Simulate retrieving two children
    retrieved_children = children[:2]

    # 6. Recover their parent documents
    recovered_parents = (
        parent_child.get_parents_from_children(
            retrieved_children
        )
    )

    print("\n========== RECOVERED PARENTS ==========")

    print("Retrieved children:", len(retrieved_children))
    print("Unique parents:", len(recovered_parents))

    for index, parent in enumerate(recovered_parents, start=1):

        print(f"\nParent {index}")
        print("Source:", parent["metadata"]["source"])
        print("Page:", parent["metadata"]["page"])
        print("Content:", parent["content"][:500])

    # 7. Verify parent-child mapping
    assert len(children) > 0

    for child in children:
        assert child["metadata"]["parent_id"] in parent_child.parents

    print("\nParent-child mapping verified successfully!")


if __name__ == "__main__":
    main()
