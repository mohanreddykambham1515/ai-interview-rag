from src.ingestion.pdf_loader import load_all_pdfs

documents = load_all_pdfs('data/raw')

print('total extracted pages',len(documents))


for document in documents[:3]:

    print('source',document['metadata']['source'])
    print('page',document['metadata']['page'])
    print('content',document['content'][:300])