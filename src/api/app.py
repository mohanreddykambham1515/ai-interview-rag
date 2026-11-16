
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.ingestion.pdf_loader import load_all_pdfs
from src.parent_child.parent_child_retriever import ParentChildRetriever
from src.parent_child.parent_child_hybrid import ParentChildHybridRetriever
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.query_transformation.query_rewriter import QueryRewriter
from src.llm.answer_generator import AnswerGenerator


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"


# Application components
pipeline = {}


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    conversation_context: str = ""


class SourceResponse(BaseModel):
    source: str
    page: int


class AskResponse(BaseModel):
    question: str
    rewritten_query: str
    answer: str
    sources: list[SourceResponse]


def build_retrieval_pipeline():

    documents = load_all_pdfs(str(DATA_DIRECTORY))

    if not documents:
        raise ValueError("No PDF documents found in data/raw.")

    chunker = ParentChildRetriever(
        parent_chunk_size=1000,
        parent_chunk_overlap=100,
        child_chunk_size=300,
        child_chunk_overlap=50
    )

    children = chunker.create_parent_child_chunks(documents)

    retriever = ParentChildHybridRetriever(
        children=children,
        parents=chunker.parents,
        embedding_model=pipeline["embedding_model"],
        reranker=pipeline["reranker"]
    )

    retriever.index_children(rebuild=True)

    return retriever


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Initializing AI Interviewer...")

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    pipeline["embedding_model"] = EmbeddingGenerator()
    pipeline["reranker"] = CrossEncoderReranker()
    pipeline["query_rewriter"] = QueryRewriter()
    pipeline["answer_generator"] = AnswerGenerator()

    pipeline["retriever"] = build_retrieval_pipeline()

    print("AI Interviewer initialized successfully!")

    yield

    if pipeline.get("retriever"):
        pipeline["retriever"].close()

    print("AI Interviewer shut down.")


app = FastAPI(
    title="AI Technical Interviewer",
    description="Advanced RAG-based technical interview assistant",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "application": "AI Technical Interviewer"
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        # 1. Rewrite the question
        rewritten_query = pipeline["query_rewriter"].rewrite_query(
            query=question,
            conversation_context=request.conversation_context
        )

        # 2. Retrieve relevant parent contexts
        results = pipeline["retriever"].search(
            query=rewritten_query,
            candidate_limit=22,
            rerank_top_k=5
        )

        contexts = results["parents"]

        if not contexts:
            raise HTTPException(
                status_code=404,
                detail="No relevant document context was found."
            )

        # 3. Generate the answer
        answer = pipeline["answer_generator"].generate_answer(
            question=question,
            contexts=contexts
        )

        # 4. Collect unique source references
        sources = []
        seen_sources = set()

        for context in contexts:

            metadata = context.get("metadata", {})

            source = metadata.get("source")
            page = metadata.get("page")

            if source and page is not None:

                key = (source, page)

                if key not in seen_sources:
                    seen_sources.add(key)

                    sources.append({
                        "source": source,
                        "page": page
                    })

        return {
            "question": question,
            "rewritten_query": rewritten_query,
            "answer": answer,
            "sources": sources
        }

    except HTTPException:
        raise

    except Exception as error:
        print(f"Question processing error: {error}")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the question."
        )


@app.post("/ingest")
async def ingest_documents(
    files: list[UploadFile] = File(...)
):

    saved_files = []

    try:

        # 1. Validate and save uploaded PDFs
        for uploaded_file in files:

            filename = Path(uploaded_file.filename or "").name

            if not filename.lower().endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Only PDF files are supported: {filename}"
                )

            destination = DATA_DIRECTORY / filename

            with destination.open("wb") as buffer:
                shutil.copyfileobj(uploaded_file.file, buffer)

            saved_files.append(filename)

        if not saved_files:
            raise HTTPException(
                status_code=400,
                detail="No PDF files were uploaded."
            )

        # 2. Close the existing local Qdrant connection
        old_retriever = pipeline.get("retriever")

        if old_retriever:
            old_retriever.close()

        # 3. Rebuild the index using all PDFs
        pipeline["retriever"] = build_retrieval_pipeline()

        return {
            "message": "Documents uploaded and indexed successfully.",
            "uploaded_files": saved_files
        }

    except HTTPException:
        raise

    except Exception as error:
        print(f"Ingestion error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed."
        )

    finally:
        for uploaded_file in files:
            await uploaded_file.close()
