import asyncio
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

from app.config import get_settings
from app.models import QueryRequest, QueryResponse

settings = get_settings()

SYSTEM_PROMPT = """You are an expert software engineer helping a developer understand a codebase.
You have access to relevant code snippets retrieved from the repository.
Use them to answer the question. Be specific — reference exact function names and file paths.
If the answer isn't in the provided code, say so honestly."""


def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _load_vectorstore(repo_id: str) -> FAISS:
    path = Path(settings.storage_path) / repo_id
    if not path.exists():
        raise FileNotFoundError(f"No index found for repo '{repo_id}'")
    return FAISS.load_local(str(path), _get_embeddings(), allow_dangerous_deserialization=True)


async def query(request: QueryRequest) -> QueryResponse:
    vectorstore = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _load_vectorstore(request.repo_id)
    )

    search_kwargs = {"k": settings.top_k_results}
    if request.file_filter:
        search_kwargs["filter"] = {"source": request.file_filter}

    docs = await asyncio.get_event_loop().run_in_executor(
        None, lambda: vectorstore.similarity_search(request.question, **search_kwargs)
    )

    context = "\n\n---\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )

    client = Groq(api_key=settings.groq_api_key)
    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Code context:\n{context}\n\nQuestion: {request.question}"},
            ],
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
    )

    answer = response.choices[0].message.content.strip()

    sources = []
    seen = set()
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append({
                "file": src,
                "language": doc.metadata.get("language", ""),
                "snippet": doc.page_content[:200].strip(),
            })

    return QueryResponse(answer=answer, sources=sources, repo_id=request.repo_id)
