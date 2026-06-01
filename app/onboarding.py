import asyncio
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

from app.config import get_settings
from app.models import OnboardingRequest, OnboardingResponse

settings = get_settings()

ONBOARDING_PROMPT = """You are a senior engineer writing an onboarding guide for a new developer.

Based on the codebase context below, write a comprehensive markdown onboarding guide covering:
1. Project Overview
2. Tech Stack
3. Project Structure
4. How to Run Locally
5. Core Concepts
6. Key Entry Points
7. Common Tasks
8. Things to Watch Out For

Codebase context:
{context}"""


def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


async def generate(request: OnboardingRequest) -> OnboardingResponse:
    index_path = Path(settings.storage_path) / request.repo_id
    if not index_path.exists():
        raise FileNotFoundError(f"Repo '{request.repo_id}' not indexed.")

    vectorstore = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: FAISS.load_local(str(index_path), _get_embeddings(), allow_dangerous_deserialization=True)
    )

    queries = [
        "project structure and main entry point",
        "how to set up and run the project",
        "core business logic and key functions",
        "database models and configuration",
        "API routes and controllers",
    ]

    all_chunks = []
    seen = set()
    for q in queries:
        docs = await asyncio.get_event_loop().run_in_executor(
            None, lambda q=q: vectorstore.similarity_search(q, k=4)
        )
        for doc in docs:
            key = doc.metadata.get("source", "") + doc.page_content[:50]
            if key not in seen:
                seen.add(key)
                all_chunks.append(f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(all_chunks[:20])
    client = Groq(api_key=settings.groq_api_key)

    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": ONBOARDING_PROMPT.format(context=context[:7000])}],
            max_tokens=2500,
            temperature=0.2,
        )
    )

    return OnboardingResponse(
        repo_id=request.repo_id,
        guide=response.choices[0].message.content.strip(),
    )
