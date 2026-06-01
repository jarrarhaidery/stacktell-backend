import asyncio
import json
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

from app.config import get_settings
from app.models import DocGenRequest, DocGenResponse

settings = get_settings()

DOC_PROMPT = """You are a senior software engineer writing clean documentation.

Below is the source code of `{file_path}`. Generate:

1. A markdown documentation block with:
   - Overview of what this file does
   - All classes and their purpose
   - All functions/methods with parameters, return values, and description

2. At the very end, output a FUNCTIONS_JSON block like this:
FUNCTIONS_JSON:
[{{"name": "fn_name", "signature": "def fn(arg) -> type", "docstring": "one-line description"}}]

Source code:
```
{source_code}
```"""


def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


async def generate(request: DocGenRequest) -> DocGenResponse:
    index_path = Path(settings.storage_path) / request.repo_id
    if not index_path.exists():
        raise FileNotFoundError(f"Repo '{request.repo_id}' not indexed.")

    vectorstore = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: FAISS.load_local(str(index_path), _get_embeddings(), allow_dangerous_deserialization=True)
    )

    docs = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: vectorstore.similarity_search(
            f"code in {request.file_path}", k=12,
            filter={"source": request.file_path},
        )
    )

    if not docs:
        raise FileNotFoundError(f"No indexed content found for '{request.file_path}'")

    combined = "\n".join(d.page_content for d in docs)
    client = Groq(api_key=settings.groq_api_key)

    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": DOC_PROMPT.format(
                file_path=request.file_path,
                source_code=combined[:5000],
            )}],
            max_tokens=2000,
            temperature=0.1,
        )
    )

    raw = response.choices[0].message.content
    if "FUNCTIONS_JSON:" in raw:
        parts = raw.split("FUNCTIONS_JSON:")
        documentation = parts[0].strip()
        try:
            functions = json.loads(parts[1].strip())
        except Exception:
            functions = []
    else:
        documentation = raw.strip()
        functions = []

    return DocGenResponse(file_path=request.file_path, documentation=documentation, functions=functions)
