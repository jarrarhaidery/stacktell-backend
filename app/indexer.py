import os
import asyncio
import tempfile
from pathlib import Path
from collections import defaultdict

import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from app.config import get_settings
from app.models import IndexRequest, IndexResponse, FileNode

settings = get_settings()

EXT_LANGUAGE_MAP = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
    ".jsx": "React (JSX)", ".tsx": "React (TSX)", ".java": "Java",
    ".go": "Go", ".rs": "Rust", ".cpp": "C++", ".c": "C",
    ".md": "Markdown", ".yml": "YAML", ".yaml": "YAML", ".toml": "TOML",
}


def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _repo_id_from_url(url: str) -> str:
    url = url.rstrip("/").replace(".git", "")
    parts = url.split("/")
    slug = f"{parts[-2]}__{parts[-1]}"
    return slug.lower().replace("-", "_")


def _index_path(repo_id: str) -> Path:
    base = Path(settings.storage_path)
    base.mkdir(parents=True, exist_ok=True)
    return base / repo_id


def _parse_owner_repo(url: str):
    url = url.rstrip("/").replace(".git", "")
    parts = url.split("/")
    return parts[-2], parts[-1]


async def _fetch_tree(owner: str, repo: str, branch: str, token: str = "") -> list:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=headers)
        r.raise_for_status()
        return r.json().get("tree", [])


async def _fetch_file(owner: str, repo: str, path: str, branch: str, token: str = "") -> str:
    headers = {"Accept": "application/vnd.github.raw+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=headers)
        if r.status_code != 200:
            return ""
        return r.text


def _build_file_tree_from_paths(paths: list[str]) -> list[FileNode]:
    tree = {}
    for path in paths:
        parts = path.split("/")
        current = tree
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = None

    def build(node, prefix=""):
        result = []
        for name, children in sorted(node.items(), key=lambda x: (x[1] is None, x[0])):
            full_path = f"{prefix}/{name}" if prefix else name
            if children is None:
                ext = Path(name).suffix
                result.append(FileNode(name=name, path=full_path, type="file", language=EXT_LANGUAGE_MAP.get(ext)))
            else:
                result.append(FileNode(name=name, path=full_path, type="dir", children=build(children, full_path)))
        return result

    return build(tree)


async def index_repo(request: IndexRequest) -> IndexResponse:
    repo_id = _repo_id_from_url(request.repo_url)
    index_dir = _index_path(repo_id)
    owner, repo_name = _parse_owner_repo(request.repo_url)
    token = settings.github_token

    # Fetch file tree from GitHub API
    tree = await _fetch_tree(owner, repo_name, request.branch, token)

    # Filter to indexable files only
    indexable = [
        item for item in tree
        if item["type"] == "blob"
        and Path(item["path"]).suffix in request.include_extensions
        and not any(ex in item["path"].split("/") for ex in request.exclude_dirs)
    ]

    if not indexable:
        raise ValueError("No indexable files found in the repository.")

    # Fetch file contents concurrently (batched to avoid rate limits)
    documents: list[Document] = []
    language_counts: dict[str, int] = defaultdict(int)
    file_paths: list[str] = []

    async def fetch_and_add(item):
        content = await _fetch_file(owner, repo_name, item["path"], request.branch, token)
        if not content.strip():
            return
        ext = Path(item["path"]).suffix
        lang = EXT_LANGUAGE_MAP.get(ext, "Other")
        language_counts[lang] += 1
        file_paths.append(item["path"])
        documents.append(Document(
            page_content=content,
            metadata={"source": item["path"], "language": lang, "repo_id": repo_id},
        ))

    # Batch fetches — 10 at a time to avoid GitHub rate limits
    batch_size = 10
    for i in range(0, len(indexable), batch_size):
        batch = indexable[i:i + batch_size]
        await asyncio.gather(*[fetch_and_add(item) for item in batch])

    if not documents:
        raise ValueError("Could not fetch any file contents.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = _get_embeddings()
    vectorstore = await asyncio.get_event_loop().run_in_executor(
        None, lambda: FAISS.from_documents(chunks, embeddings)
    )
    vectorstore.save_local(str(index_dir))

    file_tree = _build_file_tree_from_paths(file_paths)

    return IndexResponse(
        repo_id=repo_id,
        repo_name=repo_name,
        total_files=len(documents),
        total_chunks=len(chunks),
        languages=dict(language_counts),
        file_tree=file_tree,
        message=f"Successfully indexed {len(documents)} files into {len(chunks)} chunks.",
    )