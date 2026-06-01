import os
import asyncio
import tempfile
from pathlib import Path
from collections import defaultdict

import git
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


def _build_file_tree(root: Path, exclude_dirs: list[str]) -> list[FileNode]:
    nodes = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name))
    except PermissionError:
        return nodes
    for entry in entries:
        if entry.name in exclude_dirs or entry.name.startswith("."):
            continue
        if entry.is_dir():
            children = _build_file_tree(entry, exclude_dirs)
            nodes.append(FileNode(name=entry.name, path=str(entry.relative_to(root.parent)), type="dir", children=children))
        else:
            lang = EXT_LANGUAGE_MAP.get(entry.suffix)
            nodes.append(FileNode(name=entry.name, path=str(entry.relative_to(root.parent)), type="file", language=lang))
    return nodes


async def index_repo(request: IndexRequest) -> IndexResponse:
    repo_id = _repo_id_from_url(request.repo_url)
    index_dir = _index_path(repo_id)

    with tempfile.TemporaryDirectory() as tmp:
        url = request.repo_url
        if settings.github_token:
            url = url.replace("https://", f"https://{settings.github_token}@")

        git.Repo.clone_from(url, tmp, depth=1, branch=request.branch)
        repo_name = Path(request.repo_url).stem

        documents: list[Document] = []
        language_counts: dict[str, int] = defaultdict(int)
        total_files = 0

        for root, dirs, files in os.walk(tmp):
            dirs[:] = [d for d in dirs if d not in request.exclude_dirs and not d.startswith(".")]
            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix not in request.include_extensions:
                    continue
                try:
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                rel_path = str(fpath.relative_to(tmp))
                lang = EXT_LANGUAGE_MAP.get(fpath.suffix, "Other")
                language_counts[lang] += 1
                total_files += 1
                documents.append(Document(
                    page_content=content,
                    metadata={"source": rel_path, "language": lang, "repo_id": repo_id},
                ))

        if not documents:
            raise ValueError("No indexable files found in the repository.")

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

        file_tree = _build_file_tree(Path(tmp), request.exclude_dirs)

        return IndexResponse(
            repo_id=repo_id,
            repo_name=repo_name,
            total_files=total_files,
            total_chunks=len(chunks),
            languages=dict(language_counts),
            file_tree=file_tree,
            message=f"Successfully indexed {total_files} files into {len(chunks)} chunks.",
        )
