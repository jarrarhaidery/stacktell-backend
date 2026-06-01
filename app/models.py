from pydantic import BaseModel, HttpUrl
from typing import Optional


# ── Request schemas ──────────────────────────────────────────────

class IndexRequest(BaseModel):
    repo_url: str                       # e.g. https://github.com/user/repo
    branch: str = "main"
    include_extensions: list[str] = [   # file types to index
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".go", ".rs", ".cpp", ".c",
        ".md", ".yml", ".yaml", ".toml", ".env.example"
    ]
    exclude_dirs: list[str] = [         # dirs to skip
        "node_modules", ".git", "__pycache__",
        ".venv", "venv", "dist", "build", ".next"
    ]


class QueryRequest(BaseModel):
    repo_id: str                        # slug from IndexResponse
    question: str
    file_filter: Optional[str] = None  # optionally scope to a specific file


class DocGenRequest(BaseModel):
    repo_id: str
    file_path: str                      # generate docs for this specific file


class OnboardingRequest(BaseModel):
    repo_id: str


# ── Response schemas ─────────────────────────────────────────────

class FileNode(BaseModel):
    name: str
    path: str
    type: str                           # "file" | "dir"
    language: Optional[str] = None
    children: Optional[list["FileNode"]] = None

FileNode.model_rebuild()


class IndexResponse(BaseModel):
    repo_id: str
    repo_name: str
    total_files: int
    total_chunks: int
    languages: dict[str, int]           # {"Python": 24, "JavaScript": 8}
    file_tree: list[FileNode]
    message: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]                 # [{file, lines, snippet}]
    repo_id: str


class DocGenResponse(BaseModel):
    file_path: str
    documentation: str                  # markdown
    functions: list[dict]               # [{name, signature, docstring}]


class OnboardingResponse(BaseModel):
    repo_id: str
    guide: str                          # full markdown onboarding guide


class HealthResponse(BaseModel):
    status: str
    version: str
    env: str
