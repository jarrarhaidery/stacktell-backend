from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import get_settings
from app.models import (
    IndexRequest, IndexResponse,
    QueryRequest, QueryResponse,
    DocGenRequest, DocGenResponse,
    OnboardingRequest, OnboardingResponse,
    HealthResponse,
)
from app import indexer, rag, docgen, onboarding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Codebase Explainer API starting up")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Stacktell API",
    description="RAG-powered API that indexes GitHub repos and answers questions about them.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ───────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(
        status="ok",
        version="1.0.0",
        env=settings.app_env,
    )


# ── Indexing ─────────────────────────────────────────────────────

@app.post("/index", response_model=IndexResponse, tags=["Indexing"])
async def index_repository(request: IndexRequest):
    """
    Clone a GitHub repo, chunk all code files, embed them,
    and store in a FAISS index keyed by repo_id.
    """
    try:
        result = await indexer.index_repo(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise HTTPException(status_code=500, detail="Indexing failed. Check the repo URL and try again.")


# ── Chat / Query ─────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse, tags=["Chat"])
async def query_codebase(request: QueryRequest):
    """
    Ask a natural language question about an indexed repo.
    Returns an answer with source file references.
    """
    try:
        result = await rag.query(request)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Repo '{request.repo_id}' not indexed yet.")
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Query failed.")


# ── Doc Generation ───────────────────────────────────────────────

@app.post("/generate-docs", response_model=DocGenResponse, tags=["Docs"])
async def generate_file_docs(request: DocGenRequest):
    """
    Auto-generate markdown documentation for a specific file in the repo.
    """
    try:
        result = await docgen.generate(request)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Doc gen failed: {e}")
        raise HTTPException(status_code=500, detail="Documentation generation failed.")


# ── Onboarding Guide ─────────────────────────────────────────────

@app.post("/onboarding", response_model=OnboardingResponse, tags=["Docs"])
async def generate_onboarding(request: OnboardingRequest):
    """
    Generate a full onboarding guide for new developers joining the project.
    """
    try:
        result = await onboarding.generate(request)
        return result
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Repo '{request.repo_id}' not indexed yet.")
    except Exception as e:
        logger.error(f"Onboarding gen failed: {e}")
        raise HTTPException(status_code=500, detail="Onboarding generation failed.")

@app.get("/debug", tags=["System"])
def debug():
    import subprocess, shutil
    return {
        "git_in_path": shutil.which("git"),
        "git_version": subprocess.run(["git", "--version"], capture_output=True, text=True).stdout,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
