import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.models import IndexRequest, QueryRequest


# ── Indexer tests ────────────────────────────────────────────────

class TestRepoIdGeneration:
    def test_basic_url(self):
        from app.indexer import _repo_id_from_url
        rid = _repo_id_from_url("https://github.com/jarrarhaidery/university-ai-assistant")
        assert rid == "jarrarhaidery__university_ai_assistant"

    def test_trailing_slash(self):
        from app.indexer import _repo_id_from_url
        rid = _repo_id_from_url("https://github.com/user/repo/")
        assert "__" in rid

    def test_dot_git_stripped(self):
        from app.indexer import _repo_id_from_url
        rid = _repo_id_from_url("https://github.com/user/repo.git")
        assert "git" not in rid


# ── RAG tests ────────────────────────────────────────────────────

class TestRAGQuery:
    @pytest.mark.asyncio
    async def test_missing_index_raises(self):
        from app.rag import query
        req = QueryRequest(repo_id="nonexistent__repo", question="what does this do?")
        with pytest.raises(FileNotFoundError):
            await query(req)


# ── Model validation ─────────────────────────────────────────────

class TestModels:
    def test_index_request_defaults(self):
        req = IndexRequest(repo_url="https://github.com/user/repo")
        assert req.branch == "main"
        assert ".py" in req.include_extensions
        assert "node_modules" in req.exclude_dirs

    def test_query_request_requires_fields(self):
        with pytest.raises(Exception):
            QueryRequest(repo_id="x")  # missing question
