"""Bonus tests for CLI helpers and robustness (do not load the LLM)."""

import io
import os
import tempfile
from contextlib import redirect_stdout

from src.pipeline import _has_documents, _print_result

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def test_has_documents_with_real_data():
    assert _has_documents(DATA_DIR) is True


def test_has_documents_with_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        assert _has_documents(tmp) is False


def test_has_documents_with_missing_dir():
    assert _has_documents(os.path.join(DATA_DIR, "does-not-exist")) is False


def test_print_result_shows_answer_and_sources():
    result = {
        "answer": "The Growth package costs $5,500 per month.",
        "sources": ["GROWTH PACKAGE — $5,500/month"],
    }
    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_result(result)

    output = buf.getvalue()
    assert "Answer: The Growth package costs $5,500 per month." in output
    assert "GROWTH PACKAGE — $5,500/month" in output
