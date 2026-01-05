"""Pytest fixtures for Flask test client."""

import pytest

from func_to_gen import create_app


def mock_answer(prompt: str) -> str:
    """Mock answer function for testing."""
    return f"Response to: {prompt}"


@pytest.fixture
def app():
    """Create application for testing."""
    app = create_app(answer_func=mock_answer, config={"TESTING": True})
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()
