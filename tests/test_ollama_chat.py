"""Tests for Ollama /api/chat endpoint."""

import json


class TestOllamaChat:
    """Tests for the Ollama chat endpoint."""

    def test_basic_chat(self, client):
        """Test a basic chat request."""
        response = client.post(
            "/api/chat",
            data=json.dumps({
                "model": "local-llm",
                "messages": [{"role": "user", "content": "Hello"}],
            }),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = response.get_json()

        assert data["model"] == "local-llm"
        assert "created_at" in data
        assert "message" in data
        assert data["message"]["role"] == "assistant"
        assert "content" in data["message"]
        assert data["done"] is True

    def test_chat_response_structure(self, client):
        """Test that response has correct Ollama chat structure."""
        response = client.post(
            "/api/chat",
            data=json.dumps({
                "messages": [{"role": "user", "content": "Test"}],
            }),
            content_type="application/json",
        )

        data = response.get_json()

        # Check Ollama-specific fields
        assert "model" in data
        assert "created_at" in data
        assert "message" in data
        assert data["message"]["role"] == "assistant"
        assert "done" in data
        assert "done_reason" in data
        assert "total_duration" in data

    def test_chat_content(self, client):
        """Test that response content comes from answer function."""
        response = client.post(
            "/api/chat",
            data=json.dumps({
                "messages": [{"role": "user", "content": "Hello Ollama chat"}],
            }),
            content_type="application/json",
        )

        data = response.get_json()
        content = data["message"]["content"]

        # The mock_answer function returns "Response to: {prompt}"
        assert "Hello Ollama chat" in content

    def test_chat_multiple_messages(self, client):
        """Test chat with multiple messages."""
        response = client.post(
            "/api/chat",
            data=json.dumps({
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "How are you?"},
                ],
            }),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["done"] is True

    def test_chat_no_body(self, client):
        """Test error when no request body is provided."""
        response = client.post(
            "/api/chat",
            data="",
            content_type="application/json",
        )

        assert response.status_code == 400

    def test_chat_no_messages(self, client):
        """Test error when messages is missing."""
        response = client.post(
            "/api/chat",
            data=json.dumps({"model": "test"}),
            content_type="application/json",
        )

        assert response.status_code == 400

    def test_chat_empty_messages(self, client):
        """Test error when messages array is empty."""
        response = client.post(
            "/api/chat",
            data=json.dumps({"messages": []}),
            content_type="application/json",
        )

        assert response.status_code == 400
