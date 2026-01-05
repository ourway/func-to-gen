"""Tests for Ollama /api/generate endpoint."""

import json


class TestOllamaGenerate:
    """Tests for the Ollama generate endpoint."""

    def test_basic_generate(self, client):
        """Test a basic generate request."""
        response = client.post(
            "/api/generate",
            data=json.dumps({
                "model": "local-llm",
                "prompt": "Hello world",
            }),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = response.get_json()

        assert data["model"] == "local-llm"
        assert "created_at" in data
        assert "response" in data
        assert data["done"] is True
        assert data["done_reason"] == "stop"

    def test_generate_response_structure(self, client):
        """Test that response has correct Ollama structure."""
        response = client.post(
            "/api/generate",
            data=json.dumps({"prompt": "Test prompt", "model": "test-model"}),
            content_type="application/json",
        )

        data = response.get_json()

        # Check Ollama-specific fields
        assert "model" in data
        assert "created_at" in data
        assert "response" in data
        assert "done" in data
        assert "total_duration" in data
        assert "prompt_eval_count" in data
        assert "eval_count" in data

    def test_generate_content(self, client):
        """Test that response content comes from answer function."""
        response = client.post(
            "/api/generate",
            data=json.dumps({"prompt": "Hello Ollama"}),
            content_type="application/json",
        )

        data = response.get_json()

        # The mock_answer function returns "Response to: {prompt}"
        assert "Hello Ollama" in data["response"]

    def test_generate_no_body(self, client):
        """Test error when no request body is provided."""
        response = client.post(
            "/api/generate",
            data="",
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_generate_no_prompt(self, client):
        """Test error when prompt is missing."""
        response = client.post(
            "/api/generate",
            data=json.dumps({"model": "test"}),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_generate_empty_prompt(self, client):
        """Test error when prompt is empty."""
        response = client.post(
            "/api/generate",
            data=json.dumps({"prompt": ""}),
            content_type="application/json",
        )

        assert response.status_code == 400
