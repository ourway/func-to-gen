"""Tests for /v1/completions endpoint (legacy format)."""

import json


class TestCompletions:
    """Tests for the legacy completions endpoint."""

    def test_basic_completion(self, client):
        """Test a basic completion request."""
        response = client.post(
            "/v1/completions",
            data=json.dumps({
                "model": "local-llm",
                "prompt": "Hello world",
            }),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = response.get_json()

        assert "id" in data
        assert data["id"].startswith("cmpl-")
        assert data["object"] == "text_completion"
        assert "created" in data
        assert data["model"] == "local-llm"

    def test_completion_response_structure(self, client):
        """Test that response has correct legacy structure."""
        response = client.post(
            "/v1/completions",
            data=json.dumps({"prompt": "Test prompt"}),
            content_type="application/json",
        )

        data = response.get_json()

        # Check choices structure (legacy uses 'text' instead of 'message')
        assert "choices" in data
        assert len(data["choices"]) == 1

        choice = data["choices"][0]
        assert choice["index"] == 0
        assert "text" in choice
        assert choice["finish_reason"] == "stop"

        # Check usage structure
        assert "usage" in data

    def test_completion_content(self, client):
        """Test that response content comes from answer function."""
        response = client.post(
            "/v1/completions",
            data=json.dumps({"prompt": "Complete this sentence"}),
            content_type="application/json",
        )

        data = response.get_json()
        text = data["choices"][0]["text"]

        # The mock_answer function returns "Response to: {prompt}"
        assert "Complete this sentence" in text

    def test_completion_custom_model(self, client):
        """Test that custom model name is reflected in response."""
        response = client.post(
            "/v1/completions",
            data=json.dumps({
                "model": "custom-model",
                "prompt": "Test",
            }),
            content_type="application/json",
        )

        data = response.get_json()
        assert data["model"] == "custom-model"

    def test_completion_no_body(self, client):
        """Test error when no request body is provided."""
        response = client.post(
            "/v1/completions",
            data="",
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_completion_no_prompt(self, client):
        """Test error when prompt field is missing."""
        response = client.post(
            "/v1/completions",
            data=json.dumps({"model": "test"}),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "prompt" in data["error"]["message"]

    def test_completion_empty_prompt(self, client):
        """Test error when prompt is empty string."""
        response = client.post(
            "/v1/completions",
            data=json.dumps({"prompt": ""}),
            content_type="application/json",
        )

        assert response.status_code == 400
