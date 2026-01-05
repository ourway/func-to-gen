"""Tests for Ollama /api/tags and /api/show endpoints."""

import json


class TestOllamaTags:
    """Tests for the Ollama tags endpoint."""

    def test_list_tags(self, client):
        """Test listing available models."""
        response = client.get("/api/tags")

        assert response.status_code == 200
        data = response.get_json()

        assert "models" in data
        assert len(data["models"]) >= 1

        model = data["models"][0]
        assert "name" in model
        assert "model" in model
        assert "modified_at" in model
        assert "details" in model

    def test_tags_model_details(self, client):
        """Test that model details are present."""
        response = client.get("/api/tags")

        data = response.get_json()
        model = data["models"][0]

        assert "details" in model
        details = model["details"]
        assert "family" in details
        assert "format" in details


class TestOllamaShow:
    """Tests for the Ollama show endpoint."""

    def test_show_model(self, client):
        """Test showing model information."""
        response = client.post(
            "/api/show",
            data=json.dumps({"model": "local-llm"}),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = response.get_json()

        assert "modelfile" in data
        assert "details" in data
        assert "family" in data["details"]

    def test_show_no_body(self, client):
        """Test error when no request body is provided."""
        response = client.post(
            "/api/show",
            data="",
            content_type="application/json",
        )

        assert response.status_code == 400


class TestOllamaEmbeddings:
    """Tests for the Ollama embeddings endpoint (not supported)."""

    def test_embeddings_not_supported(self, client):
        """Test that embeddings returns 501 Not Implemented."""
        response = client.post(
            "/api/embeddings",
            data=json.dumps({
                "model": "local-llm",
                "prompt": "Hello world",
            }),
            content_type="application/json",
        )

        assert response.status_code == 501
        data = response.get_json()
        assert "error" in data
        assert "not supported" in data["error"].lower()
