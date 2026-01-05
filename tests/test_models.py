"""Tests for /v1/models endpoint."""


class TestModels:
    """Tests for the models endpoint."""

    def test_list_models(self, client):
        """Test listing available models."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.get_json()

        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) >= 1

        model = data["data"][0]
        assert "id" in model
        assert model["object"] == "model"
        assert "created" in model
        assert "owned_by" in model

    def test_get_model(self, client):
        """Test getting a specific model."""
        response = client.get("/v1/models/local-llm")

        assert response.status_code == 200
        data = response.get_json()

        assert data["id"] == "local-llm"
        assert data["object"] == "model"
        assert data["owned_by"] == "local"

    def test_get_model_not_found(self, client):
        """Test getting a model that doesn't exist."""
        response = client.get("/v1/models/nonexistent-model")

        assert response.status_code == 404
        data = response.get_json()
        assert "error" in data
