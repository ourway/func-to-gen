"""Tests for /v1/embeddings endpoint."""

import json


class TestEmbeddings:
    """Tests for the embeddings endpoint (not supported)."""

    def test_embeddings_not_supported(self, client):
        """Test that embeddings returns 501 Not Implemented."""
        response = client.post(
            "/v1/embeddings",
            data=json.dumps({
                "model": "text-embedding-ada-002",
                "input": "Hello world",
            }),
            content_type="application/json",
        )

        assert response.status_code == 501
        data = response.get_json()

        assert "error" in data
        assert "not supported" in data["error"]["message"].lower()
        assert data["error"]["type"] == "not_implemented_error"
