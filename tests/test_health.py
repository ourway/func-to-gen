"""Tests for health check endpoint."""


class TestHealth:
    """Tests for the health endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"
