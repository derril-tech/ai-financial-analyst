"""Test main application."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "AI Financial Analyst API"}


def test_v1_health_check():
    """Test v1 health check endpoint."""
    response = client.get("/v1/health/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
