"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "ai-financial-analyst-api"}


@router.get("/ready")
async def readiness_check() -> dict[str, str]:
    """Readiness check endpoint."""
    # TODO: Add database connectivity check
    return {"status": "ready", "service": "ai-financial-analyst-api"}
