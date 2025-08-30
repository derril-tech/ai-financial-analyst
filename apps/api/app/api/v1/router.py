"""API v1 router."""

from fastapi import APIRouter

from app.api.v1.endpoints import upload, health, exports, admin, advanced_features

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(upload.router, prefix="/upload", tags=["upload"])
api_router.include_router(exports.router, prefix="/exports", tags=["exports"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
api_router.include_router(advanced_features.router, prefix="/advanced", tags=["advanced_features"])
