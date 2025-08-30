"""Feature flags system."""

import os
from typing import Any

from pydantic import BaseModel


class FeatureFlags(BaseModel):
    """Feature flags configuration."""
    
    # Processing features
    enable_audio_processing: bool = True
    enable_xbrl_processing: bool = True
    enable_pdf_processing: bool = True
    
    # Analytics features
    enable_valuation_models: bool = True
    enable_risk_analytics: bool = True
    enable_event_studies: bool = True
    
    # UI features
    enable_research_board: bool = True
    enable_exports: bool = True
    enable_alerts: bool = True
    
    # Observability
    enable_tracing: bool = False
    enable_metrics: bool = True
    
    # Security
    enable_auth: bool = True
    enable_rbac: bool = True
    
    @classmethod
    def from_env(cls) -> "FeatureFlags":
        """Create feature flags from environment variables."""
        return cls(
            enable_audio_processing=_get_bool_env("ENABLE_AUDIO_PROCESSING", True),
            enable_xbrl_processing=_get_bool_env("ENABLE_XBRL_PROCESSING", True),
            enable_pdf_processing=_get_bool_env("ENABLE_PDF_PROCESSING", True),
            enable_valuation_models=_get_bool_env("ENABLE_VALUATION_MODELS", True),
            enable_risk_analytics=_get_bool_env("ENABLE_RISK_ANALYTICS", True),
            enable_event_studies=_get_bool_env("ENABLE_EVENT_STUDIES", True),
            enable_research_board=_get_bool_env("ENABLE_RESEARCH_BOARD", True),
            enable_exports=_get_bool_env("ENABLE_EXPORTS", True),
            enable_alerts=_get_bool_env("ENABLE_ALERTS", True),
            enable_tracing=_get_bool_env("ENABLE_TRACING", False),
            enable_metrics=_get_bool_env("ENABLE_METRICS", True),
            enable_auth=_get_bool_env("ENABLE_AUTH", True),
            enable_rbac=_get_bool_env("ENABLE_RBAC", True),
        )


def _get_bool_env(key: str, default: bool) -> bool:
    """Get boolean environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


# Global feature flags instance
feature_flags = FeatureFlags.from_env()


def is_enabled(flag_name: str) -> bool:
    """Check if a feature flag is enabled."""
    return getattr(feature_flags, flag_name, False)


def get_flag(flag_name: str, default: Any = None) -> Any:
    """Get feature flag value."""
    return getattr(feature_flags, flag_name, default)
