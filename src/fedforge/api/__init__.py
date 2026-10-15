"""Optional API boundary for offline FedForge run artifacts.

The production FastAPI/PostgreSQL/Redis service is intentionally not enabled in
this CPU slice.  This module provides a read-only run repository that works
with the standard library and an optional ``create_fastapi_app`` adapter when
FastAPI is installed by a future deployment environment.
"""

from .app import RunNotFoundError, RunRepository, create_fastapi_app, score_smoke_features

__all__ = [
    "RunNotFoundError",
    "RunRepository",
    "create_fastapi_app",
    "score_smoke_features",
]
