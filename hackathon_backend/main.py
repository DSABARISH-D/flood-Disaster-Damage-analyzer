"""
Compatibility shim for tests.
Provides `app` at `hackathon_backend.main` by re-exporting from `hackathon_backend.app`.
"""
from .app import app

__all__ = ["app"]
