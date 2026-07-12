"""
Supabase Client Singleton
===========================
Provides a single Supabase client instance for the entire application.
"""

from functools import lru_cache
from supabase import create_client, Client

from backend.config.settings import settings


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """
    Create and cache a Supabase client using service role key.

    The service role key has full access and bypasses RLS.
    Only use this server-side — NEVER expose to the frontend.
    """
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
        raise RuntimeError(
            "Supabase credentials not configured. "
            "Set SUPABASE_URL and SUPABASE_SERVICE_KEY in your .env file."
        )

    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
