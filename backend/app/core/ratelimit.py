"""Rate limit sederhana per-IP (sliding window in-memory).

Untuk API publik di free tier: tanpa ini, satu bot bisa membakar kuota
compute dan membuat demo lemot. Single-process (uvicorn 1 worker) sehingga
state in-memory cukup; kalau suatu hari multi-worker, ganti ke Redis.

Hanya membatasi path komputasi (search + preprocess). Limit dibaca dari
settings setiap request supaya bisa dimatikan via env tanpa redeploy logic.
"""

from __future__ import annotations

import time
from collections import deque

from fastapi import Request
from fastapi.responses import JSONResponse

_WINDOW_SECONDS = 60
_LIMITED_PREFIXES = ("/api/search", "/api/preprocess")

# ip -> deque[timestamp]; dibersihkan lazily per request
_hits: dict[str, deque] = {}


def _client_ip(request: Request) -> str:
    # Di belakang proxy (HF Spaces / Render), IP asli ada di X-Forwarded-For
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def rate_limit_middleware(request: Request, call_next):
    from app.core.config import settings

    limit = getattr(settings, "rate_limit_per_minute", 60)
    if limit <= 0 or not request.url.path.startswith(_LIMITED_PREFIXES):
        return await call_next(request)

    now = time.monotonic()
    ip = _client_ip(request)
    window = _hits.setdefault(ip, deque())
    while window and now - window[0] > _WINDOW_SECONDS:
        window.popleft()
    if len(window) >= limit:
        return JSONResponse(
            status_code=429,
            content={
                "detail": {
                    "error": f"Rate limit: maksimal {limit} request/menit untuk endpoint ini",
                    "retry_after_sec": _WINDOW_SECONDS,
                }
            },
            headers={"Retry-After": str(_WINDOW_SECONDS)},
        )
    window.append(now)

    # Housekeeping ringan: jangan biarkan map IP tumbuh tanpa batas
    if len(_hits) > 10_000:
        stale = [k for k, v in _hits.items() if not v or now - v[-1] > _WINDOW_SECONDS]
        for k in stale:
            _hits.pop(k, None)

    return await call_next(request)
