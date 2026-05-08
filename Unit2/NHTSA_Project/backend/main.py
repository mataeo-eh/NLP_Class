"""
Minimal Render-compatible FastAPI skeleton for the NHTSA project backend.

This module intentionally provides only a tiny placeholder API surface. The
goal is to give the backend team a deployable ASGI entrypoint and a stable URL
to target from the frontend while the real FastAPI application is still under
development.
"""

from fastapi import FastAPI


app = FastAPI(
    title="NHTSA Project Backend Placeholder",
    description=(
        "A minimal FastAPI app that exists only to prove the deployment path "
        "and provide a predictable URL for frontend integration work."
    ),
    version="0.1.0",
)


@app.get("/")
async def read_root() -> dict[str, str]:
    """
    Return a deterministic placeholder response.

    Keeping this route simple makes deployment verification easy on hosts like
    Render while the real backend architecture is still being designed.
    """

    return {
        "service": "nhtsa-project-backend",
        "status": "placeholder",
        "message": (
            "Render-compatible FastAPI skeleton is running. "
            "Real backend logic has not been implemented yet."
        ),
    }


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    """Return a simple health response for smoke tests and uptime checks."""

    return {"status": "ok"}
