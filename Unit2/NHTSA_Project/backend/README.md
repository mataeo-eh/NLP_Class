# Backend placeholder

This is a minimal Render-compatible FastAPI backend skeleton.

## Current intent

- Keep the directory deployable as a normal Python web service.
- Avoid introducing real backend behavior yet.
- Give the backend team a clean FastAPI starting point with a live URL.

## Important files

- `main.py`: minimal FastAPI app with placeholder routes
- `requirements.txt`: FastAPI plus Uvicorn for ASGI serving

## Render settings

If you create a Render web service from this repo, use these settings:

- Root Directory: `Unit2/NHTSA_Project/backend`
- Runtime: `Python`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Placeholder routes

- `GET /`: deployment smoke-test response
- `GET /health`: simple healthcheck response
