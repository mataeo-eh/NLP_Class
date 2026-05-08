# NHTSA Project Deployment Skeleton

This repository slice is set up so the frontend can deploy on Vercel while the
backend can deploy on a more traditional Python host such as Render.

## Recommended hosting split

1. Frontend on Vercel
   Root Directory: `Unit2/NHTSA_Project/frontend`
2. Backend on Render or a similar Python web-service host
   Root Directory: `Unit2/NHTSA_Project/backend`

## What exists today

- `frontend` is a minimal Next.js placeholder app for Vercel.
- `backend` is a minimal FastAPI placeholder app for a standard ASGI host.
- Neither side contains real product logic yet.

## Team handoff intent

- Frontend teammates can replace the placeholder Next.js page and expand the app normally.
- Backend teammates can expand the FastAPI skeleton into the real API service when they are ready.
