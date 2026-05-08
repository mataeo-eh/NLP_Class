# NHTSA Project Vercel Skeleton

This repository slice is set up so `frontend` and `backend` can be connected to Vercel as two separate projects from the same Git repository.

## Recommended Vercel dashboard setup

Create two Vercel projects that both point at this repository:

1. Frontend project
   Root Directory: `Unit2/NHTSA_Project/frontend`
2. Backend project
   Root Directory: `Unit2/NHTSA_Project/backend`

Once both projects are connected, pushes to `main` will trigger deployments for both Vercel projects.

## What exists today

- `frontend` is a minimal Next.js placeholder app.
- `backend` is a minimal Vercel Python function placeholder.
- Neither side contains real product logic yet.

## Team handoff intent

- Frontend teammates can replace the placeholder Next.js page and expand the app normally.
- Backend teammates can replace the placeholder Python function with the real API implementation when they are ready.
