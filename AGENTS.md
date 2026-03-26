# Data Analyst — Agent Instructions

Monorepo: app_frontend (React/Vite), app_backend (FastAPI web layer), core (shared business logic/runtime helpers), infra (Pulumi IaC). Commands are in `Taskfile.yaml` files — run `task --list` to discover them.

## Key Versions

Tailwind CSS v4, React 19, TypeScript 5.7, shadcn/ui (Radix-based), Python >=3.10.

## Conventions

- **Python**: ruff format + check, mypy strict, uv for deps
- **Frontend**: ESLint + Prettier, Tailwind class sorting, npm, shadcn/ui for form components, `data-testid` for test selectors
- **i18n**: use `t("English text as key")`, update all 5 locale files in `app_frontend/src/i18n/locales/`, run `npm run i18n:extract`
- **Testing**: Vitest + Testing Library (frontend), pytest (backend/core)

## Linting

Run `task lint` (or component-specific: `task app_backend:lint`, `task core:lint`, etc.) after completing a multi-step implementation or before committing. Not needed for trivial one-off changes. Fix all errors before finishing.

## Meta

Update this file when introducing new conventions.
