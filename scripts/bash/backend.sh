#!/bin/sh
set -e

cd /opt/project/app/infrastructure/psql/repositories
poetry --directory /opt/project/app/ run alembic upgrade head

cd /opt/project/app
poetry run uvicorn src.backend.infrastructure.web.app:app \
  --host 0.0.0.0 \
  --port 5000 \
  --workers "${WEB_WORKERS:-4}"
