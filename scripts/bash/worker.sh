#!/bin/sh
set -e

cd /opt/project/app
poetry run celery \
  -A infrastructure.celery.celery_app.celery worker \
  -P solo "$@"
