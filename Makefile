ifeq ($(OS),Windows_NT)
    CD := cd /d
else
    CD := cd
endif

# POETRY
POETRY := poetry
POETRY_PATH := --directory "$(CURDIR)"

poetry-info:
	$(POETRY) $(POETRY_PATH) env info

poetry-show:
	$(POETRY) $(POETRY_PATH) show

poetry-add:
	$(POETRY) $(POETRY_PATH) add $(package)

poetry-install:
	$(POETRY) $(POETRY_PATH) install --no-root

# BACKEND
BACKEND_PYTHON_PATH := $(CURDIR)/src/backend
APP_EXECUTE_FILE_PATH := infrastructure.web.api.app
CELERY_APP := infrastructure.celery_app.app.celery
CELERY_QUEUE := periodic_queue

run-local-back:
	sh -x ./scripts/bash/postgres.sh
	PYTHONPATH='$(BACKEND_PYTHON_PATH)' $(POETRY) $(POETRY_PATH) run python3 -m \
	uvicorn $(APP_EXECUTE_FILE_PATH):app \
		--host 127.0.0.1 \
		--port 18080 \
		--reload

migrate:
	@echo "Running migrations..."
	$(CD) src/backend/infrastructure/psql \
		&& $(POETRY) $(POETRY_PATH) run alembic upgrade head
	@echo "Migrations completed"

revision:
	@echo "Running revision..."
	$(CD) src/backend/infrastructure/psql \
		&& $(POETRY) $(POETRY_PATH) run alembic revision --autogenerate
	@echo "Revision completed"

add_cleaning_for_zeep:
	sh -x ./scripts/bash/cleaning_for_zeep.sh $(POETRY) $(POETRY_PATH)

run-beat:
	sh -x ./scripts/bash/rabbit.sh
	PYTHONPATH='$(BACKEND_PYTHON_PATH)' $(POETRY) $(POETRY_PATH) run celery \
		-A $(CELERY_APP) beat \
		-l info \
		-s "$(CURDIR)/celerybeat-schedule"

run-worker:
	sh -x ./scripts/bash/rabbit.sh
	PYTHONPATH='$(BACKEND_PYTHON_PATH)' $(POETRY) $(POETRY_PATH) run celery \
		-A $(CELERY_APP) worker \
		-P solo \
		-l info \
		-n cred-worker \
		-Q $(CELERY_QUEUE)
