ifeq ($(OS),Windows_NT)
    CD := cd /d
else
    CD := cd
endif

# POETRY
POETRY := poetry
POETRY_PATH := --directory ./

poetry-info:
	$(POETRY) $(POETRY_PATH) env info

poetry-show:
	$(POETRY) $(POETRY_PATH) show

poetry-add:
	$(POETRY) $(POETRY_PATH) add $(package)

poetry-install:
	$(POETRY) $(POETRY_PATH) install --no-root

# BACKEND
APP_EXECUTE_FILE_PATH := src.backend.infrastructure.web.api.app
BACKEND_PYTHON_PATH := src/backend/

run-local-back:
	-sh -x ./scripts/bash/postgres.sh
	PYTHONPATH=$(BACKEND_PYTHON_PATH) $(POETRY) $(POETRY_PATH) run python3 -m \
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