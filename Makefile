POETRY := poetry
POETRY_PATH := --directory ./

poetry-info:
	$(POETRY) $(POETRY_PATH) env info

poetry-show:
	$(POETRY) $(POETRY_PATH) show

poetry-add:
	$(POETRY) $(POETRY_PATH) add $(package)