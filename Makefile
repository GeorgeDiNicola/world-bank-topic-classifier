PROJECT_NAME=world_bank_topic_classifier
PYTHON := python3
PYTEST := pytest

.PHONY: help install run test test-verbose test-summary clean

install:
	@if [ -f requirements.txt ]; then \
		$(PYTHON) -m pip install --upgrade pip; \
		$(PYTHON) -m pip install -r requirements.txt; \
	else \
		echo "requirements.txt not found, skipping pip install."; \
	fi

run:
	$(PYTHON) src/$(PROJECT_NAME)/main.py

test:
	$(PYTHON) -m $(PYTEST)

test-coverage:
	$(PYTEST) --cov=$(PROJECT_NAME) tests/

test-verbose:
	$(PYTHON) -m $(PYTEST) -vv

test-summary:
	$(PYTHON) -m $(PYTEST) -vv -ra

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'