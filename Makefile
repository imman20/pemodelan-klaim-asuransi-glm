.PHONY: all test lint
all:
	python run_all.py
test:
	python -m pytest -q
lint:
	python -m ruff check .
