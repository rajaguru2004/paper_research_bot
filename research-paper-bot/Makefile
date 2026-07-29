PY := .venv/bin/python
PIP := uv pip
Q ?= What is multi-head attention?

.PHONY: install papers index experiments ask app test lint format notebook submission clean

install:
	uv venv --python 3.12 .venv
	$(PIP) install -e ".[dev]"
	@test -f .env || cp .env.example .env

papers:
	$(PY) scripts/download_papers.py

index:
	$(PY) scripts/build_index.py

experiments:
	$(PY) scripts/run_experiments.py

ask:
	$(PY) scripts/ask.py "$(Q)"

app:
	.venv/bin/streamlit run app/streamlit_app.py

test:
	.venv/bin/pytest -m "not integration"

test-all:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check src tests scripts app
	.venv/bin/mypy src

format:
	.venv/bin/ruff format src tests scripts app
	.venv/bin/ruff check --fix src tests scripts app

notebook:
	.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=3600 notebooks/capstone.ipynb

submission:
	$(PY) scripts/package_submission.py

clean:
	rm -rf chroma .cache reports/figures/*.png
