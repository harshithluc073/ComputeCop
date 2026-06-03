$ErrorActionPreference = "Stop"

python -m ruff format --check .
python -m ruff check .
python -m mypy src/computecop
python -m pytest --cov=computecop --cov-report=term-missing
