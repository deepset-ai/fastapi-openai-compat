.PHONY: style lint format-check typecheck test check_code_quality check publish

export PYTHONPATH = .
check_dirs := src tests
typed_dir := src/fastapi_openai_compat

style:
	uv run ruff format $(check_dirs)
	uv run ruff check --select I --fix $(check_dirs)

lint:
	uv run ruff check $(check_dirs)

format-check:
	uv run ruff format --check $(check_dirs)

typecheck:
	uv run ty check $(typed_dir)

test:
	uv run pytest -vv --cov=fastapi_openai_compat tests

check_code_quality: lint format-check typecheck

check: check_code_quality test

publish:
	uv build
	uv publish --token $$UV_PUBLISH_TOKEN
