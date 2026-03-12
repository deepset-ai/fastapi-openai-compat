.PHONY: style lint format-check typecheck test check_code_quality check publish

export PYTHONPATH = .
check_dirs := src tests
typed_dir := src/fastapi_openai_compat

# Pass PYTHON=3.12 to pin a specific version (used in CI matrix).
uv_run := uv run $(if $(PYTHON),--python $(PYTHON),)

style:
	$(uv_run) ruff format $(check_dirs)
	$(uv_run) ruff check --select I --fix $(check_dirs)

lint:
	$(uv_run) ruff check $(check_dirs)

format-check:
	$(uv_run) ruff format --check $(check_dirs)

typecheck:
	$(uv_run) ty check $(typed_dir)

test:
	$(uv_run) pytest -vv --cov=fastapi_openai_compat tests

check_code_quality: lint format-check typecheck

check: check_code_quality test

publish:
	uv build
	uv publish --token $$UV_PUBLISH_TOKEN
