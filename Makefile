.PHONY: install test eval eval-live playground serve lint deploy-dry-run deploy-agent-runtime tf-plan

install:
	uv sync --extra dev

test:
	uv run pytest tests/

eval:
	uv run pytest evals/test_golden_evals.py

eval-live:
	uv run adk eval travel_concierge evals/travel_concierge.evalset.json --config_file_path evals/eval_config.json

playground:
	uv run adk web . --host 127.0.0.1 --port 8080

serve:
	uv run python -m travel_concierge.app_server

lint:
	uv run ruff check travel_concierge tests evals

deploy-dry-run:
	agents-cli deploy --dry-run

deploy-agent-runtime:
	agents-cli deploy --project $${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT} --region $${GOOGLE_CLOUD_LOCATION:-us-central1}

tf-plan:
	cd infra/terraform && terraform init && terraform validate && terraform plan
