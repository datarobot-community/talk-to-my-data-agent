# OS detection
ifeq ($(OS),Windows_NT)
	SET_ENV_SCRIPT = ./set_env.bat
else
	SET_ENV_SCRIPT = . ./set_env.sh
endif

.PHONY: copyright-check apply-copyright fix-licenses check-licenses

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

copyright-check: ## Copyright checks
	docker run -it --rm -v $(CURDIR):/github/workspace apache/skywalking-eyes -c .github/.licenserc.yaml header check

apply-copyright: ## Add copyright notice to new files
	docker run -it --rm -v $(CURDIR):/github/workspace apache/skywalking-eyes -c .github/.licenserc.yaml header fix

fix-licenses: apply-copyright

check-licenses: copyright-check

fix-lint: ## Fix linting issues
	ruff format .
	ruff check . --fix
	mypy --pretty .

lint: ## Lint the code
	ruff format --check .
	ruff check .
	mypy --pretty .

check-all: check-licenses lint ## Run all checks

install-frontend:
	cd app_frontend && npm install

build-frontend:
	cd app_frontend && npm run build

run-local-dev-backend: install-frontend build-frontend
	@$(SET_ENV_SCRIPT) && \
	PYTHONPATH=app_backend SERVE_STATIC_FRONTEND=False DEV_MODE=True ./app_backend/start-app.sh

run-local-static-backend: install-frontend build-frontend
	@$(SET_ENV_SCRIPT) && \
	PYTHONPATH=app_backend DEV_MODE=True ./app_backend/start-app.sh
