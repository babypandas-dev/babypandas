.PHONY: help push lab01 lab02 lab03 lab04 build serve clean

TODAY := $(shell date +"%m-%d")

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

test: ## Run tests
	uv run pytest

build: ## Build the babypandas package
	uv build

clean: ## Remove the build directory
	rm -rf dist
