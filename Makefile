# Common commands
.PHONY: lint format test clean

lint:
	flake8 src tests

format:
	black src notebooks tests
	isort src notebooks tests

test:
	pytest -q --disable-warnings --maxfail=1

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Example: run pipeline
run:
	python -m rafads.example_pipeline
