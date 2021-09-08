install: requirements.txt
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv --cov

format:
	black *.py

lint:
	pylint --disable=R,C,E1101,W0221 model tests train.py visualization.py\
		utils.py evaluate.py search_hyperparams.py synthesize_results.py
	# lint Dockerfile
	# docker run --rm -i hadolint/hadolint < Dockerfile

clean:
	rm -rf __pycache__ .coverage

all: install lint test

.PHONY: lint clean all