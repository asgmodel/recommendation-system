PYTHON ?= python
RUN ?= run-001

.PHONY: install test inspect demo experiment

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m unittest discover -s tests -v

inspect:
	$(PYTHON) -m hlo_recommender inspect

demo:
	$(PYTHON) -m hlo_recommender demo --output artifacts/demo/$(RUN)

experiment:
	$(PYTHON) -m hlo_recommender experiment --config configs/experiment.toml
