PYTHON ?= python

.PHONY: test check build manifest release clean

test:
	PYTHONPATH=src $(PYTHON) -m pytest

check:
	$(PYTHON) -m compileall -q src validation benchmarks examples tests
	PYTHONPATH=src $(PYTHON) -m pytest
	$(PYTHON) tools/check_release_tree.py

build: check
	mkdir -p dist
	$(PYTHON) -m pip wheel . --no-deps --no-build-isolation -w dist

manifest:
	$(PYTHON) tools/generate_manifest.py

release: build manifest

clean:
	rm -rf build .pytest_cache dist/*.whl src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
