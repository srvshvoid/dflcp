.DEFAULT_GOAL := help
.ONESHELL:
SHELL := /bin/sh

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
PYTHONPATH_VALUE := $(CURDIR)/src
RUN_DIR ?= runs/smoke
DEMO_RUN_DIR ?= runs/demo
SMOKE_CONFIG ?= configs/smoke.json
PORT ?= 8000

.PHONY: help setup test smoke demo demo-security report reproduce run lint check clean

help:
	@printf '%s\n' \
		'make setup  Create an offline virtualenv and install this local package' \
		'make test   Run the repository test suite with the stdlib unittest runner' \
		'make smoke  Run the deterministic CPU smoke command into RUN_DIR' \
		'make demo   Prepare a labelled replay in DEMO_RUN_DIR' \
		'make report  Render a labelled replay index from runs/' \
		'make demo-security  Run clean/attack/defense synthetic stages' \
		'make reproduce RQ=1  Validate a preregistered RQ contract' \
		'make run    Serve dashboard/ locally on PORT' \
		'make lint   Compile Python sources without third-party tooling' \
		'make check  Validate packaging/dashboard/docs and run lint'

setup:
	@$(PYTHON) -c 'import sys; expected=(3,11,2); actual=sys.version_info[:3]; assert actual == expected, f"FedForge requires Python {expected}, found {actual}"'
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install --no-index --no-deps --no-build-isolation -r requirements.lock
	$(VENV_PYTHON) -c 'import pathlib; assert pathlib.Path("src/fedforge").is_dir(), "src/fedforge is missing"; print("offline environment ready; Make targets use PYTHONPATH=src")'

test:
	@test -d tests || { printf '%s\n' 'tests/ is missing; the source/test implementation is not present in this checkout.' >&2; exit 1; }
	PYTHONPATH="$(PYTHONPATH_VALUE)" $(PYTHON) -m unittest discover -s tests -p 'test*.py' -v

smoke:
	@test -d src/fedforge || { printf '%s\n' 'src/fedforge/ is missing; cannot claim a smoke run.' >&2; exit 1; }
	@test -f "$(SMOKE_CONFIG)" || { printf '%s\n' "$(SMOKE_CONFIG) is missing; cannot resolve the smoke configuration." >&2; exit 1; }
	mkdir -p "$(RUN_DIR)"
	PYTHONPATH="$(PYTHONPATH_VALUE)" $(PYTHON) -m fedforge "$(SMOKE_CONFIG)" "$(RUN_DIR)"
	@test -s "$(RUN_DIR)/metrics.jsonl"
	@test -s "$(RUN_DIR)/provenance.json"
	@printf '%s\n' "Synthetic CPU smoke replay written to $(RUN_DIR). No research benchmark is inferred."

demo:
	$(MAKE) smoke RUN_DIR="$(DEMO_RUN_DIR)"
	@test -s dashboard/index.html
	@printf '%s\n' \
		'Synthetic replay prepared; dashboard/index.html is the viewer.' \
		"Serve it with: $(MAKE) run PORT=$(PORT)" \
		"Then choose $(DEMO_RUN_DIR)/metrics.jsonl in the file picker."

report:
	PYTHONPATH="$(PYTHONPATH_VALUE)" $(PYTHON) scripts/render_results.py --runs runs --output docs/RESULTS.generated.md
	@printf '%s\n' 'Generated docs/RESULTS.generated.md from actual run artifacts; it is synthetic/unverified.'

demo-security:
	PYTHONPATH="$(PYTHONPATH_VALUE)" $(PYTHON) scripts/security_demo.py --output "$(DEMO_RUN_DIR)-security"
	@printf '%s\n' 'Synthetic security replay only; inspect the manifest before presenting it.'

reproduce:
	@test -n "$(RQ)" || { printf '%s\n' 'Usage: make reproduce RQ=1 (or another configured RQ number).' >&2; exit 2; }
	@test -f "configs/experiments/RQ$(RQ).yaml" || { printf '%s\n' "No preregistered contract for RQ$(RQ)." >&2; exit 2; }
	@printf '%s\n' "RQ$(RQ) contract is present. Real-data reproduction remains pending dataset/Flower/GPU prerequisites; no benchmark is claimed."

run:
	$(PYTHON) -m http.server "$(PORT)" --directory dashboard

lint:
	@test -d src || { printf '%s\n' 'src/ is missing; nothing to lint in this checkout.' >&2; exit 1; }
	$(PYTHON) -m compileall -q src
	@if test -d tests; then $(PYTHON) -m compileall -q tests; fi

check: lint
	$(PYTHON) -c 'import pathlib, tomllib; root=pathlib.Path("."); tomllib.load((root/"pyproject.toml").open("rb")); required=[root/"README.md",root/"Dockerfile",root/"dashboard/index.html",root/"docs/ARCHITECTURE.md",root/"docs/DATASET_VERIFICATION.md",root/"docs/DEMO_SCRIPT.md",root/"docs/ETHICS.md",root/"docs/LITERATURE_VERIFICATION.md",root/"docs/RESULTS.md",root/"docs/THREAT_MODEL.md"]; missing=[str(p) for p in required if not p.is_file()]; assert not missing, "missing required files: " + ", ".join(missing)'
	@test -s requirements.lock
	@test "$(shell tr -d '\r\n' < .python-version)" = "3.11.2"

clean:
	rm -rf "$(VENV)" "$(RUN_DIR)" "$(DEMO_RUN_DIR)" .pytest_cache
