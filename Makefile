# Локальное окружение хранится в .venv, конфигурация по умолчанию - configs/base.yaml.

PYTHON ?= python3.11
VENV   := .venv
PY     := $(VENV)/bin/python
CONFIG := configs/base.yaml

.PHONY: install prep split atomic canary train eval report test clean

install:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

prep:
	$(PY) -m src.cli prep --config $(CONFIG)

split:
	$(PY) -m src.cli split --config $(CONFIG)

atomic:
	$(PY) -m src.cli atomic --config $(CONFIG)

canary:
	$(PY) -m src.cli eval --model canary --split val --config $(CONFIG)

train:
	$(PY) -m src.cli train --config $(CONFIG)

eval:
	$(PY) -m src.cli eval --config $(CONFIG)

report:
	$(PY) -m src.cli report --config $(CONFIG)

test:
	$(PY) -m pytest tests/ -v

clean:
	rm -rf $(VENV)
