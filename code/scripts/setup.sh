#!/usr/bin/env bash
set -euo pipefail
python -m pip install -e '.[geometry,media,test]'
python -m roam_bench.cli --dataset ../dataset --dry-run
python -m pytest -q
