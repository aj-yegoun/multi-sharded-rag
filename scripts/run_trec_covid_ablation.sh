#!/usr/bin/env bash
set -euo pipefail

python -m src.experiment.run_ablation --config configs/full/trec_covid_ablation.yaml
