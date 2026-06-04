$ErrorActionPreference = "Stop"

python -m src.experiment.run_ablation --config configs/full/trec_covid_ablation.yaml
