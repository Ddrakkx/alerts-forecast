#!/usr/bin/env bash
# Whole pipeline for one oblast, outputs to results/<slug>/. Run from an activated virtual environment.
#   bash scripts/run_region.sh "Lvivska oblast"
set -euo pipefail
region="$1"
slug=$(echo "$region" | cut -d' ' -f1 | tr 'A-Z' 'a-z')
cd "$(dirname "$0")/.."
mkdir -p "results/$slug"
export PYTHONUNBUFFERED=1
python scripts/run_experiment.py --region "$region" > "results/$slug/experiment.txt"
python scripts/run_robustness.py --region "$region" > "results/$slug/robustness.txt"
python scripts/run_boosting.py --region "$region" > "results/$slug/boosting.txt"
echo "done: $region"
