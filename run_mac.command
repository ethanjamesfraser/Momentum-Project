#!/bin/bash
cd "$(dirname "$0")"
echo "Setting up (first run takes a minute or two)..."
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "Running backtest..."
python momentum_strategy.py | tee results.txt
open figures
echo ""
echo "Done. Results saved to results.txt, outputs/ and figures/"
