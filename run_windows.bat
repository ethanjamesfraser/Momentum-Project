@echo off
cd /d "%~dp0"
echo Setting up (first run takes a minute or two)...
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo Running backtest...
python momentum_strategy.py > results.txt
type results.txt
start "" figures
echo.
echo Done. Results saved to results.txt, outputs\ and figures\
pause
