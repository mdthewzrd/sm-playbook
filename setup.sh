#!/usr/bin/env bash

# BMAD Setup Script - Creates trading system structure
echo "Setting up BMAD trading system structure..."

# Create directories
mkdir -p scripts reports trading-journal docs/knowledge/prompts .bmad-core/agents

# 5) Report writer → Markdown  
cat > scripts/write_report.py << 'PY'
#!/usr/bin/env python3
import argparse, json, pathlib, datetime


parser = argparse.ArgumentParser()
parser.add_argument('--in', dest='inp', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()


data = json.loads(pathlib.Path(args.inp).read_text())
M = data['metrics']
md = f"""# Strategy Report — EMA Cross\n\nGenerated: {datetime.datetime.now().isoformat(timespec='seconds')}\n\n## Params\n- fast: {data['params']['fast']}\n- slow: {data['params']['slow']}\n- cash: {data['params']['cash']:,}\n- data: {data['data_path']}\n\n## Key Metrics\n| Metric | Value |\n|---|---:|\n| Return [%] | {M.get('Return [%]', '—')} |\n| Win Rate [%] | {M.get('Win Rate [%]', '—')} |\n| # Trades | {M.get('# Trades', '—')} |\n| Max. Drawdown [%] | {M.get('Max. Drawdown [%]', '—')} |\n| Sharpe Ratio | {M.get('Sharpe Ratio', '—')} |\n\n## Trades (first 20)\n```json\n""" + json.dumps(data.get('trades_preview', []), indent=2) + "\n```\n\n"""


out = pathlib.Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(md)
print("Wrote", out)
PY
chmod +x scripts/write_report.py


# 6) Journal appender → CSV
cat > scripts/append_journal.py << 'PY'
#!/usr/bin/env python3
import argparse, json, pathlib, csv, datetime


parser = argparse.ArgumentParser()
parser.add_argument('--in', dest='inp', required=True)
parser.add_argument('--csv', required=True)
args = parser.parse_args()


data = json.loads(pathlib.Path(args.inp).read_text())
row = {
'timestamp': datetime.datetime.now().isoformat(timespec='seconds'),
'run_id': data['run_id'],
'fast': data['params']['fast'],
'slow': data['params']['slow'],
'return_pct': data['metrics'].get('Return [%]'),
'win_rate_pct': data['metrics'].get('Win Rate [%]'),
'trades': data['metrics'].get('# Trades'),
'max_drawdown_pct': data['metrics'].get('Max. Drawdown [%]'),
'sharpe': data['metrics'].get('Sharpe Ratio'),
'data_path': data['data_path'],
}


csv_path = pathlib.Path(args.csv)
csv_path.parent.mkdir(parents=True, exist_ok=True)
write_header = not csv_path.exists()
with csv_path.open('a', newline='') as f:
w = csv.DictWriter(f, fieldnames=list(row.keys()))
if write_header:
w.writeheader()
w.writerow(row)
print("Appended", csv_path)
PY
chmod +x scripts/append_journal.py


# 7) Minimal knowledge doc + prompt
cat > docs/knowledge/bmad_knowledge_pack.md << 'MD'
# BMAD Knowledge Pack (Minimal)
- Golden path: `make demo-e2e` → writes `reports/strategy_report.md` and `trading-journal/journal.csv`.
- Engine: backtesting.py (offline, cached synthetic data).
- Commands (BMAD/Claude): demo, backtest, report, journal.
- Guardrails: offline by default; no live orders; short outputs.
MD


cat > docs/knowledge/prompts/system_short.txt << 'TXT'
You are the BMAD Story/Backtest Orchestrator. Follow these commands strictly.
Golden path: make demo-e2e → backtest → report → journal.
Never preload big files; ask max 2 clarifying Qs then assume defaults.
Keep replies ≤ 800 tokens unless I say /long. Always show artifact paths.
TXT


# 8) Optional: lightweight agent aliasing (BMAD style)
cat > .bmad-core/agents/sm.yaml << 'YML'
name: sm
commands:
demo: ["make", "demo-e2e"]
backtest: ["make", "backtest"]
report: ["make", "report"]
journal: ["make", "journal"]
limits:
max_output_tokens: 800
ask_missing_inputs: 2
YML


# 9) Done
echo "\nAll files created. Next steps:\n1) python -m venv .venv && source .venv/bin/activate\n2) pip install -r requirements.txt\n3) make demo-e2e\n4) Open reports/strategy_report.md\n"