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
