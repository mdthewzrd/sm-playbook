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
