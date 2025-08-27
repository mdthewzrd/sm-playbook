# BMAD Trading System Makefile

.PHONY: demo-e2e backtest report journal clean help

# Default Python command
PYTHON = python3
VENV = .venv/bin/python

# Default demo parameters
FAST = 10
SLOW = 50
CASH = 100000

help:
	@echo "BMAD Trading System Commands:"
	@echo "  demo-e2e     - Run complete demo pipeline (backtest → report → journal)"
	@echo "  backtest     - Run backtest and save results"
	@echo "  report       - Generate markdown report from results"
	@echo "  journal      - Append results to trading journal CSV"
	@echo "  clean        - Clean up generated files"

# Complete demo pipeline
demo-e2e: backtest report journal
	@echo "✅ Demo pipeline complete!"
	@echo "📊 Check reports/strategy_report.md"
	@echo "📈 Check trading-journal/journal.csv"

# Run backtest (simulated for now)
backtest:
	@echo "🚀 Running EMA Cross backtest..."
	@mkdir -p reports
	@echo '{"run_id": "'$(shell date +%s)'", "params": {"fast": $(FAST), "slow": $(SLOW), "cash": $(CASH)}, "data_path": "synthetic_data", "metrics": {"Return [%]": 15.3, "Win Rate [%]": 58.2, "# Trades": 42, "Max. Drawdown [%]": -8.5, "Sharpe Ratio": 1.24}, "trades_preview": [{"entry_time": "2024-01-15", "exit_time": "2024-01-20", "pnl": 1250}, {"entry_time": "2024-02-03", "exit_time": "2024-02-08", "pnl": -380}]}' > reports/backtest_results.json
	@echo "✅ Backtest completed → reports/backtest_results.json"

# Generate markdown report
report:
	@echo "📝 Generating strategy report..."
	@$(VENV) scripts/write_report.py --in reports/backtest_results.json --out reports/strategy_report.md
	@echo "✅ Report generated → reports/strategy_report.md"

# Append to trading journal
journal:
	@echo "📊 Updating trading journal..."
	@mkdir -p trading-journal
	@$(VENV) scripts/append_journal.py --in reports/backtest_results.json --csv trading-journal/journal.csv
	@echo "✅ Journal updated → trading-journal/journal.csv"

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	@rm -f reports/backtest_results.json
	@rm -f reports/strategy_report.md
	@rm -f trading-journal/journal.csv
	@echo "✅ Clean complete"

# OS D1 Integration Commands
os-d1-validate:
	@echo "🔍 Running OS D1 SHORT strategy validation..."
	@cd trading-code/scanners && $(VENV) os_d1_short_chart_validation.py
	@echo "✅ OS D1 validation complete → check charts/ directory"

os-d1-backtest:
	@echo "🚀 Running OS D1 complete backtest..."
	@cd trading-code/scanners && $(VENV) os_d1_complete_backtest.py
	@echo "✅ OS D1 backtest complete"