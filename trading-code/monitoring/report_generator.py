"""
Report Generation System for SM Playbook Trading System

Generates comprehensive reports for trading performance, risk analysis,
and strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
from jinja2 import Template

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Comprehensive report generation system for trading analysis.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Report templates
        self.templates = self._load_templates()
        
        logger.info(f"Report generator initialized, output: {self.output_dir}")
    
    def generate_daily_report(
        self,
        performance_data: Dict[str, Any],
        positions: List[Dict],
        trades: List[Dict],
        date: Optional[datetime] = None
    ) -> str:
        """
        Generate daily trading report.
        
        Args:
            performance_data: Performance metrics
            positions: Current positions
            trades: Today's trades
            date: Report date (default: today)
            
        Returns:
            Path to generated report file
        """
        if date is None:
            date = datetime.now()
        
        logger.info(f"Generating daily report for {date.strftime('%Y-%m-%d')}")
        
        try:
            # Prepare report data
            report_data = {
                'report_date': date.strftime('%Y-%m-%d'),
                'report_time': datetime.now().strftime('%H:%M:%S'),
                'performance': performance_data,
                'positions': self._format_positions_for_report(positions),
                'trades': self._format_trades_for_report(trades),
                'summary': self._generate_daily_summary(performance_data, positions, trades)
            }
            
            # Generate HTML report
            html_content = self._render_template('daily_report.html', report_data)
            
            # Save report
            report_filename = f"daily_report_{date.strftime('%Y%m%d')}.html"
            report_path = self.output_dir / report_filename
            
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Daily report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
            raise
    
    def generate_weekly_report(
        self,
        performance_history: List[Dict],
        start_date: Optional[datetime] = None
    ) -> str:
        """
        Generate weekly performance report.
        
        Args:
            performance_history: Historical performance data
            start_date: Week start date (default: last Monday)
            
        Returns:
            Path to generated report file
        """
        if start_date is None:
            today = datetime.now()
            start_date = today - timedelta(days=today.weekday())
        
        end_date = start_date + timedelta(days=6)
        
        logger.info(f"Generating weekly report for {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        try:
            # Filter data for the week
            week_data = [
                p for p in performance_history
                if start_date <= datetime.fromisoformat(p['timestamp']) <= end_date
            ]
            
            if not week_data:
                logger.warning("No data available for weekly report")
                return ""
            
            # Calculate weekly metrics
            weekly_metrics = self._calculate_weekly_metrics(week_data)
            
            # Prepare report data
            report_data = {
                'week_start': start_date.strftime('%Y-%m-%d'),
                'week_end': end_date.strftime('%Y-%m-%d'),
                'report_time': datetime.now().strftime('%H:%M:%S'),
                'metrics': weekly_metrics,
                'daily_data': week_data,
                'charts': self._generate_chart_data(week_data)
            }
            
            # Generate HTML report
            html_content = self._render_template('weekly_report.html', report_data)
            
            # Save report
            report_filename = f"weekly_report_{start_date.strftime('%Y%m%d')}.html"
            report_path = self.output_dir / report_filename
            
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Weekly report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")
            raise
    
    def generate_strategy_report(
        self,
        strategy_name: str,
        backtest_results: Dict[str, Any],
        live_performance: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate strategy analysis report.
        
        Args:
            strategy_name: Name of the strategy
            backtest_results: Backtesting results
            live_performance: Live trading performance data
            
        Returns:
            Path to generated report file
        """
        logger.info(f"Generating strategy report for: {strategy_name}")
        
        try:
            # Prepare report data
            report_data = {
                'strategy_name': strategy_name,
                'report_date': datetime.now().strftime('%Y-%m-%d'),
                'report_time': datetime.now().strftime('%H:%M:%S'),
                'backtest': backtest_results,
                'live_performance': live_performance,
                'analysis': self._analyze_strategy_performance(backtest_results, live_performance)
            }
            
            # Generate HTML report
            html_content = self._render_template('strategy_report.html', report_data)
            
            # Save report
            safe_name = strategy_name.replace(' ', '_').lower()
            report_filename = f"strategy_report_{safe_name}_{datetime.now().strftime('%Y%m%d')}.html"
            report_path = self.output_dir / report_filename
            
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Strategy report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating strategy report: {e}")
            raise
    
    def generate_risk_report(
        self,
        risk_metrics: Dict[str, Any],
        positions: List[Dict],
        alerts: List[Dict]
    ) -> str:
        """
        Generate risk analysis report.
        
        Args:
            risk_metrics: Risk assessment metrics
            positions: Current positions
            alerts: Risk alerts
            
        Returns:
            Path to generated report file
        """
        logger.info("Generating risk analysis report")
        
        try:
            # Prepare report data
            report_data = {
                'report_date': datetime.now().strftime('%Y-%m-%d'),
                'report_time': datetime.now().strftime('%H:%M:%S'),
                'risk_metrics': risk_metrics,
                'positions': self._analyze_position_risks(positions),
                'alerts': alerts,
                'risk_summary': self._generate_risk_summary(risk_metrics, alerts)
            }
            
            # Generate HTML report
            html_content = self._render_template('risk_report.html', report_data)
            
            # Save report
            report_filename = f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
            report_path = self.output_dir / report_filename
            
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Risk report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            raise
    
    def _format_positions_for_report(self, positions: List[Dict]) -> List[Dict]:
        """Format positions data for report display."""
        formatted = []
        
        for pos in positions:
            formatted_pos = {
                'symbol': pos.get('symbol', 'N/A'),
                'quantity': pos.get('quantity', 0),
                'entry_price': pos.get('entry_price', 0),
                'current_price': pos.get('current_price', 0),
                'market_value': pos.get('quantity', 0) * pos.get('current_price', 0),
                'unrealized_pnl': pos.get('unrealized_pnl', 0),
                'unrealized_pnl_pct': self._calculate_pnl_percentage(pos),
                'position_size_pct': pos.get('position_size_pct', 0)
            }
            formatted.append(formatted_pos)
        
        return formatted
    
    def _format_trades_for_report(self, trades: List[Dict]) -> List[Dict]:
        """Format trades data for report display."""
        formatted = []
        
        for trade in trades:
            formatted_trade = {
                'symbol': trade.get('symbol', 'N/A'),
                'side': trade.get('side', 'N/A'),
                'quantity': trade.get('quantity', 0),
                'entry_price': trade.get('entry_price', 0),
                'exit_price': trade.get('exit_price', 0),
                'pnl': trade.get('pnl', 0),
                'pnl_pct': trade.get('pnl_percent', 0),
                'entry_time': trade.get('entry_date', ''),
                'exit_time': trade.get('exit_date', ''),
                'duration': self._calculate_trade_duration(trade)
            }
            formatted.append(formatted_trade)
        
        return formatted
    
    def _generate_daily_summary(
        self,
        performance_data: Dict[str, Any],
        positions: List[Dict],
        trades: List[Dict]
    ) -> Dict[str, Any]:
        """Generate daily summary statistics."""
        summary = {
            'total_positions': len(positions),
            'total_trades': len(trades),
            'winning_trades': len([t for t in trades if t.get('pnl', 0) > 0]),
            'losing_trades': len([t for t in trades if t.get('pnl', 0) < 0]),
            'win_rate': 0,
            'total_pnl': sum(t.get('pnl', 0) for t in trades),
            'largest_winner': max([t.get('pnl', 0) for t in trades], default=0),
            'largest_loser': min([t.get('pnl', 0) for t in trades], default=0),
            'avg_trade': 0
        }
        
        if trades:
            summary['win_rate'] = summary['winning_trades'] / len(trades) * 100
            summary['avg_trade'] = summary['total_pnl'] / len(trades)
        
        return summary
    
    def _calculate_weekly_metrics(self, week_data: List[Dict]) -> Dict[str, Any]:
        """Calculate weekly performance metrics."""
        if not week_data:
            return {}
        
        first_day = week_data[0]
        last_day = week_data[-1]
        
        weekly_return = 0
        if first_day.get('portfolio_value', 0) > 0:
            weekly_return = ((last_day.get('portfolio_value', 0) / 
                            first_day.get('portfolio_value', 1)) - 1) * 100
        
        # Calculate daily P&L changes
        daily_pnls = [day.get('daily_pnl', 0) for day in week_data]
        
        metrics = {
            'weekly_return': weekly_return,
            'best_day': max(daily_pnls),
            'worst_day': min(daily_pnls),
            'avg_daily_pnl': np.mean(daily_pnls),
            'volatility': np.std(daily_pnls),
            'total_trades': last_day.get('total_trades', 0) - first_day.get('total_trades', 0),
            'win_rate': last_day.get('win_rate', 0) * 100,
            'sharpe_ratio': last_day.get('sharpe_ratio', 0)
        }
        
        return metrics
    
    def _analyze_strategy_performance(
        self,
        backtest_results: Dict[str, Any],
        live_performance: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze strategy performance comparing backtest vs live."""
        analysis = {
            'backtest_metrics': backtest_results.get('performance_summary', {}),
            'live_metrics': live_performance or {},
            'performance_comparison': {},
            'recommendations': []
        }
        
        if live_performance:
            # Compare key metrics
            bt_return = backtest_results.get('performance_summary', {}).get('total_return', 0)
            live_return = live_performance.get('total_return', 0)
            
            performance_diff = live_return - bt_return
            
            analysis['performance_comparison'] = {
                'return_difference': performance_diff,
                'is_outperforming': performance_diff > 0,
                'performance_consistency': abs(performance_diff) < 5  # Within 5%
            }
            
            # Generate recommendations
            if performance_diff < -10:
                analysis['recommendations'].append(
                    "Live performance significantly below backtest - review strategy parameters"
                )
            elif abs(performance_diff) > 15:
                analysis['recommendations'].append(
                    "High variance between backtest and live - investigate market regime changes"
                )
            else:
                analysis['recommendations'].append(
                    "Performance within expected range - continue monitoring"
                )
        
        return analysis
    
    def _analyze_position_risks(self, positions: List[Dict]) -> Dict[str, Any]:
        """Analyze risks in current positions."""
        if not positions:
            return {'total_risk': 0, 'position_risks': [], 'concentration': {}}
        
        total_value = sum(abs(p.get('quantity', 0) * p.get('current_price', 0)) for p in positions)
        
        position_risks = []
        for pos in positions:
            pos_value = abs(pos.get('quantity', 0) * pos.get('current_price', 0))
            risk_pct = (pos_value / total_value * 100) if total_value > 0 else 0
            
            position_risks.append({
                'symbol': pos.get('symbol'),
                'risk_percentage': risk_pct,
                'unrealized_pnl': pos.get('unrealized_pnl', 0),
                'risk_level': 'HIGH' if risk_pct > 20 else 'MEDIUM' if risk_pct > 10 else 'LOW'
            })
        
        # Sector concentration analysis (simplified)
        concentration = {
            'largest_position': max(position_risks, key=lambda x: x['risk_percentage']),
            'position_count': len(positions),
            'avg_position_size': sum(p['risk_percentage'] for p in position_risks) / len(position_risks)
        }
        
        return {
            'total_risk': sum(p['risk_percentage'] for p in position_risks),
            'position_risks': position_risks,
            'concentration': concentration
        }
    
    def _generate_risk_summary(self, risk_metrics: Dict[str, Any], alerts: List[Dict]) -> Dict[str, Any]:
        """Generate risk summary for report."""
        active_alerts = [a for a in alerts if not a.get('acknowledged', False)]
        
        risk_level = 'LOW'
        if len(active_alerts) > 5:
            risk_level = 'HIGH'
        elif len(active_alerts) > 2:
            risk_level = 'MEDIUM'
        
        return {
            'overall_risk_level': risk_level,
            'active_alerts': len(active_alerts),
            'total_alerts': len(alerts),
            'portfolio_risk': risk_metrics.get('portfolio_risk', 0),
            'max_drawdown': risk_metrics.get('current_drawdown', 0),
            'key_concerns': [a['message'] for a in active_alerts[:3]]  # Top 3 concerns
        }
    
    def _generate_chart_data(self, data: List[Dict]) -> Dict[str, List]:
        """Generate data for charts in reports."""
        return {
            'dates': [d.get('timestamp', '') for d in data],
            'portfolio_values': [d.get('portfolio_value', 0) for d in data],
            'daily_pnls': [d.get('daily_pnl', 0) for d in data],
            'drawdowns': [d.get('drawdown', 0) for d in data]
        }
    
    def _calculate_pnl_percentage(self, position: Dict) -> float:
        """Calculate P&L percentage for a position."""
        entry_price = position.get('entry_price', 0)
        current_price = position.get('current_price', 0)
        
        if entry_price <= 0:
            return 0
        
        return ((current_price / entry_price) - 1) * 100
    
    def _calculate_trade_duration(self, trade: Dict) -> str:
        """Calculate trade duration as human-readable string."""
        entry_date = trade.get('entry_date')
        exit_date = trade.get('exit_date')
        
        if not entry_date or not exit_date:
            return 'N/A'
        
        try:
            if isinstance(entry_date, str):
                entry_date = datetime.fromisoformat(entry_date)
            if isinstance(exit_date, str):
                exit_date = datetime.fromisoformat(exit_date)
            
            duration = exit_date - entry_date
            return f"{duration.days}d {duration.seconds//3600}h"
        except:
            return 'N/A'
    
    def _load_templates(self) -> Dict[str, str]:
        """Load HTML report templates."""
        templates = {}
        
        # Basic HTML template for daily report
        templates['daily_report.html'] = """
<!DOCTYPE html>
<html>
<head>
    <title>Daily Trading Report - {{ report_date }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .metrics { display: flex; justify-content: space-around; }
        .metric { text-align: center; padding: 15px; background: #e8f4f8; border-radius: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .positive { color: green; }
        .negative { color: red; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Daily Trading Report</h1>
        <p>Date: {{ report_date }} | Generated: {{ report_time }}</p>
    </div>
    
    <div class="section">
        <h2>Performance Summary</h2>
        <div class="metrics">
            <div class="metric">
                <h3>Portfolio Value</h3>
                <p>${{ "%.2f"|format(performance.current_value) }}</p>
            </div>
            <div class="metric">
                <h3>Daily P&L</h3>
                <p class="{% if performance.daily_pnl >= 0 %}positive{% else %}negative{% endif %}">
                    ${{ "%.2f"|format(performance.daily_pnl) }}
                </p>
            </div>
            <div class="metric">
                <h3>Total Return</h3>
                <p class="{% if performance.total_return >= 0 %}positive{% else %}negative{% endif %}">
                    {{ "%.2f"|format(performance.total_return) }}%
                </p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Current Positions</h2>
        <table>
            <tr>
                <th>Symbol</th>
                <th>Quantity</th>
                <th>Entry Price</th>
                <th>Current Price</th>
                <th>Market Value</th>
                <th>Unrealized P&L</th>
            </tr>
            {% for pos in positions %}
            <tr>
                <td>{{ pos.symbol }}</td>
                <td>{{ pos.quantity }}</td>
                <td>${{ "%.2f"|format(pos.entry_price) }}</td>
                <td>${{ "%.2f"|format(pos.current_price) }}</td>
                <td>${{ "%.2f"|format(pos.market_value) }}</td>
                <td class="{% if pos.unrealized_pnl >= 0 %}positive{% else %}negative{% endif %}">
                    ${{ "%.2f"|format(pos.unrealized_pnl) }}
                </td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="section">
        <h2>Today's Trades</h2>
        <table>
            <tr>
                <th>Symbol</th>
                <th>Side</th>
                <th>Quantity</th>
                <th>Entry Price</th>
                <th>Exit Price</th>
                <th>P&L</th>
                <th>Duration</th>
            </tr>
            {% for trade in trades %}
            <tr>
                <td>{{ trade.symbol }}</td>
                <td>{{ trade.side }}</td>
                <td>{{ trade.quantity }}</td>
                <td>${{ "%.2f"|format(trade.entry_price) }}</td>
                <td>${{ "%.2f"|format(trade.exit_price) }}</td>
                <td class="{% if trade.pnl >= 0 %}positive{% else %}negative{% endif %}">
                    ${{ "%.2f"|format(trade.pnl) }}
                </td>
                <td>{{ trade.duration }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
</body>
</html>
        """
        
        # Add other templates here as needed
        templates['weekly_report.html'] = "<!-- Weekly Report Template -->"
        templates['strategy_report.html'] = "<!-- Strategy Report Template -->"
        templates['risk_report.html'] = "<!-- Risk Report Template -->"
        
        return templates
    
    def _render_template(self, template_name: str, data: Dict[str, Any]) -> str:
        """Render HTML template with data."""
        try:
            template_str = self.templates.get(template_name, '')
            if not template_str:
                return f"<html><body><h1>Template not found: {template_name}</h1></body></html>"
            
            template = Template(template_str)
            return template.render(**data)
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {e}")
            return f"<html><body><h1>Error rendering report: {e}</h1></body></html>"


# Example usage
def main():
    """Example usage of report generator."""
    generator = ReportGenerator()
    
    # Sample data
    performance_data = {
        'current_value': 105000,
        'daily_pnl': 2500,
        'total_return': 5.0
    }
    
    positions = [
        {
            'symbol': 'AAPL',
            'quantity': 100,
            'entry_price': 150.00,
            'current_price': 155.00,
            'unrealized_pnl': 500
        }
    ]
    
    trades = [
        {
            'symbol': 'MSFT',
            'side': 'buy',
            'quantity': 50,
            'entry_price': 300.00,
            'exit_price': 310.00,
            'pnl': 500,
            'entry_date': '2024-01-15 09:30:00',
            'exit_date': '2024-01-15 15:30:00'
        }
    ]
    
    # Generate daily report
    report_path = generator.generate_daily_report(performance_data, positions, trades)
    print(f"Daily report generated: {report_path}")


if __name__ == '__main__':
    main()