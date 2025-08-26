#!/usr/bin/env python3
"""
BMad Trading System Setup Script

This script sets up the complete SM Playbook BMAT trading system,
integrating with existing directories and ensuring all components
are properly configured.
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BMadSystemSetup:
    """Setup and integration manager for the BMad trading system."""
    
    def __init__(self, project_root: Path):
        """Initialize setup manager."""
        self.project_root = Path(project_root)
        self.bmad_core_path = self.project_root / '.bmad-core'
        self.trading_code_path = self.project_root / 'trading-code'
        
        # Existing directories to integrate
        self.existing_dirs = {
            'backtest-results': self.project_root / 'backtest-results',
            'playbook': self.project_root / 'playbook',
            'trading-data': self.project_root / 'trading-data',
            'trading-journal': self.project_root / 'trading-journal',
            'trading-logs': self.project_root / 'trading-logs'
        }
        
        logger.info(f"BMad setup initialized for: {self.project_root}")
    
    def setup_complete_system(self):
        """Set up the complete BMad trading system."""
        logger.info("Starting complete BMad system setup...")
        
        try:
            # 1. Verify BMad core structure
            self.verify_bmad_core()
            
            # 2. Set up Python environment
            self.setup_python_environment()
            
            # 3. Integrate with existing directories
            self.integrate_existing_directories()
            
            # 4. Create integration scripts
            self.create_integration_scripts()
            
            # 5. Set up data pipelines
            self.setup_data_pipelines()
            
            # 6. Create configuration files
            self.create_configuration_files()
            
            # 7. Set up monitoring and logging
            self.setup_monitoring()
            
            # 8. Create startup scripts
            self.create_startup_scripts()
            
            # 9. Validate installation
            self.validate_installation()
            
            logger.info("✅ BMad trading system setup completed successfully!")
            self.print_setup_summary()
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise
    
    def verify_bmad_core(self):
        """Verify BMad core structure exists."""
        logger.info("Verifying BMad core structure...")
        
        required_paths = [
            self.bmad_core_path / 'agents',
            self.bmad_core_path / 'workflows',
            self.bmad_core_path / 'data',
            self.bmad_core_path / 'agent-teams'
        ]
        
        for path in required_paths:
            if not path.exists():
                logger.error(f"Missing required path: {path}")
                raise FileNotFoundError(f"BMad core path missing: {path}")
        
        logger.info("✅ BMad core structure verified")
    
    def setup_python_environment(self):
        """Set up Python environment and dependencies."""
        logger.info("Setting up Python environment...")
        
        # Create requirements.txt
        requirements = [
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "scipy>=1.9.0",
            "pyyaml>=6.0",
            "asyncio",
            "aiofiles",
            "websockets",
            "requests",
            "python-dotenv",
            "ta-lib",  # Technical analysis library
            "ccxt",    # Cryptocurrency exchange connectivity
            "yfinance", # Yahoo Finance data
            "alpaca-trade-api",  # Alpaca trading API
            "ib-insync",  # Interactive Brokers
        ]
        
        requirements_file = self.project_root / 'requirements.txt'
        with open(requirements_file, 'w') as f:
            f.write('\n'.join(requirements))
        
        logger.info("✅ Requirements file created")
        
        # Create virtual environment setup script
        venv_script = self.project_root / 'setup_venv.sh'
        venv_content = '''#!/bin/bash
# BMad Trading System - Virtual Environment Setup

echo "Setting up Python virtual environment for BMad trading system..."

# Create virtual environment
python3 -m venv bmad_env

# Activate virtual environment
source bmad_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "✅ Virtual environment setup complete!"
echo "To activate: source bmad_env/bin/activate"
echo "To start BMad: python bmad_interface.py"
'''
        
        with open(venv_script, 'w') as f:
            f.write(venv_content)
        
        # Make executable
        os.chmod(venv_script, 0o755)
        
        logger.info("✅ Virtual environment setup script created")
    
    def integrate_existing_directories(self):
        """Integrate with existing trading directories."""
        logger.info("Integrating with existing directories...")
        
        # Create integration mappings
        integrations = {
            'playbook': {
                'source': self.existing_dirs['playbook'],
                'bmad_link': self.bmad_core_path / 'data' / 'active-playbook'
            },
            'backtest-results': {
                'source': self.existing_dirs['backtest-results'],
                'bmad_link': self.trading_code_path / 'backtesting' / 'results'
            },
            'trading-data': {
                'source': self.existing_dirs['trading-data'],
                'bmad_link': self.trading_code_path / 'data'
            },
            'trading-logs': {
                'source': self.existing_dirs['trading-logs'],
                'bmad_link': self.project_root / 'logs'
            }
        }
        
        for name, config in integrations.items():
            source = config['source']
            link = config['bmad_link']
            
            # Ensure source directory exists
            source.mkdir(parents=True, exist_ok=True)
            
            # Create symbolic link if it doesn't exist
            if not link.exists():
                link.parent.mkdir(parents=True, exist_ok=True)
                try:
                    link.symlink_to(source, target_is_directory=True)
                    logger.info(f"✅ Linked {name}: {link} -> {source}")
                except OSError as e:
                    logger.warning(f"Could not create symlink for {name}: {e}")
                    # Fall back to copying directory structure
                    if not link.exists():
                        shutil.copytree(source, link, dirs_exist_ok=True)
        
        logger.info("✅ Directory integration completed")
    
    def create_integration_scripts(self):
        """Create scripts for data integration."""
        logger.info("Creating integration scripts...")
        
        scripts_dir = self.project_root / 'scripts'
        scripts_dir.mkdir(exist_ok=True)
        
        # Market data integration script
        market_data_script = scripts_dir / 'update_market_data.py'
        market_data_content = '''#!/usr/bin/env python3
"""
Market Data Update Script
Updates market data for the BMad trading system
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_market_data():
    """Update market data for trading system."""
    data_dir = Path(__file__).parent.parent / 'trading-data'
    data_dir.mkdir(exist_ok=True)
    
    # Default symbols to update
    symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    logger.info(f"Updating market data for {len(symbols)} symbols...")
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            
            # Get 2 years of data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            data = ticker.history(start=start_date, end=end_date)
            
            if not data.empty:
                output_file = data_dir / f'{symbol}_daily.csv'
                data.to_csv(output_file)
                logger.info(f"✅ Updated {symbol}: {len(data)} records")
            else:
                logger.warning(f"❌ No data for {symbol}")
                
        except Exception as e:
            logger.error(f"Error updating {symbol}: {e}")
    
    logger.info("Market data update completed")

if __name__ == '__main__':
    update_market_data()
'''
        
        with open(market_data_script, 'w') as f:
            f.write(market_data_content)
        
        os.chmod(market_data_script, 0o755)
        
        # Playbook sync script
        playbook_sync_script = scripts_dir / 'sync_playbook.py'
        playbook_sync_content = '''#!/usr/bin/env python3
"""
Playbook Sync Script
Synchronizes trading playbook with BMad system
"""

import json
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sync_playbook():
    """Sync playbook data with BMad system."""
    playbook_dir = Path(__file__).parent.parent / 'playbook'
    bmad_data_dir = Path(__file__).parent.parent / '.bmad-core' / 'data'
    
    # Create playbook index
    strategies = []
    
    for strategy_file in playbook_dir.glob('*.md'):
        strategy = {
            'name': strategy_file.stem,
            'file': str(strategy_file.relative_to(playbook_dir)),
            'last_modified': strategy_file.stat().st_mtime
        }
        strategies.append(strategy)
    
    # Save playbook index
    index_file = bmad_data_dir / 'playbook-index.json'
    with open(index_file, 'w') as f:
        json.dump(strategies, f, indent=2)
    
    logger.info(f"✅ Synced {len(strategies)} strategies to playbook index")

if __name__ == '__main__':
    sync_playbook()
'''
        
        with open(playbook_sync_script, 'w') as f:
            f.write(playbook_sync_content)
        
        os.chmod(playbook_sync_script, 0o755)
        
        logger.info("✅ Integration scripts created")
    
    def setup_data_pipelines(self):
        """Set up data processing pipelines."""
        logger.info("Setting up data pipelines...")
        
        pipelines_dir = self.project_root / 'pipelines'
        pipelines_dir.mkdir(exist_ok=True)
        
        # Create data pipeline configuration
        pipeline_config = {
            'market_data': {
                'sources': ['yahoo_finance', 'alpha_vantage'],
                'symbols': ['SPY', 'QQQ', 'IWM'],
                'intervals': ['1d', '1h'],
                'storage': 'trading-data/market'
            },
            'fundamental_data': {
                'sources': ['yahoo_finance'],
                'metrics': ['pe_ratio', 'market_cap', 'dividend_yield'],
                'storage': 'trading-data/fundamentals'
            },
            'alternative_data': {
                'sources': ['sentiment', 'news'],
                'storage': 'trading-data/alternative'
            }
        }
        
        config_file = pipelines_dir / 'pipeline_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(pipeline_config, f, default_flow_style=False)
        
        logger.info("✅ Data pipelines configured")
    
    def create_configuration_files(self):
        """Create system configuration files."""
        logger.info("Creating configuration files...")
        
        config_dir = self.project_root / 'config'
        config_dir.mkdir(exist_ok=True)
        
        # Main BMad configuration
        bmad_config = {
            'system': {
                'name': 'SM Playbook BMad Trading System',
                'version': '1.0.0',
                'mode': 'development'
            },
            'agents': {
                'path': '.bmad-core/agents',
                'default_agent': 'bmad-orchestrator'
            },
            'workflows': {
                'path': '.bmad-core/workflows',
                'auto_save': True
            },
            'data': {
                'market_data_path': 'trading-data',
                'playbook_path': 'playbook',
                'results_path': 'backtest-results'
            },
            'execution': {
                'default_mode': 'paper',
                'risk_limits': {
                    'max_portfolio_risk': 0.08,
                    'max_position_risk': 0.02,
                    'max_daily_loss': 5000
                }
            },
            'logging': {
                'level': 'INFO',
                'file': 'trading-logs/bmad.log'
            }
        }
        
        config_file = config_dir / 'bmad_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(bmad_config, f, default_flow_style=False)
        
        # Environment variables template
        env_template = '''# BMad Trading System Environment Variables
# Copy this file to .env and fill in your API keys

# Market Data APIs
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
YAHOO_FINANCE_API_KEY=your_yahoo_key

# Broker APIs
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Interactive Brokers
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=1

# Database
DATABASE_URL=sqlite:///trading_data.db

# Risk Management
MAX_PORTFOLIO_RISK=0.08
MAX_POSITION_RISK=0.02
MAX_DAILY_LOSS=5000

# Logging
LOG_LEVEL=INFO
'''
        
        env_file = self.project_root / '.env.template'
        with open(env_file, 'w') as f:
            f.write(env_template)
        
        logger.info("✅ Configuration files created")
    
    def setup_monitoring(self):
        """Set up monitoring and logging systems."""
        logger.info("Setting up monitoring systems...")
        
        # Ensure log directory exists
        logs_dir = self.existing_dirs['trading-logs']
        logs_dir.mkdir(exist_ok=True)
        
        # Create log configuration
        log_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                },
                'detailed': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
                }
            },
            'handlers': {
                'default': {
                    'level': 'INFO',
                    'formatter': 'standard',
                    'class': 'logging.StreamHandler'
                },
                'file': {
                    'level': 'DEBUG',
                    'formatter': 'detailed',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': 'trading-logs/bmad.log',
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5
                },
                'trading_file': {
                    'level': 'INFO',
                    'formatter': 'detailed',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': 'trading-logs/trading.log',
                    'maxBytes': 10485760,
                    'backupCount': 10
                }
            },
            'loggers': {
                'bmad': {
                    'handlers': ['default', 'file'],
                    'level': 'INFO',
                    'propagate': False
                },
                'trading': {
                    'handlers': ['default', 'trading_file'],
                    'level': 'INFO',
                    'propagate': False
                }
            },
            'root': {
                'level': 'INFO',
                'handlers': ['default']
            }
        }
        
        log_config_file = self.project_root / 'config' / 'logging_config.yaml'
        with open(log_config_file, 'w') as f:
            yaml.dump(log_config, f, default_flow_style=False)
        
        logger.info("✅ Monitoring systems configured")
    
    def create_startup_scripts(self):
        """Create startup scripts for the system."""
        logger.info("Creating startup scripts...")
        
        # Main startup script
        startup_script = self.project_root / 'start_bmad.sh'
        startup_content = '''#!/bin/bash
# BMad Trading System Startup Script

echo "🚀 Starting BMad Trading System..."

# Check if virtual environment exists
if [ ! -d "bmad_env" ]; then
    echo "Virtual environment not found. Run ./setup_venv.sh first."
    exit 1
fi

# Activate virtual environment
source bmad_env/bin/activate

# Update market data (optional)
read -p "Update market data? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📊 Updating market data..."
    python scripts/update_market_data.py
fi

# Sync playbook
echo "📋 Syncing playbook..."
python scripts/sync_playbook.py

# Start BMad interface
echo "🎭 Starting BMad interface..."
python bmad_interface.py

echo "👋 BMad system shutdown complete"
'''
        
        with open(startup_script, 'w') as f:
            f.write(startup_content)
        
        os.chmod(startup_script, 0o755)
        
        # Quick start script
        quick_start = self.project_root / 'quick_start.sh'
        quick_start_content = '''#!/bin/bash
# Quick Start BMad Trading System

echo "⚡ Quick Start - BMad Trading System"

# Activate virtual environment
source bmad_env/bin/activate

# Start BMad interface directly
python bmad_interface.py --agent trading-orchestrator

echo "Session complete"
'''
        
        with open(quick_start, 'w') as f:
            f.write(quick_start_content)
        
        os.chmod(quick_start, 0o755)
        
        logger.info("✅ Startup scripts created")
    
    def validate_installation(self):
        """Validate the BMad installation."""
        logger.info("Validating installation...")
        
        validations = []
        
        # Check core directories
        required_dirs = [
            self.bmad_core_path,
            self.trading_code_path,
            self.project_root / 'config'
        ]
        
        for directory in required_dirs:
            if directory.exists():
                validations.append(f"✅ {directory.name}")
            else:
                validations.append(f"❌ {directory.name}")
        
        # Check key files
        key_files = [
            self.project_root / 'bmad_interface.py',
            self.project_root / 'requirements.txt',
            self.project_root / 'config' / 'bmad_config.yaml'
        ]
        
        for file_path in key_files:
            if file_path.exists():
                validations.append(f"✅ {file_path.name}")
            else:
                validations.append(f"❌ {file_path.name}")
        
        # Check Python syntax
        try:
            interface_file = self.project_root / 'bmad_interface.py'
            result = subprocess.run([
                sys.executable, '-m', 'py_compile', str(interface_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                validations.append("✅ Python syntax valid")
            else:
                validations.append("❌ Python syntax errors")
                logger.error(f"Syntax error: {result.stderr}")
        except Exception as e:
            validations.append(f"❌ Python validation failed: {e}")
        
        # Print validation results
        logger.info("Validation Results:")
        for validation in validations:
            logger.info(f"  {validation}")
        
        # Check if all validations passed
        failed_count = sum(1 for v in validations if v.startswith('❌'))
        if failed_count == 0:
            logger.info("✅ All validations passed!")
        else:
            logger.warning(f"⚠️  {failed_count} validation(s) failed")
    
    def print_setup_summary(self):
        """Print setup completion summary."""
        summary = f'''
{'='*60}
    🎭 BMad TRADING SYSTEM SETUP COMPLETE
{'='*60}

📁 Project Structure:
   ├── .bmad-core/           # BMad core system
   ├── trading-code/         # Trading algorithms
   ├── config/               # Configuration files  
   ├── scripts/              # Integration scripts
   ├── pipelines/            # Data pipelines
   └── bmad_interface.py     # Main interface

🚀 Quick Start:
   1. Set up Python environment:  ./setup_venv.sh
   2. Start BMad system:          ./start_bmad.sh
   3. Or quick start:             ./quick_start.sh

🔧 Configuration:
   - Copy .env.template to .env and add your API keys
   - Modify config/bmad_config.yaml as needed
   - Update scripts/update_market_data.py with your symbols

📚 Available Agents:
   - bmad-orchestrator        (Master coordinator)
   - trading-orchestrator     (Trading system manager)
   - strategy-designer        (Strategy development)
   - backtesting-engineer     (Validation & testing)
   - execution-engineer       (Trade execution)
   - indicator-developer      (Technical indicators)

🔄 Available Workflows:
   - strategy-development     (Complete strategy lifecycle)
   - signal-generation        (Generate trading signals)
   - market-analysis          (Market condition analysis)
   - performance-review       (Strategy optimization)

💡 Next Steps:
   1. Configure your market data sources
   2. Set up broker connections (paper trading recommended)
   3. Import your existing strategies to playbook/
   4. Run your first strategy backtest

📖 For help: python bmad_interface.py
             Then type: *help

{'='*60}
    HAPPY TRADING! 🚀📈
{'='*60}
'''
        print(summary)


def main():
    """Main setup function."""
    project_root = Path(__file__).parent
    
    print("🎭 Starting BMad Trading System Setup...")
    print(f"📁 Project root: {project_root}")
    
    setup = BMadSystemSetup(project_root)
    setup.setup_complete_system()
    
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)