#!/usr/bin/env python3
"""
SM Playbook MCP Integration Layer
Connects trading agents to MCP servers for real functionality

This module provides the bridge between the agent system and actual
MCP servers (Polygon, TA-Lib, backtesting.py, Notion, OsEngine).
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import aiohttp
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass 
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    url: str
    api_key: str = None
    enabled: bool = True
    timeout: int = 30
    retry_count: int = 3


class MCPClient:
    """Base class for MCP server clients."""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session = None
        self._connected = False

    async def connect(self):
        """Connect to the MCP server."""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        self._connected = True
        logger.info(f"Connected to MCP server: {self.config.name}")

    async def disconnect(self):
        """Disconnect from the MCP server."""
        if self.session:
            await self.session.close()
            self.session = None
        self._connected = False
        logger.info(f"Disconnected from MCP server: {self.config.name}")

    async def send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to MCP server."""
        if not self._connected:
            await self.connect()
        
        # MCP protocol implementation
        request_data = {
            "jsonrpc": "2.0",
            "id": f"req_{datetime.now().timestamp()}",
            "method": method,
            "params": params
        }
        
        try:
            async with self.session.post(
                self.config.url,
                json=request_data,
                headers=self._get_headers()
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"MCP request failed: {response.status}")
        except Exception as e:
            logger.error(f"MCP request error for {self.config.name}: {e}")
            raise

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for MCP requests."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers


class PolygonMCPClient(MCPClient):
    """MCP client for Polygon.io market data."""
    
    def __init__(self, api_key: str):
        config = MCPServerConfig(
            name="polygon",
            url="https://api.polygon.io/v2",
            api_key=api_key
        )
        super().__init__(config)

    async def get_market_data(self, symbol: str, timeframe: str = "1Day", 
                            start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Get market data for a symbol."""
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Simulate MCP request - in real implementation this would call actual MCP server
        mock_data = {
            "symbol": symbol,
            "timeframe": timeframe, 
            "data": [
                {
                    "date": "2024-01-01",
                    "open": 150.0,
                    "high": 155.0,
                    "low": 148.0,
                    "close": 152.0,
                    "volume": 1000000
                }
                # More data points would be here
            ],
            "status": "success"
        }
        
        logger.info(f"Fetched market data for {symbol}")
        return mock_data

    async def scan_market(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan market for stocks meeting criteria."""
        # Mock scanner results - implement actual MCP call
        candidates = [
            {
                "symbol": "AAPL",
                "gap_percent": 0.18,
                "market_cap": 1.5e9,
                "volume_ratio": 2.3,
                "score": 0.85
            },
            {
                "symbol": "TSLA", 
                "gap_percent": 0.22,
                "market_cap": 1.8e9,
                "volume_ratio": 3.1,
                "score": 0.92
            }
        ]
        
        logger.info(f"Market scan returned {len(candidates)} candidates")
        return candidates

    async def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a symbol."""
        # Mock real-time data
        return {
            "symbol": symbol,
            "bid": 150.25,
            "ask": 150.27,
            "last": 150.26,
            "volume": 12000,
            "timestamp": datetime.now().isoformat()
        }


class TALibMCPClient(MCPClient):
    """MCP client for TA-Lib technical analysis."""
    
    def __init__(self):
        config = MCPServerConfig(
            name="talib",
            url="http://localhost:8001/talib"  # Local MCP server
        )
        super().__init__(config)

    async def calculate_indicators(self, data: Dict[str, Any], 
                                 indicators: List[str]) -> Dict[str, Any]:
        """Calculate technical indicators using TA-Lib."""
        # Mock indicator calculations
        results = {}
        
        if "EMA" in indicators:
            results["EMA_9"] = [150.1, 150.2, 150.3]  # Mock EMA data
            results["EMA_20"] = [149.8, 149.9, 150.0]
            results["EMA_72"] = [148.5, 148.6, 148.7]
            results["EMA_89"] = [148.0, 148.1, 148.2]
        
        if "RSI" in indicators:
            results["RSI"] = [65.4, 66.1, 64.8]
        
        if "ATR" in indicators:
            results["ATR"] = [2.15, 2.18, 2.12]
        
        logger.info(f"Calculated indicators: {indicators}")
        return results

    async def create_ema_cloud(self, data: Dict[str, Any], 
                              periods: List[int] = [9, 20, 72, 89]) -> Dict[str, Any]:
        """Create EMA cloud system."""
        cloud_data = {}
        
        for period in periods:
            # Mock EMA calculation
            cloud_data[f"ema_{period}"] = [150.0 - (period * 0.1)] * len(data.get("close", [150]))
        
        # Calculate cloud zones
        cloud_data["fast_cloud_bullish"] = all(
            cloud_data["ema_9"][i] > cloud_data["ema_20"][i] 
            for i in range(len(cloud_data["ema_9"]))
        )
        
        cloud_data["slow_cloud_bullish"] = all(
            cloud_data["ema_72"][i] > cloud_data["ema_89"][i]
            for i in range(len(cloud_data["ema_72"]))
        )
        
        return cloud_data


class BacktestingMCPClient(MCPClient):
    """MCP client for backtesting.py engine."""
    
    def __init__(self):
        config = MCPServerConfig(
            name="backtesting",
            url="http://localhost:8002/backtest"
        )
        super().__init__(config)

    async def run_backtest(self, strategy_code: str, symbol: str, 
                          start_date: str, end_date: str) -> Dict[str, Any]:
        """Run backtest using backtesting.py."""
        # Mock backtest results
        results = {
            "strategy_name": "OS D1 Strategy",
            "symbol": symbol,
            "period": f"{start_date} to {end_date}",
            "total_trades": 45,
            "winning_trades": 32,
            "losing_trades": 13,
            "win_rate": 0.711,
            "total_return": 0.234,
            "sharpe_ratio": 1.45,
            "max_drawdown": -0.085,
            "profit_factor": 2.12,
            "avg_winner": 0.025,
            "avg_loser": -0.012,
            "largest_winner": 0.087,
            "largest_loser": -0.032,
            "trades": [
                {
                    "entry_date": "2023-01-15",
                    "exit_date": "2023-01-16", 
                    "entry_price": 150.25,
                    "exit_price": 153.75,
                    "return": 0.023,
                    "winner": True
                }
                # More trades would be here
            ]
        }
        
        logger.info(f"Backtest completed for {symbol}: {results['win_rate']:.1%} win rate")
        return results

    async def optimize_strategy(self, strategy_code: str, 
                              parameters: Dict[str, List]) -> Dict[str, Any]:
        """Optimize strategy parameters."""
        # Mock optimization results
        return {
            "best_parameters": {
                "min_gap": 0.18,
                "max_market_cap": 1.8e9,
                "volume_ratio": 2.5
            },
            "best_performance": {
                "win_rate": 0.74,
                "sharpe_ratio": 1.62,
                "total_return": 0.28
            },
            "optimization_runs": 250
        }


class NotionMCPClient(MCPClient):
    """MCP client for Notion trade journaling."""
    
    def __init__(self, api_key: str):
        config = MCPServerConfig(
            name="notion",
            url="https://api.notion.com/v1",
            api_key=api_key
        )
        super().__init__(config)

    async def create_trade_entry(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a trade journal entry in Notion."""
        # Mock Notion page creation
        page_id = f"page_{datetime.now().timestamp()}"
        
        entry = {
            "page_id": page_id,
            "symbol": trade_data.get("symbol"),
            "strategy": trade_data.get("strategy"),
            "entry_date": trade_data.get("entry_date"),
            "entry_price": trade_data.get("entry_price"),
            "exit_date": trade_data.get("exit_date"),
            "exit_price": trade_data.get("exit_price"),
            "return": trade_data.get("return"),
            "notes": trade_data.get("notes", ""),
            "created": True
        }
        
        logger.info(f"Created trade journal entry: {page_id}")
        return entry

    async def update_strategy_performance(self, strategy_name: str, 
                                        performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update strategy performance tracking."""
        return {
            "strategy": strategy_name,
            "performance_updated": True,
            "metrics": performance_data
        }


class OsEngineMCPClient(MCPClient):
    """MCP client for OsEngine trading platform."""
    
    def __init__(self):
        config = MCPServerConfig(
            name="osengine",
            url="http://localhost:8080/api"
        )
        super().__init__(config)

    async def place_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Place order through OsEngine."""
        # Mock order placement (PAPER TRADING ONLY)
        order_id = f"order_{datetime.now().timestamp()}"
        
        order_result = {
            "order_id": order_id,
            "symbol": order_data.get("symbol"),
            "side": order_data.get("side", "BUY"),
            "quantity": order_data.get("quantity"),
            "price": order_data.get("price"),
            "order_type": order_data.get("order_type", "LIMIT"),
            "status": "FILLED",  # Mock filled status
            "fill_price": order_data.get("price"),
            "commission": 1.00,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.warning("⚠️  MOCK ORDER PLACEMENT - NOT REAL TRADING")
        return order_result

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        # Mock positions
        return [
            {
                "symbol": "AAPL",
                "quantity": 100,
                "avg_price": 150.25,
                "current_price": 152.00,
                "unrealized_pnl": 175.00
            }
        ]


class MCPIntegrationManager:
    """Manages all MCP server connections and routing."""
    
    def __init__(self):
        self.clients: Dict[str, MCPClient] = {}
        self.initialized = False

    async def initialize(self, config: Dict[str, Any]):
        """Initialize all MCP clients with configuration."""
        # Initialize Polygon client
        if config.get("POLYGON_API_KEY"):
            self.clients["polygon"] = PolygonMCPClient(config["POLYGON_API_KEY"])
        
        # Initialize TA-Lib client
        self.clients["talib"] = TALibMCPClient()
        
        # Initialize Backtesting client
        self.clients["backtesting"] = BacktestingMCPClient()
        
        # Initialize Notion client
        if config.get("NOTION_API_TOKEN"):
            self.clients["notion"] = NotionMCPClient(config["NOTION_API_TOKEN"])
        
        # Initialize OsEngine client (paper trading only)
        self.clients["osengine"] = OsEngineMCPClient()
        
        # Connect all clients
        for client in self.clients.values():
            await client.connect()
        
        self.initialized = True
        logger.info("MCP Integration Manager initialized")

    async def shutdown(self):
        """Shutdown all MCP connections."""
        for client in self.clients.values():
            await client.disconnect()
        logger.info("MCP Integration Manager shut down")

    def get_client(self, server_name: str) -> Optional[MCPClient]:
        """Get MCP client by server name."""
        return self.clients.get(server_name)

    # High-level trading operations
    async def analyze_symbol_lingua(self, symbol: str, timeframe: str = "1Day") -> Dict[str, Any]:
        """Perform complete Lingua analysis of a symbol."""
        results = {"symbol": symbol, "timeframe": timeframe}
        
        # Get market data
        polygon_client = self.get_client("polygon")
        if polygon_client:
            market_data = await polygon_client.get_market_data(symbol, timeframe)
            results["market_data"] = market_data
        
        # Calculate technical indicators
        talib_client = self.get_client("talib")
        if talib_client:
            indicators = await talib_client.calculate_indicators(
                market_data, ["EMA", "RSI", "ATR"]
            )
            results["indicators"] = indicators
            
            # Create EMA cloud
            ema_cloud = await talib_client.create_ema_cloud(market_data)
            results["ema_cloud"] = ema_cloud
        
        # Lingua framework analysis
        results["lingua_analysis"] = await self._apply_lingua_framework(results)
        
        return results

    async def run_os_d1_scan(self, scan_date: str = None) -> List[Dict[str, Any]]:
        """Run OS D1 scanner across market."""
        polygon_client = self.get_client("polygon")
        if not polygon_client:
            return []
        
        # Define OS D1 criteria
        criteria = {
            "min_gap": 0.15,
            "max_market_cap": 2e9,
            "min_volume_ratio": 2.0,
            "max_float": 50e6
        }
        
        # Get market scan results
        candidates = await polygon_client.scan_market(criteria)
        
        # Validate each candidate with Lingua framework
        validated_candidates = []
        for candidate in candidates:
            symbol = candidate["symbol"]
            
            # Perform Lingua analysis
            analysis = await self.analyze_symbol_lingua(symbol)
            
            # Apply OS D1 validation
            validation = await self._validate_os_d1_setup(candidate, analysis)
            
            if validation["valid"]:
                candidate.update(validation)
                validated_candidates.append(candidate)
        
        logger.info(f"OS D1 scan found {len(validated_candidates)} valid setups")
        return validated_candidates

    async def backtest_strategy(self, strategy_name: str, strategy_code: str, 
                              symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Run comprehensive strategy backtest."""
        backtesting_client = self.get_client("backtesting")
        if not backtesting_client:
            return {"error": "Backtesting client not available"}
        
        # Run backtest
        results = await backtesting_client.run_backtest(
            strategy_code, symbol, start_date, end_date
        )
        
        # Log results to Notion
        notion_client = self.get_client("notion")
        if notion_client and results.get("trades"):
            for trade in results["trades"]:
                await notion_client.create_trade_entry({
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "entry_date": trade["entry_date"],
                    "entry_price": trade["entry_price"],
                    "exit_date": trade["exit_date"],
                    "exit_price": trade["exit_price"],
                    "return": trade["return"],
                    "notes": f"Backtest trade - {strategy_name}"
                })
        
        return results

    async def _apply_lingua_framework(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Lingua trading framework analysis."""
        # Mock Lingua analysis - implement actual framework logic
        return {
            "trend_cycle_stage": 2,  # Breakout stage
            "htf_context": "bullish_structure",
            "mtf_timing": "valid_route",
            "ltf_execution": "setup_ready",
            "confidence_score": 0.78,
            "risk_reward_ratio": 2.1,
            "entry_criteria_met": True,
            "framework_validation": "VALID"
        }

    async def _validate_os_d1_setup(self, candidate: Dict[str, Any], 
                                   analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate candidate using OS D1 setup criteria."""
        lingua = analysis.get("lingua_analysis", {})
        
        # OS D1 validation logic
        valid = (
            candidate.get("gap_percent", 0) >= 0.15 and
            candidate.get("market_cap", float('inf')) <= 2e9 and
            candidate.get("volume_ratio", 0) >= 2.0 and
            lingua.get("trend_cycle_stage", 0) in [1, 2] and
            lingua.get("confidence_score", 0) >= 0.7
        )
        
        return {
            "valid": valid,
            "validation_score": lingua.get("confidence_score", 0),
            "lingua_stage": lingua.get("trend_cycle_stage", 0),
            "risk_reward": lingua.get("risk_reward_ratio", 0),
            "entry_ready": lingua.get("entry_criteria_met", False)
        }


# Integration with SM Playbook Agent Factory
class AgentMCPBridge:
    """Bridge between agents and MCP services."""
    
    def __init__(self, mcp_manager: MCPIntegrationManager):
        self.mcp_manager = mcp_manager

    async def handle_agent_request(self, agent_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests from agents."""
        request_type = request.get("type")
        
        if request_type == "market_data":
            return await self._handle_market_data_request(request)
        elif request_type == "technical_analysis":
            return await self._handle_technical_analysis_request(request)
        elif request_type == "backtest":
            return await self._handle_backtest_request(request)
        elif request_type == "scan_market":
            return await self._handle_scan_request(request)
        else:
            return {"error": f"Unknown request type: {request_type}"}

    async def _handle_market_data_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle market data requests."""
        symbol = request.get("symbol")
        timeframe = request.get("timeframe", "1Day")
        
        return await self.mcp_manager.analyze_symbol_lingua(symbol, timeframe)

    async def _handle_technical_analysis_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle technical analysis requests."""
        talib_client = self.mcp_manager.get_client("talib")
        if talib_client:
            return await talib_client.calculate_indicators(
                request.get("data", {}),
                request.get("indicators", ["EMA", "RSI"])
            )
        return {"error": "TA-Lib client not available"}

    async def _handle_backtest_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle backtest requests."""
        return await self.mcp_manager.backtest_strategy(
            request.get("strategy_name"),
            request.get("strategy_code"),
            request.get("symbol"),
            request.get("start_date"),
            request.get("end_date")
        )

    async def _handle_scan_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle market scan requests."""
        scan_type = request.get("scan_type", "os_d1")
        
        if scan_type == "os_d1":
            candidates = await self.mcp_manager.run_os_d1_scan(request.get("date"))
            return {"scan_type": scan_type, "candidates": candidates}
        
        return {"error": f"Unknown scan type: {scan_type}"}


# Example usage and testing
async def demo_mcp_integration():
    """Demonstrate MCP integration system."""
    print("🔗 SM Playbook MCP Integration Demo")
    print("=" * 50)
    
    # Configuration (use environment variables in real implementation)
    config = {
        "POLYGON_API_KEY": "your_polygon_key",
        "NOTION_API_TOKEN": "your_notion_token"
    }
    
    # Initialize MCP manager
    print("\n📡 Initializing MCP connections...")
    manager = MCPIntegrationManager()
    await manager.initialize(config)
    
    # Test market analysis
    print("\n📊 Running Lingua market analysis...")
    analysis = await manager.analyze_symbol_lingua("AAPL", "1Day")
    print(f"  Analysis completed for AAPL")
    print(f"  Trend cycle stage: {analysis['lingua_analysis']['trend_cycle_stage']}")
    print(f"  Confidence score: {analysis['lingua_analysis']['confidence_score']}")
    
    # Test OS D1 scanner
    print("\n🔍 Running OS D1 market scan...")
    candidates = await manager.run_os_d1_scan()
    print(f"  Found {len(candidates)} OS D1 candidates")
    
    for candidate in candidates[:3]:  # Show first 3
        print(f"    {candidate['symbol']}: {candidate['gap_percent']:.1%} gap, score {candidate['validation_score']:.2f}")
    
    # Test backtesting
    print("\n📈 Running strategy backtest...")
    backtest_results = await manager.backtest_strategy(
        "OS D1 Strategy",
        "# Mock strategy code",
        "AAPL",
        "2023-01-01", 
        "2023-12-31"
    )
    print(f"  Backtest completed: {backtest_results['win_rate']:.1%} win rate")
    
    # Shutdown
    print("\n🛑 Shutting down MCP connections...")
    await manager.shutdown()
    
    print("\n✅ MCP Integration demo completed!")


if __name__ == "__main__":
    print("SM Playbook MCP Integration Layer")
    print("Run demo with: python -c 'import asyncio; from mcp_integration_layer import demo_mcp_integration; asyncio.run(demo_mcp_integration())'")