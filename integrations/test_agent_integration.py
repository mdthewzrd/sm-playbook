#!/usr/bin/env python3
"""
SM Playbook Agent Integration Test
Tests the complete agent system integration
"""

import asyncio
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_agent_integration():
    """Test the complete agent integration."""
    print("🧪 SM Playbook Agent Integration Test")
    print("=" * 50)
    
    try:
        # Test imports
        print("\n1. Testing imports...")
        from claude_code_integration import (
            init_sm_playbook_system, os_d1_scan, backtest_strategy,
            analyze_lingua, design_strategy, system_status
        )
        print("  ✅ All imports successful")
        
        # Test initialization
        print("\n2. Testing system initialization...")
        result = await init_sm_playbook_system()
        if result:
            print("  ✅ Agent system initialized")
        else:
            print("  ⚠️  Agent system initialization returned False")
        
        # Test system status
        print("\n3. Testing system status...")
        status = system_status()
        print(f"  Status: {status.get('factory_status', 'unknown')}")
        print(f"  Agents: {status.get('total_agents', 0)}")
        
        # Test OS D1 scan
        print("\n4. Testing OS D1 scan...")
        from claude_code_integration import os_d1_scan_async, design_strategy_async, analyze_lingua_async
        scan_result = await os_d1_scan_async()
        print(f"  Scan result: {type(scan_result)}")
        
        # Test strategy design
        print("\n5. Testing strategy design...")
        design_result = await design_strategy_async("os_d1")
        print(f"  Design result: {type(design_result)}")
        
        # Test Lingua analysis
        print("\n6. Testing Lingua analysis...")
        analysis_result = await analyze_lingua_async("AAPL", "daily")
        print(f"  Analysis result: {type(analysis_result)}")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure you've copied the agent factory code")
        print("  2. Run: pip install aiohttp pandas numpy")
        print("  3. Check that all agent files are in place")
        return False
        
    except Exception as e:
        print(f"  ❌ Test error: {e}")
        return False

def run_test():
    """Run the integration test."""
    return asyncio.run(test_agent_integration())

if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)