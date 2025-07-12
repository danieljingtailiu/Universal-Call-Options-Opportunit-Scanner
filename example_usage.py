#!/usr/bin/env python3
"""
Example showing how the dynamic market cap approach works
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from config import Config

def show_example_usage():
    """Show example usage with different market cap ranges"""
    print("🚀 DYNAMIC MARKET CAP APPROACH")
    print("="*50)
    
    # Example 1: Small caps ($1B - $10B)
    print("\n📊 Example 1: Small Caps ($1B - $10B)")
    config_small = Config()
    config_small.trading.market_cap_min = 1_000_000_000  # $1B
    config_small.trading.market_cap_max = 10_000_000_000  # $10B
    print(f"   Market Cap Range: ${config_small.trading.market_cap_min/1e9:.0f}B - ${config_small.trading.market_cap_max/1e9:.0f}B")
    print("   Expected stocks: SOFI, HOOD, AFRM, UPST, etc.")
    
    # Example 2: Mid caps ($10B - $100B)
    print("\n📊 Example 2: Mid Caps ($10B - $100B)")
    config_mid = Config()
    config_mid.trading.market_cap_min = 10_000_000_000  # $10B
    config_mid.trading.market_cap_max = 100_000_000_000  # $100B
    print(f"   Market Cap Range: ${config_mid.trading.market_cap_min/1e9:.0f}B - ${config_mid.trading.market_cap_max/1e9:.0f}B")
    print("   Expected stocks: NVDA, TSLA, META, AMZN, etc.")
    
    # Example 3: Large caps ($100B+)
    print("\n📊 Example 3: Large Caps ($100B+)")
    config_large = Config()
    config_large.trading.market_cap_min = 100_000_000_000  # $100B
    config_large.trading.market_cap_max = 1_000_000_000_000  # $1T
    print(f"   Market Cap Range: ${config_large.trading.market_cap_min/1e9:.0f}B - ${config_large.trading.market_cap_max/1e9:.0f}B")
    print("   Expected stocks: AAPL, MSFT, GOOGL, AMZN, etc.")
    
    print("\n" + "="*50)
    print("💡 HOW TO USE:")
    print("1. Edit config.json to set your desired market cap range")
    print("2. Run: python main.py --scan")
    print("3. The system will dynamically fetch stocks in your range")
    print("4. No hardcoded lists - completely configurable!")
    print("="*50)
    
    # Show current config
    current_config = Config()
    print(f"\n🔧 CURRENT CONFIG:")
    print(f"   Market Cap: ${current_config.trading.market_cap_min/1e9:.1f}B - ${current_config.trading.market_cap_max/1e9:.0f}B")
    print(f"   Min Volume: {current_config.trading.min_volume:,}")
    print(f"   Revenue Growth: {current_config.scanner.min_revenue_growth:.0%}")
    print(f"   Earnings Growth: {current_config.scanner.min_earnings_growth:.0%}")

if __name__ == '__main__':
    show_example_usage() 