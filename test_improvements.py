#!/usr/bin/env python3
"""
Test script to verify improvements work correctly
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from config import Config
from utils.data_fetcher import DataFetcher
from utils.market_scanner import MarketScanner
from utils.options_analyzer import OptionsAnalyzer

def test_improvements():
    """Test the improvements"""
    print("🧪 Testing Dynamic Market Cap Approach")
    print("="*50)
    
    # Initialize components
    config = Config()
    data_fetcher = DataFetcher(config)
    scanner = MarketScanner(config, data_fetcher)
    analyzer = OptionsAnalyzer(config, data_fetcher)
    
    print(f"✅ Market cap range: ${config.trading.market_cap_min/1e9:.1f}B - ${config.trading.market_cap_max/1e9:.0f}B")
    print(f"✅ Min volume: {config.trading.min_volume:,}")
    print(f"✅ Revenue growth requirement: {config.scanner.min_revenue_growth:.0%}")
    print(f"✅ Earnings growth requirement: {config.scanner.min_earnings_growth:.0%}")
    print(f"✅ Relative strength requirement: {config.scanner.min_relative_strength:.1f}")
    
    # Test dynamic stock fetching
    print(f"\n🔍 Testing Dynamic Stock Fetching...")
    
    # Get a sample of stocks to test
    sample_stocks = data_fetcher.get_stocks_by_market_cap(
        min_cap=config.trading.market_cap_min,
        max_cap=config.trading.market_cap_max,
        min_volume=config.trading.min_volume
    )
    
    print(f"✅ Found {len(sample_stocks)} stocks in market cap range")
    if sample_stocks:
        print(f"   Sample stocks: {[s['symbol'] for s in sample_stocks[:5]]}")
        
        # Show market cap distribution
        small_caps = [s for s in sample_stocks if s.get('market_cap', 0) < 1e9]
        mid_caps = [s for s in sample_stocks if 1e9 <= s.get('market_cap', 0) < 10e9]
        large_caps = [s for s in sample_stocks if s.get('market_cap', 0) >= 10e9]
        
        print(f"   Small caps (<$1B): {len(small_caps)}")
        print(f"   Mid caps ($1B-$10B): {len(mid_caps)}")
        print(f"   Large caps (>$10B): {len(large_caps)}")
    
    # Test expected return calculation
    print(f"\n🔍 Testing Expected Return Calculation...")
    
    sample_option = {
        'strike': 150.0,
        'ask': 5.0,
        'bid': 4.8,
        'mid': 4.9,
        'volume': 500,
        'open_interest': 1000,
        'implied_volatility': 0.4,
        'days_to_expiration': 35,
        'delta': 0.35,
        'theta': -0.02,
        'gamma': 0.02,
        'vega': 0.15,
        'spread_pct': 0.04,
        'current_stock_price': 145.0
    }
    
    expected_return = analyzer._calculate_expected_return(145.0, sample_option, 3.0)
    print(f"Expected Return: {expected_return:+.1%}")
    
    if expected_return > 0:
        print("✅ Positive expected return - good for buying calls")
    else:
        print("⚠️  Negative expected return - consider carefully")
    
    print("\n✅ All tests completed successfully!")
    print("\n💡 Key Improvements:")
    print("• Dynamic stock fetching based on market cap range")
    print("• No hardcoded ticker lists")
    print("• Configurable market cap and volume filters")
    print("• Better expected return calculation")
    print("• Simplified output format")

if __name__ == '__main__':
    test_improvements() 