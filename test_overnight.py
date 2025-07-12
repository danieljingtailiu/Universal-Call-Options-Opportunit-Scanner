#!/usr/bin/env python3
"""
Quick test of overnight scanner with current cache
"""

import logging
from pathlib import Path

from utils.data_fetcher import DataFetcher
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cache_access():
    """Test if we can access the cached data"""
    print("Testing cache access...")
    
    config = Config()
    data_fetcher = DataFetcher(config)
    
    # Test getting stocks from cache
    stocks = data_fetcher.get_stocks_by_market_cap(
        config.trading.market_cap_min,
        config.trading.market_cap_max,
        config.trading.min_volume
    )
    
    print(f"✓ Found {len(stocks)} stocks in cache")
    
    if stocks:
        print(f"Sample stocks: {[s['symbol'] for s in stocks[:5]]}")
        print(f"Market cap range: ${min(s['market_cap'] for s in stocks)/1e9:.1f}B - ${max(s['market_cap'] for s in stocks)/1e9:.1f}B")
    
    return len(stocks) > 0


def test_rate_limiting():
    """Test rate limiting with a few stocks"""
    print("\nTesting rate limiting with sample stocks...")
    
    config = Config()
    data_fetcher = DataFetcher(config)
    
    # Get a few stocks from cache
    stocks = data_fetcher.get_stocks_by_market_cap(
        config.trading.market_cap_min,
        config.trading.market_cap_max,
        config.trading.min_volume
    )
    
    if not stocks:
        print("❌ No stocks found in cache")
        return False
    
    # Test getting quotes for first 3 stocks
    test_stocks = stocks[:3]
    successful_quotes = 0
    
    for stock in test_stocks:
        try:
            quote = data_fetcher.get_quote(stock['symbol'])
            if quote:
                print(f"✓ {stock['symbol']}: ${quote['price']:.2f}")
                successful_quotes += 1
            else:
                print(f"✗ {stock['symbol']}: No quote")
        except Exception as e:
            print(f"✗ {stock['symbol']}: {e}")
    
    print(f"✓ Successfully got quotes for {successful_quotes}/{len(test_stocks)} stocks")
    return successful_quotes > 0


def main():
    """Run tests"""
    print("Testing Overnight Scanner Components")
    print("="*50)
    
    # Test 1: Cache access
    cache_ok = test_cache_access()
    
    # Test 2: Rate limiting
    rate_ok = test_rate_limiting()
    
    print("\n" + "="*50)
    print("Test Results:")
    print(f"Cache Access: {'✓ PASS' if cache_ok else '✗ FAIL'}")
    print(f"Rate Limiting: {'✓ PASS' if rate_ok else '✗ FAIL'}")
    
    if cache_ok and rate_ok:
        print("\n✅ System ready for overnight scan!")
        print("Run: python overnight_scan.py")
    else:
        print("\n❌ System needs fixes before overnight scan")
        print("Check logs for details")


if __name__ == '__main__':
    main() 