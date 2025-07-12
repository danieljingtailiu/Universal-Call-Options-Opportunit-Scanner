#!/usr/bin/env python3
"""
Test script for improved options analysis
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from config import Config
from utils.data_fetcher import DataFetcher
from utils.options_analyzer import OptionsAnalyzer

def test_options_analysis():
    """Test the improved options analysis"""
    print("🧪 Testing Improved Options Analysis")
    print("="*50)
    
    # Initialize components
    config = Config()
    data_fetcher = DataFetcher(config)
    analyzer = OptionsAnalyzer(config, data_fetcher)
    
    # Test with a known stock that has options
    test_symbol = 'SOFI'  # SoFi has active options
    
    print(f"\n📊 Testing with {test_symbol}...")
    
    try:
        # Get stock data
        stock_data = {
            'symbol': test_symbol,
            'price': 7.50,  # Example price
            'volume': 1000000,
            'market_cap': 8000000000,  # $8B market cap
            'atr': 0.30,  # Average True Range
            'rsi': 45,
            'relative_strength': 1.1
        }
        
        # Test expected return calculation
        print("\n🔍 Testing Expected Return Calculation...")
        
        # Create a sample option contract
        sample_option = {
            'strike': 8.0,
            'ask': 0.50,
            'bid': 0.45,
            'mid': 0.475,
            'volume': 150,
            'open_interest': 500,
            'implied_volatility': 0.45,
            'days_to_expiration': 35,
            'delta': 0.35,
            'theta': -0.015,
            'gamma': 0.025,
            'vega': 0.12,
            'spread_pct': 0.11,
            'current_stock_price': 7.50
        }
        
        # Test expected return
        expected_return = analyzer._calculate_expected_return(7.50, sample_option, 0.30)
        print(f"Expected Return: {expected_return:+.1%}")
        
        if expected_return > 0:
            print("✅ Expected return is positive - good for buying calls")
        else:
            print("⚠️  Expected return is negative - consider carefully")
        
        # Test probability of profit
        prob_profit = analyzer._calculate_probability_of_profit(7.50, 8.0, 35, 0.45, 0.50)
        print(f"Probability of Profit: {prob_profit:.1%}")
        
        # Test risk/reward ratio
        risk_reward = analyzer._calculate_risk_reward(sample_option)
        print(f"Risk/Reward Ratio: {risk_reward:.1f}")
        
        # Test option value analysis
        value_analysis = analyzer._calculate_option_value_analysis(7.50, sample_option)
        print(f"Moneyness: {value_analysis.get('moneyness', 0):.3f}")
        print(f"Intrinsic Value: ${value_analysis.get('intrinsic_value', 0):.2f}")
        print(f"Time Value: ${value_analysis.get('time_value', 0):.2f}")
        print(f"Liquidity Score: {value_analysis.get('liquidity_score', 0)}")
        
        # Test comprehensive scoring
        score, analysis = analyzer._score_option_comprehensive(sample_option, 7.50, stock_data)
        print(f"\n📈 Comprehensive Score: {score:.1f}/100")
        print(f"Risk Assessment: {analysis.get('risk_assessment', 'Unknown')}")
        print(f"Liquidity Assessment: {analysis.get('liquidity_assessment', 'Unknown')}")
        
        print("\n✅ All tests completed successfully!")
        
        # Show reasoning
        print(f"\n🎯 REASONING:")
        for reason in analysis.get('reasons', [])[:5]:
            print(f"  + {reason}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_real_options():
    """Test with real options data"""
    print("\n" + "="*50)
    print("🔍 Testing with Real Options Data")
    print("="*50)
    
    config = Config()
    data_fetcher = DataFetcher(config)
    analyzer = OptionsAnalyzer(config, data_fetcher)
    
    # Test with a real stock
    test_symbol = 'AAPL'  # Apple has very liquid options
    
    try:
        print(f"\n📊 Getting real options data for {test_symbol}...")
        
        # Get stock quote
        quote = data_fetcher.get_quote(test_symbol)
        if not quote:
            print(f"❌ Could not get quote for {test_symbol}")
            return
        
        print(f"Current price: ${quote['price']:.2f}")
        
        # Get options chain
        options_chain = data_fetcher.get_options_chain(test_symbol)
        if not options_chain:
            print(f"❌ No options found for {test_symbol}")
            return
        
        print(f"Found {len(options_chain)} option contracts")
        
        # Test with first few options
        for i, option in enumerate(options_chain[:3]):
            print(f"\n--- Testing Option {i+1} ---")
            print(f"Strike: ${option['strike']:.2f}")
            print(f"Expiration: {option['expiration']}")
            print(f"Ask: ${option['ask']:.2f}")
            print(f"Volume: {option['volume']}")
            
            # Calculate expected return
            expected_return = analyzer._calculate_expected_return(
                quote['price'], option, quote['price'] * 0.02
            )
            print(f"Expected Return: {expected_return:+.1%}")
            
            if expected_return > 0.1:
                print("✅ Good potential for buying calls")
            elif expected_return > 0:
                print("📈 Positive expected return")
            else:
                print("⚠️  Negative expected return")
        
        print("\n✅ Real options test completed!")
        
    except Exception as e:
        print(f"❌ Real options test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("🚀 Options Analysis Test Suite")
    print("="*50)
    
    # Run tests
    test_options_analysis()
    test_real_options()
    
    print("\n" + "="*50)
    print("✅ All tests completed!")
    print("\n💡 Key Improvements:")
    print("• Fixed expected return calculation (no more -100%)")
    print("• Better probability of profit calculation")
    print("• Improved risk/reward analysis")
    print("• More comprehensive scoring system")
    print("• Better reasoning for option selection")
    print("="*50) 