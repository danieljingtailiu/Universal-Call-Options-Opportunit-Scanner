#!/usr/bin/env python3
"""
Improved Small-Cap Options Tracker with Better Recommendations
"""

import json
import logging
from datetime import datetime
import argparse
import os
from pathlib import Path
from typing import List, Dict

from utils.market_scanner import MarketScanner
from utils.options_analyzer import OptionsAnalyzer
from utils.data_fetcher import DataFetcher
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/tracker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add performance monitoring
import time
from contextlib import contextmanager

@contextmanager
def timer(operation_name: str):
    """Context manager for timing operations"""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"{operation_name} completed in {elapsed:.2f} seconds")


class PerformanceMonitor:
    """Simple performance monitoring"""
    
    def __init__(self):
        self.operations = {}
        
    def start_operation(self, name: str):
        """Start timing an operation"""
        self.operations[name] = time.time()
        
    def end_operation(self, name: str) -> float:
        """End timing an operation and return elapsed time"""
        if name in self.operations:
            elapsed = time.time() - self.operations[name]
            logger.info(f"{name} took {elapsed:.2f} seconds")
            del self.operations[name]
            return elapsed
        return 0.0


class ImprovedOptionsTracker:
    """Improved tracker focused on finding and monitoring specific options"""
    
    def __init__(self, config_path: str = 'config.json'):
        """Initialize the tracker with configuration"""
        try:
            self.config = Config(config_path)
            
            # Validate configuration
            issues = self.config.validate()
            if issues:
                logger.warning("Configuration issues found:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
            
            # Use improved modules
            self.data_fetcher = DataFetcher(self.config)
            self.scanner = MarketScanner(self.config, self.data_fetcher)
            self.options_analyzer = OptionsAnalyzer(self.config, self.data_fetcher)
            
        except Exception as e:
            logger.error(f"Failed to initialize tracker: {e}")
            raise
    
    def find_opportunities(self, top_n: int = 10):
        """Find top options opportunities with improved error handling"""
        logger.info("="*80)
        logger.info("SCANNING FOR OPTIONS OPPORTUNITIES")
        logger.info("="*80)
        
        # Show market cap range being used
        min_cap_b = self.config.trading.market_cap_min / 1e9
        max_cap_b = self.config.trading.market_cap_max / 1e9
        logger.info(f"Market Cap Range: ${min_cap_b:.1f}B - ${max_cap_b:.0f}B")
        logger.info(f"Min Volume: {self.config.trading.min_volume:,}")
        
        try:
            # Step 1: Get stocks within market cap range
            logger.info("\n1. Finding stocks within market cap range...")
            candidates = self.scanner.find_small_caps()
            logger.info(f"   Found {len(candidates)} stocks in range")
            
            if not candidates:
                logger.warning("No candidates found. Check market hours and data availability.")
                return []
            
            # Step 2: Apply technical filters
            logger.info("\n2. Applying technical analysis...")
            filtered = self.scanner.apply_filters(candidates)
            logger.info(f"   {len(filtered)} stocks passed technical filters")
            
            # If too few, relax filters further
            if len(filtered) < 5:
                logger.info("   Too few results - relaxing filters...")
                # Get top movers even if they don't pass all filters
                filtered = self._get_top_movers(candidates, 30)
            
            # Step 3: Analyze options for each
            logger.info(f"\n3. Analyzing options for {len(filtered)} stocks...")
            all_recommendations = []
            
            # Analyze more stocks to get better diversification
            stocks_to_analyze = min(len(filtered), 25)  # Analyze up to 25 stocks
            
            for i, stock in enumerate(filtered[:stocks_to_analyze], 1):
                logger.info(f"\n   [{i}/{stocks_to_analyze}] Analyzing {stock['symbol']}...")
                
                try:
                    # Get multiple recommendations per stock
                    recommendations = self.options_analyzer.analyze_stock(stock)
                    
                    if recommendations:
                        all_recommendations.extend(recommendations)
                        logger.info(f"   + Found {len(recommendations)} option opportunities")
                    else:
                        logger.info(f"   - No suitable options found")
                        
                except Exception as e:
                    logger.error(f"   ✗ Error analyzing {stock['symbol']}: {e}")
                    # Continue with next stock instead of failing completely
                    continue
            
            # Step 4: Sort and diversify recommendations
            if all_recommendations:
                # Sort by score first
                all_recommendations.sort(key=lambda x: x['score'], reverse=True)
                
                # Take only the best option from each stock to ensure diversification
                diversified_recommendations = []
                seen_stocks = set()
                
                for rec in all_recommendations:
                    if rec['symbol'] not in seen_stocks:
                        diversified_recommendations.append(rec)
                        seen_stocks.add(rec['symbol'])
                        
                        # Stop when we have enough different stocks
                        if len(diversified_recommendations) >= 5:
                            break
                
                # Step 5: Display top recommendations
                self._display_top_recommendations(diversified_recommendations)
                self._prompt_for_monitoring(diversified_recommendations)
            else:
                logger.warning("\nNo option opportunities found. Try:")
                logger.warning("1. Clear cache: python main.py --clear-cache")
                logger.warning("2. Adjust market cap range in config.json")
                logger.warning("3. Check market hours")
                logger.warning("4. Wait for rate limits to reset")
            
            return all_recommendations
            
        except Exception as e:
            logger.error(f"Fatal error in find_opportunities: {e}")
            return []
    
    def _get_top_movers(self, stocks: List[Dict], n: int = 20) -> List[Dict]:
        """Get stocks with best momentum regardless of all filters"""
        # Calculate simple momentum score
        for stock in stocks:
            score = 0
            
            # Volume score
            if stock.get('volume', 0) > stock.get('avg_volume', 0):
                score += 20
            
            # Price action (simplified)
            if stock.get('price', 0) > 5:  # Not penny stock
                score += 10
            
            # Has options
            if stock.get('has_options', False):
                score += 30
            
            # Market cap in sweet spot
            if 1e9 <= stock.get('market_cap', 0) <= 5e9:
                score += 20
            
            stock['momentum_score'] = score
        
        # Sort by score and return top N
        stocks.sort(key=lambda x: x.get('momentum_score', 0), reverse=True)
        return stocks[:n]
    
    def _display_top_recommendations(self, recommendations: List[Dict]):
        """Display top recommendations in a clean, simple format"""
        print("\n" + "="*60)
        print("🚀 TOP 5 CALL OPTIONS RECOMMENDATIONS")
        print("="*60)
        
        if not recommendations:
            print("❌ No suitable options found. Try:")
            print("   • Clear cache: python main.py --clear-cache")
            print("   • Check market hours")
            print("   • Wait for rate limits to reset")
            return
        
        for i, rec in enumerate(recommendations[:5], 1):
            # Calculate key metrics
            moneyness = rec['current_stock_price'] / rec['strike']
            breakeven_move = ((rec['strike'] + rec['entry_price']) / rec['current_stock_price'] - 1) * 100
            
            print(f"\n{i}. {rec['symbol']} @ ${rec['current_stock_price']:.2f}")
            print(f"   ${rec['strike']}C {rec['expiration']} ({rec['days_to_expiration']}d)")
            print(f"   Entry: ${rec['entry_price']:.2f} | BE: +{breakeven_move:.1f}% | Score: {rec['score']:.0f}")
            
            # Show top reason only
            if rec['recommendation_reasons']:
                print(f"   ✓ {rec['recommendation_reasons'][0]}")
            
            # Risk indicator
            if rec['expected_return'] > 0.2:
                print(f"   🚀 High potential")
            elif rec['expected_return'] > 0:
                print(f"   📈 Good potential")
            else:
                print(f"   ⚠️  Caution")
        
        print("\n" + "="*60)
        print("💡 Quick Tips:")
        print("• Only buy when bullish on the stock")
        print("• Use 1-5% position sizing")
        print("• Set stop losses and profit targets")
        print("• Monitor theta decay near expiration")
        print("="*60)
    
    def _prompt_for_monitoring(self, recommendations: List[Dict]):
        """Prompt user to add positions to monitoring"""
        print("\n📊 MONITOR POSITIONS?")
        print("Enter numbers (1-5) separated by commas, or 'n' for none:")
        
        try:
            user_input = input("> ").strip()
            
            if user_input.lower() == 'n':
                return
            
            # Parse selections
            selections = [int(x.strip()) - 1 for x in user_input.split(',')]
            
            for idx in selections:
                if 0 <= idx < len(recommendations):
                    rec = recommendations[idx]
                    
                    # Ask for number of contracts
                    print(f"\nHow many contracts of {rec['symbol']} ${rec['strike']}C?")
                    contracts = int(input("> ") or "1")
                    
                    # Add to monitoring
                    self.options_analyzer.add_to_monitoring(rec, contracts)
                    print(f"✅ Added {contracts} contracts to monitoring")
                    
        except Exception as e:
            logger.error(f"Error processing selection: {e}")
    
    def monitor_positions(self):
        """Monitor existing positions"""
        logger.info("\n" + "="*80)
        logger.info("MONITORING POSITIONS")
        logger.info("="*80)
        
        # Get exit signals
        exit_signals = self.options_analyzer.monitor_positions()
        
        # Display any urgent actions
        if exit_signals:
            print("\n" + "!"*80)
            print("ACTION REQUIRED")
            print("!"*80)
            
            for signal in exit_signals:
                print(f"\n{signal['symbol']} ${signal['strike']} {signal['expiration']}")
                print(f"ACTION: {signal['action']} - {signal.get('urgency', 'RECOMMENDED')}")
                print(f"Recommendation: {signal['recommendation']}")
    
    def run_analysis(self, scan_new: bool = True, monitor: bool = True):
        """Run complete analysis cycle"""
        
        # Monitor existing positions first
        if monitor:
            self.monitor_positions()
        
        # Scan for new opportunities
        if scan_new:
            self.find_opportunities()
    
    def clear_cache(self):
        """Clear all cached data"""
        logger.info("Clearing cache...")
        self.data_fetcher.clear_cache()
        logger.info("Cache cleared successfully")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Improved Small-Cap Options Tracker')
    parser.add_argument('--scan', action='store_true', help='Scan for new opportunities')
    parser.add_argument('--monitor', action='store_true', help='Monitor existing positions')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cached data')
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Create necessary directories
    for dir in ['logs', 'reports', 'data', 'data/cache']:
        Path(dir).mkdir(exist_ok=True)
    
    # Initialize tracker
    tracker = ImprovedOptionsTracker(args.config)
    
    # Handle commands
    if args.clear_cache:
        tracker.clear_cache()
    elif args.monitor and not args.scan:
        tracker.run_analysis(scan_new=False, monitor=True)
    elif args.scan and not args.monitor:
        tracker.run_analysis(scan_new=True, monitor=False)
    else:
        # Default: do both
        tracker.run_analysis(scan_new=True, monitor=True)


if __name__ == '__main__':
    main()