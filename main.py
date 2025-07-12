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


class ImprovedOptionsTracker:
    """Improved tracker focused on finding and monitoring specific options"""
    
    def __init__(self, config_path: str = 'config.json'):
        """Initialize the tracker with configuration"""
        self.config = Config(config_path)
        
        # Use improved modules
        self.data_fetcher = DataFetcher(self.config)
        self.scanner = MarketScanner(self.config, self.data_fetcher)
        self.options_analyzer = OptionsAnalyzer(self.config, self.data_fetcher)
        
    def find_opportunities(self, top_n: int = 10):
        """Find top options opportunities"""
        logger.info("="*80)
        logger.info("SCANNING FOR OPTIONS OPPORTUNITIES")
        logger.info("="*80)
        
        # Step 1: Get small-cap stocks
        logger.info("\n1. Finding small-cap stocks...")
        candidates = self.scanner.find_small_caps()
        logger.info(f"   Found {len(candidates)} small-cap candidates")
        
        # Step 2: Apply technical filters (RELAXED)
        logger.info("\n2. Applying technical analysis...")
        filtered = self.scanner.apply_filters(candidates)
        logger.info(f"   {len(filtered)} stocks passed technical filters")
        
        # If too few, relax filters further
        if len(filtered) < 5:
            logger.info("   Too few results - relaxing filters...")
            # Get top movers even if they don't pass all filters
            filtered = self._get_top_movers(candidates, 20)
        
        # Step 3: Analyze options for each
        logger.info(f"\n3. Analyzing options for {len(filtered)} stocks...")
        all_recommendations = []
        
        for i, stock in enumerate(filtered[:top_n], 1):
            logger.info(f"\n   [{i}/{min(len(filtered), top_n)}] Analyzing {stock['symbol']}...")
            
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
        
        # Step 4: Sort all recommendations by score
        all_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Step 5: Display top recommendations
        if all_recommendations:
            self._display_top_recommendations(all_recommendations[:5])
            self._prompt_for_monitoring(all_recommendations[:5])
        else:
            logger.warning("\nNo option opportunities found. Try:")
            logger.warning("1. Clear cache: python main.py --clear-cache")
            logger.warning("2. Relax filters in config.json")
            logger.warning("3. Check market hours")
        
        return all_recommendations
    
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
        """Display top recommendations in detail"""
        print("\n" + "="*80)
        print("TOP OPTIONS RECOMMENDATIONS")
        print("="*80)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n#{i}. {rec['symbol']} - Score: {rec['score']:.1f}/100")
            print("-" * 40)
            print(f"Stock Price: ${rec['current_stock_price']:.2f}")
            print(f"CALL: ${rec['strike']} strike, Exp: {rec['expiration']} ({rec['days_to_expiration']} days)")
            print(f"Entry Price: ${rec['entry_price']:.2f} (Bid: ${rec['bid_price']:.2f}, Ask: ${rec['ask_price']:.2f})")
            
            # Key metrics
            print(f"\nMetrics:")
            print(f"  • Probability of Profit: {rec['probability_of_profit']:.1%}")
            print(f"  • Expected Return: {rec['expected_return']:+.1%}")
            print(f"  • Delta: {rec['delta']:.3f} | Theta: ${rec['theta']:.3f}")
            print(f"  • Volume: {rec['volume']} | OI: {rec['open_interest']}")
            
            # Trading plan
            print(f"\nTrading Plan:")
            print(f"  • BUY: {rec['symbol']} ${rec['strike']}C {rec['expiration']}")
            print(f"  • Entry: ${rec['entry_price']:.2f}")
            print(f"  • Stop Loss: ${rec['stop_loss']:.2f} (-50%)")
            print(f"  • Target 1: ${rec['target_1']:.2f} (+50%)")
            print(f"  • Target 2: ${rec['target_2']:.2f} (+100%)")
            
            # Reasons
            print(f"\nWhy Buy:")
            for reason in rec['recommendation_reasons'][:3]:
                print(f"  + {reason}")
    
    def _prompt_for_monitoring(self, recommendations: List[Dict]):
        """Prompt user to add positions to monitoring"""
        print("\n" + "="*80)
        print("Would you like to monitor any of these positions?")
        print("Enter the number(s) separated by commas (e.g., 1,3,5) or 'n' for none:")
        
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
                    print(f"+ Added to monitoring: {contracts} contracts")
                    
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