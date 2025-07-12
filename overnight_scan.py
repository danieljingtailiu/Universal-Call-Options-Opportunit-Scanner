#!/usr/bin/env python3
"""
Overnight Comprehensive Options Scanner
Fetches a large universe of stocks and finds the best call options across different stocks
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from utils.market_scanner import MarketScanner
from utils.options_analyzer import OptionsAnalyzer
from utils.data_fetcher import DataFetcher
from config import Config

# Configure logging for overnight run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/overnight_scan.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OvernightScanner:
    """Comprehensive scanner for overnight analysis"""
    
    def __init__(self, config_path: str = 'config.json'):
        """Initialize the scanner"""
        self.config = Config(config_path)
        self.data_fetcher = DataFetcher(self.config)
        self.scanner = MarketScanner(self.config, self.data_fetcher)
        self.options_analyzer = OptionsAnalyzer(self.config, self.data_fetcher)
        
        # Results storage
        self.results = {
            'scan_time': datetime.now().isoformat(),
            'stocks_analyzed': 0,
            'options_found': 0,
            'top_recommendations': []
        }
    
    def run_comprehensive_scan(self):
        """Run comprehensive scan with detailed logging"""
        logger.info("="*80)
        logger.info("STARTING OVERNIGHT COMPREHENSIVE SCAN")
        logger.info("="*80)
        
        start_time = time.time()
        
        try:
            # Step 1: Get comprehensive stock universe
            logger.info("\n1. FETCHING COMPREHENSIVE STOCK UNIVERSE")
            logger.info(f"Market Cap Range: ${self.config.trading.market_cap_min/1e9:.1f}B - ${self.config.trading.market_cap_max/1e9:.0f}B")
            logger.info(f"Min Volume: {self.config.trading.min_volume:,}")
            
            stocks = self.scanner.find_small_caps()
            logger.info(f"✓ Found {len(stocks)} stocks in universe")
            
            if len(stocks) < 10:
                logger.error("❌ Too few stocks found. Check market hours and data availability.")
                return
            
            # Step 2: Apply filters to get quality candidates
            logger.info("\n2. APPLYING FILTERS FOR QUALITY CANDIDATES")
            filtered_stocks = self.scanner.apply_filters(stocks)
            logger.info(f"✓ {len(filtered_stocks)} stocks passed filters")
            
            # If too few, use top movers
            if len(filtered_stocks) < 5:
                logger.info("⚠️  Too few filtered results, using top movers...")
                filtered_stocks = self._get_top_movers(stocks, 50)
                logger.info(f"✓ Using {len(filtered_stocks)} top movers")
            
            # Step 3: Analyze options for each stock
            logger.info(f"\n3. ANALYZING OPTIONS FOR {len(filtered_stocks)} STOCKS")
            all_recommendations = []
            
            for i, stock in enumerate(filtered_stocks, 1):
                logger.info(f"\n[{i}/{len(filtered_stocks)}] Analyzing {stock['symbol']} @ ${stock.get('price', 0):.2f}")
                
                try:
                    recommendations = self.options_analyzer.analyze_stock(stock)
                    
                    if recommendations:
                        all_recommendations.extend(recommendations)
                        logger.info(f"✓ Found {len(recommendations)} option opportunities")
                    else:
                        logger.info("✗ No suitable options found")
                        
                except Exception as e:
                    logger.error(f"✗ Error analyzing {stock['symbol']}: {e}")
                    continue
                
                # Rate limiting between stocks
                if i < len(filtered_stocks):
                    time.sleep(2)
            
            # Step 4: Find best diversified recommendations
            logger.info(f"\n4. FINDING BEST DIVERSIFIED RECOMMENDATIONS")
            logger.info(f"Total options found: {len(all_recommendations)}")
            
            if all_recommendations:
                # Sort by score
                all_recommendations.sort(key=lambda x: x['score'], reverse=True)
                
                # Take best option from each stock for diversification
                diversified_recommendations = []
                seen_stocks = set()
                
                for rec in all_recommendations:
                    if rec['symbol'] not in seen_stocks:
                        diversified_recommendations.append(rec)
                        seen_stocks.add(rec['symbol'])
                        
                        # Stop when we have enough different stocks
                        if len(diversified_recommendations) >= 10:
                            break
                
                # Step 5: Display results
                self._display_comprehensive_results(diversified_recommendations)
                
                # Save results
                self._save_results(diversified_recommendations, len(stocks), len(all_recommendations))
                
            else:
                logger.warning("\n❌ No option opportunities found.")
                logger.warning("Possible reasons:")
                logger.warning("1. Market is closed")
                logger.warning("2. Rate limits exceeded")
                logger.warning("3. No stocks meet criteria")
                logger.warning("4. Options data unavailable")
            
            elapsed_time = time.time() - start_time
            logger.info(f"\n✓ Scan completed in {elapsed_time/60:.1f} minutes")
            
        except Exception as e:
            logger.error(f"❌ Fatal error in comprehensive scan: {e}")
            raise
    
    def _get_top_movers(self, stocks: List[Dict], n: int = 50) -> List[Dict]:
        """Get stocks with best momentum regardless of filters"""
        # Calculate momentum score
        for stock in stocks:
            score = 0
            
            # Volume score
            if stock.get('volume', 0) > stock.get('avg_volume', 0):
                score += 20
            
            # Price action
            if stock.get('price', 0) > 5:  # Not penny stock
                score += 10
            
            # Has options
            if stock.get('has_options', False):
                score += 30
            
            # Market cap in sweet spot
            market_cap = stock.get('market_cap', 0)
            if 1e9 <= market_cap <= 5e9:
                score += 20
            elif 5e9 < market_cap <= 10e9:
                score += 15
            
            stock['momentum_score'] = score
        
        # Sort by score and return top N
        stocks.sort(key=lambda x: x.get('momentum_score', 0), reverse=True)
        return stocks[:n]
    
    def _display_comprehensive_results(self, recommendations: List[Dict]):
        """Display comprehensive results"""
        print("\n" + "="*80)
        print("🚀 TOP 10 CALL OPTIONS RECOMMENDATIONS")
        print("="*80)
        
        if not recommendations:
            print("❌ No suitable options found.")
            return
        
        for i, rec in enumerate(recommendations, 1):
            # Calculate key metrics
            moneyness = rec['current_stock_price'] / rec['strike']
            breakeven_move = ((rec['strike'] + rec['entry_price']) / rec['current_stock_price'] - 1) * 100
            
            print(f"\n{i:2d}. {rec['symbol']} @ ${rec['current_stock_price']:.2f}")
            print(f"    ${rec['strike']}C {rec['expiration']} ({rec['days_to_expiration']}d)")
            print(f"    Entry: ${rec['entry_price']:.2f} | BE: +{breakeven_move:.1f}% | Score: {rec['score']:.0f}")
            
            # Show key reasoning
            if rec['recommendation_reasons']:
                for reason in rec['recommendation_reasons'][:2]:
                    print(f"    ✓ {reason}")
            
            # Risk indicator
            if rec['expected_return'] > 0.2:
                print(f"    🚀 High potential")
            elif rec['expected_return'] > 0:
                print(f"    📈 Good potential")
            else:
                print(f"    ⚠️  Caution")
        
        print("\n" + "="*80)
        print("💡 Overnight Scan Summary:")
        print(f"• Analyzed {self.results['stocks_analyzed']} stocks")
        print(f"• Found {self.results['options_found']} total options")
        print(f"• Selected {len(recommendations)} best diversified options")
        print("• Each option is from a different stock for portfolio diversity")
        print("="*80)
    
    def _save_results(self, recommendations: List[Dict], stocks_analyzed: int, options_found: int):
        """Save results to file"""
        self.results.update({
            'stocks_analyzed': stocks_analyzed,
            'options_found': options_found,
            'top_recommendations': recommendations
        })
        
        # Create reports directory
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Save to JSON file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/overnight_scan_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"✓ Results saved to {filename}")


def main():
    """Main entry point for overnight scan"""
    print("Starting Overnight Comprehensive Options Scanner...")
    print("This will take several hours to complete.")
    print("Results will be saved to reports/ directory.")
    print()
    
    # Create necessary directories
    for dir in ['logs', 'reports', 'data', 'data/cache']:
        Path(dir).mkdir(exist_ok=True)
    
    # Initialize scanner
    scanner = OvernightScanner()
    
    # Run comprehensive scan
    scanner.run_comprehensive_scan()


if __name__ == '__main__':
    main() 