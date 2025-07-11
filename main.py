#!/usr/bin/env python3
"""
Small-Cap Market Tracker with Options Trading Signals
Main entry point for the application
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import schedule
import time
import argparse

from utils.market_scanner import MarketScanner
from utils.options_analyzer import OptionsAnalyzer
from utils.portfolio_manager import PortfolioManager
from utils.risk_manager import RiskManager
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


class SmallCapOptionsTracker:
    """Main application class for small-cap options tracking"""
    
    def __init__(self, config_path: str = 'config.json'):
        """Initialize the tracker with configuration"""
        self.config = Config(config_path)
        self.data_fetcher = DataFetcher(self.config)
        self.scanner = MarketScanner(self.config, self.data_fetcher)
        self.options_analyzer = OptionsAnalyzer(self.config, self.data_fetcher)
        self.portfolio = PortfolioManager(self.config)
        self.risk_manager = RiskManager(self.config)
        
    def scan_for_opportunities(self) -> List[Dict]:
        """Scan market for small-cap stocks with potential"""
        logger.info("Starting market scan for small-cap opportunities...")
        
        # Get small-cap stocks (500M - 10B market cap)
        candidates = self.scanner.find_small_caps()
        logger.info(f"Found {len(candidates)} small-cap candidates")
        
        # Apply technical and fundamental filters
        filtered = self.scanner.apply_filters(candidates)
        logger.info(f"Filtered to {len(filtered)} stocks meeting criteria")
        
        # Analyze options for each candidate
        opportunities = []
        for stock in filtered:
            try:
                option_signal = self.options_analyzer.analyze_stock(stock)
                if option_signal and option_signal['recommendation'] == 'BUY':
                    opportunities.append(option_signal)
            except Exception as e:
                logger.error(f"Error analyzing {stock['symbol']}: {e}")
                
        return opportunities
    
    def monitor_positions(self) -> List[Dict]:
        """Monitor existing positions and generate exit signals"""
        logger.info("Monitoring existing positions...")
        
        positions = self.portfolio.get_open_positions()
        exit_signals = []
        
        for position in positions:
            try:
                # Check if position needs adjustment or exit
                signal = self.options_analyzer.evaluate_position(position)
                
                if signal['action'] in ['SELL', 'ROLL']:
                    exit_signals.append(signal)
                    logger.info(f"Exit signal for {position['symbol']}: {signal['reason']}")
                    
            except Exception as e:
                logger.error(f"Error monitoring position {position['symbol']}: {e}")
                
        return exit_signals
    
    def execute_trades(self, opportunities: List[Dict], exit_signals: List[Dict]):
        """Execute trades based on signals (paper trading)"""
        # Process exit signals first
        for signal in exit_signals:
            if signal['action'] == 'SELL':
                self.portfolio.close_position(
                    signal['symbol'],
                    signal['current_price'],
                    signal['reason']
                )
            elif signal['action'] == 'ROLL':
                # Close current and open new position
                self.portfolio.roll_position(
                    signal['symbol'],
                    signal['new_strike'],
                    signal['new_expiration'],
                    signal['current_price']
                )
        
        # Process new opportunities
        for opp in opportunities:
            # Check risk limits before entering
            if self.risk_manager.can_enter_position(opp, self.portfolio):
                self.portfolio.open_position(opp)
                logger.info(f"Opened position: {opp['symbol']} {opp['strike']}C exp {opp['expiration']}")
            else:
                logger.warning(f"Risk limits prevent entering {opp['symbol']}")
    
    def generate_report(self):
        """Generate daily performance report"""
        logger.info("Generating daily report...")
        
        report = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'portfolio_value': self.portfolio.get_total_value(),
            'open_positions': self.portfolio.get_open_positions(),
            'closed_today': self.portfolio.get_positions_closed_today(),
            'performance': self.portfolio.calculate_performance(),
            'risk_metrics': self.risk_manager.calculate_portfolio_risk(self.portfolio)
        }
        
        # Save report
        report_path = f"reports/daily_{report['date']}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Report saved to {report_path}")
        return report
    
    def run_analysis(self):
        """Run complete analysis cycle"""
        logger.info("="*50)
        logger.info("Starting analysis cycle...")
        
        try:
            # 1. Monitor existing positions
            exit_signals = self.monitor_positions()
            
            # 2. Scan for new opportunities
            opportunities = self.scan_for_opportunities()
            
            # 3. Execute trades
            self.execute_trades(opportunities, exit_signals)
            
            # 4. Generate report
            report = self.generate_report()
            
            # 5. Display summary
            self.display_summary(report)
            
        except Exception as e:
            logger.error(f"Error in analysis cycle: {e}")
            
    def display_summary(self, report: Dict):
        """Display summary of current status"""
        print("\n" + "="*60)
        print(f"SMALL-CAP OPTIONS TRACKER - {report['date']}")
        print("="*60)
        print(f"Portfolio Value: ${report['portfolio_value']:,.2f}")
        print(f"Open Positions: {len(report['open_positions'])}")
        print(f"Today's P&L: ${report['performance']['daily_pnl']:,.2f}")
        print(f"Total P&L: ${report['performance']['total_pnl']:,.2f}")
        print(f"Win Rate: {report['performance']['win_rate']:.1f}%")
        print(f"Sharpe Ratio: {report['risk_metrics']['sharpe_ratio']:.2f}")
        print("="*60)
        
        if report['open_positions']:
            print("\nOPEN POSITIONS:")
            for pos in report['open_positions']:
                days_held = (datetime.now() - datetime.fromisoformat(pos['entry_date'])).days
                days_to_exp = (datetime.fromisoformat(pos['expiration']) - datetime.now()).days
                print(f"- {pos['symbol']} ${pos['strike']}C exp {pos['expiration'][:10]}")
                print(f"  Entry: ${pos['entry_price']:.2f} | Current: ${pos['current_price']:.2f}")
                print(f"  P&L: ${pos['unrealized_pnl']:.2f} ({pos['pnl_percent']:.1f}%)")
                print(f"  Days held: {days_held} | Days to exp: {days_to_exp}")
                print(f"  Theta: ${pos['theta']:.2f} | IV: {pos['iv']:.1f}%")
                
    def run_continuous(self):
        """Run tracker continuously with scheduled scans"""
        logger.info("Starting continuous tracking mode...")
        
        # Schedule scans
        schedule.every().day.at("09:35").do(self.run_analysis)  # After market open
        schedule.every().day.at("15:30").do(self.run_analysis)  # Before market close
        
        # Run initial analysis
        self.run_analysis()
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Small-Cap Options Tracker')
    parser.add_argument('--mode', choices=['once', 'continuous'], default='once',
                      help='Run mode: once for single scan, continuous for scheduled scans')
    parser.add_argument('--config', default='config.json',
                      help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Create necessary directories
    import os
    for dir in ['logs', 'reports', 'data']:
        os.makedirs(dir, exist_ok=True)
    
    # Initialize and run tracker
    tracker = SmallCapOptionsTracker(args.config)
    
    if args.mode == 'once':
        tracker.run_analysis()
    else:
        tracker.run_continuous()


if __name__ == '__main__':
    main()