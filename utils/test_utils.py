import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config
from utils.risk_manager import RiskManager
from utils.market_scanner import MarketScanner
from utils.options_analyzer import OptionsAnalyzer
from utils.portfolio_manager import PortfolioManager
from utils.data_fetcher import DataFetcher


def main():
    config = Config()
    print("Config loaded.")

    # Instantiate all utils
    try:
        risk_manager = RiskManager(config)
        print("RiskManager instantiated.")
        data_fetcher = DataFetcher(config)
        print("DataFetcher instantiated.")
        market_scanner = MarketScanner(config, data_fetcher)
        print("MarketScanner instantiated.")
        options_analyzer = OptionsAnalyzer(config, data_fetcher)
        print("OptionsAnalyzer instantiated.")
        portfolio_manager = PortfolioManager(config)
        print("PortfolioManager instantiated.")
    except Exception as e:
        print("Error instantiating utils:", e)
        return

    # Optionally, call a simple method on each
    try:
        # RiskManager: test risk level
        print("RiskManager _get_risk_level(50):", risk_manager._get_risk_level(50))
        # MarketScanner: test scan (if method exists)
        if hasattr(market_scanner, "scan"):
            print("MarketScanner scan():", market_scanner.scan())
        # OptionsAnalyzer: test analyze (if method exists)
        if hasattr(options_analyzer, "analyze"):
            print("OptionsAnalyzer analyze():", options_analyzer.analyze())
        # PortfolioManager: test get_total_value (if method exists)
        if hasattr(portfolio_manager, "get_total_value"):
            print("PortfolioManager get_total_value():", portfolio_manager.get_total_value())
        # DataFetcher: test get_quote for a known symbol
        print("DataFetcher get_quote('ASTS'):", data_fetcher.get_quote('ASTS'))
    except Exception as e:
        print("Error calling util methods:", e)

if __name__ == "__main__":
    main() 