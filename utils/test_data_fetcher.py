import sys
import logging
from data_fetcher import DataFetcher

# Set up logging to print to console
logging.basicConfig(level=logging.INFO)

# Minimal mock config with required attributes
class MockTradingConfig:
    min_days_to_expiration = 7
    max_days_to_expiration = 60

class MockConfig:
    trading = MockTradingConfig()

if __name__ == "__main__":
    config = MockConfig()
    fetcher = DataFetcher(config)
    try:
        print("Testing get_quote for 'ASTS'...")
        quote = fetcher.get_quote('ASTS')
        print("Quote for ASTS:", quote)
    except Exception as e:
        print("Error during test:", e)
        sys.exit(1)
    print("Test completed successfully.") 