#!/usr/bin/env python3
"""
Create an expanded ticker universe for better options opportunities
Run this to generate a comprehensive list of tickers for options scanning
"""

import pandas as pd
import yfinance as yf
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_comprehensive_ticker_list():
    """Create a comprehensive ticker list from multiple sources"""
    
    # Popular options-friendly stocks by category
    ticker_categories = {
        'mega_cap_tech': [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE',
            'CRM', 'ORCL', 'INTC', 'CSCO', 'AMD', 'QCOM', 'TXN', 'AVGO', 'MU', 'MRVL'
        ],
        'large_cap_growth': [
            'UNH', 'JNJ', 'V', 'WMT', 'PG', 'HD', 'MA', 'DIS', 'PYPL', 'BAC',
            'ABBV', 'TMO', 'CVX', 'KO', 'PEP', 'COST', 'DHR', 'ABT', 'VZ', 'CMCSA'
        ],
        'mid_cap_growth': [
            'ROKU', 'SNOW', 'CRWD', 'DDOG', 'NET', 'OKTA', 'TWLO', 'DOCU', 'ZM', 'SHOP',
            'SQ', 'PYPL', 'AFRM', 'SOFI', 'HOOD', 'PATH', 'AI', 'PLTR', 'RBLX', 'U'
        ],
        'small_cap_volatile': [
            'GME', 'AMC', 'BB', 'NOK', 'WISH', 'CLOV', 'SPCE', 'TLRY', 'SNDL', 'MVIS',
            'CLNE', 'LCID', 'RIVN', 'PTON', 'BYND', 'COIN', 'OPEN', 'FUBO', 'SKLZ', 'HOOD'
        ],
        'biotech_pharma': [
            'MRNA', 'PFE', 'BNTX', 'NVAX', 'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'BMRN',
            'CRSP', 'EDIT', 'NTLA', 'BEAM', 'PACB', 'TDOC', 'VEEV', 'ZTS', 'INCY', 'EXAS'
        ],
        'financial_services': [
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF',
            'AXP', 'BLK', 'SPGI', 'ICE', 'CME', 'MCO', 'MSCI', 'TRV', 'AIG', 'AFL'
        ],
        'energy_commodities': [
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'OXY', 'MRO', 'DVN', 'FANG',
            'PSX', 'VLO', 'MPC', 'HES', 'KMI', 'WMB', 'EPD', 'ET', 'ENB', 'TRP'
        ],
        'industrial_aerospace': [
            'BA', 'CAT', 'DE', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX',
            'NOC', 'GD', 'LHX', 'TDG', 'LDOS', 'HWM', 'TXT', 'ITW', 'EMR', 'ETN'
        ],
        'consumer_discretionary': [
            'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'TGT', 'BKNG',
            'GM', 'F', 'CCL', 'RCL', 'NCLH', 'MAR', 'HLT', 'MGM', 'WYNN', 'LVS'
        ],
        'etfs_popular': [
            'SPY', 'QQQ', 'IWM', 'VTI', 'VEA', 'VWO', 'ARKK', 'ARKQ', 'ARKG', 'ARKW',
            'TQQQ', 'SQQQ', 'UPRO', 'SPXL', 'TNA', 'TECL', 'SOXL', 'CURE', 'JETS', 'XLF'
        ],
        'reits_utilities': [
            'SPG', 'AMT', 'CCI', 'EQIX', 'PLD', 'EXR', 'WELL', 'DLR', 'O', 'REG',
            'NEE', 'DUK', 'SO', 'D', 'EXC', 'SRE', 'AEP', 'XEL', 'PEG', 'ES'
        ]
    }
    
    # Combine all symbols
    all_symbols = []
    for category, symbols in ticker_categories.items():
        all_symbols.extend(symbols)
    
    # Remove duplicates and sort
    unique_symbols = sorted(list(set(all_symbols)))
    
    logger.info(f"Created universe of {len(unique_symbols)} unique tickers")
    
    # Create DataFrame
    df = pd.DataFrame({'symbol': unique_symbols})
    
    # Save to CSV
    output_file = Path('tickers.csv')
    df.to_csv(output_file, index=False)
    logger.info(f"Saved ticker universe to {output_file}")
    
    # Create additional universe files for different strategies
    create_specialty_lists(ticker_categories)
    
    return unique_symbols

def create_specialty_lists(ticker_categories):
    """Create specialty ticker lists for different strategies"""
    
    # High-volume options universe (most liquid for day trading)
    high_volume = (
        ticker_categories['mega_cap_tech'] + 
        ticker_categories['etfs_popular'] + 
        ticker_categories['financial_services'][:10] +
        ticker_categories['small_cap_volatile'][:15]
    )
    
    pd.DataFrame({'symbol': high_volume}).to_csv('tickers_high_volume.csv', index=False)
    logger.info(f"Created high-volume universe: {len(high_volume)} tickers")
    
    # Small-cap growth universe (for aggressive strategies)
    small_cap = (
        ticker_categories['small_cap_volatile'] + 
        ticker_categories['mid_cap_growth'] + 
        ticker_categories['biotech_pharma'][:15]
    )
    
    pd.DataFrame({'symbol': small_cap}).to_csv('tickers_small_cap.csv', index=False)
    logger.info(f"Created small-cap universe: {len(small_cap)} tickers")
    
    # Sector rotation universe (for sector plays)
    sector_rotation = (
        ticker_categories['energy_commodities'] + 
        ticker_categories['financial_services'] + 
        ticker_categories['biotech_pharma'] + 
        ticker_categories['industrial_aerospace']
    )
    
    pd.DataFrame({'symbol': sector_rotation}).to_csv('tickers_sector_rotation.csv', index=False)
    logger.info(f"Created sector rotation universe: {len(sector_rotation)} tickers")

def validate_tickers():
    """Validate that tickers are real and tradeable"""
    
    # Load the main ticker file
    ticker_file = Path('tickers.csv')
    if not ticker_file.exists():
        logger.error("No tickers.csv file found. Run create_comprehensive_ticker_list() first.")
        return
    
    df = pd.read_csv(ticker_file)
    symbols = df['symbol'].tolist()
    
    logger.info(f"Validating {len(symbols)} tickers...")
    
    # Test a sample of tickers
    sample_size = min(20, len(symbols))
    test_symbols = symbols[:sample_size]
    
    valid_count = 0
    for symbol in test_symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if info.get('regularMarketPrice') or info.get('previousClose'):
                valid_count += 1
                logger.debug(f"✓ {symbol} - ${info.get('regularMarketPrice', 'N/A')}")
            else:
                logger.warning(f"✗ {symbol} - No price data")
                
        except Exception as e:
            logger.warning(f"✗ {symbol} - Error: {e}")
    
    logger.info(f"Validation complete: {valid_count}/{sample_size} tickers valid")
    return valid_count / sample_size

if __name__ == '__main__':
    print("Creating comprehensive ticker universe for options trading...")
    
    # Create the main universe
    symbols = create_comprehensive_ticker_list()
    
    # Validate a sample
    validation_rate = validate_tickers()
    
    print(f"\n✅ Created ticker universe with {len(symbols)} symbols")
    print(f"✅ Validation rate: {validation_rate:.1%}")
    print(f"✅ Files created:")
    print(f"   • tickers.csv (main universe)")
    print(f"   • tickers_high_volume.csv (liquid options)")
    print(f"   • tickers_small_cap.csv (growth/volatile)")
    print(f"   • tickers_sector_rotation.csv (sector plays)")
    print(f"\nTo use: Place any of these CSV files in your project root")
    print(f"The scanner will automatically load tickers from tickers.csv if present")