"""
Data Fetcher module for retrieving market data
Automatically fetches all small-cap stocks from online sources
"""

import logging
import json
import time
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from functools import lru_cache
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from pathlib import Path
from scipy.stats import norm
import math

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches market data from various sources"""
    
    def __init__(self, config):
        self.config = config
        self.cache = {}
        self.cache_expiry = {}
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def get_stocks_by_market_cap(self, min_cap: float, max_cap: float, 
                                 min_volume: int) -> List[Dict]:
        """Get stocks filtered by market cap and volume"""
        
        # Check cache first
        cache_file = self.cache_dir / "small_caps_universe.pkl"
        cache_age_hours = 24  # Refresh daily
        
        if cache_file.exists():
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if (datetime.now() - cache_time).total_seconds() < cache_age_hours * 3600:
                logger.info("Loading small-caps from cache")
                with open(cache_file, 'rb') as f:
                    all_stocks = pickle.load(f)
                    # Apply filters
                    return [s for s in all_stocks 
                           if min_cap <= s.get('market_cap', 0) <= max_cap 
                           and s.get('volume', 0) >= min_volume]
        
        logger.info("Fetching fresh small-cap universe from online sources...")
        
        # Fetch from multiple sources
        all_tickers = set()
        
        # 1. Fetch from NASDAQ API
        nasdaq_tickers = self._fetch_nasdaq_screener()
        all_tickers.update(nasdaq_tickers)
        logger.info(f"Found {len(nasdaq_tickers)} tickers from NASDAQ")
        
        # 2. Fetch from Yahoo Finance screener
        yahoo_tickers = self._fetch_yahoo_screener()
        all_tickers.update(yahoo_tickers)
        logger.info(f"Found {len(yahoo_tickers)} tickers from Yahoo")
        
        # 3. Fetch from FinViz (via API workaround)
        finviz_tickers = self._fetch_finviz_tickers()
        all_tickers.update(finviz_tickers)
        logger.info(f"Found {len(finviz_tickers)} tickers from FinViz")
        
        # 4. Fetch from FMP free tier
        fmp_tickers = self._fetch_fmp_tickers()
        all_tickers.update(fmp_tickers)
        logger.info(f"Found {len(fmp_tickers)} tickers from FMP")
        
        logger.info(f"Total unique tickers to verify: {len(all_tickers)}")
        
        # Verify and enrich tickers
        verified_stocks = self._verify_and_enrich_tickers(list(all_tickers), min_cap, max_cap)
        
        # Cache the results
        with open(cache_file, 'wb') as f:
            pickle.dump(verified_stocks, f)
        
        # Apply volume filter
        filtered_stocks = [s for s in verified_stocks if s.get('volume', 0) >= min_volume]
        
        logger.info(f"Returning {len(filtered_stocks)} stocks after all filters")
        return filtered_stocks
    
    def _fetch_nasdaq_screener(self) -> List[str]:
        """Fetch stock list from NASDAQ screener API"""
        tickers = []
        
        try:
            url = "https://api.nasdaq.com/api/screener/stocks"
            params = {
                'tableonly': 'true',
                'limit': 25000,
                'offset': 0,
                'download': 'true'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                for row in data.get('data', {}).get('rows', []):
                    ticker = row.get('symbol', '')
                    if ticker and not any(c in ticker for c in ['^', '.', '=']):  # Skip special symbols
                        tickers.append(ticker)
                        
        except Exception as e:
            logger.error(f"Error fetching NASDAQ data: {e}")
            
        return tickers
    
    def _fetch_yahoo_screener(self) -> List[str]:
        """Fetch from Yahoo Finance screener API"""
        tickers = []
        
        try:
            # Yahoo Finance query API
            base_url = "https://query2.finance.yahoo.com/v1/finance/screener"
            
            # Create screener query for small caps
            screener_data = {
                "size": 250,
                "offset": 0,
                "sortField": "marketcap",
                "sortType": "DESC",
                "quoteType": "EQUITY",
                "query": {
                    "operator": "AND",
                    "operands": [
                        {"operator": "GT", "operands": ["marketcap", 500000000]},
                        {"operator": "LT", "operands": ["marketcap", 10000000000]},
                        {"operator": "GT", "operands": ["avgdailyvol3m", 100000]}
                    ]
                },
                "userId": "",
                "userIdType": "guid"
            }
            
            # Try predefined screeners
            predefined_screeners = [
                "small_cap_gainers",
                "most_actives_small_cap", 
                "growth_technology_small_cap"
            ]
            
            for screener in predefined_screeners:
                try:
                    url = f"{base_url}/predefined/{screener}"
                    response = self.session.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        quotes = data.get('finance', {}).get('result', [{}])[0].get('quotes', [])
                        
                        for quote in quotes:
                            symbol = quote.get('symbol', '')
                            if symbol:
                                tickers.append(symbol)
                                
                except Exception as e:
                    logger.debug(f"Error with screener {screener}: {e}")
                    
        except Exception as e:
            logger.error(f"Error fetching Yahoo data: {e}")
            
        return tickers
    
    def _fetch_finviz_tickers(self) -> List[str]:
        """Fetch tickers using FinViz-style filtering"""
        tickers = []
        
        try:
            # Alternative: Use Yahoo Finance trending and related APIs
            urls = [
                "https://query1.finance.yahoo.com/v1/finance/trending/US",
                "https://query1.finance.yahoo.com/v1/finance/screener/instrument/equity/fields",
            ]
            
            for url in urls:
                try:
                    response = self.session.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Extract symbols from different response formats
                        if 'finance' in data:
                            for item in data.get('finance', {}).get('result', [])[0].get('quotes', []):
                                tickers.append(item.get('symbol', ''))
                                
                except Exception as e:
                    logger.debug(f"Error with URL {url}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in FinViz-style fetch: {e}")
            
        return tickers
    
    def _fetch_fmp_tickers(self) -> List[str]:
        """Fetch from Financial Modeling Prep free tier"""
        tickers = []
        
        try:
            # FMP stock list endpoint (limited free tier)
            url = "https://financialmodelingprep.com/api/v3/stock/list"
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, list):
                    for stock in data[:1000]:  # Free tier limit
                        symbol = stock.get('symbol', '')
                        if symbol and stock.get('exchangeShortName') in ['NASDAQ', 'NYSE', 'AMEX']:
                            tickers.append(symbol)
                            
        except Exception as e:
            logger.error(f"Error fetching FMP data: {e}")
            
        return tickers
    
    def _verify_and_enrich_tickers(self, tickers: List[str], min_cap: float, max_cap: float) -> List[Dict]:
        """Verify tickers and get their market data"""
        verified_stocks = []
        
        logger.info(f"Verifying {len(tickers)} tickers...")
        
        # Remove duplicates and sort
        unique_tickers = sorted(list(set(tickers)))
        
        # Process in batches to avoid overwhelming the API
        batch_size = 50
        max_workers = 10
        
        for i in range(0, len(unique_tickers), batch_size):
            batch = unique_tickers[i:i + batch_size]
            batch_stocks = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ticker = {
                    executor.submit(self._verify_single_ticker, ticker, min_cap, max_cap): ticker 
                    for ticker in batch
                }
                
                for future in as_completed(future_to_ticker):
                    result = future.result()
                    if result:
                        batch_stocks.append(result)
            
            verified_stocks.extend(batch_stocks)
            
            # Progress update
            processed = min(i + batch_size, len(unique_tickers))
            logger.info(f"Verified {processed}/{len(unique_tickers)} tickers, found {len(verified_stocks)} valid stocks")
            
            # Rate limiting
            time.sleep(1)
        
        return verified_stocks
    
    def _verify_single_ticker(self, ticker: str, min_cap: float, max_cap: float) -> Optional[Dict]:
        """Verify a single ticker meets our criteria"""
        try:
            # Skip if ticker has special characters
            if any(c in ticker for c in ['^', '.', '=', ' ', '/']):
                return None
                
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if valid stock with market cap
            market_cap = info.get('marketCap', 0)
            if not market_cap or market_cap < min_cap or market_cap > max_cap:
                return None
            
            # Get recent price/volume data
            hist = stock.history(period="5d")
            if hist.empty:
                return None
            
            current_price = hist['Close'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            
            # Skip penny stocks and illiquid stocks
            if current_price < 1 or avg_volume < 50000:
                return None
            
            return {
                'symbol': ticker,
                'name': info.get('shortName', ticker),
                'market_cap': market_cap,
                'price': current_price,
                'volume': int(avg_volume),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'exchange': info.get('exchange', 'Unknown'),
                'pe_ratio': info.get('forwardPE'),
                'employees': info.get('fullTimeEmployees'),
                'country': info.get('country', 'US')
            }
            
        except Exception as e:
            logger.debug(f"Error verifying {ticker}: {e}")
            return None
    
    @lru_cache(maxsize=100)
    def get_quote(self, symbol: str) -> Dict:
        """Get current quote for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            
            if data.empty:
                raise ValueError(f"No data available for {symbol}")
                
            current_price = float(data['Close'].iloc[-1])
            volume = int(data['Volume'].sum())
            
            # Get additional info
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': float(current_price),
                'volume': int(volume),
                'avg_volume': int(info.get('averageVolume', volume)),
                'bid': float(info.get('bid', current_price)),
                'ask': float(info.get('ask', current_price)),
                'day_high': float(data['High'].max()),
                'day_low': float(data['Low'].min()),
                'prev_close': float(info.get('previousClose', current_price)),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            raise
    
    def get_fundamentals(self, symbol: str) -> Dict:
        """Get fundamental data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get financial data
            financials = ticker.quarterly_financials
            
            # Calculate growth metrics
            revenue_growth = None
            if not financials.empty and len(financials.columns) >= 2:
                recent_revenue = financials.iloc[0, 0]
                previous_revenue = financials.iloc[0, 1]
                if previous_revenue and previous_revenue != 0:
                    revenue_growth = (recent_revenue - previous_revenue) / previous_revenue
                    
            return {
                'pe_ratio': info.get('forwardPE', info.get('trailingPE')),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'revenue_growth': revenue_growth,
                'earnings_growth': info.get('earningsQuarterlyGrowth'),
                'profit_margin': info.get('profitMargins'),
                'institutional_ownership': info.get('heldPercentInstitutions', 0),
                'insider_ownership': info.get('heldPercentInsiders', 0),
                'short_ratio': info.get('shortRatio'),
                'beta': info.get('beta')
            }
            
        except Exception as e:
            logger.error(f"Error getting fundamentals for {symbol}: {e}")
            return {}
    
    def get_price_history(self, symbol: str, days: int = 100) -> List[Dict]:
        """Get price history for technical analysis"""
        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                return []
                
            history = []
            for date, row in data.iterrows():
                history.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': row['Open'],
                    'high': row['High'],
                    'low': row['Low'],
                    'close': row['Close'],
                    'volume': row['Volume']
                })
                
            return history
            
        except Exception as e:
            logger.error(f"Error getting price history for {symbol}: {e}")
            return []
    
    def get_options_chain(self, symbol: str) -> List[Dict]:
        """Get options chain for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                return []
                
            options_data = []
            current_price = self.get_quote(symbol)['price']
            
            for exp_date in expirations:
                # Convert expiration to datetime
                exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                days_to_exp = (exp_datetime - datetime.now()).days
                
                # Skip if outside our target range
                if (days_to_exp < self.config.trading.min_days_to_expiration or
                    days_to_exp > self.config.trading.max_days_to_expiration):
                    continue
                    
                # Get options data
                opt_chain = ticker.option_chain(exp_date)
                calls = opt_chain.calls
                
                # Process calls
                for _, call in calls.iterrows():
                    strike = call['strike']
                    
                    # Skip deep ITM/OTM
                    if strike < current_price * 0.9 or strike > current_price * 1.15:
                        continue
                        
                    # Calculate Greeks (simplified)
                    iv = call.get('impliedVolatility', 0.3)
                    
                    option_data = {
                        'symbol': symbol,
                        'type': 'CALL',
                        'strike': strike,
                        'expiration': exp_date,
                        'days_to_expiration': days_to_exp,
                        'bid': call['bid'],
                        'ask': call['ask'],
                        'last': call['lastPrice'],
                        'volume': call['volume'],
                        'open_interest': call['openInterest'],
                        'implied_volatility': iv,
                        'in_the_money': call['inTheMoney'],
                        'contract_symbol': call['contractSymbol'],
                        # Greeks (would calculate properly in production)
                        'delta': self._estimate_delta(current_price, strike, days_to_exp, iv),
                        'theta': self._estimate_theta(current_price, strike, days_to_exp, iv, call['ask']),
                        'gamma': self._estimate_gamma(current_price, strike, days_to_exp, iv),
                        'vega': self._estimate_vega(current_price, strike, days_to_exp, iv),
                        'iv_percentile': self._calculate_iv_percentile(symbol, iv)
                    }
                    
                    options_data.append(option_data)
                    
            return options_data
            
        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            return []
    
    def get_option_quote(self, symbol: str, strike: float, expiration: str, 
                        option_type: str = 'CALL') -> Optional[Dict]:
        """Get quote for specific option contract"""
        try:
            ticker = yf.Ticker(symbol)
            opt_chain = ticker.option_chain(expiration)
            
            if option_type == 'CALL':
                options = opt_chain.calls
            else:
                options = opt_chain.puts
                
            # Find the specific strike
            option = options[options['strike'] == strike]
            
            if option.empty:
                return None
                
            opt = option.iloc[0]
            current_price = self.get_quote(symbol)['price']
            days_to_exp = (datetime.strptime(expiration, '%Y-%m-%d') - datetime.now()).days
            iv = opt.get('impliedVolatility', 0.3)
            
            return {
                'bid': opt['bid'],
                'ask': opt['ask'],
                'mid': (opt['bid'] + opt['ask']) / 2,
                'last': opt['lastPrice'],
                'volume': opt['volume'],
                'open_interest': opt['openInterest'],
                'implied_volatility': iv,
                'delta': self._estimate_delta(current_price, strike, days_to_exp, iv),
                'theta': self._estimate_theta(current_price, strike, days_to_exp, iv, opt['ask']),
                'gamma': self._estimate_gamma(current_price, strike, days_to_exp, iv),
                'vega': self._estimate_vega(current_price, strike, days_to_exp, iv)
            }
            
        except Exception as e:
            logger.error(f"Error getting option quote: {e}")
            return None
    
    def _estimate_delta(self, spot: float, strike: float, days: int, iv: float) -> float:
        """Estimate delta using Black-Scholes approximation"""
        # Simplified delta calculation
        moneyness = spot / strike
        time_factor = days / 365
        
        if moneyness > 1.05:  # ITM
            return 0.6 + min(0.3, (moneyness - 1) * 2)
        elif moneyness < 0.95:  # OTM
            return max(0.1, 0.4 * moneyness)
        else:  # ATM
            return 0.5
    
    def _estimate_theta(self, spot: float, strike: float, days: int, 
                       iv: float, option_price: float) -> float:
        """Estimate theta (time decay)"""
        # Simplified theta - increases as expiration approaches
        time_factor = 1 / max(days, 1)
        atm_factor = 1 - abs(spot - strike) / spot
        
        # Theta as percentage of option value
        theta_rate = -0.01 * time_factor * atm_factor * (1 + iv)
        
        return option_price * theta_rate
    
    def _estimate_gamma(self, spot: float, strike: float, days: int, iv: float) -> float:
        """Estimate gamma"""
        # Highest at ATM, decreases away from strike
        moneyness = abs(spot - strike) / spot
        time_factor = 1 / max(math.sqrt(days), 1)
        
        return max(0, (0.05 - moneyness) * time_factor)
    
    def _estimate_vega(self, spot: float, strike: float, days: int, iv: float) -> float:
        """Estimate vega"""
        # Vega highest for ATM options with time remaining
        moneyness = abs(spot - strike) / spot
        time_factor = math.sqrt(days / 365)
        
        return spot * 0.01 * (1 - moneyness) * time_factor
    
    def _calculate_iv_percentile(self, symbol: str, current_iv: float) -> float:
        """Calculate IV percentile based on historical data"""
        # Simplified - in production would use historical IV data
        # For now, return a random percentile
        return min(95, max(5, current_iv * 100))
    
    def clear_cache(self):
        """Clear data cache"""
        self.cache.clear()
        self.cache_expiry.clear()
        
        # Clear file cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            
        logger.info("Data cache cleared")