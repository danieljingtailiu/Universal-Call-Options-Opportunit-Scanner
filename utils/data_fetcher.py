"""
Improved Data Fetcher with better filtering and options analysis
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
    """Enhanced data fetcher with improved filtering"""
    
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
        
        # RELAXED FILTERS for better results
        self.min_option_volume = 1  # Accept even 1 contract volume
        self.min_option_oi = 10  # Minimum 10 open interest
        self.max_bid_ask_spread = 0.50  # Max 50% spread for low volume
        
    def get_stocks_by_market_cap(self, min_cap: float, max_cap: float, 
                                 min_volume: int) -> List[Dict]:
        """Get stocks filtered by market cap and volume with RELAXED criteria"""
        
        # Check cache first
        cache_file = self.cache_dir / "small_caps_universe.pkl"
        cache_age_hours = 24  # Refresh daily
        
        if cache_file.exists():
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if (datetime.now() - cache_time).total_seconds() < cache_age_hours * 3600:
                logger.info("Loading small-caps from cache")
                with open(cache_file, 'rb') as f:
                    all_stocks = pickle.load(f)
                    # Apply RELAXED filters
                    filtered = []
                    for s in all_stocks:
                        if min_cap <= s.get('market_cap', 0) <= max_cap:
                            # RELAXED volume requirement
                            if s.get('volume', 0) >= min_volume * 0.5:  # Allow 50% of min volume
                                filtered.append(s)
                    return filtered
        
        logger.info("Fetching fresh small-cap universe from online sources...")
        
        # Get high-quality small-cap list
        quality_tickers = self._get_quality_small_caps()
        
        # Fetch from multiple sources
        all_tickers = set(quality_tickers)
        
        # Add from other sources
        nasdaq_tickers = self._fetch_nasdaq_screener()
        all_tickers.update(nasdaq_tickers)
        
        yahoo_tickers = self._fetch_yahoo_screener()
        all_tickers.update(yahoo_tickers)
        
        logger.info(f"Total unique tickers to verify: {len(all_tickers)}")
        
        # Verify with relaxed criteria
        verified_stocks = self._verify_and_enrich_tickers_relaxed(list(all_tickers), min_cap, max_cap)
        
        # Cache the results
        with open(cache_file, 'wb') as f:
            pickle.dump(verified_stocks, f)
        
        # Apply relaxed volume filter
        filtered_stocks = []
        for s in verified_stocks:
            # More lenient volume filtering
            if s.get('volume', 0) >= min_volume * 0.3:  # Allow 30% of min volume
                filtered_stocks.append(s)
            elif s.get('avg_volume', 0) >= min_volume * 0.5:  # Or 50% of average volume
                filtered_stocks.append(s)
        
        logger.info(f"Returning {len(filtered_stocks)} stocks after relaxed filters")
        return filtered_stocks
    
    def _get_quality_small_caps(self) -> List[str]:
        """Get high-quality small-cap stocks that are likely to have options"""
        # Focus on stocks with good options liquidity
        quality_small_caps = [
            # High-volume small caps with active options
            'SOFI', 'HOOD', 'AFRM', 'UPST', 'OPEN', 'RKLB', 'IONQ', 'ASTS',
            'PLTR', 'DKNG', 'COIN', 'MARA', 'RIOT', 'CLSK', 'HUT', 'BITF',
            'CHPT', 'BLNK', 'EVGO', 'LCID', 'RIVN', 'NKLA', 'GOEV', 'FSR',
            'SPCE', 'JOBY', 'LILM', 'ACHR', 'EVTL', 'ARCHER', 'BBAI', 'AI',
            'PATH', 'DOCN', 'GTLB', 'S', 'CFLT', 'NET', 'DDOG', 'SNOW',
            'CRWD', 'ZS', 'OKTA', 'TWLO', 'MDB', 'ESTC', 'FSLY', 'APPS',
            'WISH', 'CLOV', 'SKLZ', 'FUBO', 'GENI', 'RSI', 'PENN', 'CZR',
            'WYNN', 'MGM', 'CHWY', 'W', 'ETSY', 'SHOP', 'BIGC', 'CPNG',
            'DASH', 'ABNB', 'LYFT', 'UBER', 'GRAB', 'SE', 'MELI', 'PINS',
            'SNAP', 'MTCH', 'BMBL', 'RBLX', 'U', 'EA', 'TTWO', 'ATVI',
            'ZM', 'DOCU', 'DBX', 'BOX', 'ASAN', 'MNDY', 'BILL', 'TOST',
            'SQ', 'PYPL', 'AFRM', 'UPST', 'SOFI', 'HOOD', 'COMP', 'OPEN',
            'NVAX', 'MRNA', 'BNTX', 'CVAC', 'INO', 'VXRT', 'ALT', 'OCGN',
            'RUN', 'ENPH', 'SEDG', 'NOVA', 'MAXN', 'SPWR', 'CSIQ', 'JKS',
            'PLUG', 'FCEL', 'BE', 'BLDP', 'CHPT', 'BLNK', 'EVGO', 'VLDR'
        ]
        
        return quality_small_caps
    
    def _verify_and_enrich_tickers_relaxed(self, tickers: List[str], min_cap: float, max_cap: float) -> List[Dict]:
        """Verify tickers with RELAXED criteria"""
        verified_stocks = []
        
        logger.info(f"Verifying {len(tickers)} tickers with relaxed criteria...")
        
        # Remove duplicates and sort
        unique_tickers = sorted(list(set(tickers)))
        
        # Process in batches
        batch_size = 50
        max_workers = 10
        
        for i in range(0, len(unique_tickers), batch_size):
            batch = unique_tickers[i:i + batch_size]
            batch_stocks = []
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ticker = {
                    executor.submit(self._verify_single_ticker_relaxed, ticker, min_cap, max_cap): ticker 
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
    
    def _verify_single_ticker_relaxed(self, ticker: str, min_cap: float, max_cap: float) -> Optional[Dict]:
        """Verify ticker with RELAXED criteria"""
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
            
            # RELAXED: Allow stocks above $0.50 instead of $1
            if current_price < 0.50:
                return None
            
            # RELAXED: Allow lower volume stocks that might have options
            if avg_volume < 10000:  # Very low threshold
                return None
            
            # Check if has options (important!)
            try:
                options_dates = stock.options
                has_options = len(options_dates) > 0
            except:
                has_options = False
            
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
                'has_options': has_options,
                'options_expirations': len(options_dates) if has_options else 0
            }
            
        except Exception as e:
            logger.debug(f"Error verifying {ticker}: {e}")
            return None
    
    def get_options_chain(self, symbol: str) -> List[Dict]:
        """Get options chain with IMPROVED analysis for low-volume options"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                logger.warning(f"No options available for {symbol}")
                return []
                
            options_data = []
            current_price = self.get_quote(symbol)['price']
            
            # Log available expirations
            logger.info(f"{symbol} has {len(expirations)} expiration dates")
            
            for exp_date in expirations:
                # Convert expiration to datetime
                exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                days_to_exp = (exp_datetime - datetime.now()).days
                
                # RELAXED: Accept 20-70 days instead of 30-60
                if days_to_exp < 20 or days_to_exp > 70:
                    continue
                    
                # Get options data
                opt_chain = ticker.option_chain(exp_date)
                calls = opt_chain.calls
                
                logger.info(f"  Expiration {exp_date} ({days_to_exp} days): {len(calls)} strikes")
                
                # Process calls with RELAXED criteria
                for _, call in calls.iterrows():
                    strike = call['strike']
                    
                    # RELAXED: Allow wider range (85% to 120% of current price)
                    if strike < current_price * 0.85 or strike > current_price * 1.20:
                        continue
                    
                    # Get option data
                    bid = call['bid']
                    ask = call['ask']
                    volume = call['volume'] if pd.notna(call['volume']) else 0
                    open_interest = call['openInterest'] if pd.notna(call['openInterest']) else 0
                    
                    # Calculate spread
                    if ask > 0:
                        spread_pct = (ask - bid) / ask
                    else:
                        spread_pct = 1.0
                    
                    # RELAXED: Accept if ANY of these conditions are met
                    acceptable = False
                    liquidity_score = 0
                    
                    # Condition 1: Has some volume
                    if volume >= 1:
                        acceptable = True
                        liquidity_score += 30
                    
                    # Condition 2: Has open interest
                    if open_interest >= 10:
                        acceptable = True
                        liquidity_score += 40
                    
                    # Condition 3: Reasonable spread (even for low volume)
                    if spread_pct <= 0.50 and ask > 0:  # 50% spread acceptable for small caps
                        acceptable = True
                        liquidity_score += 30
                    
                    # Condition 4: Near the money options (more liquid)
                    moneyness = abs(strike - current_price) / current_price
                    if moneyness <= 0.05:  # Within 5% of current price
                        liquidity_score += 20
                    
                    if not acceptable:
                        continue
                    
                    # Calculate IMPLIED volatility if missing
                    iv = call.get('impliedVolatility', 0)
                    if iv == 0 or pd.isna(iv):
                        # Estimate IV based on bid-ask spread and time
                        iv = 0.3 + (spread_pct * 0.5)  # Higher spread = higher IV estimate
                    
                    option_data = {
                        'symbol': symbol,
                        'type': 'CALL',
                        'strike': strike,
                        'expiration': exp_date,
                        'days_to_expiration': days_to_exp,
                        'bid': bid,
                        'ask': ask,
                        'mid': (bid + ask) / 2 if ask > 0 else call.get('lastPrice', 0),
                        'last': call.get('lastPrice', 0),
                        'volume': int(volume),
                        'open_interest': int(open_interest),
                        'implied_volatility': iv,
                        'in_the_money': call.get('inTheMoney', False),
                        'contract_symbol': call.get('contractSymbol', ''),
                        'spread_pct': spread_pct,
                        'liquidity_score': liquidity_score,
                        # Greeks estimates
                        'delta': self._estimate_delta(current_price, strike, days_to_exp, iv),
                        'theta': self._estimate_theta(current_price, strike, days_to_exp, iv, ask if ask > 0 else 0.01),
                        'gamma': self._estimate_gamma(current_price, strike, days_to_exp, iv),
                        'vega': self._estimate_vega(current_price, strike, days_to_exp, iv),
                        'iv_percentile': min(95, max(5, iv * 150))  # Rough estimate
                    }
                    
                    options_data.append(option_data)
                    
            logger.info(f"Found {len(options_data)} acceptable option contracts for {symbol}")
            return options_data
            
        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            return []
    
    @lru_cache(maxsize=100)
    def get_quote(self, symbol: str) -> Dict:
        """Get current quote for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Try to get intraday data first
            data = ticker.history(period='1d', interval='5m')
            
            if data.empty:
                # Fall back to daily data
                data = ticker.history(period='5d')
                if data.empty:
                    raise ValueError(f"No data available for {symbol}")
                    
            current_price = data['Close'].iloc[-1]
            volume = data['Volume'].iloc[-1] if len(data) == 1 else data['Volume'].sum()
            
            # Get additional info
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': current_price,
                'volume': volume,
                'avg_volume': info.get('averageVolume', volume),
                'bid': info.get('bid', current_price * 0.995),  # Estimate if missing
                'ask': info.get('ask', current_price * 1.005),  # Estimate if missing
                'day_high': data['High'].max(),
                'day_low': data['Low'].min(),
                'prev_close': info.get('previousClose', current_price),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            raise
    
    def get_fundamentals(self, symbol: str) -> Dict:
        """Get fundamental data with defaults for missing values"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get financial data with error handling
            try:
                financials = ticker.quarterly_financials
                revenue_growth = None
                if not financials.empty and len(financials.columns) >= 2:
                    recent_revenue = financials.iloc[0, 0]
                    previous_revenue = financials.iloc[0, 1]
                    if previous_revenue and previous_revenue != 0:
                        revenue_growth = (recent_revenue - previous_revenue) / previous_revenue
            except:
                revenue_growth = None
                    
            return {
                'pe_ratio': info.get('forwardPE', info.get('trailingPE', 25)),  # Default PE
                'peg_ratio': info.get('pegRatio', 1.5),
                'price_to_book': info.get('priceToBook', 2),
                'revenue_growth': revenue_growth if revenue_growth else 0.10,  # Default 10% growth
                'earnings_growth': info.get('earningsQuarterlyGrowth', 0.10),
                'profit_margin': info.get('profitMargins', 0.10),
                'institutional_ownership': info.get('heldPercentInstitutions', 0.20),  # Default 20%
                'insider_ownership': info.get('heldPercentInsiders', 0.10),
                'short_ratio': info.get('shortRatio', 2),
                'beta': info.get('beta', 1.2)  # Default beta for growth stocks
            }
            
        except Exception as e:
            logger.error(f"Error getting fundamentals for {symbol}: {e}")
            # Return reasonable defaults
            return {
                'pe_ratio': 25,
                'peg_ratio': 1.5,
                'price_to_book': 2,
                'revenue_growth': 0.10,
                'earnings_growth': 0.10,
                'profit_margin': 0.10,
                'institutional_ownership': 0.20,
                'insider_ownership': 0.10,
                'short_ratio': 2,
                'beta': 1.2
            }
    
    def get_price_history(self, symbol: str, days: int = 100) -> List[Dict]:
        """Get price history for technical analysis"""
        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                # Try shorter period
                data = ticker.history(period="1mo")
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
    
    def _estimate_delta(self, spot: float, strike: float, days: int, iv: float) -> float:
        """Estimate delta using Black-Scholes approximation"""
        try:
            # More accurate delta calculation
            moneyness = spot / strike
            time_to_exp = days / 365.0
            
            # Simplified Black-Scholes delta approximation
            d1 = (np.log(moneyness) + (0.02 + 0.5 * iv * iv) * time_to_exp) / (iv * np.sqrt(time_to_exp))
            delta = norm.cdf(d1)
            
            return round(delta, 3)
        except:
            # Fallback to simple calculation
            if moneyness > 1.05:
                return 0.7
            elif moneyness < 0.95:
                return 0.3
            else:
                return 0.5
    
    def _estimate_theta(self, spot: float, strike: float, days: int, 
                       iv: float, option_price: float) -> float:
        """Estimate theta (time decay) - NEGATIVE value"""
        try:
            # Theta increases as expiration approaches
            time_factor = np.sqrt(365 / max(days, 1))
            
            # ATM options have highest theta
            moneyness = spot / strike
            atm_factor = np.exp(-((moneyness - 1) ** 2) / 0.02)
            
            # Base theta as percentage of option value
            base_theta = -0.005 * time_factor * atm_factor * (1 + iv)
            
            # Convert to dollar amount
            theta_dollars = option_price * base_theta
            
            return round(theta_dollars, 4)
        except:
            return -0.01  # Default small theta
    
    def _estimate_gamma(self, spot: float, strike: float, days: int, iv: float) -> float:
        """Estimate gamma"""
        try:
            # Gamma highest at ATM
            moneyness = spot / strike
            time_to_exp = days / 365.0
            
            # ATM factor
            atm_factor = np.exp(-((moneyness - 1) ** 2) / 0.02)
            
            # Time factor - gamma increases near expiration
            time_factor = 1 / np.sqrt(max(time_to_exp, 0.01))
            
            gamma = 0.01 * atm_factor * time_factor / (spot * iv * np.sqrt(2 * np.pi))
            
            return round(max(0, gamma), 4)
        except:
            return 0.01
    
    def _estimate_vega(self, spot: float, strike: float, days: int, iv: float) -> float:
        """Estimate vega"""
        try:
            # Vega highest for ATM options with time remaining
            moneyness = spot / strike
            time_to_exp = days / 365.0
            
            # ATM factor
            atm_factor = np.exp(-((moneyness - 1) ** 2) / 0.02)
            
            # Time factor - vega decreases near expiration
            time_factor = np.sqrt(time_to_exp)
            
            vega = spot * 0.01 * atm_factor * time_factor * np.sqrt(2 / np.pi)
            
            return round(vega, 3)
        except:
            return 0.1
    
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
                    if ticker and not any(c in ticker for c in ['^', '.', '=']):
                        tickers.append(ticker)
                        
        except Exception as e:
            logger.error(f"Error fetching NASDAQ data: {e}")
            
        return tickers
    
    def _fetch_yahoo_screener(self) -> List[str]:
        """Fetch from Yahoo Finance screener API"""
        tickers = []
        
        try:
            # Get trending and active stocks
            urls = [
                "https://query1.finance.yahoo.com/v1/finance/trending/US",
                "https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=day_gainers&count=100",
                "https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=day_losers&count=100",
                "https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives&count=100"
            ]
            
            for url in urls:
                try:
                    response = self.session.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Different response formats
                        if 'finance' in data:
                            results = data.get('finance', {}).get('result', [])
                            if results and 'quotes' in results[0]:
                                for quote in results[0]['quotes']:
                                    if 'symbol' in quote:
                                        tickers.append(quote['symbol'])
                                        
                except Exception as e:
                    logger.debug(f"Error with URL {url}: {e}")
                    
        except Exception as e:
            logger.error(f"Error fetching Yahoo data: {e}")
            
        return tickers
    
    def clear_cache(self):
        """Clear data cache"""
        self.cache.clear()
        self.cache_expiry.clear()
        
        # Clear file cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            
        logger.info("Data cache cleared")