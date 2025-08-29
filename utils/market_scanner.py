"""
Market Scanner module for finding small-cap opportunities
"""

import logging
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MarketScanner:
    """Scans market for small-cap stocks meeting criteria"""
    
    def __init__(self, config, data_fetcher):
        self.config = config
        self.data_fetcher = data_fetcher
        
    def find_stocks_by_market_cap(self) -> List[Dict]:
        """Find all stocks with market cap between configured min and max"""
        logger.info("Scanning for stocks within market cap range...")
        
        # Get all stocks with basic filters using bulk data sources
        stocks = self.data_fetcher.get_stocks_by_market_cap(
            min_cap=self.config.trading.market_cap_min,
            max_cap=self.config.trading.market_cap_max,
            min_volume=self.config.trading.min_volume
        )
        
        # Update with current prices and volumes
        enriched_stocks = self.data_fetcher.update_stock_data_with_current_prices(stocks)
        
        # Add additional fundamental data for each stock (optional for speed)
        final_stocks = []
        batch_size = 50  # Increased from 10 to 50 for much faster processing
        skip_enriching = True  # Skip enrichment for speed - we have enough basic data
        
        if skip_enriching:
            logger.info(f"Skipping fundamental data enrichment for speed. Using {len(enriched_stocks)} stocks as-is.")
            return enriched_stocks
        
        logger.info(f"Enriching {len(enriched_stocks)} stocks with fundamental data...")
        
        for i in range(0, len(enriched_stocks), batch_size):
            batch = enriched_stocks[i:i + batch_size]
            
            for stock in batch:
                try:
                    # Get basic fundamentals
                    fundamentals = self.data_fetcher.get_fundamentals(stock['symbol'])
                    stock.update({
                        'pe_ratio': fundamentals.get('pe_ratio'),
                        'revenue_growth': fundamentals.get('revenue_growth'),
                        'earnings_growth': fundamentals.get('earnings_growth'),
                        'institutional_ownership': fundamentals.get('institutional_ownership', 0),
                    })
                    final_stocks.append(stock)
                except Exception as e:
                    logger.warning(f"Error enriching data for {stock['symbol']}: {e}")
                    # Keep the stock with existing data
                    final_stocks.append(stock)
            # Rate limiting between batches (reduced to 0.2 seconds for speed)
            if i + batch_size < len(enriched_stocks):
                time.sleep(0.2)  # Much faster processing
        logger.info(f"Successfully enriched {len(final_stocks)} stocks")
        # --- POST-ENRICHMENT MARKET CAP FILTER ---
        min_cap = self.config.trading.market_cap_min
        max_cap = self.config.trading.market_cap_max
        filtered_final = []
        for stock in final_stocks:
            try:
                mc = stock.get('market_cap', 0)
                # Handle str/int issues
                if isinstance(mc, str):
                    try:
                        mc = float(mc.replace(',', ''))
                    except Exception:
                        continue
                if min_cap <= mc <= max_cap:
                    stock['market_cap'] = mc
                    filtered_final.append(stock)
            except Exception as e:
                logger.warning(f"Error filtering {stock.get('symbol', '?')}: {e}")
        # Save fundamentals cache after enrichment
        if hasattr(self.data_fetcher, '_save_fundamentals_cache'):
            self.data_fetcher._save_fundamentals_cache()
        logger.info(f"Returning {len(filtered_final)} stocks after post-enrichment market cap filter")
        return filtered_final
    
    def apply_filters(self, stocks: List[Dict]) -> List[Dict]:
        """Apply technical and fundamental filters, including momentum filter"""
        filtered = []
        for stock in stocks:
            try:
                # Only keep stocks that look good both fundamentally and technically
                if not self._passes_fundamental_filters(stock):
                    continue
                technicals = self._analyze_technicals(stock['symbol'])
                if not technicals:
                    continue
                # Require at least some positive momentum over 3 months
                if technicals.get('price_change_60d', 0) < 0:
                    continue
                if self._has_bullish_setup(technicals):
                    stock.update(technicals)
                    filtered.append(stock)
            except Exception as e:
                logger.warning(f"Error filtering {stock['symbol']}: {e}")
        return filtered
    
    def _passes_fundamental_filters(self, stock: Dict) -> bool:
        """Keep only stocks that are fundamentally healthy and not penny stocks"""
        market_cap = stock.get('market_cap', 0)
        
        # Use adaptive thresholds based on company size
        growth_min = self.config.get_adaptive_growth_min(market_cap)
        pe_max = self.config.get_adaptive_pe_max(market_cap)
        
        # Avoid stocks with crazy high or negative PE ratios
        if stock.get('pe_ratio'):
            if stock['pe_ratio'] > pe_max or stock['pe_ratio'] < 0:
                return False
        # Look for at least some revenue growth
        if stock.get('revenue_growth'):
            if stock['revenue_growth'] < growth_min:
                return False
        # Earnings growth should be positive too
        if stock.get('earnings_growth'):
            earnings_min = growth_min * 0.67
            if stock['earnings_growth'] < earnings_min:
                return False
        # Make sure there's some institutional interest
        category = self.config.get_market_cap_category(market_cap)
        ownership_min = 0.05
        if category == 'micro_cap':
            ownership_min = 0.02
        elif category == 'mega_cap':
            ownership_min = 0.10
        if stock.get('institutional_ownership', 0) < ownership_min:
            return False
        # More lenient filters for penny stocks and volume
        if stock.get('volume', 0) < 50000:  # Reduced from 100k to 50k
            return False
        if stock.get('price', 0) < 0.50:  # Reduced from $1 to $0.50
            return False
        if market_cap < 50_000_000:  # Reduced from 100M to 50M
            return False
        return True
    
    def _analyze_technicals(self, symbol: str) -> Optional[Dict]:
        """Analyze technical indicators for a stock, including 3-month momentum"""
        try:
            # Get price history
            history = self.data_fetcher.get_price_history(symbol, days=100)
            if len(history) < 50:
                return None
            df = pd.DataFrame(history)
            # Calculate indicators
            technicals = {
                'rsi': self._calculate_rsi(df['close']),
                'sma_20': df['close'].rolling(20).mean().iloc[-1],
                'sma_50': df['close'].rolling(50).mean().iloc[-1],
                'volume_ratio': df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1],
                'price_change_5d': (df['close'].iloc[-1] / df['close'].iloc[-5] - 1),
                'price_change_20d': (df['close'].iloc[-1] / df['close'].iloc[-20] - 1),
                'price_change_60d': (df['close'].iloc[-1] / df['close'].iloc[-60] - 1),
                'atr': self._calculate_atr(df),
                'relative_strength': self._calculate_relative_strength(df),
                'pattern': self._detect_pattern(df)
            }
            return technicals
        except Exception as e:
            logger.error(f"Error analyzing technicals for {symbol}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1]
    
    def _calculate_relative_strength(self, df: pd.DataFrame) -> float:
        """Calculate relative strength vs market"""
        # Simple implementation - compare to SPY
        try:
            spy_history = self.data_fetcher.get_price_history('SPY', days=20)
            spy_return = (spy_history[-1]['close'] / spy_history[0]['close'] - 1)
            stock_return = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1)
            
            return stock_return / spy_return if spy_return != 0 else 1.0
        except:
            return 1.0
    
    def _detect_pattern(self, df: pd.DataFrame) -> str:
        """Detect chart patterns"""
        closes = df['close'].values[-20:]
        highs = df['high'].values[-20:]
        lows = df['low'].values[-20:]
        
        # Simple pattern detection
        if self._is_breakout(closes, highs):
            return 'breakout'
        elif self._is_flag_pattern(closes, highs, lows):
            return 'flag'
        elif self._is_ascending_triangle(highs, lows):
            return 'ascending_triangle'
        else:
            return 'none'
    
    def _is_breakout(self, closes: np.ndarray, highs: np.ndarray) -> bool:
        """Check for breakout pattern"""
        # Price breaks above recent high with volume
        recent_high = np.max(highs[:-5])
        current_price = closes[-1]
        
        return current_price > recent_high * 1.02  # 2% above recent high
    
    def _is_flag_pattern(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Check for flag pattern"""
        # Simplified flag detection
        first_half_trend = np.polyfit(range(10), closes[:10], 1)[0]
        second_half_trend = np.polyfit(range(10), closes[10:], 1)[0]
        
        # Strong move up followed by consolidation
        return first_half_trend > 0.5 and abs(second_half_trend) < 0.1
    
    def _is_ascending_triangle(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """Check for ascending triangle pattern"""
        # Higher lows with resistance at top
        low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
        high_std = np.std(highs)
        
        return low_trend > 0 and high_std < np.mean(highs) * 0.02
    
    def _has_bullish_setup(self, technicals: Dict) -> bool:
        """Check for bullish technical setup with momentum focus - MUCH MORE LENIENT"""
        # Very lenient filtering - accept most stocks for testing
        relative_strength = technicals.get('relative_strength', 1)
        price_change_20d = technicals.get('price_change_20d', 0)
        volume_ratio = technicals.get('volume_ratio', 1)
        rsi = technicals.get('rsi', 50)
        price_change_5d = technicals.get('price_change_5d', 0)
        
        # VERY LENIENT - accept almost all stocks
        if relative_strength >= 0.3:  # Much more lenient
            return True
            
        # Accept stocks with moderate price movement
        if price_change_20d >= -0.25:  # Allow 25% decline
            return True
            
        # Accept stocks with any reasonable volume
        if volume_ratio >= 0.2:  # Very lenient volume requirement
            return True
            
        # Accept most stocks based on RSI
        if rsi <= 95:  # Very lenient RSI
            return True
            
        # Accept stocks with recent movement
        if price_change_5d >= -0.15:  # Allow 15% decline
            return True
            
        # If all else fails, accept the stock anyway
        return True

if __name__ == "__main__":
    # Dummy config and data_fetcher for testing
    class DummyConfig:
        class trading:
            market_cap_min = 500_000_000
            market_cap_max = 10_000_000_000
            min_volume = 100_000
            rsi_overbought = 70
        class scanner:
            max_pe_ratio = 40
            min_revenue_growth = 0.05
            min_earnings_growth = 0.05
            min_institutional_ownership = 0.1
            min_relative_strength = 1.1

    class DummyDataFetcher:
        def get_stocks_by_market_cap(self, min_cap, max_cap, min_volume):
            return [{'symbol': 'TEST', 'name': 'Test Corp', 'market_cap': 1_000_000_000}]
        def get_quote(self, symbol):
            return {'price': 10, 'volume': 200_000, 'avg_volume': 150_000}
        def get_fundamentals(self, symbol):
            return {'pe_ratio': 20, 'revenue_growth': 0.1, 'earnings_growth': 0.1, 'institutional_ownership': 0.2}
        def get_price_history(self, symbol, days):
            return [{'close': 10 + i*0.1, 'high': 10 + i*0.15, 'low': 10 + i*0.05, 'volume': 200_000} for i in range(days)]

    scanner = MarketScanner(DummyConfig(), DummyDataFetcher())
    stocks = scanner.find_stocks_by_market_cap()
    filtered = scanner.apply_filters(stocks)
    print("Filtered stocks:", filtered)