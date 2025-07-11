"""
Options Analyzer module for analyzing and recommending options trades
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
import math

logger = logging.getLogger(__name__)


class OptionsAnalyzer:
    """Analyzes options chains and generates trading signals"""
    
    def __init__(self, config, data_fetcher):
        self.config = config
        self.data_fetcher = data_fetcher
        
    def analyze_stock(self, stock: Dict) -> Optional[Dict]:
        """Analyze options for a stock and generate buy signal if appropriate"""
        try:
            symbol = stock['symbol']
            current_price = stock['price']
            
            # Get options chain
            options_chain = self.data_fetcher.get_options_chain(symbol)
            if not options_chain:
                return None
                
            # Find optimal option contract
            optimal_contract = self._find_optimal_contract(
                options_chain, 
                current_price,
                stock
            )
            
            if not optimal_contract:
                return None
                
            # Calculate probability of profit
            prob_profit = self._calculate_probability_of_profit(
                current_price,
                optimal_contract['strike'],
                optimal_contract['days_to_expiration'],
                optimal_contract['implied_volatility'],
                optimal_contract['ask']
            )
            
            # Check if meets minimum criteria
            if prob_profit < 0.60:  # Require 60% probability of profit
                return None
                
            # Build recommendation
            signal = {
                'symbol': symbol,
                'recommendation': 'BUY',
                'contract_type': 'CALL',
                'strike': optimal_contract['strike'],
                'expiration': optimal_contract['expiration'],
                'days_to_expiration': optimal_contract['days_to_expiration'],
                'ask_price': optimal_contract['ask'],
                'bid_price': optimal_contract['bid'],
                'mid_price': (optimal_contract['ask'] + optimal_contract['bid']) / 2,
                'implied_volatility': optimal_contract['implied_volatility'],
                'delta': optimal_contract['delta'],
                'theta': optimal_contract['theta'],
                'gamma': optimal_contract['gamma'],
                'vega': optimal_contract['vega'],
                'volume': optimal_contract['volume'],
                'open_interest': optimal_contract['open_interest'],
                'probability_of_profit': prob_profit,
                'expected_return': self._calculate_expected_return(
                    current_price, 
                    optimal_contract,
                    stock.get('atr', current_price * 0.02)
                ),
                'risk_reward_ratio': self._calculate_risk_reward(optimal_contract),
                'entry_price': optimal_contract['ask'],  # Use ask for conservative entry
                'stop_loss': optimal_contract['ask'] * (1 - self.config.trading.stop_loss_percent),
                'take_profit': optimal_contract['ask'] * (1 + self.config.trading.take_profit_percent),
                'position_score': self._calculate_position_score(
                    optimal_contract, 
                    prob_profit, 
                    stock
                ),
                'timestamp': datetime.now().isoformat()
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing options for {stock['symbol']}: {e}")
            return None
    
    def _find_optimal_contract(self, options_chain: List[Dict], 
                              current_price: float, 
                              stock: Dict) -> Optional[Dict]:
        """Find the optimal call option contract"""
        
        # Filter for calls only
        calls = [opt for opt in options_chain if opt['type'] == 'CALL']
        
        # Apply filters
        filtered_calls = []
        for call in calls:
            # Days to expiration filter
            if (call['days_to_expiration'] < self.config.trading.min_days_to_expiration or
                call['days_to_expiration'] > self.config.trading.max_days_to_expiration):
                continue
                
            # Strike price filter (slightly OTM to ATM)
            if call['strike'] < current_price or call['strike'] > current_price * 1.10:
                continue
                
            # Liquidity filters
            if (call['volume'] < self.config.trading.min_option_volume or
                call['open_interest'] < self.config.trading.min_option_oi):
                continue
                
            # Delta filter
            if call['delta'] < self.config.trading.min_delta:
                continue
                
            # IV percentile filter
            if call.get('iv_percentile', 0) > self.config.trading.max_iv_percentile:
                continue
                
            filtered_calls.append(call)
        
        if not filtered_calls:
            return None
            
        # Score and rank contracts
        scored_calls = []
        for call in filtered_calls:
            score = self._score_contract(call, current_price, stock)
            call['score'] = score
            scored_calls.append(call)
            
        # Sort by score and return best
        scored_calls.sort(key=lambda x: x['score'], reverse=True)
        return scored_calls[0]
    
    def _score_contract(self, contract: Dict, current_price: float, stock: Dict) -> float:
        """Score an option contract based on multiple factors"""
        score = 0.0
        
        # Moneyness score (prefer slightly OTM)
        moneyness = (contract['strike'] - current_price) / current_price
        if 0 <= moneyness <= 0.05:
            score += 30
        elif 0.05 < moneyness <= 0.10:
            score += 20
        else:
            score += 10
            
        # Days to expiration score (prefer target)
        dte_diff = abs(contract['days_to_expiration'] - self.config.trading.target_days_to_expiration)
        score += max(0, 20 - dte_diff)
        
        # Liquidity score
        liquidity_score = min(20, (contract['volume'] / 100) + (contract['open_interest'] / 500))
        score += liquidity_score
        
        # Greeks score
        if contract['delta'] >= 0.30:
            score += 10
        if contract['theta'] / contract['ask'] < self.config.trading.max_theta_decay_daily:
            score += 10
            
        # IV score (lower is better)
        iv_score = max(0, 20 - (contract.get('iv_percentile', 50) / 5))
        score += iv_score
        
        return score
    
    def _calculate_probability_of_profit(self, spot: float, strike: float, 
                                       days: int, iv: float, premium: float) -> float:
        """Calculate probability of profit for a call option"""
        # Break-even price
        breakeven = strike + premium
        
        # Convert IV to daily
        daily_vol = iv / math.sqrt(252)
        
        # Calculate probability using log-normal distribution
        expected_return = 0  # Assume neutral
        variance = (daily_vol ** 2) * days
        
        d1 = (math.log(spot / breakeven) + expected_return + 0.5 * variance) / math.sqrt(variance)
        
        # Probability that price > breakeven
        prob = norm.cdf(d1)
        
        return prob
    
    def _calculate_expected_return(self, spot: float, contract: Dict, atr: float) -> float:
        """Calculate expected return based on price targets"""
        strike = contract['strike']
        premium = contract['ask']
        
        # Simple expected move based on ATR
        expected_move = atr * math.sqrt(contract['days_to_expiration'] / 20)  # Scale ATR by time
        expected_price = spot + expected_move * 0.7  # Conservative estimate
        
        if expected_price > strike:
            profit = expected_price - strike - premium
            return profit / premium
        else:
            return -1.0  # Total loss
    
    def _calculate_risk_reward(self, contract: Dict) -> float:
        """Calculate risk/reward ratio"""
        max_loss = contract['ask']
        
        # Potential profit at 50% move
        potential_profit = contract['ask'] * self.config.trading.take_profit_percent
        
        return potential_profit / max_loss
    
    def _calculate_position_score(self, contract: Dict, prob_profit: float, stock: Dict) -> float:
        """Calculate overall position score for ranking"""
        score = 0.0
        
        # Probability weight (40%)
        score += prob_profit * 40
        
        # Technical setup weight (30%)
        if stock.get('pattern') in ['breakout', 'flag']:
            score += 20
        elif stock.get('pattern') == 'ascending_triangle':
            score += 15
        else:
            score += 10
            
        # Momentum weight (20%)
        if stock.get('relative_strength', 1.0) > 1.5:
            score += 20
        elif stock.get('relative_strength', 1.0) > 1.2:
            score += 15
        else:
            score += 10
            
        # Greeks weight (10%)
        if contract['delta'] > 0.35 and contract['theta'] / contract['ask'] < 0.015:
            score += 10
        else:
            score += 5
            
        return score
    
    def evaluate_position(self, position: Dict) -> Dict:
        """Evaluate an existing position for exit signals"""
        try:
            symbol = position['symbol']
            
            # Get current option data
            current_contract = self.data_fetcher.get_option_quote(
                symbol,
                position['strike'],
                position['expiration'],
                'CALL'
            )
            
            if not current_contract:
                return {'action': 'HOLD', 'reason': 'Unable to get current data'}
                
            # Get current stock price
            stock_quote = self.data_fetcher.get_quote(symbol)
            current_stock_price = stock_quote['price']
            
            # Calculate current metrics
            days_held = (datetime.now() - datetime.fromisoformat(position['entry_date'])).days
            days_to_expiration = (datetime.fromisoformat(position['expiration']) - datetime.now()).days
            
            # Current P&L
            entry_price = position['entry_price']
            current_price = (current_contract['bid'] + current_contract['ask']) / 2
            pnl_percent = (current_price - entry_price) / entry_price
            
            # Update position data
            position['current_price'] = current_price
            position['unrealized_pnl'] = current_price - entry_price
            position['pnl_percent'] = pnl_percent * 100
            position['theta'] = current_contract['theta']
            position['iv'] = current_contract['implied_volatility']
            
            # Check exit conditions
            exit_signal = self._check_exit_conditions(
                position,
                current_contract,
                current_stock_price,
                days_to_expiration,
                pnl_percent
            )
            
            return exit_signal
            
        except Exception as e:
            logger.error(f"Error evaluating position {position['symbol']}: {e}")
            return {'action': 'HOLD', 'reason': 'Error in evaluation'}
    
    def _check_exit_conditions(self, position: Dict, current_contract: Dict,
                             stock_price: float, days_to_exp: int, pnl_percent: float) -> Dict:
        """Check if position meets exit criteria"""
        
        # Take profit hit
        if pnl_percent >= self.config.trading.profit_exit_threshold:
            return {
                'action': 'SELL',
                'reason': f'Take profit target reached ({pnl_percent:.1%})',
                'symbol': position['symbol'],
                'current_price': current_contract['mid']
            }
            
        # Stop loss hit
        if pnl_percent <= -self.config.trading.stop_loss_percent:
            return {
                'action': 'SELL',
                'reason': f'Stop loss triggered ({pnl_percent:.1%})',
                'symbol': position['symbol'],
                'current_price': current_contract['mid']
            }
            
        # Time decay exit
        if days_to_exp <= self.config.trading.days_before_exp_exit:
            return {
                'action': 'SELL',
                'reason': f'Approaching expiration ({days_to_exp} days left)',
                'symbol': position['symbol'],
                'current_price': current_contract['mid']
            }
            
        # Theta decay too high
        theta_decay_rate = abs(current_contract['theta']) / current_contract['mid']
        if theta_decay_rate > self.config.trading.theta_exit_threshold:
            return {
                'action': 'SELL',
                'reason': f'Theta decay too high ({theta_decay_rate:.1%} daily)',
                'symbol': position['symbol'],
                'current_price': current_contract['mid']
            }
            
        # IV spike exit (potential to sell high IV)
        if 'entry_iv' in position:
            iv_change = current_contract['implied_volatility'] / position['entry_iv']
            if iv_change >= self.config.trading.iv_spike_exit:
                return {
                    'action': 'SELL',
                    'reason': f'IV spike ({iv_change:.1f}x increase)',
                    'symbol': position['symbol'],
                    'current_price': current_contract['mid']
                }
                
        # Check for roll opportunity
        if days_to_exp <= 15 and pnl_percent > 0.20:
            # Profitable position approaching expiration - consider rolling
            return {
                'action': 'ROLL',
                'reason': 'Profitable position to roll forward',
                'symbol': position['symbol'],
                'current_price': current_contract['mid'],
                'new_expiration': self._suggest_roll_expiration(),
                'new_strike': self._suggest_roll_strike(stock_price, position['strike'])
            }
            
        # Default: Hold position
        return {
            'action': 'HOLD',
            'reason': 'Position within normal parameters',
            'days_to_exp': days_to_exp,
            'pnl_percent': pnl_percent,
            'theta_decay': theta_decay_rate
        }
    
    def _suggest_roll_expiration(self) -> str:
        """Suggest expiration date for rolling position"""
        target_date = datetime.now() + timedelta(days=self.config.trading.target_days_to_expiration)
        # Round to Friday (options expiration)
        days_to_friday = (4 - target_date.weekday()) % 7
        target_date += timedelta(days=days_to_friday)
        return target_date.strftime('%Y-%m-%d')
    
    def _suggest_roll_strike(self, current_price: float, old_strike: float) -> float:
        """Suggest strike price for rolling position"""
        # If stock has moved up significantly, adjust strike up
        if current_price > old_strike * 1.05:
            # Round to nearest $5 increment
            new_strike = round(current_price * 1.03 / 5) * 5
            return new_strike
        else:
            return old_strike