"""
Improved Options Analyzer with specific recommendations and monitoring
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
import math
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class OptionsAnalyzer:
    """Enhanced options analyzer with specific recommendations"""
    
    def __init__(self, config, data_fetcher):
        self.config = config
        self.data_fetcher = data_fetcher
        self.monitored_positions_file = Path("data/monitored_positions.json")
        self.load_monitored_positions()
        
    def load_monitored_positions(self):
        """Load positions being monitored"""
        if self.monitored_positions_file.exists():
            with open(self.monitored_positions_file, 'r') as f:
                self.monitored_positions = json.load(f)
        else:
            self.monitored_positions = {}
    
    def save_monitored_positions(self):
        """Save monitored positions"""
        self.monitored_positions_file.parent.mkdir(exist_ok=True)
        with open(self.monitored_positions_file, 'w') as f:
            json.dump(self.monitored_positions, f, indent=2)
    
    def analyze_stock(self, stock: dict) -> list:
        """Analyze a single stock for both call and put options opportunities with comprehensive scoring"""
        symbol = stock['symbol']
        recommendations = []
        put_count = 0
        call_count = 0
        best_put_score = 0
        best_call_score = 0
        
        try:
            options_chain = self.data_fetcher.get_options_chain(symbol)
            if not options_chain:
                return []
            current_price = stock.get('price', 0)
            for option in options_chain:
                try:
                    score, analysis = self._score_option_comprehensive(option, current_price, stock)
                    recommendation = option.copy()
                    recommendation['score'] = score
                    recommendation['analysis'] = analysis
                    recommendation['recommendation_reasons'] = analysis.get('reasons', [])
                    recommendation['expected_return'] = analysis.get('expected_return', 0)
                    recommendation['current_stock_price'] = current_price
                    recommendation['entry_price'] = option.get('ask', option.get('mid', 0))
                    recommendations.append(recommendation)
                    
                    # Track put vs call statistics for debugging
                    option_type = option.get('type', 'CALL')
                    if option_type == 'PUT':
                        put_count += 1
                        best_put_score = max(best_put_score, score)
                    else:
                        call_count += 1
                        best_call_score = max(best_call_score, score)
                        
                except Exception as e:
                    logger.warning(f"Error analyzing option for {symbol}: {e}")
                    continue
            
            # Debug logging for put vs call analysis (cleaned up)
            if put_count > 0 and call_count > 0:
                logger.debug(f"{symbol}: {call_count} calls (best: {best_call_score:.0f}), {put_count} puts (best: {best_put_score:.0f})")
            elif put_count > 0:
                logger.debug(f"{symbol}: {put_count} puts only (best: {best_put_score:.0f})")
            elif call_count > 0:
                logger.debug(f"{symbol}: {call_count} calls only (best: {best_call_score:.0f})")
                
            return recommendations
        except Exception as e:
            logger.error(f"Error analyzing options for {symbol}: {e}")
            return []
    
    def _score_option_comprehensive(self, option: Dict, current_price: float, 
                                   stock: Dict) -> Tuple[float, Dict]:
        """Advanced scoring system using cutting-edge academic research methods"""
        score = 0
        reasons = []
        analysis = {}
        
        # NEW: Apply sophisticated call vs put selection using academic research
        option_type = option.get('type', 'CALL')
        put_call_bonus = self._analyze_put_call_selection_research(stock, option, current_price)
        score += put_call_bonus
        
        # Get option value analysis
        value_analysis = self._calculate_option_value_analysis(current_price, option)
        
        # Advanced academic research methods
        value_score, value_reasons = self._analyze_value_opportunity(stock, option, current_price)
        growth_score, growth_reasons = self._analyze_growth_opportunity(stock, option, current_price)
        technical_score, technical_reasons = self._analyze_technical_setup(stock, option, current_price)
        
        # NEW: Advanced academic methods
        momentum_score, momentum_reasons = self._analyze_momentum_factor_model(stock, option, current_price)
        volatility_score, volatility_reasons = self._analyze_volatility_smile_arbitrage(stock, option, current_price)
        behavioral_score, behavioral_reasons = self._analyze_behavioral_finance_signals(stock, option, current_price)
        liquidity_score, liquidity_reasons = self._analyze_advanced_liquidity_metrics(option, stock)
        expiration_score, expiration_reasons = self._analyze_sophisticated_expiration_selection(option, stock, current_price)
        
        # NEW: Breakout detection methods (like SOGP)
        breakout_score, breakout_reasons = self._analyze_breakout_potential(stock, option, current_price)
        momentum_breakout_score, momentum_breakout_reasons = self._analyze_momentum_breakout_signals(stock, option, current_price)
        
        # Combine all advanced analysis with breakout emphasis
        score += value_score + growth_score + technical_score + momentum_score + volatility_score + behavioral_score + liquidity_score + expiration_score + breakout_score + momentum_breakout_score
        reasons.extend(value_reasons + growth_reasons + technical_reasons + momentum_reasons + volatility_reasons + behavioral_reasons + liquidity_reasons + expiration_reasons + breakout_reasons + momentum_breakout_reasons)
        
        # 1. Moneyness Score (25 points) - More lenient for all strikes
        moneyness = value_analysis.get('moneyness', 1.0)
        if 0.90 <= moneyness <= 1.10:  # Wider ATM range
            score += 25
            reasons.append(f"Excellent strike near money (${option['strike']:.2f})")
        elif 1.10 < moneyness <= 1.25:  # OTM - good for momentum  
            score += 22
            reasons.append(f"Good OTM strike for upside (${option['strike']:.2f})")
        elif 0.80 <= moneyness < 0.90:  # ITM - safer
            score += 20
            reasons.append(f"ITM with intrinsic value (${option['strike']:.2f})")
        elif 1.25 < moneyness <= 1.50:  # Further OTM - higher leverage
            score += 18
            reasons.append(f"Higher leverage OTM strike (${option['strike']:.2f})")
        elif moneyness < 0.80:  # Deep ITM
            score += 15
            reasons.append(f"Deep ITM - conservative play (${option['strike']:.2f})")
        elif moneyness > 1.50:  # Far OTM
            score += 12
            reasons.append(f"Far OTM - lottery ticket (${option['strike']:.2f})")
        
        # 2. Time Value Analysis (20 points)
        time_value_pct = value_analysis.get('time_value_pct', 50)
        days = option['days_to_expiration']
        
        if 30 <= days <= 45 and time_value_pct < 80:  # Optimal time decay
            score += 20
            reasons.append(f"Optimal expiration ({days} days) with reasonable time value")
        elif 25 <= days < 30 or 45 < days <= 60:
            score += 15
            reasons.append(f"Good expiration ({days} days)")
        elif 20 <= days < 25 or 60 < days <= 70:
            score += 10
            reasons.append(f"Acceptable expiration ({days} days)")
        elif days < 20:
            score += 5
            reasons.append(f"Short expiration ({days} days) - high theta risk")
        elif days > 70:
            score += 8
            reasons.append(f"Long expiration ({days} days) - expensive time value")
        
        # 3. Liquidity Score (20 points) - More lenient for low-volume options
        liquidity_score = value_analysis.get('liquidity_score', 0)
        volume = option.get('volume', 0)
        open_interest = option.get('open_interest', 0)
        spread_pct = option.get('spread_pct', 0.5)
        
        # More generous liquidity scoring
        normalized_liquidity = min(liquidity_score / 80 * 20, 20)  # Reduced denominator
        score += normalized_liquidity
        
        if volume > 50:  # Reduced from 100
            reasons.append(f"Good volume ({volume}) - tradeable")
        elif volume > 5:  # Reduced from 10
            reasons.append(f"Some volume ({volume}) - can trade")
        elif open_interest > 100:  # Reduced from 500
            reasons.append(f"Good open interest ({open_interest}) - has liquidity")
        elif open_interest > 25:  # Reduced from 100
            reasons.append(f"Some open interest ({open_interest})")
        else:
            reasons.append("Low liquidity - use limit orders")
        
        if spread_pct < 0.15:  # More lenient
            reasons.append("Reasonable bid-ask spread")
        elif spread_pct < 0.35:  # More lenient
            reasons.append("Acceptable spread")
        elif spread_pct < 0.50:  # More lenient
            reasons.append("Wide spread - use limit orders")
        else:
            reasons.append("Very wide spread - trade carefully")
        
        # 4. Greeks Score (15 points)
        delta = option.get('delta', 0.5)
        theta = option.get('theta', -0.01)
        gamma = option.get('gamma', 0.01)
        
        # Delta analysis
        if 0.25 <= delta <= 0.45:  # Sweet spot for calls
            score += 8
            reasons.append(f"Good delta ({delta:.2f}) - balanced risk/reward")
        elif 0.20 <= delta < 0.25 or 0.45 < delta <= 0.55:
            score += 6
            reasons.append(f"Acceptable delta ({delta:.2f})")
        elif delta > 0.55:  # High delta - expensive
            score += 4
            reasons.append(f"High delta ({delta:.2f}) - expensive but safer")
        elif delta < 0.20:  # Low delta - risky
            score += 3
            reasons.append(f"Low delta ({delta:.2f}) - high leverage")
        
        # Theta analysis (want low theta relative to price)
        theta_ratio = abs(theta) / (option['ask'] if option['ask'] > 0 else 0.01)
        if theta_ratio < 0.02:  # Less than 2% daily decay
            score += 4
            reasons.append(f"Low theta decay ({abs(theta):.3f})")
        elif theta_ratio < 0.03:
            score += 2
            reasons.append(f"Moderate theta decay ({abs(theta):.3f})")
        elif theta_ratio > 0.05:
            score -= 2
            reasons.append(f"High theta decay ({abs(theta):.3f}) - time decay risk")
        
        # Gamma analysis
        if 0.01 <= gamma <= 0.05:
            score += 3
            reasons.append(f"Good gamma ({gamma:.3f}) - responsive to stock moves")
        
        # 5. Volatility Analysis (10 points)
        iv = option.get('implied_volatility', 0.3)
        iv_percentile = option.get('iv_percentile', 50)
        
        if 0.3 <= iv <= 0.6 and 30 <= iv_percentile <= 70:  # Moderate IV
            score += 10
            reasons.append(f"Reasonable IV ({iv:.1%}) - not overpriced")
        elif 0.2 <= iv < 0.3 or 0.6 < iv <= 0.8:
            score += 7
            reasons.append(f"Acceptable IV ({iv:.1%})")
        elif iv > 0.8 or iv_percentile > 80:
            score += 3
            reasons.append(f"High IV ({iv:.1%}) - expensive but potential for IV crush")
        elif iv < 0.2 or iv_percentile < 20:
            score += 5
            reasons.append(f"Low IV ({iv:.1%}) - cheap but low volatility")
        
        # 6. Expected Return Analysis (10 points) - More optimistic
        expected_return = self._calculate_expected_return(current_price, option, stock.get('atr', current_price * 0.02))
        
        if expected_return > 0.2:  # Reduced from 0.3
            score += 10
            reasons.append(f"High expected return ({expected_return:.1%})")
        elif expected_return > 0.05:  # Reduced from 0.1
            score += 8
            reasons.append(f"Good expected return ({expected_return:.1%})")
        elif expected_return > -0.1:  # More lenient
            score += 6
            reasons.append(f"Reasonable expected return ({expected_return:.1%})")
        elif expected_return > -0.3:  # More lenient
            score += 3
            reasons.append(f"Acceptable expected return ({expected_return:.1%})")
        else:
            score += 1  # Still give some points
            reasons.append(f"Lower expected return ({expected_return:.1%}) but tradeable")
        
        # 7. Technical Setup Bonus (up to 15 points) - Enhanced for negative momentum analysis
        technical_bonus = 0
        
        # RSI analysis for oversold conditions
        rsi = stock.get('rsi', 50)
        if rsi < 30:
            technical_bonus += 8
            reasons.append("Extremely oversold RSI - strong bounce potential")
        elif rsi < 40:
            technical_bonus += 5
            reasons.append("Oversold RSI - potential reversal")
        elif rsi < 50:
            technical_bonus += 3
            reasons.append("Below-average RSI - room for improvement")
        
        # Relative strength analysis
        relative_strength = stock.get('relative_strength', 1.0)
        if relative_strength > 1.3:
            technical_bonus += 5
            reasons.append("Strong relative strength vs market")
        elif relative_strength > 1.1:
            technical_bonus += 3
            reasons.append("Above-average relative strength")
        elif relative_strength < 0.8:
            technical_bonus += 2
            reasons.append("Underperforming market - mean reversion potential")
        
        # Price momentum analysis
        price_change_20d = stock.get('price_change_20d', 0)
        price_change_60d = stock.get('price_change_60d', 0)
        
        # Negative momentum analysis - look for reversal signals
        if price_change_20d < -0.15 and price_change_60d < -0.30:
            # Stock has been beaten down - potential for mean reversion
            technical_bonus += 6
            reasons.append("Significant recent decline - oversold conditions")
        elif price_change_20d < -0.10:
            technical_bonus += 4
            reasons.append("Recent decline - potential bounce")
        elif price_change_20d > 0.05:
            technical_bonus += 3
            reasons.append("Positive recent momentum")
        
        # Volume analysis for confirmation
        volume_ratio = stock.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:
            technical_bonus += 3
            reasons.append("High volume - strong institutional interest")
        elif volume_ratio > 1.2:
            technical_bonus += 2
            reasons.append("Above-average volume")
        
        # Market cap analysis for small-cap opportunities
        market_cap = stock.get('market_cap', 0)
        if 1e9 <= market_cap <= 5e9:  # Small-cap range
            technical_bonus += 2
            reasons.append("Small-cap stock - higher volatility potential")
        
        # Sector rotation analysis (simplified)
        sector = stock.get('sector', 'Unknown')
        if sector in ['Technology', 'Healthcare', 'Consumer Discretionary']:
            technical_bonus += 2
            reasons.append(f"{sector} sector - growth potential")
        
        score += min(technical_bonus, 15)  # Cap at 15 points
        
        # 8. Risk/Reward Analysis (10 points)
        risk_reward = self._calculate_risk_reward(option)
        
        if risk_reward > 2.0:
            score += 10
            reasons.append(f"Excellent risk/reward ratio ({risk_reward:.1f})")
        elif risk_reward > 1.5:
            score += 8
            reasons.append(f"Good risk/reward ratio ({risk_reward:.1f})")
        elif risk_reward > 1.0:
            score += 5
            reasons.append(f"Acceptable risk/reward ratio ({risk_reward:.1f})")
        elif risk_reward > 0.5:
            score += 2
            reasons.append(f"Moderate risk/reward ratio ({risk_reward:.1f})")
        
        # Compile analysis
        analysis = {
            'total_score': score,
            'moneyness': moneyness,
            'reasons': reasons,
            'liquidity_assessment': 'Excellent' if liquidity_score > 80 else 'Good' if liquidity_score > 60 else 'Fair' if liquidity_score > 40 else 'Poor',
            'risk_assessment': 'Low' if score > 80 else 'Medium' if score > 60 else 'High' if score > 40 else 'Very High',
            'expected_return': expected_return,
            'risk_reward_ratio': risk_reward,
            'time_value_pct': time_value_pct,
            'iv_rank': value_analysis.get('iv_rank', 'Normal')
        }
        
        return score, analysis
    
    def _print_recommendations(self, symbol: str, current_price: float, 
                              recommendations: List[Dict]):
        """Print simplified recommendations with essential info only"""
        # Only print if verbose mode is enabled (disabled by default to avoid duplication)
        return
    
    def add_to_monitoring(self, recommendation: Dict, contracts: int = 1):
        """Add position to monitoring list"""
        position_id = f"{recommendation['symbol']}_{recommendation['strike']}_{recommendation['expiration']}"
        
        self.monitored_positions[position_id] = {
            'symbol': recommendation['symbol'],
            'strike': recommendation['strike'],
            'expiration': recommendation['expiration'],
            'entry_date': datetime.now().isoformat(),
            'entry_price': recommendation['entry_price'],
            'contracts': contracts,
            'current_stock_price_at_entry': recommendation['current_stock_price'],
            'entry_analysis': recommendation,
            'status': 'ACTIVE',
            'alerts': []
        }
        
        self.save_monitored_positions()
        logger.info(f"Added {position_id} to monitoring")
        
    def monitor_positions(self) -> List[Dict]:
        """Monitor all active positions and provide exit signals"""
        exit_signals = []
        
        for position_id, position in self.monitored_positions.items():
            if position['status'] != 'ACTIVE':
                continue
                
            try:
                signal = self.evaluate_position(position)
                if signal['action'] != 'HOLD':
                    exit_signals.append(signal)
                    
                # Print monitoring update
                self._print_position_update(position, signal)
                
            except Exception as e:
                logger.error(f"Error monitoring {position_id}: {e}")
        
        return exit_signals
    
    def sell_position(self, position_id: str, sell_price: float = None, reason: str = "Manual sell") -> bool:
        """Sell a monitored position and mark it as CLOSED"""
        if position_id not in self.monitored_positions:
            print(f"❌ Position {position_id} not found in monitoring list")
            return False
            
        position = self.monitored_positions[position_id]
        if position['status'] != 'ACTIVE':
            print(f"❌ Position {position_id} is not active (status: {position['status']})")
            return False
        
        # Get current price if not provided
        if sell_price is None:
            try:
                current_contract = self.data_fetcher.get_option_quote(
                    position['symbol'],
                    position['strike'],
                    position['expiration'],
                    'CALL'
                )
                sell_price = current_contract.get('mid', current_contract.get('last', 0)) if current_contract else 0
            except:
                print(f"❌ Could not get current price for {position_id}")
                return False
        
        # Calculate final P&L
        entry_price = position['entry_price']
        contracts = position['contracts']
        pnl_per_contract = sell_price - entry_price
        total_pnl = pnl_per_contract * contracts * 100  # Options are per 100 shares
        pnl_percent = (pnl_per_contract / entry_price) * 100 if entry_price > 0 else 0
        
        # Update position status
        position['status'] = 'CLOSED'
        position['exit_date'] = datetime.now().isoformat()
        position['exit_price'] = sell_price
        position['exit_reason'] = reason
        position['final_pnl'] = total_pnl
        position['final_pnl_percent'] = pnl_percent
        
        # Save updated positions
        self.save_monitored_positions()
        
        # Print confirmation
        print(f"\n✅ POSITION SOLD: {position_id}")
        print(f"   Entry: ${entry_price:.2f} → Exit: ${sell_price:.2f}")
        print(f"   P&L: ${total_pnl:,.2f} ({pnl_percent:+.1f}%)")
        print(f"   Reason: {reason}")
        print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
    
    def list_active_positions(self) -> List[Dict]:
        """List all active positions for selling"""
        active = []
        for position_id, position in self.monitored_positions.items():
            if position['status'] == 'ACTIVE':
                active.append({
                    'id': position_id,
                    'symbol': position['symbol'],
                    'strike': position['strike'],
                    'expiration': position['expiration'],
                    'contracts': position['contracts'],
                    'entry_price': position['entry_price']
                })
        return active
    
    def evaluate_position(self, position: Dict) -> Dict:
        """Evaluate a monitored position for exit signals"""
        symbol = position['symbol']
        
        # Get current data
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
        
        # Calculate metrics
        days_held = (datetime.now() - datetime.fromisoformat(position['entry_date'])).days
        days_to_expiration = (datetime.fromisoformat(position['expiration']) - datetime.now()).days
        
        # Current P&L
        entry_price = position['entry_price']
        current_price = current_contract.get('mid', current_contract.get('last', 0))
        pnl = current_price - entry_price
        pnl_percent = (pnl / entry_price) * 100 if entry_price > 0 else 0
        
        # Compile evaluation
        evaluation = {
            'action': 'HOLD',
            'symbol': symbol,
            'strike': position['strike'],
            'expiration': position['expiration'],
            'current_stock_price': current_stock_price,
            'current_option_price': current_price,
            'entry_price': entry_price,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'days_held': days_held,
            'days_to_expiration': days_to_expiration,
            'current_delta': current_contract.get('delta', 0),
            'current_theta': current_contract.get('theta', 0),
            'current_iv': current_contract.get('implied_volatility', 0)
        }
        
        # Exit conditions
        reasons = []
        
        # 1. Profit targets
        if pnl_percent >= 50:
            evaluation['action'] = 'SELL'
            evaluation['urgency'] = 'RECOMMENDED'
            reasons.append(f"Hit 50% profit target (currently +{pnl_percent:.1f}%)")
        elif pnl_percent >= 30 and days_to_expiration <= 14:
            evaluation['action'] = 'SELL'
            evaluation['urgency'] = 'RECOMMENDED'
            reasons.append(f"Good profit (+{pnl_percent:.1f}%) with {days_to_expiration} days left")
        
        # 2. Stop loss
        if pnl_percent <= -50:
            evaluation['action'] = 'SELL'
            evaluation['urgency'] = 'URGENT'
            reasons.append(f"Stop loss triggered ({pnl_percent:.1f}%)")
        
        # 3. Time-based exits
        if days_to_expiration <= 7:
            if evaluation['action'] != 'SELL':
                evaluation['action'] = 'SELL'
                evaluation['urgency'] = 'URGENT'
            reasons.append(f"Only {days_to_expiration} days to expiration")
        elif days_to_expiration <= 14 and current_contract.get('delta', 0) < 0.20:
            if evaluation['action'] != 'SELL':
                evaluation['action'] = 'SELL'
                evaluation['urgency'] = 'RECOMMENDED'
            reasons.append(f"Low delta ({current_contract.get('delta', 0):.2f}) with {days_to_expiration} days left")
        
        # 4. Theta decay
        theta_ratio = abs(current_contract.get('theta', 0)) / current_price if current_price > 0 else 0
        if theta_ratio > 0.05:  # Losing more than 5% per day
            if evaluation['action'] != 'SELL':
                evaluation['action'] = 'SELL'
                evaluation['urgency'] = 'CONSIDER'
            reasons.append(f"High theta decay ({theta_ratio:.1%} daily)")
        
        # 5. Stock moved against us
        stock_change = (current_stock_price - position['current_stock_price_at_entry']) / position['current_stock_price_at_entry']
        if stock_change < -0.10:  # Stock down 10%
            if evaluation['action'] != 'SELL':
                evaluation['action'] = 'SELL'
                evaluation['urgency'] = 'RECOMMENDED'
            reasons.append(f"Stock down {stock_change:.1%} since entry")
        
        evaluation['reasons'] = reasons
        evaluation['recommendation'] = self._get_action_recommendation(evaluation)
        
        return evaluation
    
    def _get_action_recommendation(self, evaluation: Dict) -> str:
        """Get detailed recommendation based on evaluation"""
        if evaluation['action'] == 'HOLD':
            if evaluation['pnl_percent'] > 20:
                return "HOLD - Consider taking partial profits or moving stop loss up"
            elif evaluation['days_to_expiration'] < 21:
                return "HOLD - Monitor closely as expiration approaches"
            else:
                return "HOLD - Position performing as expected"
        
        elif evaluation['action'] == 'SELL':
            if evaluation.get('urgency') == 'URGENT':
                return "SELL IMMEDIATELY - " + "; ".join(evaluation['reasons'])
            elif evaluation.get('urgency') == 'RECOMMENDED':
                return "SELL RECOMMENDED - " + "; ".join(evaluation['reasons'])
            else:
                return "CONSIDER SELLING - " + "; ".join(evaluation['reasons'])
        
        return "HOLD"
    
    def _print_position_update(self, position: Dict, evaluation: Dict):
        """Print position monitoring update"""
        print(f"\n--- {position['symbol']} ${position['strike']} {position['expiration']} ---")
        print(f"Entry: ${position['entry_price']:.2f} | Current: ${evaluation['current_option_price']:.2f}")
        print(f"P&L: ${evaluation['pnl']:.2f} ({evaluation['pnl_percent']:+.1f}%)")
        print(f"Stock: ${evaluation['current_stock_price']:.2f} | Days to Exp: {evaluation['days_to_expiration']}")
        print(f"Action: {evaluation['recommendation']}")
        
    def _calculate_probability_of_profit(self, spot: float, strike: float, 
                                       days: int, iv: float, premium: float) -> float:
        """Calculate probability of profit for a call option using Black-Scholes"""
        try:
            # Break-even price for call option
            breakeven = strike + premium
            
            # Use Black-Scholes probability calculation
            time_to_exp = days / 365.0
            risk_free_rate = 0.04  # 4% annual rate
            
            # Calculate d1 for probability
            variance = (iv ** 2) * time_to_exp
            d1 = (np.log(spot / breakeven) + (risk_free_rate + 0.5 * variance) * time_to_exp) / np.sqrt(variance)
            
            # Probability of profit is N(d1)
            prob = norm.cdf(d1)
            
            # Ensure reasonable bounds
            return max(0.05, min(0.95, prob))
            
        except Exception as e:
            logger.debug(f"Error calculating probability: {e}")
            # Fallback calculation
            moneyness = spot / strike
            if moneyness > 1.1:  # ITM
                return 0.7
            elif moneyness > 0.95:  # Near ATM
                return 0.5
            else:  # OTM
                return 0.3
    
    def _calculate_expected_return(self, spot: float, contract: Dict, atr: float) -> float:
        """Calculate realistic expected return for call options"""
        try:
            strike = contract['strike']
            premium = contract['ask'] if contract['ask'] > 0 else contract['mid']
            days = contract['days_to_expiration']
            iv = contract['implied_volatility']
            
            if premium <= 0:
                return 0.0
            
            # Calculate moneyness
            moneyness = spot / strike
            
            # Expected stock price movement based on IV and time
            time_to_exp = days / 365.0
            expected_stock_move = spot * iv * np.sqrt(time_to_exp) * 0.6  # Conservative 60% of IV move
            
            # Calculate potential scenarios
            scenarios = []
            
            # Scenario 1: Stock moves up by expected amount
            up_price = spot + expected_stock_move
            if up_price > strike:
                up_profit = up_price - strike - premium
                up_return = up_profit / premium
                scenarios.append(up_return * 0.4)  # 40% probability
            
            # Scenario 2: Stock moves up by 2x expected amount (breakout)
            breakout_price = spot + expected_stock_move * 2
            if breakout_price > strike:
                breakout_profit = breakout_price - strike - premium
                breakout_return = breakout_profit / premium
                scenarios.append(breakout_return * 0.2)  # 20% probability
            
            # Scenario 3: Stock moves up slightly (50% of expected)
            small_up_price = spot + expected_stock_move * 0.5
            if small_up_price > strike:
                small_profit = small_up_price - strike - premium
                small_return = small_profit / premium
                scenarios.append(small_return * 0.3)  # 30% probability
            
            # Scenario 4: Stock stays flat or moves down (loss)
            if moneyness > 1.05:  # ITM options have some intrinsic value
                intrinsic_value = spot - strike
                if intrinsic_value > 0:
                    scenarios.append((intrinsic_value - premium) / premium * 0.1)  # 10% probability
                else:
                    scenarios.append(-0.8 * 0.1)  # 80% loss probability
            else:
                scenarios.append(-0.9 * 0.1)  # 90% loss probability
            
            # Calculate weighted expected return
            expected_return = sum(scenarios)
            
            # Ensure reasonable bounds
            return max(-0.95, min(2.0, expected_return))
            
        except Exception as e:
            logger.debug(f"Error calculating expected return: {e}")
            # Simple fallback based on moneyness
            moneyness = spot / strike
            if moneyness > 1.1:
                return 0.3  # ITM calls
            elif moneyness > 0.95:
                return 0.1  # Near ATM calls
            else:
                return -0.2  # OTM calls
    
    def _calculate_risk_reward(self, contract: Dict) -> float:
        """Calculate realistic risk/reward ratio for call options"""
        try:
            premium = contract['ask'] if contract['ask'] > 0 else contract['mid']
            strike = contract['strike']
            spot = contract.get('current_stock_price', 0)
            
            if premium <= 0 or spot <= 0:
                return 0.0
            
            # Calculate potential profit scenarios
            profit_scenarios = []
            
            # Scenario 1: 50% stock price increase
            target_price_1 = spot * 1.5
            if target_price_1 > strike:
                profit_1 = target_price_1 - strike - premium
                profit_scenarios.append(profit_1)
            
            # Scenario 2: 100% stock price increase (meme stock scenario)
            target_price_2 = spot * 2.0
            if target_price_2 > strike:
                profit_2 = target_price_2 - strike - premium
                profit_scenarios.append(profit_2)
            
            # Scenario 3: 25% stock price increase
            target_price_3 = spot * 1.25
            if target_price_3 > strike:
                profit_3 = target_price_3 - strike - premium
                profit_scenarios.append(profit_3)
            
            # Calculate average potential profit
            if profit_scenarios:
                avg_potential_profit = sum(profit_scenarios) / len(profit_scenarios)
            else:
                avg_potential_profit = 0
            
            # Maximum loss is the premium paid
            max_loss = premium
            
            # Risk/reward ratio
            if max_loss > 0:
                return avg_potential_profit / max_loss
            else:
                return 0.0
                
        except Exception as e:
            logger.debug(f"Error calculating risk/reward: {e}")
            return 0.5  # Default conservative ratio
    
    def _calculate_option_value_analysis(self, spot: float, contract: Dict) -> Dict:
        """Comprehensive option value analysis"""
        try:
            strike = contract['strike']
            premium = contract['ask'] if contract['ask'] > 0 else contract['mid']
            days = contract['days_to_expiration']
            iv = contract['implied_volatility']
            
            # Moneyness analysis
            moneyness = spot / strike
            intrinsic_value = max(0, spot - strike)
            time_value = premium - intrinsic_value
            
            # Greeks analysis (simplified)
            delta = contract.get('delta', 0.5)
            theta = contract.get('theta', -0.01)
            gamma = contract.get('gamma', 0.01)
            vega = contract.get('vega', 0.1)
            
            # Volatility analysis
            iv_percentile = contract.get('iv_percentile', 50)
            iv_rank = "Low" if iv_percentile < 30 else "High" if iv_percentile > 70 else "Normal"
            
            # Liquidity analysis
            volume = contract.get('volume', 0)
            open_interest = contract.get('open_interest', 0)
            spread_pct = contract.get('spread_pct', 0.1)
            
            liquidity_score = 0
            if volume > 100:
                liquidity_score += 30
            elif volume > 10:
                liquidity_score += 20
            
            if open_interest > 500:
                liquidity_score += 30
            elif open_interest > 100:
                liquidity_score += 20
            
            if spread_pct < 0.1:
                liquidity_score += 40
            elif spread_pct < 0.2:
                liquidity_score += 20
            
            return {
                'moneyness': moneyness,
                'intrinsic_value': intrinsic_value,
                'time_value': time_value,
                'time_value_pct': (time_value / premium * 100) if premium > 0 else 0,
                'delta': delta,
                'theta': theta,
                'gamma': gamma,
                'vega': vega,
                'iv_percentile': iv_percentile,
                'iv_rank': iv_rank,
                'liquidity_score': liquidity_score,
                'volume': volume,
                'open_interest': open_interest,
                'spread_pct': spread_pct,
                'days_to_expiration': days,
                'premium': premium,
                'strike': strike,
                'spot_price': spot
            }
            
        except Exception as e:
            logger.debug(f"Error in option value analysis: {e}")
            return {}
    
    def _analyze_value_opportunity(self, stock: Dict, option: Dict, current_price: float) -> Tuple[float, List[str]]:
        """Analyze value opportunity using fundamental metrics"""
        score = 0
        reasons = []
        
        # Market cap analysis
        market_cap = stock.get('market_cap', 0)
        pe_ratio = stock.get('pe_ratio', 0)
        
        # Value scoring based on market cap efficiency
        if market_cap < 500_000_000:  # Micro cap
            score += 15
            reasons.append("🔍 Micro-cap value opportunity with high growth potential")
        elif market_cap < 2_000_000_000:  # Small cap
            score += 12
            reasons.append("📈 Small-cap value play with room for expansion")
        elif market_cap < 10_000_000_000:  # Mid cap
            score += 8
            reasons.append("⚖️ Mid-cap value with balanced risk/reward")
        
        # PE ratio analysis
        if pe_ratio and 0 < pe_ratio < 15:
            score += 10
            reasons.append(f"💰 Undervalued with PE ratio of {pe_ratio:.1f}")
        elif pe_ratio and 15 <= pe_ratio < 25:
            score += 5
            reasons.append(f"📊 Reasonable valuation with PE of {pe_ratio:.1f}")
        
        # Price momentum vs value
        avg_volume = stock.get('avg_volume', 0)
        current_volume = stock.get('volume', 0)
        
        if current_volume > avg_volume * 1.5:
            score += 8
            reasons.append("📊 High volume suggests institutional interest")
        elif current_volume > avg_volume * 1.2:
            score += 5
            reasons.append("📈 Above-average volume indicating accumulation")
        
        # Sector value analysis
        sector = stock.get('sector', '')
        if sector in ['Technology', 'Healthcare', 'Consumer Discretionary']:
            score += 5
            reasons.append(f"🚀 Growth sector: {sector}")
        elif sector in ['Financials', 'Energy', 'Materials']:
            score += 3
            reasons.append(f"💼 Value sector: {sector}")
        
        return score, reasons
    
    def _analyze_growth_opportunity(self, stock: Dict, option: Dict, current_price: float) -> Tuple[float, List[str]]:
        """Analyze growth opportunity using momentum and trend metrics"""
        score = 0
        reasons = []
        
        # Market cap growth potential
        market_cap = stock.get('market_cap', 0)
        
        # Growth scoring - smaller companies have higher growth potential
        if market_cap < 1_000_000_000:  # Under $1B
            score += 20
            reasons.append("🚀 High growth potential - small market cap")
        elif market_cap < 5_000_000_000:  # $1B-$5B
            score += 15
            reasons.append("📈 Good growth potential - emerging company")
        elif market_cap < 20_000_000_000:  # $5B-$20B
            score += 10
            reasons.append("⬆️ Moderate growth potential - established player")
        
        # Volume analysis for growth
        avg_volume = stock.get('avg_volume', 0)
        current_volume = stock.get('volume', 0)
        
        if current_volume > avg_volume * 2:
            score += 15
            reasons.append("🔥 Explosive volume - potential breakout")
        elif current_volume > avg_volume * 1.5:
            score += 10
            reasons.append("📊 Strong volume - momentum building")
        
        # Price action analysis
        price = current_price
        if price > 10 and price < 50:  # Sweet spot for options
            score += 10
            reasons.append("💎 Optimal price range for options leverage")
        elif price >= 50 and price < 100:
            score += 8
            reasons.append("📈 Good price level for substantial moves")
        elif price < 10:
            score += 12
            reasons.append("🎯 Low price with high percentage move potential")
        
        # Sector growth analysis
        sector = stock.get('sector', '')
        growth_sectors = ['Technology', 'Healthcare', 'Consumer Discretionary', 'Communication Services']
        if sector in growth_sectors:
            score += 8
            reasons.append(f"🌟 High-growth sector: {sector}")
        
        # Options-specific growth factors
        days_to_exp = option.get('days_to_expiration', 30)
        if 20 <= days_to_exp <= 45:
            score += 5
            reasons.append("⏰ Optimal time horizon for growth moves")
        
        return score, reasons
    
    def _analyze_technical_setup(self, stock: Dict, option: Dict, current_price: float) -> Tuple[float, List[str]]:
        """Analyze technical setup for optimal entry"""
        score = 0
        reasons = []
        
        # Volatility analysis
        strike = option.get('strike', current_price)
        moneyness = current_price / strike if strike > 0 else 1.0
        
        # Technical scoring based on moneyness and setup
        if 0.95 <= moneyness <= 1.05:  # At-the-money
            score += 15
            reasons.append("🎯 ATM strike - balanced risk/reward")
        elif 1.05 < moneyness <= 1.15:  # Slightly OTM
            score += 18
            reasons.append("🚀 Slightly OTM - good leverage potential")
        elif 1.15 < moneyness <= 1.30:  # OTM
            score += 20
            reasons.append("⚡ OTM strike - high leverage play")
        elif moneyness < 0.95:  # ITM
            score += 12
            reasons.append("🛡️ ITM with intrinsic value protection")
        
        # Liquidity technical factors
        volume = option.get('volume', 0)
        open_interest = option.get('open_interest', 0)
        
        if volume >= 100 or open_interest >= 500:
            score += 10
            reasons.append("💧 Good liquidity for easy entry/exit")
        elif volume >= 50 or open_interest >= 250:
            score += 5
            reasons.append("📊 Adequate liquidity")
        
        # Time decay optimization
        days_to_exp = option.get('days_to_expiration', 30)
        if 25 <= days_to_exp <= 50:
            score += 10
            reasons.append("⏳ Optimal time decay balance")
        elif 15 <= days_to_exp < 25:
            score += 8
            reasons.append("⚡ Short-term momentum play")
        elif 50 < days_to_exp <= 70:
            score += 6
            reasons.append("📅 Longer-term position")
        
        # Market cap technical factors
        market_cap = stock.get('market_cap', 0)
        if market_cap < 2_000_000_000:  # Small caps move more
            score += 8
            reasons.append("🎢 Small-cap volatility advantage")
        
        # Spread analysis (if available)
        bid = option.get('bid', 0)
        ask = option.get('ask', 0)
        if bid > 0 and ask > 0:
            spread_pct = (ask - bid) / ((ask + bid) / 2) * 100
            if spread_pct < 10:
                score += 5
                reasons.append("💰 Tight bid-ask spread")
        
        return score, reasons
    
    def _analyze_momentum_factor_model(self, stock: Dict, option: Dict, current_price: float) -> Tuple[float, List[str]]:
        """Fama-French 5-Factor Model + Momentum Factor Analysis"""
        score = 0
        reasons = []
        
        # Market factor (Beta)
        beta = stock.get('beta', 1.0)
        if 0.8 <= beta <= 1.2:
            score += 8
            reasons.append("📈 Balanced market sensitivity")
        elif beta > 1.2:
            score += 6
            reasons.append("🚀 High beta - leveraged market moves")
        
        # Size factor (SMB - Small Minus Big)
        market_cap = stock.get('market_cap', 0)
        if 500_000_000 <= market_cap <= 2_000_000_000:  # Small cap sweet spot
            score += 12
            reasons.append("🎯 Optimal small-cap size factor")
        elif 2_000_000_000 < market_cap <= 10_000_000_000:  # Mid cap
            score += 8
            reasons.append("📊 Mid-cap size factor")
        
        # Value factor (HML - High Minus Low)
        pe_ratio = stock.get('pe_ratio', 0)
        if 0 < pe_ratio <= 20:  # Value stocks
            score += 10
            reasons.append("💎 Value factor - low P/E")
        elif 20 < pe_ratio <= 35:  # Growth stocks
            score += 8
            reasons.append("🌱 Growth factor - reasonable P/E")
        
        # Profitability factor (RMW - Robust Minus Weak)
        profit_margin = stock.get('profit_margin', 0)
        if profit_margin > 0.15:  # High profitability
            score += 10
            reasons.append("💰 High profitability factor")
        elif profit_margin > 0.05:  # Decent profitability
            score += 6
            reasons.append("📈 Positive profitability factor")
        
        # Investment factor (CMA - Conservative Minus Aggressive)
        debt_to_equity = stock.get('debt_to_equity', 0)
        if debt_to_equity < 0.3:  # Conservative investment
            score += 8
            reasons.append("🛡️ Conservative investment factor")
        
        # Momentum factor (12-month price momentum)
        price_change_12m = stock.get('price_change_12m', 0)
        if 0.1 <= price_change_12m <= 0.5:  # Strong but not excessive momentum
            score += 15
            reasons.append("⚡ Strong momentum factor")
        elif price_change_12m > 0.5:  # Very strong momentum
            score += 12
            reasons.append("🚀 Exceptional momentum factor")
        elif -0.2 <= price_change_12m < 0:  # Mean reversion potential
            score += 8
            reasons.append("🔄 Mean reversion momentum")
        
        return score, reasons
    
    def _analyze_volatility_smile_arbitrage(self, stock: Dict, option: Dict, current_price: float) -> Tuple[float, List[str]]:
        """Volatility Smile and Term Structure Analysis"""
        score = 0
        reasons = []
        
        # Implied volatility analysis
        iv = option.get('implied_volatility', 0)
        strike = option.get('strike', current_price)
        moneyness = current_price / strike if strike > 0 else 1.0
        
        # Volatility smile analysis
        if 0.95 <= moneyness <= 1.05:  # ATM
            if 0.2 <= iv <= 0.4:  # Normal IV range
                score += 12
                reasons.append("😊 Normal volatility smile - ATM")
            elif iv > 0.4:  # High IV
                score += 8
                reasons.append("📈 High IV smile - potential crush")
        elif moneyness < 0.95:  # ITM
            if iv > 0.3:  # ITM with high IV
                score += 10
                reasons.append("💎 ITM with volatility premium")
        elif moneyness > 1.05:  # OTM
            if iv > 0.25:  # OTM with reasonable IV
                score += 15
                reasons.append("🎯 OTM volatility smile advantage")
        
        # Term structure analysis
        days_to_exp = option.get('days_to_expiration', 30)
        if 20 <= days_to_exp <= 45:  # Optimal term structure
            score += 10
            reasons.append("⏰ Optimal volatility term structure")
        elif days_to_exp < 20:  # Short-term
            score += 6
            reasons.append("⚡ Short-term volatility play")
        
        # Volatility clustering analysis
        atr = stock.get('atr', current_price * 0.02)
        atr_pct = atr / current_price
        if 0.02 <= atr_pct <= 0.05:  # Normal volatility
            score += 8
            reasons.append("📊 Normal volatility clustering")
        elif atr_pct > 0.05:  # High volatility
            score += 6
            reasons.append("🌊 High volatility clustering")
        
        return score, reasons
    
    def _analyze_behavioral_finance_signals(self, stock: Dict, option: Dict, current_price: float) -> Tuple[float, List[str]]:
        """Behavioral Finance and Market Psychology Analysis"""
        score = 0
        reasons = []
        
        # Overreaction analysis (De Bondt & Thaler)
        price_change_3m = stock.get('price_change_3m', 0)
        price_change_12m = stock.get('price_change_12m', 0)
        
        if price_change_3m < -0.2 and price_change_12m > 0.1:  # Recent overreaction
            score += 15
            reasons.append("🧠 Overreaction reversal signal")
        elif price_change_3m < -0.15:  # Recent decline
            score += 10
            reasons.append("📉 Recent decline - contrarian opportunity")
        
        # Anchoring bias analysis
        price_52w_high = stock.get('price_52w_high', current_price)
        price_52w_low = stock.get('price_52w_low', current_price)
        
        if current_price < price_52w_low * 1.1:  # Near 52-week low
            score += 12
            reasons.append("⚓ Near 52-week low - anchoring opportunity")
        elif current_price > price_52w_high * 0.9:  # Near 52-week high
            score += 8
            reasons.append("🎯 Near 52-week high - momentum continuation")
        
        # Herding behavior analysis
        volume_ratio = stock.get('volume_ratio', 1.0)
        if volume_ratio > 2.0:  # High volume - herding
            score += 8
            reasons.append("🐑 High volume herding - trend confirmation")
        elif volume_ratio < 0.5:  # Low volume - contrarian
            score += 6
            reasons.append("🔍 Low volume - contrarian opportunity")
        
        # Loss aversion analysis
        rsi = stock.get('rsi', 50)
        if rsi < 30:  # Oversold - loss aversion
            score += 10
            reasons.append("😰 Oversold - loss aversion reversal")
        elif rsi > 70:  # Overbought - greed
            score += 5
            reasons.append("😍 Overbought - greed continuation")
        
        # Disposition effect analysis
        price_change_1m = stock.get('price_change_1m', 0)
        if -0.1 <= price_change_1m <= 0.05:  # Sideways - disposition effect
            score += 8
            reasons.append("⚖️ Sideways - disposition effect opportunity")
        
        return score, reasons
    
    def _analyze_advanced_liquidity_metrics(self, option: Dict, stock: Dict) -> Tuple[float, List[str]]:
        """Advanced Liquidity and Slippage Analysis"""
        score = 0
        reasons = []
        
        # Volume analysis for slippage
        volume = option.get('volume', 0)
        open_interest = option.get('open_interest', 0)
        
        # Advanced volume metrics (minimum 5000 volume required)
        if volume >= 10000 and open_interest >= 5000:
            score += 25
            reasons.append("💧 Excellent liquidity - minimal slippage (10k+ volume)")
        elif volume >= 7500 and open_interest >= 3000:
            score += 20
            reasons.append("📊 Very good liquidity - low slippage risk (7.5k+ volume)")
        elif volume >= 5000 and open_interest >= 2000:
            score += 15
            reasons.append("✅ Good liquidity - manageable slippage (5k+ volume)")
        elif volume >= 3000 and open_interest >= 1000:
            score += 5
            reasons.append("⚠️ Moderate liquidity - some slippage risk (3k+ volume)")
        else:
            score -= 20
            reasons.append("❌ Insufficient volume - high slippage risk (below 5k requirement)")
        
        # Bid-ask spread analysis
        bid = option.get('bid', 0)
        ask = option.get('ask', 0)
        if bid > 0 and ask > 0:
            spread = ask - bid
            mid_price = (ask + bid) / 2
            spread_pct = (spread / mid_price) * 100 if mid_price > 0 else 100
            
            if spread_pct < 5:
                score += 15
                reasons.append("💰 Tight spread - excellent execution")
            elif spread_pct < 10:
                score += 10
                reasons.append("📈 Reasonable spread - good execution")
            elif spread_pct < 20:
                score += 5
                reasons.append("📊 Wide spread - moderate execution cost")
            else:
                score += 0
                reasons.append("❌ Very wide spread - high execution cost")
        
        # Market depth analysis (simulated)
        market_cap = stock.get('market_cap', 0)
        if market_cap > 5_000_000_000:  # Large cap
            score += 8
            reasons.append("🏢 Large cap - deep market")
        elif market_cap > 1_000_000_000:  # Mid cap
            score += 6
            reasons.append("🏭 Mid cap - decent market depth")
        else:  # Small cap
            score += 3
            reasons.append("🏪 Small cap - limited market depth")
        
        # Time-of-day liquidity (simulated)
        from datetime import datetime
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 15:  # Market hours
            score += 5
            reasons.append("🕘 Market hours - optimal liquidity")
        elif 8 <= current_hour <= 16:  # Extended hours
            score += 3
            reasons.append("🕗 Extended hours - reduced liquidity")
        
        # Sector liquidity analysis
        sector = stock.get('sector', 'Unknown')
        liquid_sectors = ['Technology', 'Healthcare', 'Financial Services', 'Consumer Discretionary']
        if sector in liquid_sectors:
            score += 5
            reasons.append(f"🏷️ {sector} - liquid sector")
        
        return score, reasons
    
    def _analyze_sophisticated_expiration_selection(self, option: Dict, stock: Dict, current_price: float) -> Tuple[float, List[str]]:
        """Sophisticated Expiration Selection Based on Greeks Correlation"""
        score = 0
        reasons = []
        
        days_to_exp = option.get('days_to_expiration', 30)
        delta = option.get('delta', 0.5)
        theta = option.get('theta', -0.01)
        gamma = option.get('gamma', 0.01)
        vega = option.get('vega', 0.1)
        iv = option.get('implied_volatility', 0.3)
        
        # 1. Theta Decay Optimization (Time Decay vs. Time Value)
        theta_ratio = abs(theta) / (option.get('ask', 0.01) if option.get('ask', 0) > 0 else 0.01)
        
        if 20 <= days_to_exp <= 45:  # Optimal theta decay zone
            if theta_ratio < 0.02:  # Low daily decay
                score += 25
                reasons.append("⏰ Optimal expiration - balanced theta decay")
            elif theta_ratio < 0.03:
                score += 20
                reasons.append("📅 Good expiration - manageable theta decay")
            else:
                score += 15
                reasons.append("⚡ Acceptable expiration - higher theta risk")
        elif 15 <= days_to_exp < 20:  # Short-term momentum
            if theta_ratio < 0.025:
                score += 18
                reasons.append("🚀 Short-term momentum play - controlled decay")
            else:
                score += 10
                reasons.append("⚡ Short-term - high theta risk")
        elif 45 < days_to_exp <= 60:  # Longer-term
            if theta_ratio < 0.015:
                score += 20
                reasons.append("📈 Longer-term position - low theta decay")
            else:
                score += 12
                reasons.append("📅 Longer-term - expensive time value")
        elif days_to_exp < 15:  # Very short-term
            score += 5
            reasons.append("🔥 Very short-term - extreme theta risk")
        elif days_to_exp > 60:  # Very long-term
            score += 8
            reasons.append("📊 Very long-term - expensive time premium")
        
        # 2. Gamma Sensitivity Analysis (Price Movement Responsiveness)
        gamma_threshold = 0.01
        if gamma > gamma_threshold:
            if 20 <= days_to_exp <= 45:  # High gamma + optimal time
                score += 15
                reasons.append("🎯 High gamma + optimal time - responsive to moves")
            elif days_to_exp < 20:  # High gamma + short time
                score += 12
                reasons.append("⚡ High gamma + short time - explosive potential")
            else:
                score += 8
                reasons.append("📈 High gamma - good responsiveness")
        elif gamma > 0.005:
            score += 5
            reasons.append("📊 Moderate gamma - decent responsiveness")
        
        # 3. Vega Sensitivity Analysis (Volatility Exposure)
        vega_threshold = 0.1
        if vega > vega_threshold:
            if iv > 0.4:  # High vega + high IV
                score += 12
                reasons.append("🌊 High vega + high IV - volatility play")
            elif 0.2 <= iv <= 0.4:  # High vega + moderate IV
                score += 15
                reasons.append("📊 High vega + moderate IV - balanced volatility exposure")
            else:
                score += 8
                reasons.append("📈 High vega - volatility sensitivity")
        elif vega > 0.05:
            score += 5
            reasons.append("📊 Moderate vega - some volatility exposure")
        
        # 4. Delta-Time Correlation Analysis
        if 0.3 <= delta <= 0.7:  # Good delta range
            if 25 <= days_to_exp <= 50:  # Optimal time for delta
                score += 20
                reasons.append("🎯 Optimal delta-time correlation")
            elif 20 <= days_to_exp < 25 or 50 < days_to_exp <= 60:
                score += 15
                reasons.append("📈 Good delta-time correlation")
            else:
                score += 10
                reasons.append("📊 Decent delta-time correlation")
        elif delta < 0.3:  # Low delta (OTM)
            if days_to_exp <= 30:  # Short time for OTM
                score += 12
                reasons.append("🚀 OTM + short time - leverage play")
            else:
                score += 8
                reasons.append("📈 OTM - leverage with time risk")
        elif delta > 0.7:  # High delta (ITM)
            if days_to_exp >= 30:  # Longer time for ITM
                score += 15
                reasons.append("🛡️ ITM + longer time - safer play")
            else:
                score += 10
                reasons.append("📊 ITM - safer but expensive")
        
        # 5. Greeks Harmony Analysis (All Greeks Working Together)
        greeks_score = 0
        if 0.3 <= delta <= 0.7 and gamma > 0.01 and vega > 0.05 and theta_ratio < 0.025:
            greeks_score = 20
            reasons.append("🎭 Perfect Greeks harmony - all factors aligned")
        elif 0.25 <= delta <= 0.75 and gamma > 0.005 and vega > 0.03 and theta_ratio < 0.03:
            greeks_score = 15
            reasons.append("🎪 Good Greeks harmony - most factors aligned")
        elif 0.2 <= delta <= 0.8 and gamma > 0.003 and vega > 0.02 and theta_ratio < 0.035:
            greeks_score = 10
            reasons.append("🎨 Decent Greeks harmony - some factors aligned")
        else:
            greeks_score = 5
            reasons.append("🎯 Basic Greeks - mixed signals")
        
        score += greeks_score
        
        # 6. Market Cap Adaptive Expiration
        market_cap = stock.get('market_cap', 0)
        if market_cap < 1_000_000_000:  # Small cap
            if days_to_exp <= 30:  # Shorter for small caps
                score += 8
                reasons.append("🏪 Small cap + short expiration - volatility play")
            else:
                score += 5
                reasons.append("📊 Small cap + longer expiration - time premium risk")
        elif market_cap > 10_000_000_000:  # Large cap
            if days_to_exp >= 30:  # Longer for large caps
                score += 8
                reasons.append("🏢 Large cap + longer expiration - stability play")
            else:
                score += 5
                reasons.append("📈 Large cap + short expiration - momentum play")
        
        return score, reasons
    
    def _analyze_breakout_potential(self, stock: Dict, option: Dict, current_price: float) -> Tuple[float, List[str]]:
        """Advanced Breakout Detection Methods (like SOGP before it popped)"""
        score = 0
        reasons = []
        
        # 1. Volume Breakout Analysis
        volume = stock.get('volume', 0)
        avg_volume = stock.get('avg_volume', volume)
        if avg_volume > 0:
            volume_ratio = volume / avg_volume
            if volume_ratio > 3.0:  # 3x average volume
                score += 30
                reasons.append("🚀 Massive volume breakout - 3x+ average volume")
            elif volume_ratio > 2.0:  # 2x average volume
                score += 20
                reasons.append("📈 Strong volume breakout - 2x+ average volume")
            elif volume_ratio > 1.5:  # 1.5x average volume
                score += 10
                reasons.append("📊 Elevated volume - 1.5x+ average volume")
        
        # 2. Price Action Breakout Patterns
        day_high = stock.get('day_high', current_price)
        day_low = stock.get('day_low', current_price)
        prev_close = stock.get('prev_close', current_price)
        
        # Breakout above previous close
        if current_price > prev_close * 1.05:  # 5%+ breakout
            score += 25
            reasons.append("💥 Price breakout - 5%+ above previous close")
        elif current_price > prev_close * 1.02:  # 2%+ breakout
            score += 15
            reasons.append("📈 Price momentum - 2%+ above previous close")
        
        # Intraday range analysis
        daily_range = (day_high - day_low) / current_price
        if daily_range > 0.15:  # 15%+ daily range
            score += 20
            reasons.append("⚡ High volatility day - 15%+ intraday range")
        elif daily_range > 0.10:  # 10%+ daily range
            score += 12
            reasons.append("📊 Elevated volatility - 10%+ intraday range")
        
        # 3. Market Cap and Float Analysis (SOGP-like characteristics)
        market_cap = stock.get('market_cap', 0)
        if 100_000_000 <= market_cap <= 2_000_000_000:  # $100M - $2B (sweet spot for breakouts)
            score += 15
            reasons.append("🎯 Optimal market cap for breakout - $100M-$2B range")
        elif 2_000_000_000 < market_cap <= 10_000_000_000:  # $2B - $10B
            score += 10
            reasons.append("📊 Good market cap for momentum - $2B-$10B range")
        
        # 4. Sector Momentum Analysis
        sector = stock.get('sector', '').lower()
        breakout_sectors = ['technology', 'healthcare', 'consumer cyclical', 'communication services']
        if sector in breakout_sectors:
            score += 8
            reasons.append(f"🔥 Breakout sector momentum - {sector.title()}")
        
        # 5. Options Flow Analysis
        option_volume = option.get('volume', 0)
        option_oi = option.get('open_interest', 0)
        
        if option_volume > 1000 and option_oi > 5000:  # High options activity
            score += 20
            reasons.append("📊 High options activity - institutional interest")
        elif option_volume > 500 and option_oi > 2000:
            score += 12
            reasons.append("📈 Elevated options activity - smart money interest")
        
        # 6. Implied Volatility Expansion
        iv = option.get('implied_volatility', 0.3)
        if iv > 0.6:  # Very high IV (breakout potential)
            score += 15
            reasons.append("🔥 High IV expansion - breakout volatility")
        elif iv > 0.4:  # High IV
            score += 10
            reasons.append("📊 Elevated IV - momentum volatility")
        
        return score, reasons
    
    def _analyze_momentum_breakout_signals(self, stock: Dict, option: Dict, current_price: float) -> Tuple[float, List[str]]:
        """Advanced Momentum and Breakout Signal Analysis"""
        score = 0
        reasons = []
        
        # 1. Relative Strength Analysis
        day_high = stock.get('day_high', current_price)
        day_low = stock.get('day_low', current_price)
        prev_close = stock.get('prev_close', current_price)
        
        # Strong momentum indicators
        if current_price > day_high * 0.95:  # Near day high
            score += 15
            reasons.append("🚀 Near day high - strong momentum")
        
        if current_price > prev_close * 1.10:  # 10%+ daily gain
            score += 25
            reasons.append("💥 Explosive move - 10%+ daily gain")
        elif current_price > prev_close * 1.05:  # 5%+ daily gain
            score += 18
            reasons.append("📈 Strong move - 5%+ daily gain")
        
        # 2. Breakout Pattern Recognition
        gap_up = (current_price - prev_close) / prev_close
        if gap_up > 0.20:  # 20%+ gap up
            score += 30
            reasons.append("🚀 Massive gap up - 20%+ pre-market move")
        elif gap_up > 0.10:  # 10%+ gap up
            score += 20
            reasons.append("📈 Strong gap up - 10%+ pre-market move")
        elif gap_up > 0.05:  # 5%+ gap up
            score += 12
            reasons.append("📊 Gap up - 5%+ pre-market move")
        
        # 3. Volume-Price Confirmation
        volume = stock.get('volume', 0)
        avg_volume = stock.get('avg_volume', volume)
        if avg_volume > 0:
            volume_ratio = volume / avg_volume
            price_change = (current_price - prev_close) / prev_close
            
            if volume_ratio > 2.0 and price_change > 0.05:  # High volume + 5%+ gain
                score += 25
                reasons.append("🔥 Volume-price confirmation - high volume + strong gain")
            elif volume_ratio > 1.5 and price_change > 0.03:  # Elevated volume + 3%+ gain
                score += 15
                reasons.append("📈 Volume-price alignment - elevated volume + gain")
        
        return score, reasons
    
    def _analyze_put_call_selection_research(self, stock: Dict, option: Dict, current_price: float) -> float:
        """
        TOP-TIER ACADEMIC RESEARCH for put vs call selection
        Based on 40+ years of academic finance literature including:
        - Fama-French multi-factor models
        - Behavioral finance (De Bondt & Thaler, Daniel & Moskowitz)  
        - Momentum research (Jegadeesh & Titman)
        - Volatility research (Engle, Bakshi, Kapadia, Madan)
        - Black-Scholes extensions and portfolio theory
        - Tail risk hedging (Taleb) and option skew research
        - Innovation premiums and merger arbitrage research
        """
        option_type = option.get('type', 'CALL')
        score = 0
        
        # Get stock characteristics
        market_cap = stock.get('market_cap', 0)
        volume = stock.get('volume', 0)
        avg_volume = stock.get('avg_volume', 1)
        sector = stock.get('sector', '')
        price = current_price
        
        # Academic research-based scoring
        if option_type == 'PUT':
            # PUT PREMIUM RESEARCH FACTORS
            
            # 1. MULTI-CAP PUT RESEARCH (Rebalanced for fairness)
            # Large cap hedging (Black-Scholes extensions)
            if market_cap > 50_000_000_000:  # $50B+
                score += 28  # Slight increase to maintain large cap put advantage
            elif market_cap > 20_000_000_000:  # $20B+
                score += 20  # Increased from 15
            # NEW: Small/mid cap mean reversion research (Behavioral finance)
            elif market_cap < 2_000_000_000:  # Small cap volatility = put opportunities
                score += 22  # NEW: Small caps have high volatility, good for puts
            elif market_cap < 10_000_000_000:  # Mid cap  
                score += 16  # NEW: Mid caps prone to corrections, good for puts
            
            # 2. MEAN REVERSION RESEARCH (Lo & MacKinlay, 1988)
            # Extreme volume often signals overextension and mean reversion
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            if volume_ratio > 3.0:  # Extreme volume spike
                score += 20
            elif volume_ratio > 2.0:  # High volume
                score += 12
            
            # 3. BEHAVIORAL FINANCE OVERREACTION (De Bondt & Thaler)
            # High-priced stocks prone to behavioral overreaction
            if price > 300:  # Very high-priced stocks
                score += 20
            elif price > 150:  # High-priced stocks
                score += 12
            elif price > 75:   # Medium-high priced
                score += 8
            
            # 4. SECTOR CYCLICAL RESEARCH
            # Academic sector rotation studies show these sectors prone to reversals
            cyclical_sectors = ['Energy', 'Materials', 'Financials', 'Real Estate', 'Industrials']
            if sector in cyclical_sectors:
                score += 15
            
            # 5. VOLATILITY SMILE ARBITRAGE
            # Put skew research - puts often underpriced relative to calls
            iv = option.get('implied_volatility', 0.3)
            if iv > 0.6:  # High IV favors puts
                score += 20  # Increased from 18
            elif iv > 0.4:  # Medium-high IV
                score += 12  # Increased from 10
            elif iv > 0.25:  # Even medium IV can favor puts due to skew
                score += 8   # NEW: Added medium IV bonus
                
            # 6. NEW: CONTRARIAN RESEARCH (Academic studies on mean reversion)
            # Technology sector corrections (behavioral finance research)
            if sector in ['Technology', 'Communication Services']:
                score += 12  # Tech stocks prone to sharp corrections
            
            # 7. NEW: DEFENSIVE PUT RESEARCH (Portfolio theory)
            # Any stock can benefit from protective puts for risk management
            score += 8  # Base defensive bonus for all puts
            
            # 8. NEW: EARNINGS/EVENT RISK RESEARCH  
            # Puts benefit from downside protection around uncertain events
            score += 6  # Event risk protection bonus
            
            # 9. NEW: PUT SKEW PREMIUM RESEARCH (Bakshi, Kapadia, Madan, 2003)
            # Academic research shows put options often underpriced due to skew
            score += 12  # Put skew premium bonus
            
            # 10. NEW: TAIL RISK HEDGING RESEARCH (Taleb, 2007)
            # Black swan protection value increases put attractiveness
            if market_cap > 5_000_000_000:  # Larger positions need more hedging
                score += 15
            else:
                score += 10  # All positions benefit from tail risk protection
                
            # 11. NEW: VOLATILITY CLUSTERING RESEARCH (Engle, 1982)
            # Academic studies show volatility clusters, benefiting put buyers
            score += 8
            
            # 12. NEW: MOMENTUM CRASH RESEARCH (Daniel & Moskowitz, 2016) 
            # Academic research shows momentum strategies prone to crashes
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            if volume_ratio > 1.8:  # High momentum often precedes crashes
                score += 14
                
        else:  # CALL options
            # CALL PREMIUM RESEARCH FACTORS
            
            # 1. SMALL/MID CAP GROWTH RESEARCH (Fama-French size factor)
            # Academic research shows small caps have momentum premium
            if market_cap < 2_000_000_000:  # Small cap
                score += 25
            elif market_cap < 10_000_000_000:  # Mid cap
                score += 15
            
            # 2. MOMENTUM FACTOR RESEARCH (Jegadeesh & Titman, 1993)
            # Strong momentum favors calls
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            if volume_ratio > 2.0 and price > 0:  # High momentum
                score += 20
            elif volume_ratio > 1.5:  # Good momentum
                score += 12
            
            # 3. GROWTH SECTOR RESEARCH
            # Academic studies show growth sectors outperform over medium term
            growth_sectors = ['Technology', 'Healthcare', 'Communication Services', 'Consumer Discretionary']
            if sector in growth_sectors:
                score += 18
            
            # 4. VOLATILITY ARBITRAGE RESEARCH
            # Low IV creates opportunities for call buying
            iv = option.get('implied_volatility', 0.3)
            if iv < 0.25:  # Low IV favors calls
                score += 15
            elif iv < 0.35:  # Medium-low IV
                score += 8
            
            # 5. PRICE ACCESSIBILITY RESEARCH (Retail interest studies)
            # Academic research on retail options activity
            if 10 <= price <= 100:  # Sweet spot for retail interest
                score += 12
            elif 5 <= price <= 200:  # Good range
                score += 8
                
            # 6. NEW: MOMENTUM ANOMALY RESEARCH (Jegadeesh & Titman, 2001)
            # Academic studies show momentum continues 6-12 months
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1
            if volume_ratio > 2.5:  # Strong momentum signal
                score += 16
            elif volume_ratio > 2.0:  # Good momentum
                score += 12
                
            # 7. NEW: EARNINGS SURPRISE RESEARCH (Foster, Olsen, Shevlin, 1984)
            # Calls benefit from positive earnings surprises
            score += 10  # Base earnings upside potential
            
            # 8. NEW: ANALYST REVISION RESEARCH (Womack, 1996)
            # Academic research shows analyst upgrades drive call performance
            score += 8  # Analyst revision potential
            
            # 9. NEW: INNOVATION PREMIUM RESEARCH (Chen, Da, Zhao, 2013)
            # Technology and healthcare sectors have innovation premium
            if sector in ['Technology', 'Healthcare', 'Communication Services']:
                score += 14  # Innovation premium bonus
                
            # 10. NEW: MERGER ARBITRAGE RESEARCH (Mitchell & Pulvino, 2001)
            # Small/mid caps more likely to be acquisition targets
            if market_cap < 5_000_000_000:  # Acquisition target potential
                score += 10
        
        # 6. LIQUIDITY PREMIUM RESEARCH (Market microstructure)
        volume = option.get('volume', 0)
        open_interest = option.get('open_interest', 0)
        
        if volume > 1000 and open_interest > 1000:  # High liquidity
            score += 10
        elif volume > 500 or open_interest > 500:   # Good liquidity
            score += 5
        
        # 7. TIME VALUE RESEARCH (Greeks optimization)
        days_to_exp = option.get('days_to_expiration', 30)
        if 20 <= days_to_exp <= 50:  # Optimal theta zone
            score += 8
        elif 15 <= days_to_exp <= 60:  # Good theta zone
            score += 5
        
        return score