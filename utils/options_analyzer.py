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
        """Analyze a single stock for call options opportunities (scoring and enrichment logic restored)"""
        symbol = stock['symbol']
        recommendations = []
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
                except Exception as e:
                    logger.warning(f"Error analyzing option for {symbol}: {e}")
                    continue
            return recommendations
        except Exception as e:
            logger.error(f"Error analyzing options for {symbol}: {e}")
            return []
    
    def _score_option_comprehensive(self, option: Dict, current_price: float, 
                                   stock: Dict) -> Tuple[float, Dict]:
        """Comprehensive scoring system for options with better reasoning"""
        score = 0
        reasons = []
        analysis = {}
        
        # Get option value analysis
        value_analysis = self._calculate_option_value_analysis(current_price, option)
        
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