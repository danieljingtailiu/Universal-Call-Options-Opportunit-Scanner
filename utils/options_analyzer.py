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
    
    def analyze_stock(self, stock: Dict) -> List[Dict]:
        """Analyze options and return TOP 3 recommendations"""
        try:
            symbol = stock['symbol']
            current_price = stock['price']
            
            logger.info(f"\nAnalyzing {symbol} at ${current_price:.2f}")
            
            # Get options chain
            options_chain = self.data_fetcher.get_options_chain(symbol)
            if not options_chain:
                logger.warning(f"No options found for {symbol}")
                return []
            
            # Score and rank ALL options
            scored_options = []
            
            for option in options_chain:
                score, analysis = self._score_option_comprehensive(option, current_price, stock)
                
                if score > 40:  # Minimum score threshold
                    recommendation = {
                        'symbol': symbol,
                        'current_stock_price': current_price,
                        'recommendation': 'BUY',
                        'contract_type': 'CALL',
                        'strike': option['strike'],
                        'expiration': option['expiration'],
                        'days_to_expiration': option['days_to_expiration'],
                        'ask_price': option['ask'],
                        'bid_price': option['bid'],
                        'spread_pct': option['spread_pct'],
                        'mid_price': option['mid'],
                        'volume': option['volume'],
                        'open_interest': option['open_interest'],
                        'implied_volatility': option['implied_volatility'],
                        'delta': option['delta'],
                        'theta': option['theta'],
                        'gamma': option['gamma'],
                        'vega': option['vega'],
                        'score': score,
                        'analysis': analysis,
                        'probability_of_profit': self._calculate_probability_of_profit(
                            current_price, option['strike'], option['days_to_expiration'],
                            option['implied_volatility'], option['ask']
                        ),
                        'expected_return': self._calculate_expected_return(
                            current_price, option, stock.get('atr', current_price * 0.02)
                        ),
                        'risk_reward_ratio': self._calculate_risk_reward(option),
                        'recommendation_reasons': analysis['reasons'],
                        'entry_price': option['ask'] if option['ask'] > 0 else option['last'],
                        'stop_loss': option['ask'] * 0.50,  # 50% stop loss
                        'target_1': option['ask'] * 1.50,  # 50% profit target
                        'target_2': option['ask'] * 2.00,  # 100% profit target
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    scored_options.append(recommendation)
            
            # Sort by score and return top 3
            scored_options.sort(key=lambda x: x['score'], reverse=True)
            top_recommendations = scored_options[:3]
            
            # Print detailed recommendations
            if top_recommendations:
                self._print_recommendations(symbol, current_price, top_recommendations)
            
            return top_recommendations
            
        except Exception as e:
            logger.error(f"Error analyzing options for {stock['symbol']}: {e}")
            return []
    
    def _score_option_comprehensive(self, option: Dict, current_price: float, 
                                   stock: Dict) -> Tuple[float, Dict]:
        """Comprehensive scoring system for options"""
        score = 0
        reasons = []
        analysis = {}
        
        # 1. Moneyness Score (20 points)
        moneyness = (option['strike'] - current_price) / current_price
        if 0 <= moneyness <= 0.05:  # ATM to slightly OTM
            score += 20
            reasons.append(f"Ideal strike near money (${option['strike']})")
        elif 0.05 < moneyness <= 0.10:
            score += 15
            reasons.append(f"Good OTM strike (${option['strike']})")
        elif -0.05 <= moneyness < 0:  # Slightly ITM
            score += 10
            reasons.append(f"Slightly ITM (${option['strike']})")
        
        # 2. Time Value Score (15 points)
        days = option['days_to_expiration']
        if 30 <= days <= 45:
            score += 15
            reasons.append(f"Optimal expiration ({days} days)")
        elif 25 <= days < 30 or 45 < days <= 60:
            score += 10
            reasons.append(f"Good expiration ({days} days)")
        elif 20 <= days < 25 or 60 < days <= 70:
            score += 5
        
        # 3. Liquidity Score (25 points)
        liquidity_score = option.get('liquidity_score', 0)
        normalized_liquidity = min(liquidity_score / 100 * 25, 25)
        score += normalized_liquidity
        
        if option['volume'] > 100:
            reasons.append(f"High volume ({option['volume']})")
        elif option['volume'] > 10:
            reasons.append(f"Decent volume ({option['volume']})")
        elif option['open_interest'] > 100:
            reasons.append(f"Good open interest ({option['open_interest']})")
        
        # 4. Greeks Score (20 points)
        # Delta
        if 0.25 <= option['delta'] <= 0.45:
            score += 8
            reasons.append(f"Good delta ({option['delta']:.2f})")
        elif 0.20 <= option['delta'] < 0.25 or 0.45 < option['delta'] <= 0.55:
            score += 5
        
        # Theta (want low theta relative to price)
        theta_ratio = abs(option['theta']) / (option['ask'] if option['ask'] > 0 else 0.01)
        if theta_ratio < 0.02:  # Less than 2% daily decay
            score += 7
            reasons.append(f"Low theta decay ({abs(option['theta']):.3f})")
        elif theta_ratio < 0.03:
            score += 4
        
        # Gamma (moderate gamma is good)
        if 0.01 <= option['gamma'] <= 0.05:
            score += 5
        
        # 5. Volatility Score (10 points)
        iv = option['implied_volatility']
        if 0.3 <= iv <= 0.6:  # Moderate IV
            score += 10
            reasons.append(f"Reasonable IV ({iv:.1%})")
        elif 0.2 <= iv < 0.3 or 0.6 < iv <= 0.8:
            score += 5
        
        # 6. Spread Score (10 points)
        spread_pct = option['spread_pct']
        if spread_pct <= 0.10:  # Tight spread
            score += 10
            reasons.append("Tight bid-ask spread")
        elif spread_pct <= 0.20:
            score += 7
            reasons.append("Acceptable spread")
        elif spread_pct <= 0.35:
            score += 4
        elif spread_pct <= 0.50:
            score += 2
        
        # 7. Technical Setup Bonus (up to 10 points)
        if stock.get('rsi', 50) < 40:
            score += 5
            reasons.append("Oversold RSI")
        
        if stock.get('relative_strength', 1) > 1.2:
            score += 5
            reasons.append("Strong relative strength")
        
        # Compile analysis
        analysis = {
            'total_score': score,
            'moneyness': moneyness,
            'reasons': reasons,
            'liquidity_assessment': 'Good' if liquidity_score > 50 else 'Fair' if liquidity_score > 30 else 'Low',
            'risk_assessment': 'Low' if score > 70 else 'Medium' if score > 50 else 'High'
        }
        
        return score, analysis
    
    def _print_recommendations(self, symbol: str, current_price: float, 
                              recommendations: List[Dict]):
        """Print detailed recommendations in a user-friendly format"""
        print("\n" + "="*80)
        print(f"OPTIONS RECOMMENDATIONS FOR {symbol}")
        print(f"Current Stock Price: ${current_price:.2f}")
        print("="*80)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n--- Recommendation #{i} (Score: {rec['score']:.1f}/100) ---")
            print(f"CALL Option: ${rec['strike']} Strike")
            print(f"Expiration: {rec['expiration']} ({rec['days_to_expiration']} days)")
            print(f"Entry Price: ${rec['entry_price']:.2f} (Ask: ${rec['ask_price']:.2f}, Bid: ${rec['bid_price']:.2f})")
            
            print(f"\nKey Metrics:")
            print(f"  • Delta: {rec['delta']:.3f}")
            print(f"  • Theta: ${rec['theta']:.3f}/day")
            print(f"  • IV: {rec['implied_volatility']:.1%}")
            print(f"  • Volume: {rec['volume']} | Open Interest: {rec['open_interest']}")
            print(f"  • Probability of Profit: {rec['probability_of_profit']:.1%}")
            print(f"  • Expected Return: {rec['expected_return']:.1%}")
            
            print(f"\nTrading Plan:")
            print(f"  • Entry: ${rec['entry_price']:.2f}")
            print(f"  • Stop Loss: ${rec['stop_loss']:.2f} (-50%)")
            print(f"  • Target 1: ${rec['target_1']:.2f} (+50%)")
            print(f"  • Target 2: ${rec['target_2']:.2f} (+100%)")
            
            print(f"\nReasons to Buy:")
            for reason in rec['recommendation_reasons']:
                print(f"  + {reason}")
        
        print("\n" + "="*80)
        print("Would you like to monitor any of these positions? (Enter 1, 2, 3, or 'n' for none)")
        
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
        """Calculate probability of profit for a call option"""
        # Break-even price
        breakeven = strike + premium
        
        # Use log-normal distribution
        time_to_exp = days / 365.0
        
        # Expected return (neutral assumption)
        drift = 0.0
        
        # Calculate probability
        try:
            variance = (iv ** 2) * time_to_exp
            d1 = (np.log(spot / breakeven) + drift + 0.5 * variance) / np.sqrt(variance)
            prob = norm.cdf(d1)
            return prob
        except:
            return 0.5  # Default 50%
    
    def _calculate_expected_return(self, spot: float, contract: Dict, atr: float) -> float:
        """Calculate expected return based on price targets"""
        strike = contract['strike']
        premium = contract['ask'] if contract['ask'] > 0 else contract['mid']
        days = contract['days_to_expiration']
        
        # Expected move based on ATR and time
        expected_move = atr * np.sqrt(days / 20)  # Scale ATR by time
        expected_price = spot + expected_move * 0.6  # Conservative 60% of expected move
        
        if expected_price > strike:
            profit = expected_price - strike - premium
            return profit / premium if premium > 0 else 0
        else:
            return -1.0  # Total loss
    
    def _calculate_risk_reward(self, contract: Dict) -> float:
        """Calculate risk/reward ratio"""
        premium = contract['ask'] if contract['ask'] > 0 else contract['mid']
        
        # Potential profit at 50% option price increase
        potential_profit = premium * 0.50
        max_loss = premium
        
        return potential_profit / max_loss if max_loss > 0 else 0