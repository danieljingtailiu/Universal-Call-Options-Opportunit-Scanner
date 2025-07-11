"""
Risk Manager module for portfolio risk management
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages portfolio risk and position sizing"""
    
    def __init__(self, config):
        self.config = config
        self.risk_free_rate = 0.04  # 4% annual risk-free rate
        
    def can_enter_position(self, signal: Dict, portfolio) -> bool:
        """Check if new position meets risk criteria"""
        try:
            # Get current portfolio state
            current_risk = self.calculate_portfolio_risk(portfolio)
            open_positions = portfolio.get_open_positions()
            portfolio_value = portfolio.get_total_value()
            
            # Check 1: Portfolio risk limit
            if current_risk['total_risk_percent'] >= self.config.trading.max_portfolio_risk * 100:
                logger.warning(f"Portfolio risk limit reached: {current_risk['total_risk_percent']:.1f}%")
                return False
            
            # Check 2: Maximum number of positions
            max_positions = 20  # Maximum 20 open positions
            if len(open_positions) >= max_positions:
                logger.warning(f"Maximum number of positions reached: {len(open_positions)}")
                return False
            
            # Check 3: Position size check
            position_cost = signal.get('ask_price', 0) * 100  # 1 contract = 100 shares
            max_position_value = portfolio_value * self.config.trading.max_position_size
            
            if position_cost > max_position_value:
                logger.warning(f"Position too large: ${position_cost:.2f} > ${max_position_value:.2f}")
                return False
            
            # Check 4: Duplicate position
            for pos in open_positions:
                if pos['symbol'] == signal['symbol']:
                    logger.warning(f"Already have position in {signal['symbol']}")
                    return False
            
            # Check 5: Sector concentration
            if self._check_concentration_risk(signal['symbol'], open_positions, portfolio_value):
                logger.warning(f"Concentration risk too high for {signal['symbol']}")
                return False
            
            # Check 6: Correlation risk
            if self._check_correlation_risk(signal, open_positions):
                logger.warning(f"Correlation risk too high for {signal['symbol']}")
                return False
            
            # Check 7: Greeks risk
            if not self._check_greeks_risk(signal):
                logger.warning(f"Greeks risk outside acceptable range for {signal['symbol']}")
                return False
            
            # Check 8: Liquidity check
            if signal.get('volume', 0) < self.config.trading.min_option_volume:
                logger.warning(f"Option liquidity too low for {signal['symbol']}")
                return False
            
            # All checks passed
            logger.info(f"Risk checks passed for {signal['symbol']}")
            return True
            
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            return False
    
    def calculate_portfolio_risk(self, portfolio) -> Dict:
        """Calculate overall portfolio risk metrics"""
        try:
            open_positions = portfolio.get_open_positions()
            portfolio_value = portfolio.get_total_value()
            
            if not open_positions:
                return {
                    'total_risk_percent': 0,
                    'var_95': 0,
                    'cvar_95': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0,
                    'calmar_ratio': 0,
                    'position_count': 0,
                    'avg_position_size': 0,
                    'largest_position_pct': 0,
                    'correlation_risk': 0,
                    'theta_risk': 0,
                    'positions_at_risk': []
                }
            
            # Calculate total capital at risk
            total_risk = sum(pos['total_cost'] for pos in open_positions)
            total_risk_percent = (total_risk / portfolio_value) * 100
            
            # Position sizes
            position_sizes = [pos['total_cost'] for pos in open_positions]
            avg_position_size = np.mean(position_sizes)
            largest_position_pct = (max(position_sizes) / portfolio_value) * 100
            
            # Calculate VaR and CVaR
            var_95, cvar_95 = self._calculate_var_cvar(portfolio, confidence=0.95)
            
            # Calculate historical metrics
            historical_returns = self._calculate_historical_returns(portfolio)
            
            # Performance ratios
            sharpe_ratio = self._calculate_sharpe_ratio(historical_returns)
            sortino_ratio = self._calculate_sortino_ratio(historical_returns)
            max_drawdown = self._calculate_max_drawdown(portfolio)
            calmar_ratio = self._calculate_calmar_ratio(historical_returns, max_drawdown)
            
            # Greeks aggregation
            total_theta = sum(pos.get('theta', 0) for pos in open_positions)
            theta_risk = abs(total_theta) / portfolio_value * 100  # Daily theta as % of portfolio
            
            # Correlation risk
            correlation_risk = self._calculate_correlation_risk(open_positions)
            
            # Identify positions at risk
            positions_at_risk = self._identify_positions_at_risk(open_positions)
            
            return {
                'total_risk_percent': total_risk_percent,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'position_count': len(open_positions),
                'avg_position_size': avg_position_size,
                'largest_position_pct': largest_position_pct,
                'correlation_risk': correlation_risk,
                'theta_risk': theta_risk,
                'positions_at_risk': positions_at_risk,
                'risk_score': self._calculate_overall_risk_score(
                    total_risk_percent, var_95, correlation_risk, theta_risk
                )
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return self._get_default_risk_metrics()
    
    def _check_concentration_risk(self, symbol: str, open_positions: List[Dict], 
                                 portfolio_value: float) -> bool:
        """Check if adding position would create concentration risk"""
        # Get sector concentrations
        sector_exposure = defaultdict(float)
        
        for pos in open_positions:
            sector = pos.get('sector', 'Unknown')
            sector_exposure[sector] += pos['total_cost']
        
        # Check if any sector exceeds 30% of portfolio
        for sector, exposure in sector_exposure.items():
            if exposure / portfolio_value > 0.30:
                return True
        
        # Check correlation cluster risk
        similar_positions = 0
        for pos in open_positions:
            # Simple correlation check based on price similarity
            if 'strike' in pos and abs(pos['strike'] - 50) < 10:  # Placeholder logic
                similar_positions += 1
        
        return similar_positions >= 5
    
    def _check_correlation_risk(self, signal: Dict, open_positions: List[Dict]) -> bool:
        """Check if new position is too correlated with existing positions"""
        if len(open_positions) == 0:
            return False
        
        # Count positions in similar price ranges (simplified correlation check)
        signal_price = signal.get('strike', 0)
        correlated_count = 0
        
        for pos in open_positions:
            pos_price = pos.get('strike', 0)
            if abs(pos_price - signal_price) / signal_price < 0.20:  # Within 20%
                correlated_count += 1
        
        # Don't allow more than 3 highly correlated positions
        return correlated_count >= 3
    
    def _check_greeks_risk(self, signal: Dict) -> bool:
        """Check if option Greeks are within acceptable ranges"""
        # Delta check
        delta = signal.get('delta', 0)
        if delta < self.config.trading.min_delta:
            return False
        
        # Theta check (as percentage of option value)
        theta = abs(signal.get('theta', 0))
        option_price = signal.get('ask_price', 1)
        theta_percent = theta / option_price if option_price > 0 else 1
        
        if theta_percent > self.config.trading.max_theta_decay_daily:
            return False
        
        # IV check
        iv_percentile = signal.get('iv_percentile', 50)
        if iv_percentile > self.config.trading.max_iv_percentile:
            return False
        
        return True
    
    def _calculate_var_cvar(self, portfolio, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Value at Risk and Conditional Value at Risk"""
        try:
            # Get historical returns or simulate
            returns = self._calculate_historical_returns(portfolio)
            
            if len(returns) < 20:
                # Not enough data, use position-based estimate
                positions = portfolio.get_open_positions()
                position_values = [pos['total_cost'] for pos in positions]
                
                # Assume max loss is premium paid
                var_95 = np.percentile(position_values, 95) if position_values else 0
                cvar_95 = np.mean([v for v in position_values if v >= var_95]) if position_values else 0
                
                return var_95, cvar_95
            
            # Calculate VaR
            returns_array = np.array(returns)
            var_percentile = (1 - confidence) * 100
            var = np.percentile(returns_array, var_percentile)
            
            # Calculate CVaR (expected loss beyond VaR)
            cvar = returns_array[returns_array <= var].mean() if len(returns_array[returns_array <= var]) > 0 else var
            
            # Convert to dollar values
            portfolio_value = portfolio.get_total_value()
            var_dollar = abs(var * portfolio_value)
            cvar_dollar = abs(cvar * portfolio_value)
            
            return var_dollar, cvar_dollar
            
        except Exception as e:
            logger.error(f"Error calculating VaR/CVaR: {e}")
            return 0, 0
    
    def _calculate_historical_returns(self, portfolio) -> List[float]:
        """Calculate historical daily returns"""
        returns = []
        
        try:
            # Get closed positions
            closed_positions = portfolio.positions.get('closed_positions', [])
            
            if not closed_positions:
                return returns
            
            # Group by date
            daily_pnl = defaultdict(float)
            for pos in closed_positions:
                if 'exit_date' in pos and 'pnl' in pos:
                    date = datetime.fromisoformat(pos['exit_date']).date()
                    daily_pnl[date] += pos['pnl']
            
            # Calculate returns
            if daily_pnl:
                sorted_dates = sorted(daily_pnl.keys())
                portfolio_value = portfolio.initial_capital
                
                for date in sorted_dates:
                    pnl = daily_pnl[date]
                    daily_return = pnl / portfolio_value if portfolio_value > 0 else 0
                    returns.append(daily_return)
                    portfolio_value += pnl
                    
        except Exception as e:
            logger.error(f"Error calculating historical returns: {e}")
            
        return returns
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe Ratio"""
        if len(returns) < 2:
            return 0
        
        returns_array = np.array(returns)
        
        # Annualized metrics
        mean_return = np.mean(returns_array) * 252
        std_return = np.std(returns_array) * np.sqrt(252)
        
        if std_return == 0:
            return 0
        
        sharpe = (mean_return - self.risk_free_rate) / std_return
        return round(sharpe, 2)
    
    def _calculate_sortino_ratio(self, returns: List[float]) -> float:
        """Calculate Sortino Ratio (downside deviation)"""
        if len(returns) < 2:
            return 0
        
        returns_array = np.array(returns)
        
        # Annualized metrics
        mean_return = np.mean(returns_array) * 252
        
        # Downside deviation
        negative_returns = returns_array[returns_array < 0]
        if len(negative_returns) == 0:
            return 3.0  # Max value if no negative returns
        
        downside_std = np.std(negative_returns) * np.sqrt(252)
        
        if downside_std == 0:
            return 3.0
        
        sortino = (mean_return - self.risk_free_rate) / downside_std
        return round(min(sortino, 3.0), 2)
    
    def _calculate_max_drawdown(self, portfolio) -> float:
        """Calculate maximum drawdown"""
        try:
            # Build equity curve
            equity_curve = [portfolio.initial_capital]
            
            # Add P&L from closed positions chronologically
            closed_positions = sorted(
                portfolio.positions.get('closed_positions', []),
                key=lambda x: x.get('exit_date', '')
            )
            
            for pos in closed_positions:
                if 'pnl' in pos:
                    equity_curve.append(equity_curve[-1] + pos['pnl'])
            
            # Add unrealized P&L from open positions
            open_positions = portfolio.get_open_positions()
            current_equity = equity_curve[-1]
            for pos in open_positions:
                current_equity += pos.get('unrealized_pnl', 0)
            equity_curve.append(current_equity)
            
            if len(equity_curve) < 2:
                return 0
            
            # Calculate drawdown
            peak = equity_curve[0]
            max_dd = 0
            
            for value in equity_curve[1:]:
                if value > peak:
                    peak = value
                else:
                    dd = (peak - value) / peak
                    max_dd = max(max_dd, dd)
            
            return round(max_dd * 100, 2)  # Return as percentage
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0
    
    def _calculate_calmar_ratio(self, returns: List[float], max_drawdown: float) -> float:
        """Calculate Calmar Ratio (return/max drawdown)"""
        if len(returns) == 0 or max_drawdown == 0:
            return 0
        
        annualized_return = np.mean(returns) * 252 * 100  # As percentage
        
        if max_drawdown == 0:
            return 0
        
        calmar = annualized_return / max_drawdown
        return round(calmar, 2)
    
    def _calculate_correlation_risk(self, positions: List[Dict]) -> float:
        """Calculate portfolio correlation risk score (0-100)"""
        if len(positions) < 2:
            return 0
        
        # Simplified correlation based on sectors and price ranges
        sector_counts = defaultdict(int)
        price_buckets = defaultdict(int)
        
        for pos in positions:
            sector_counts[pos.get('sector', 'Unknown')] += 1
            
            # Price buckets
            strike = pos.get('strike', 0)
            if strike < 20:
                price_buckets['micro'] += 1
            elif strike < 50:
                price_buckets['small'] += 1
            elif strike < 100:
                price_buckets['mid'] += 1
            else:
                price_buckets['large'] += 1
        
        # Calculate concentration scores
        max_sector_concentration = max(sector_counts.values()) / len(positions) if positions else 0
        max_price_concentration = max(price_buckets.values()) / len(positions) if positions else 0
        
        # Combined correlation risk (0-100 scale)
        correlation_risk = ((max_sector_concentration + max_price_concentration) / 2) * 100
        
        return round(correlation_risk, 1)
    
    def _identify_positions_at_risk(self, positions: List[Dict]) -> List[Dict]:
        """Identify positions that are at risk"""
        at_risk = []
        
        for pos in positions:
            try:
                risk_factors = []
                risk_score = 0
                
                # Get position details
                days_held = (datetime.now() - datetime.fromisoformat(pos['entry_date'])).days
                days_to_exp = (datetime.fromisoformat(pos['expiration']) - datetime.now()).days
                pnl_percent = pos.get('pnl_percent', 0)
                
                # Risk factor 1: Near expiration
                if days_to_exp <= 7:
                    risk_factors.append("Expires in < 7 days")
                    risk_score += 40
                elif days_to_exp <= 14:
                    risk_factors.append("Expires in < 14 days")
                    risk_score += 25
                
                # Risk factor 2: Large loss
                if pnl_percent <= -25:
                    risk_factors.append(f"Large loss: {pnl_percent:.1f}%")
                    risk_score += 30
                elif pnl_percent <= -15:
                    risk_factors.append(f"Moderate loss: {pnl_percent:.1f}%")
                    risk_score += 15
                
                # Risk factor 3: Held too long
                if days_held > 30:
                    risk_factors.append(f"Held {days_held} days")
                    risk_score += 20
                
                # Risk factor 4: High theta decay
                theta = pos.get('theta', 0)
                theta_percent = abs(theta) / pos.get('current_price', 1) * 100
                if theta_percent > 3:
                    risk_factors.append(f"High theta: {theta_percent:.1f}%/day")
                    risk_score += 25
                
                if risk_factors:
                    at_risk.append({
                        'symbol': pos['symbol'],
                        'strike': pos['strike'],
                        'expiration': pos['expiration'],
                        'risk_factors': risk_factors,
                        'risk_score': min(risk_score, 100),
                        'days_to_expiration': days_to_exp,
                        'pnl_percent': pnl_percent
                    })
                    
            except Exception as e:
                logger.error(f"Error evaluating risk for position {pos.get('symbol', 'Unknown')}: {e}")
        
        # Sort by risk score
        at_risk.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return at_risk
    
    def _calculate_overall_risk_score(self, total_risk_pct: float, var_95: float,
                                    correlation_risk: float, theta_risk: float) -> float:
        """Calculate overall portfolio risk score (0-100)"""
        # Weight different risk components
        weights = {
            'total_risk': 0.30,
            'var': 0.25,
            'correlation': 0.25,
            'theta': 0.20
        }
        
        # Normalize components to 0-100 scale
        total_risk_score = min(total_risk_pct * 5, 100)  # 20% risk = 100 score
        var_score = min(var_95 / 1000 * 10, 100)  # $10k VaR = 100 score
        
        # Calculate weighted score
        risk_score = (
            weights['total_risk'] * total_risk_score +
            weights['var'] * var_score +
            weights['correlation'] * correlation_risk +
            weights['theta'] * theta_risk
        )
        
        return round(min(risk_score, 100), 1)
    
    def calculate_position_risk_metrics(self, position: Dict, current_data: Dict) -> Dict:
        """Calculate risk metrics for a specific position"""
        try:
            # Greeks-based risk
            delta_risk = current_data['delta'] * current_data['stock_price'] * position['contracts'] * 100
            gamma_risk = 0.5 * current_data['gamma'] * (current_data['stock_price'] ** 2) * position['contracts'] * 100
            theta_risk = current_data['theta'] * position['contracts'] * 100
            vega_risk = current_data['vega'] * position['contracts'] * 100
            
            # Time decay risk
            days_to_exp = (datetime.fromisoformat(position['expiration']) - datetime.now()).days
            time_decay_accelerating = days_to_exp < 21
            
            # Volatility risk
            iv_percentile = current_data.get('iv_percentile', 50)
            high_iv_risk = iv_percentile > 80
            
            # Calculate position-specific risk score
            risk_score = self._calculate_position_risk_score(
                days_to_exp,
                abs(theta_risk),
                iv_percentile,
                position.get('pnl_percent', 0)
            )
            
            return {
                'delta_exposure': round(delta_risk, 2),
                'gamma_exposure': round(gamma_risk, 2),
                'theta_decay_daily': round(theta_risk, 2),
                'vega_exposure': round(vega_risk, 2),
                'days_to_expiration': days_to_exp,
                'time_decay_accelerating': time_decay_accelerating,
                'high_iv_risk': high_iv_risk,
                'iv_percentile': iv_percentile,
                'total_risk_score': risk_score,
                'max_loss': position['total_cost'],
                'risk_reward_ratio': self._calculate_risk_reward_ratio(position, current_data)
            }
            
        except Exception as e:
            logger.error(f"Error calculating position risk metrics: {e}")
            return {}
    
    def _calculate_position_risk_score(self, days_to_exp: int, theta_risk: float,
                                     iv_percentile: float, pnl_percent: float) -> float:
        """Calculate risk score for a position (0-100)"""
        score = 0
        
        # Time risk (0-40 points)
        if days_to_exp < 7:
            score += 40
        elif days_to_exp < 14:
            score += 30
        elif days_to_exp < 21:
            score += 20
        elif days_to_exp < 30:
            score += 10
        
        # Theta risk (0-30 points)
        theta_percent = abs(theta_risk) / 100  # As percentage
        if theta_percent > 5:
            score += 30
        elif theta_percent > 3:
            score += 20
        elif theta_percent > 2:
            score += 10
        
        # IV risk (0-20 points)
        if iv_percentile > 90:
            score += 20
        elif iv_percentile > 75:
            score += 10
        
        # P&L risk (0-10 points)
        if pnl_percent < -20:
            score += 10
        elif pnl_percent < -10:
            score += 5
        
        return min(score, 100)
    
    def _calculate_risk_reward_ratio(self, position: Dict, current_data: Dict) -> float:
        """Calculate risk/reward ratio for position"""
        try:
            max_loss = position['total_cost']
            current_value = current_data.get('mid', 0) * position['contracts'] * 100
            
            # Target based on config
            target_price = position['entry_price'] * (1 + self.config.trading.take_profit_percent)
            potential_profit = (target_price - position['entry_price']) * position['contracts'] * 100
            
            if max_loss == 0:
                return 0
            
            return round(potential_profit / max_loss, 2)
            
        except Exception as e:
            logger.error(f"Error calculating risk/reward: {e}")
            return 0
    
    def generate_risk_report(self, portfolio) -> Dict:
        """Generate comprehensive risk report"""
        try:
            risk_metrics = self.calculate_portfolio_risk(portfolio)
            open_positions = portfolio.get_open_positions()
            
            # Position-level risks
            position_risks = []
            for pos in open_positions:
                position_risks.append({
                    'symbol': pos['symbol'],
                    'strike': pos['strike'],
                    'expiration': pos['expiration'],
                    'cost_basis': pos['total_cost'],
                    'current_value': pos.get('current_value', pos['total_cost']),
                    'percent_of_portfolio': (pos['total_cost'] / portfolio.get_total_value()) * 100,
                    'days_held': (datetime.now() - datetime.fromisoformat(pos['entry_date'])).days,
                    'days_to_expiration': (datetime.fromisoformat(pos['expiration']) - datetime.now()).days,
                    'pnl_percent': pos.get('pnl_percent', 0)
                })
            
            # Generate warnings and recommendations
            warnings = self._generate_risk_warnings(risk_metrics, position_risks)
            recommendations = self._generate_risk_recommendations(risk_metrics)
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_metrics': risk_metrics,
                'position_risks': position_risks,
                'risk_warnings': warnings,
                'recommendations': recommendations,
                'risk_summary': {
                    'overall_risk_level': self._get_risk_level(risk_metrics['risk_score']),
                    'immediate_actions_required': len([w for w in warnings if 'URGENT' in w]),
                    'positions_requiring_attention': len(risk_metrics['positions_at_risk'])
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'portfolio_metrics': self._get_default_risk_metrics()
            }
    
    def _generate_risk_warnings(self, metrics: Dict, position_risks: List[Dict]) -> List[str]:
        """Generate risk warnings based on current metrics"""
        warnings = []
        
        # Portfolio-level warnings
        if metrics['total_risk_percent'] > 18:
            warnings.append(f"URGENT: Portfolio risk very high: {metrics['total_risk_percent']:.1f}% of capital at risk")
        elif metrics['total_risk_percent'] > 15:
            warnings.append(f"High portfolio risk: {metrics['total_risk_percent']:.1f}% of capital at risk")
        
        if metrics['sharpe_ratio'] < 0:
            warnings.append("Negative Sharpe ratio - portfolio underperforming risk-free rate")
        
        if metrics['max_drawdown'] > 25:
            warnings.append(f"URGENT: Maximum drawdown exceeds 25%: {metrics['max_drawdown']:.1f}%")
        elif metrics['max_drawdown'] > 20:
            warnings.append(f"High maximum drawdown: {metrics['max_drawdown']:.1f}%")
        
        if metrics['correlation_risk'] > 70:
            warnings.append(f"High correlation risk: {metrics['correlation_risk']:.1f}% - positions too similar")
        
        if metrics['theta_risk'] > 2:
            warnings.append(f"High daily theta decay: {metrics['theta_risk']:.1f}% of portfolio value")
        
        # Position-specific warnings
        for pos in position_risks:
            if pos['percent_of_portfolio'] > 8:
                warnings.append(f"URGENT: {pos['symbol']} exceeds 8% of portfolio ({pos['percent_of_portfolio']:.1f}%)")
            elif pos['percent_of_portfolio'] > 5:
                warnings.append(f"{pos['symbol']}: Large position size ({pos['percent_of_portfolio']:.1f}% of portfolio)")
            
            if pos['days_to_expiration'] < 7:
                warnings.append(f"URGENT: {pos['symbol']} expires in {pos['days_to_expiration']} days")
            
            if pos['pnl_percent'] < -25:
                warnings.append(f"{pos['symbol']}: Large loss ({pos['pnl_percent']:.1f}%)")
        
        # At-risk positions
        if len(metrics['positions_at_risk']) > 0:
            warnings.append(f"{len(metrics['positions_at_risk'])} positions identified as high risk")
        
        return warnings
    
    def _generate_risk_recommendations(self, metrics: Dict) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if metrics['total_risk_percent'] > 15:
            recommendations.append("Reduce position sizes or close some positions to lower portfolio risk")
        
        if metrics['sharpe_ratio'] < 0.5 and metrics['position_count'] > 0:
            recommendations.append("Review entry criteria - risk-adjusted returns are suboptimal")
        
        if metrics['correlation_risk'] > 60:
            recommendations.append("Diversify into different sectors or price ranges to reduce correlation")
        
        if metrics['theta_risk'] > 1.5:
            recommendations.append("Consider closing positions with high theta decay")
        
        if len(metrics['positions_at_risk']) > 3:
            recommendations.append("Review and potentially exit high-risk positions")
        
        if metrics['max_drawdown'] > 15:
            recommendations.append("Implement tighter stop-losses to reduce drawdown risk")
        
        if metrics['largest_position_pct'] > 7:
            recommendations.append("Rebalance portfolio to reduce concentration in largest position")
        
        if metrics['position_count'] > 15:
            recommendations.append("Consider reducing number of positions for better focus and management")
        
        return recommendations
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score < 20:
            return "LOW"
        elif risk_score < 40:
            return "MODERATE"
        elif risk_score < 60:
            return "HIGH"
        else:
            return "CRITICAL"