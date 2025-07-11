"""
Portfolio Manager module for tracking positions and performance
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages portfolio positions and tracks performance"""
    
    def __init__(self, config, portfolio_file: str = 'data/portfolio.json'):
        self.config = config
        self.portfolio_file = portfolio_file
        self.positions = self._load_portfolio()
        self.initial_capital = 100000  # $100k starting capital
        self.cash = self.positions.get('cash', self.initial_capital)
        
    def _load_portfolio(self) -> Dict:
        """Load portfolio from file"""
        try:
            with open(self.portfolio_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Initialize new portfolio
            return {
                'cash': self.initial_capital,
                'open_positions': [],
                'closed_positions': [],
                'transactions': [],
                'performance_history': []
            }
    
    def _save_portfolio(self):
        """Save portfolio to file"""
        self.positions['cash'] = self.cash
        with open(self.portfolio_file, 'w') as f:
            json.dump(self.positions, f, indent=2, default=str)
    
    def open_position(self, signal: Dict) -> bool:
        """Open a new position based on signal"""
        try:
            # Calculate position size
            position_size = self._calculate_position_size(signal)
            if position_size == 0:
                logger.warning(f"Position size is 0 for {signal['symbol']}")
                return False
                
            # Calculate cost
            contracts = position_size // 100  # Each contract is 100 shares
            if contracts == 0:
                contracts = 1
                
            total_cost = contracts * 100 * signal['entry_price']
            
            # Check if we have enough cash
            if total_cost > self.cash:
                logger.warning(f"Insufficient cash for {signal['symbol']}. Need ${total_cost:.2f}, have ${self.cash:.2f}")
                return False
                
            # Create position entry
            position = {
                'id': f"{signal['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'symbol': signal['symbol'],
                'type': 'CALL',
                'strike': signal['strike'],
                'expiration': signal['expiration'],
                'entry_date': datetime.now().isoformat(),
                'entry_price': signal['entry_price'],
                'entry_iv': signal['implied_volatility'],
                'contracts': contracts,
                'position_size': contracts * 100,
                'total_cost': total_cost,
                'stop_loss': signal['stop_loss'],
                'take_profit': signal['take_profit'],
                'entry_signal': signal,
                'status': 'OPEN'
            }
            
            # Add to positions
            self.positions['open_positions'].append(position)
            
            # Update cash
            self.cash -= total_cost
            
            # Record transaction
            transaction = {
                'date': datetime.now().isoformat(),
                'type': 'BUY',
                'symbol': signal['symbol'],
                'contracts': contracts,
                'price': signal['entry_price'],
                'total': total_cost,
                'position_id': position['id']
            }
            self.positions['transactions'].append(transaction)
            
            # Save portfolio
            self._save_portfolio()
            
            logger.info(f"Opened position: {contracts} contracts of {signal['symbol']} ${signal['strike']}C @ ${signal['entry_price']}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
            return False
    
    def close_position(self, symbol: str, exit_price: float, reason: str) -> bool:
        """Close an existing position"""
        try:
            # Find position
            position = None
            for pos in self.positions['open_positions']:
                if pos['symbol'] == symbol and pos['status'] == 'OPEN':
                    position = pos
                    break
                    
            if not position:
                logger.warning(f"No open position found for {symbol}")
                return False
                
            # Calculate P&L
            exit_value = position['contracts'] * 100 * exit_price
            pnl = exit_value - position['total_cost']
            pnl_percent = (pnl / position['total_cost']) * 100
            
            # Update position
            position['exit_date'] = datetime.now().isoformat()
            position['exit_price'] = exit_price
            position['exit_reason'] = reason
            position['pnl'] = pnl
            position['pnl_percent'] = pnl_percent
            position['status'] = 'CLOSED'
            position['days_held'] = (datetime.now() - datetime.fromisoformat(position['entry_date'])).days
            
            # Move to closed positions
            self.positions['open_positions'].remove(position)
            self.positions['closed_positions'].append(position)
            
            # Update cash
            self.cash += exit_value
            
            # Record transaction
            transaction = {
                'date': datetime.now().isoformat(),
                'type': 'SELL',
                'symbol': symbol,
                'contracts': position['contracts'],
                'price': exit_price,
                'total': exit_value,
                'pnl': pnl,
                'position_id': position['id']
            }
            self.positions['transactions'].append(transaction)
            
            # Save portfolio
            self._save_portfolio()
            
            logger.info(f"Closed position: {symbol} @ ${exit_price} | P&L: ${pnl:.2f} ({pnl_percent:.1f}%) | Reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def roll_position(self, symbol: str, new_strike: float, new_expiration: str, current_price: float) -> bool:
        """Roll a position to new strike/expiration"""
        try:
            # Close current position
            if self.close_position(symbol, current_price, "Rolling position"):
                # Open new position with rolled parameters
                # This would need proper option chain lookup in real implementation
                logger.info(f"Position rolled: {symbol} to ${new_strike}C exp {new_expiration}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error rolling position: {e}")
            return False
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        return [pos for pos in self.positions['open_positions'] if pos['status'] == 'OPEN']
    
    def get_positions_closed_today(self) -> List[Dict]:
        """Get positions closed today"""
        today = datetime.now().date()
        closed_today = []
        
        for pos in self.positions['closed_positions']:
            if 'exit_date' in pos:
                exit_date = datetime.fromisoformat(pos['exit_date']).date()
                if exit_date == today:
                    closed_today.append(pos)
                    
        return closed_today
    
    def get_total_value(self) -> float:
        """Calculate total portfolio value"""
        total = self.cash
        
        # Add value of open positions (simplified - would need real-time quotes)
        for pos in self.get_open_positions():
            # Assume positions maintain entry value for simplicity
            # In production, would fetch current option prices
            total += pos['total_cost']
            
        return total
    
    def calculate_performance(self) -> Dict:
        """Calculate portfolio performance metrics"""
        # Daily P&L
        daily_pnl = 0
        for pos in self.get_positions_closed_today():
            daily_pnl += pos['pnl']
            
        # Total P&L
        total_pnl = 0
        total_wins = 0
        total_losses = 0
        win_sum = 0
        loss_sum = 0
        
        for pos in self.positions['closed_positions']:
            pnl = pos.get('pnl', 0)
            total_pnl += pnl
            
            if pnl > 0:
                total_wins += 1
                win_sum += pnl
            else:
                total_losses += 1
                loss_sum += abs(pnl)
                
        # Win rate
        total_trades = total_wins + total_losses
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        # Average win/loss
        avg_win = win_sum / total_wins if total_wins > 0 else 0
        avg_loss = loss_sum / total_losses if total_losses > 0 else 0
        
        # Profit factor
        profit_factor = win_sum / loss_sum if loss_sum > 0 else float('inf')
        
        return {
            'daily_pnl': daily_pnl,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'winning_trades': total_wins,
            'losing_trades': total_losses,
            'win_rate': win_rate,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'current_equity': self.get_total_value(),
            'return_percent': ((self.get_total_value() - self.initial_capital) / self.initial_capital) * 100
        }
    
    def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size based on Kelly Criterion and risk management"""
        # Get total portfolio value
        portfolio_value = self.get_total_value()
        
        # Maximum position size from config
        max_position = portfolio_value * self.config.trading.max_position_size
        
        # Kelly Criterion calculation (simplified)
        win_prob = signal.get('probability_of_profit', 0.6)
        avg_win = self.config.trading.take_profit_percent
        avg_loss = self.config.trading.stop_loss_percent
        
        # Kelly percentage
        kelly_pct = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        kelly_pct = max(0, min(kelly_pct, 0.25))  # Cap at 25%
        
        # Use half Kelly for safety
        position_size = portfolio_value * kelly_pct * 0.5
        
        # Apply maximum position size limit
        position_size = min(position_size, max_position)
        
        # Round to nearest 100 (option contract size)
        position_size = round(position_size / 100) * 100
        
        return position_size
    
    def get_portfolio_summary(self) -> pd.DataFrame:
        """Get portfolio summary as DataFrame"""
        positions = []
        
        for pos in self.get_open_positions():
            positions.append({
                'Symbol': pos['symbol'],
                'Type': f"${pos['strike']}C",
                'Expiration': pos['expiration'][:10],
                'Entry Price': pos['entry_price'],
                'Contracts': pos['contracts'],
                'Total Cost': pos['total_cost'],
                'Days Held': (datetime.now() - datetime.fromisoformat(pos['entry_date'])).days,
                'Status': 'OPEN'
            })
            
        if positions:
            return pd.DataFrame(positions)
        else:
            return pd.DataFrame()