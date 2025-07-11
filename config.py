"""
Configuration module for Small-Cap Options Tracker
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class TradingConfig:
    """Trading parameters configuration"""
    market_cap_min: float = 500_000_000  # $500M
    market_cap_max: float = 10_000_000_000  # $10B
    
    # Options parameters
    min_days_to_expiration: int = 30
    max_days_to_expiration: int = 60
    target_days_to_expiration: int = 45
    
    # Risk parameters
    max_position_size: float = 0.05  # 5% of portfolio per position
    max_portfolio_risk: float = 0.20  # 20% total portfolio risk
    stop_loss_percent: float = 0.30  # 30% stop loss on options
    take_profit_percent: float = 0.50  # 50% take profit
    
    # Greeks thresholds
    min_delta: float = 0.25  # Minimum delta for call options
    max_theta_decay_daily: float = 0.02  # Max 2% daily theta decay
    max_iv_percentile: float = 75  # Max IV percentile (avoid overpriced)
    
    # Technical indicators
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    min_volume: int = 1_000_000  # Minimum daily volume
    min_option_volume: int = 100  # Minimum option volume
    min_option_oi: int = 500  # Minimum open interest
    
    # Exit conditions
    theta_exit_threshold: float = 0.03  # Exit if theta > 3% daily
    profit_exit_threshold: float = 0.40  # Exit at 40% profit
    days_before_exp_exit: int = 7  # Exit 7 days before expiration
    iv_spike_exit: float = 1.5  # Exit if IV increases 50%


@dataclass
class ScannerConfig:
    """Market scanner configuration"""
    # Technical patterns to look for
    patterns: List[str] = None
    
    # Fundamental filters
    min_revenue_growth: float = 0.10  # 10% revenue growth
    min_earnings_growth: float = 0.05  # 5% earnings growth
    max_pe_ratio: float = 50  # Maximum P/E ratio
    min_institutional_ownership: float = 0.10  # 10% institutional ownership
    
    # Momentum indicators
    min_relative_strength: float = 0.60  # RS vs market
    min_price_above_ma: float = 0.02  # 2% above 20-day MA
    
    def __post_init__(self):
        if self.patterns is None:
            self.patterns = [
                'breakout',
                'flag',
                'ascending_triangle',
                'cup_and_handle',
                'momentum_surge'
            ]


@dataclass
class DataConfig:
    """Data source configuration"""
    # API keys (set these in environment variables)
    polygon_api_key: Optional[str] = None
    alpha_vantage_key: Optional[str] = None
    yahoo_finance_enabled: bool = True
    
    # Data refresh intervals (minutes)
    quote_refresh_interval: int = 5
    options_refresh_interval: int = 15
    fundamentals_refresh_interval: int = 1440  # Daily
    
    # Cache settings
    use_cache: bool = True
    cache_expiry_minutes: int = 60


class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.trading = TradingConfig()
        self.scanner = ScannerConfig()
        self.data = DataConfig()
        
        if config_path:
            self.load_from_file(config_path)
            
        # Load API keys from environment
        self._load_api_keys()
        
    def load_from_file(self, path: str):
        """Load configuration from JSON file"""
        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
                
            # Update trading config
            if 'trading' in config_dict:
                for key, value in config_dict['trading'].items():
                    if hasattr(self.trading, key):
                        setattr(self.trading, key, value)
                        
            # Update scanner config
            if 'scanner' in config_dict:
                for key, value in config_dict['scanner'].items():
                    if hasattr(self.scanner, key):
                        setattr(self.scanner, key, value)
                        
            # Update data config
            if 'data' in config_dict:
                for key, value in config_dict['data'].items():
                    if hasattr(self.data, key):
                        setattr(self.data, key, value)
                        
        except FileNotFoundError:
            print(f"Config file {path} not found, using defaults")
        except json.JSONDecodeError:
            print(f"Error parsing config file {path}, using defaults")
            
    def _load_api_keys(self):
        """Load API keys from environment variables"""
        import os
        
        self.data.polygon_api_key = os.getenv('POLYGON_API_KEY')
        self.data.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY')
        
    def save_to_file(self, path: str):
        """Save current configuration to JSON file"""
        config_dict = {
            'trading': asdict(self.trading),
            'scanner': asdict(self.scanner),
            'data': {k: v for k, v in asdict(self.data).items() 
                    if not k.endswith('_key')}  # Don't save API keys
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate trading config
        if self.trading.market_cap_min >= self.trading.market_cap_max:
            issues.append("market_cap_min must be less than market_cap_max")
            
        if self.trading.min_days_to_expiration > self.trading.max_days_to_expiration:
            issues.append("min_days_to_expiration must be <= max_days_to_expiration")
            
        if self.trading.max_position_size > 0.10:
            issues.append("Warning: max_position_size > 10% is risky")
            
        # Validate data config
        if not self.data.yahoo_finance_enabled and not self.data.polygon_api_key:
            issues.append("No data source configured")
            
        return issues


# Default configuration instance
default_config = Config()

# Example configuration file content
EXAMPLE_CONFIG_JSON = {
    "trading": {
        "market_cap_min": 500000000,
        "market_cap_max": 10000000000,
        "min_days_to_expiration": 30,
        "max_days_to_expiration": 60,
        "target_days_to_expiration": 45,
        "max_position_size": 0.05,
        "max_portfolio_risk": 0.20,
        "stop_loss_percent": 0.30,
        "take_profit_percent": 0.50,
        "min_delta": 0.25,
        "max_theta_decay_daily": 0.02,
        "max_iv_percentile": 75,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "min_volume": 1000000,
        "min_option_volume": 100,
        "min_option_oi": 500,
        "theta_exit_threshold": 0.03,
        "profit_exit_threshold": 0.40,
        "days_before_exp_exit": 7,
        "iv_spike_exit": 1.5
    },
    "scanner": {
        "patterns": ["breakout", "flag", "ascending_triangle", "cup_and_handle", "momentum_surge"],
        "min_revenue_growth": 0.10,
        "min_earnings_growth": 0.05,
        "max_pe_ratio": 50,
        "min_institutional_ownership": 0.10,
        "min_relative_strength": 0.60,
        "min_price_above_ma": 0.02
    },
    "data": {
        "yahoo_finance_enabled": True,
        "quote_refresh_interval": 5,
        "options_refresh_interval": 15,
        "fundamentals_refresh_interval": 1440,
        "use_cache": True,
        "cache_expiry_minutes": 60
    }
}


if __name__ == '__main__':
    # Create example config file
    import json
    with open('config.json', 'w') as f:
        json.dump(EXAMPLE_CONFIG_JSON, f, indent=2)
    print("Created example config.json file")