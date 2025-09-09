"""
Configuration for Market Cap Options Tracker
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class TradingConfig:
    """Trading parameters"""
    market_cap_min: float = 100_000_000  # $100M minimum
    market_cap_max: float = 100_000_000_000  # $100B maximum
    
    min_days_to_expiration: int = 14
    max_days_to_expiration: int = 365
    target_days_to_expiration: int = 60
    
    max_position_size: float = 0.05
    max_portfolio_risk: float = 0.20
    stop_loss_percent: float = 0.30
    take_profit_percent: float = 0.50
    
    min_delta: float = 0.25
    max_theta_decay_daily: float = 0.02
    max_iv_percentile: float = 75
    
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    min_volume: int = 2_000_000
    min_option_volume: int = 5000
    min_option_oi: int = 1000
    
    theta_exit_threshold: float = 0.03
    profit_exit_threshold: float = 0.40
    days_before_exp_exit: int = 7
    iv_spike_exit: float = 1.5


@dataclass
class ScannerConfig:
    """Market scanner settings"""
    patterns: List[str] = None
    
    min_revenue_growth: float = 0.15
    min_earnings_growth: float = 0.10
    max_pe_ratio: float = 100
    min_institutional_ownership: float = 0.05
    
    min_relative_strength: float = 1.1
    min_price_above_ma: float = 0.05
    
    # Market cap adaptive settings
    micro_cap_volume_min: int = 500_000
    small_cap_volume_min: int = 1_000_000
    mid_cap_volume_min: int = 2_000_000
    large_cap_volume_min: int = 5_000_000
    mega_cap_volume_min: int = 10_000_000
    
    micro_cap_growth_min: float = 0.25
    small_cap_growth_min: float = 0.15
    mid_cap_growth_min: float = 0.10
    large_cap_growth_min: float = 0.05
    mega_cap_growth_min: float = 0.03
    
    micro_cap_pe_max: float = 50
    small_cap_pe_max: float = 75
    mid_cap_pe_max: float = 100
    large_cap_pe_max: float = 150
    mega_cap_pe_max: float = 200
    
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
    """Data source settings"""
    polygon_api_key: Optional[str] = None
    alpha_vantage_key: Optional[str] = None
    yahoo_finance_enabled: bool = True
    finnhub_api_token: Optional[str] = None
    quote_refresh_interval: int = 5
    options_refresh_interval: int = 15
    fundamentals_refresh_interval: int = 1440
    use_cache: bool = True
    cache_expiry_minutes: int = 60


class Config:
    """Main configuration"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.trading = TradingConfig()
        self.scanner = ScannerConfig()
        self.data = DataConfig()
        
        if config_path:
            self.load_from_file(config_path)
            
        # Load API keys from environment
        self._load_api_keys()
        
    def load_from_file(self, path: str):
        """Load configuration from file"""
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
        """Load API keys from environment"""
        import os
        
        self.data.polygon_api_key = os.getenv('POLYGON_API_KEY')
        self.data.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY')
        
    def save_to_file(self, path: str):
        """Save configuration to file"""
        config_dict = {
            'trading': asdict(self.trading),
            'scanner': asdict(self.scanner),
            'data': {k: v for k, v in asdict(self.data).items() 
                    if not k.endswith('_key')}
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_market_cap_category(self, market_cap: float) -> str:
        """Get market cap category for adaptive filtering"""
        if market_cap < 500_000_000:
            return 'micro_cap'
        elif market_cap < 2_000_000_000:
            return 'small_cap'
        elif market_cap < 10_000_000_000:
            return 'mid_cap'
        elif market_cap < 100_000_000_000:
            return 'large_cap'
        else:
            return 'mega_cap'
    
    def get_adaptive_volume_min(self, market_cap: float) -> int:
        """Get adaptive volume minimum based on market cap"""
        category = self.get_market_cap_category(market_cap)
        
        volume_map = {
            'micro_cap': self.scanner.micro_cap_volume_min,
            'small_cap': self.scanner.small_cap_volume_min,
            'mid_cap': self.scanner.mid_cap_volume_min,
            'large_cap': self.scanner.large_cap_volume_min,
            'mega_cap': self.scanner.mega_cap_volume_min
        }
        
        return volume_map.get(category, self.trading.min_volume)
    
    def get_adaptive_growth_min(self, market_cap: float) -> float:
        """Get adaptive growth minimum based on market cap"""
        category = self.get_market_cap_category(market_cap)
        
        growth_map = {
            'micro_cap': self.scanner.micro_cap_growth_min,
            'small_cap': self.scanner.small_cap_growth_min,
            'mid_cap': self.scanner.mid_cap_growth_min,
            'large_cap': self.scanner.large_cap_growth_min,
            'mega_cap': self.scanner.mega_cap_growth_min
        }
        
        return growth_map.get(category, self.scanner.min_revenue_growth)
    
    def get_adaptive_pe_max(self, market_cap: float) -> float:
        """Get adaptive PE maximum based on market cap"""
        category = self.get_market_cap_category(market_cap)
        
        pe_map = {
            'micro_cap': self.scanner.micro_cap_pe_max,
            'small_cap': self.scanner.small_cap_pe_max,
            'mid_cap': self.scanner.mid_cap_pe_max,
            'large_cap': self.scanner.large_cap_pe_max,
            'mega_cap': self.scanner.mega_cap_pe_max
        }
        
        return pe_map.get(category, self.scanner.max_pe_ratio)
    
    def get_adaptive_risk_settings(self, market_cap: float) -> Dict:
        """Get adaptive risk settings based on market cap"""
        category = self.get_market_cap_category(market_cap)
        
        risk_settings = {
            'micro_cap': {
                'max_portfolio_risk': 0.15,
                'max_position_size': 0.03,
                'stop_loss_percent': 0.40,
                'take_profit_percent': 0.60,
                'max_theta_decay_daily': 0.03,
                'min_liquidity_score': 40
            },
            'small_cap': {
                'max_portfolio_risk': 0.18,
                'max_position_size': 0.04,
                'stop_loss_percent': 0.35,
                'take_profit_percent': 0.55,
                'max_theta_decay_daily': 0.025,
                'min_liquidity_score': 50
            },
            'mid_cap': {
                'max_portfolio_risk': 0.20,
                'max_position_size': 0.05,
                'stop_loss_percent': 0.30,
                'take_profit_percent': 0.50,
                'max_theta_decay_daily': 0.02,
                'min_liquidity_score': 60
            },
            'large_cap': {
                'max_portfolio_risk': 0.22,
                'max_position_size': 0.06,
                'stop_loss_percent': 0.25,
                'take_profit_percent': 0.45,
                'max_theta_decay_daily': 0.015,
                'min_liquidity_score': 70
            },
            'mega_cap': {
                'max_portfolio_risk': 0.25,
                'max_position_size': 0.08,
                'stop_loss_percent': 0.20,
                'take_profit_percent': 0.40,
                'max_theta_decay_daily': 0.01,
                'min_liquidity_score': 80
            }
        }
        
        return risk_settings.get(category, {
            'max_portfolio_risk': 0.20,
            'max_position_size': 0.05,
            'stop_loss_percent': 0.30,
            'take_profit_percent': 0.50,
            'max_theta_decay_daily': 0.02,
            'min_liquidity_score': 60
        })
            
    def validate(self) -> List[str]:
        """Validate configuration"""
        issues = []
        
        if self.trading.market_cap_min >= self.trading.market_cap_max:
            issues.append("market_cap_min must be less than market_cap_max")
            
        if self.trading.min_days_to_expiration > self.trading.max_days_to_expiration:
            issues.append("min_days_to_expiration must be <= max_days_to_expiration")
            
        if self.trading.max_position_size > 0.10:
            issues.append("Warning: max_position_size > 10% is risky")
            
        if not self.data.yahoo_finance_enabled and not self.data.polygon_api_key:
            issues.append("No data source configured")
            
        if self.trading.min_volume <= 0:
            issues.append("min_volume must be positive")
            
        if self.trading.stop_loss_percent <= 0 or self.trading.stop_loss_percent > 1:
            issues.append("stop_loss_percent must be between 0 and 1")
            
        if self.trading.take_profit_percent <= 0:
            issues.append("take_profit_percent must be positive")
            
        if self.trading.market_cap_min < 50_000_000:
            issues.append("Warning: market_cap_min < $50M may include very illiquid stocks")
            
        if self.trading.max_days_to_expiration > 365:
            issues.append("Warning: max_days_to_expiration > 365 days may have low liquidity")
            
        return issues
    
    def get_safe_config(self) -> Dict:
        """Get configuration with safe defaults"""
        safe_config = {
            'trading': asdict(self.trading),
            'scanner': asdict(self.scanner),
            'data': {k: v for k, v in asdict(self.data).items() 
                    if not k.endswith('_key')}
        }
        
        required_fields = {
            'trading': ['market_cap_min', 'market_cap_max', 'min_volume', 'max_position_size'],
            'scanner': ['max_pe_ratio', 'min_revenue_growth', 'min_earnings_growth'],
            'data': ['yahoo_finance_enabled', 'use_cache']
        }
        
        for section, fields in required_fields.items():
            for field in fields:
                if field not in safe_config[section]:
                    print(f"Missing required field: {section}.{field}")
        
        return safe_config


default_config = Config()

EXAMPLE_CONFIG_JSON = {
    "trading": {
        "market_cap_min": 100000000,
        "market_cap_max": 1000000000000,
        "min_days_to_expiration": 7,
        "max_days_to_expiration": 365,
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
        "min_volume": 2000000,
        "min_option_volume": 100,
        "min_option_oi": 500,
        "theta_exit_threshold": 0.03,
        "profit_exit_threshold": 0.40,
        "days_before_exp_exit": 7,
        "iv_spike_exit": 1.5
    },
    "scanner": {
        "patterns": ["breakout", "flag", "ascending_triangle", "cup_and_handle", "momentum_surge"],
        "min_revenue_growth": 0.15,
        "min_earnings_growth": 0.10,
        "max_pe_ratio": 100,
        "min_institutional_ownership": 0.05,
        "min_relative_strength": 1.1,
        "min_price_above_ma": 0.05,
        "micro_cap_volume_min": 500000,
        "small_cap_volume_min": 1000000,
        "mid_cap_volume_min": 2000000,
        "large_cap_volume_min": 5000000,
        "mega_cap_volume_min": 10000000,
        "micro_cap_growth_min": 0.25,
        "small_cap_growth_min": 0.15,
        "mid_cap_growth_min": 0.10,
        "large_cap_growth_min": 0.05,
        "mega_cap_growth_min": 0.03,
        "micro_cap_pe_max": 50,
        "small_cap_pe_max": 75,
        "mid_cap_pe_max": 100,
        "large_cap_pe_max": 150,
        "mega_cap_pe_max": 200
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
    import json
    with open('config.json', 'w') as f:
        json.dump(EXAMPLE_CONFIG_JSON, f, indent=2)
    print("Created example config.json file")