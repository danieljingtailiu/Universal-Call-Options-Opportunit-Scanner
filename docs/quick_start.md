# 🚀 Quick Start Guide

## Prerequisites

- Python 3.8 or higher
- Git
- Internet connection for data fetching

## Step 1: Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/Market-Cap-Options-Tracker.git
cd Market-Cap-Options-Tracker
```

### Set Up Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Step 2: Configuration

### Basic Configuration
Create or edit `config.json`:

```json
{
  "trading": {
    "market_cap_min": 100000000,     // $100M minimum (micro-cap)
    "market_cap_max": 1000000000000, // $1T maximum (mega-cap)
    "min_volume": 2000000,           // Minimum daily volume
    "max_position_size": 0.05,       // 5% max position size
    "stop_loss_percent": 0.30,       // 30% stop loss
    "take_profit_percent": 0.50      // 50% profit target
  },
  "scanner": {
    "min_revenue_growth": 0.15,      // 15% revenue growth
    "min_earnings_growth": 0.10,     // 10% earnings growth
    "max_pe_ratio": 100             // Maximum PE ratio
  }
}
```

### Market Cap Strategy Examples

#### Micro-Cap Strategy (High Risk/High Reward)
```json
{
  "trading": {
    "market_cap_min": 100000000,     // $100M
    "market_cap_max": 500000000,     // $500M
    "min_volume": 500000,            // Lower volume requirement
    "max_position_size": 0.03        // Smaller positions (3%)
  },
  "scanner": {
    "min_revenue_growth": 0.25,      // Higher growth requirement
    "max_pe_ratio": 50              // Lower PE requirement
  }
}
```

#### Large-Cap Strategy (Lower Risk/Stable Returns)
```json
{
  "trading": {
    "market_cap_min": 10000000000,   // $10B
    "market_cap_max": 100000000000,  // $100B
    "min_volume": 5000000,           // Higher volume requirement
    "max_position_size": 0.06        // Larger positions (6%)
  },
  "scanner": {
    "min_revenue_growth": 0.05,      // Lower growth requirement
    "max_pe_ratio": 150             // Higher PE allowance
  }
}
```

### Advanced Configuration
For more control, you can modify the configuration programmatically:

```python
from config import Config

# Micro-Cap Strategy
micro_config = Config()
micro_config.trading.market_cap_min = 100_000_000   # $100M
micro_config.trading.market_cap_max = 500_000_000   # $500M
micro_config.trading.min_volume = 500_000           # Lower volume requirement
micro_config.scanner.min_revenue_growth = 0.25      # Higher growth requirement
micro_config.save_to_file('micro_cap_config.json')

# Large-Cap Strategy
large_config = Config()
large_config.trading.market_cap_min = 10_000_000_000  # $10B
large_config.trading.market_cap_max = 100_000_000_000 # $100B
large_config.trading.min_volume = 5_000_000           # Higher volume requirement
large_config.scanner.min_revenue_growth = 0.05        # Lower growth requirement
large_config.save_to_file('large_cap_config.json')
```

## Step 3: First Run

### Clear Cache (Recommended for First Run)
```bash
python main.py --clear-cache
```

### Scan for Opportunities
```bash
python main.py --scan
```

Expected output:
```
================================================================================
SCANNING FOR OPTIONS OPPORTUNITIES
================================================================================

1. Finding stocks within market cap range...
   Found 2,847 stocks in range

2. Applying technical analysis...
   156 stocks passed technical filters

3. Analyzing options for 25 stocks...
   Analyzing NVDA...
   Analyzing AMD...
   Analyzing TSLA...

================================================================================
🚀 TOP 10 CALL OPTIONS RECOMMENDATIONS
================================================================================

1. NVDA @ $450.25 (Large-Cap)
   $460C 2024-02-16 (45d)
   Entry: $12.50 | BE: +4.9% | Score: 87
   ✓ Optimal strike near money ($460.00)

2. AMD @ $145.80 (Large-Cap)
   $150C 2024-02-16 (45d)
   Entry: $8.75 | BE: +8.9% | Score: 82
   ✓ Good OTM strike for momentum ($150.00)

3. CRWD @ $85.20 (Mid-Cap)
   $90C 2024-02-16 (45d)
   Entry: $4.50 | BE: +11.1% | Score: 76
   ✓ High growth cybersecurity play
```

### Monitor Positions
```bash
python main.py --monitor
```

## Step 4: Understanding the Output

### Recommendation Format
```
1. SYMBOL @ $CURRENT_PRICE (Market-Cap-Category)
   $STRIKE C EXPIRATION_DATE (DAYS_TO_EXPIRATION)
   Entry: $ENTRY_PRICE | BE: +BREAKEVEN_PERCENT% | Score: SCORE
   ✓ REASON_FOR_RECOMMENDATION
```

### Key Metrics Explained
- **Entry Price**: Recommended entry price for the option
- **BE (Breakeven)**: Stock price increase needed to break even
- **Score**: Overall recommendation score (0-100)
- **Days to Expiration**: Time remaining until option expires
- **Market Cap Category**: Micro/Small/Mid/Large/Mega-Cap classification

## Step 5: Position Monitoring

### Adding Positions to Monitor
When you see recommendations you like, you can add them to monitoring:

```
📊 MONITOR POSITIONS?
Enter numbers (1-10) separated by commas, or 'n' for none:
> 1,3,5

How many contracts of NVDA $460C?
> 2

✅ Added 2 contracts to monitoring
```

### Monitoring Output
```
--- NVDA $460 2024-02-16 ---
Entry: $12.50 | Current: $18.75
P&L: $6.25 (+50.0%)
Stock: $475.25 | Days to Exp: 35
Action: HOLD - Consider taking partial profits
```

## Step 6: Customization

### Adjusting Filters by Market Cap
```python
# Modify scanner settings for different market caps
config = Config()

# For Micro-Cap stocks
config.scanner.micro_cap_growth_min = 0.30      # 30% growth requirement
config.scanner.micro_cap_volume_min = 300000    # Lower volume requirement
config.scanner.micro_cap_pe_max = 40           # Lower PE requirement

# For Large-Cap stocks
config.scanner.large_cap_growth_min = 0.03      # 3% growth requirement
config.scanner.large_cap_volume_min = 10000000  # Higher volume requirement
config.scanner.large_cap_pe_max = 200          # Higher PE allowance

# Save changes
config.save_to_file('config.json')
```

### Adding Custom Filters
```python
# Add market cap specific filters
def market_cap_filter(stock_data):
    market_cap = stock_data['market_cap']
    
    if market_cap < 500_000_000:  # Micro-Cap
        return (
            stock_data['volume'] > 500_000 and
            stock_data['revenue_growth'] > 0.25 and
            stock_data['pe_ratio'] < 50
        )
    elif market_cap < 2_000_000_000:  # Small-Cap
        return (
            stock_data['volume'] > 1_000_000 and
            stock_data['revenue_growth'] > 0.15 and
            stock_data['pe_ratio'] < 75
        )
    else:  # Mid/Large/Mega-Cap
        return (
            stock_data['volume'] > 2_000_000 and
            stock_data['revenue_growth'] > 0.10 and
            stock_data['pe_ratio'] < 100
        )
```

## Step 7: Advanced Usage

### Programmatic Usage
```python
from main import OptionsTracker

# Initialize tracker
tracker = OptionsTracker('config.json')

# Find opportunities
opportunities = tracker.find_opportunities(top_n=10)

# Monitor positions
exit_signals = tracker.monitor_positions()

# Process exit signals
for signal in exit_signals:
    if signal['action'] == 'SELL':
        print(f"SELL {signal['symbol']}: {signal['recommendation']}")
```

### Batch Processing with Different Strategies
```python
import schedule
import time

def micro_cap_scan():
    tracker = OptionsTracker('micro_cap_config.json')
    tracker.find_opportunities()

def large_cap_scan():
    tracker = OptionsTracker('large_cap_config.json')
    tracker.find_opportunities()

def monitor_positions():
    tracker = OptionsTracker('config.json')
    tracker.monitor_positions()

# Schedule tasks
schedule.every().day.at("09:30").do(micro_cap_scan)
schedule.every().day.at("10:00").do(large_cap_scan)
schedule.every().hour.do(monitor_positions)

# Run scheduler
while True:
    schedule.run_pending()
    time.sleep(60)
```

## Troubleshooting

### Common Issues

#### 1. No Opportunities Found
```bash
# Clear cache and retry
python main.py --clear-cache
python main.py --scan
```

#### 2. Rate Limit Errors
```bash
# Wait and retry, or adjust rate limits in config
# The system automatically handles rate limiting
```

#### 3. Data Fetching Errors
```bash
# Check internet connection
# Verify API keys (if using premium data sources)
# Check market hours (some data may not be available outside market hours)
```

### Debug Mode
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
tracker = OptionsTracker('config.json')
tracker.find_opportunities()
```

## Market Cap Strategy Guide

### Micro-Cap Strategy ($100M - $500M)
- **Risk Level**: High
- **Growth Requirement**: 25%+ revenue growth
- **Volume Requirement**: 500K+ daily volume
- **Position Size**: 3% max
- **Stop Loss**: 40%
- **Profit Target**: 60%

### Small-Cap Strategy ($500M - $2B)
- **Risk Level**: Medium-High
- **Growth Requirement**: 15%+ revenue growth
- **Volume Requirement**: 1M+ daily volume
- **Position Size**: 4% max
- **Stop Loss**: 35%
- **Profit Target**: 55%

### Mid-Cap Strategy ($2B - $10B)
- **Risk Level**: Medium
- **Growth Requirement**: 10%+ revenue growth
- **Volume Requirement**: 2M+ daily volume
- **Position Size**: 5% max
- **Stop Loss**: 30%
- **Profit Target**: 50%

### Large-Cap Strategy ($10B - $100B)
- **Risk Level**: Medium-Low
- **Growth Requirement**: 5%+ revenue growth
- **Volume Requirement**: 5M+ daily volume
- **Position Size**: 6% max
- **Stop Loss**: 25%
- **Profit Target**: 45%

### Mega-Cap Strategy ($100B+)
- **Risk Level**: Low
- **Growth Requirement**: 3%+ revenue growth
- **Volume Requirement**: 10M+ daily volume
- **Position Size**: 8% max
- **Stop Loss**: 20%
- **Profit Target**: 40%

## Next Steps

1. **Review Recommendations**: Understand why each option was recommended
2. **Paper Trading**: Test the system with paper trading before real money
3. **Customize Filters**: Adjust parameters based on your risk tolerance
4. **Monitor Performance**: Track your results and adjust strategies
5. **Join Community**: Share experiences and learn from others

## Support

- **Documentation**: Check the main README.md for detailed information
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Ask questions on GitHub Discussions
- **Contributing**: Submit improvements via Pull Requests

---

**Happy Trading! 🚀** 