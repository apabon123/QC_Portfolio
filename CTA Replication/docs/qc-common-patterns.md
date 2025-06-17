# QuantConnect Common Patterns for CTA Framework

> **Best practices and proven patterns for QuantConnect CTA development**

---

## 🏗️ **ALGORITHM STRUCTURE PATTERNS**

### **Standard QC Algorithm Template**
```python
from AlgorithmImports import *

class OptimizedCTAFramework(QCAlgorithm):
    
    def Initialize(self):
        """Standard QC initialization pattern"""
        # Basic setup using LEAN's methods
        self.SetStartDate(2020, 1, 1)
        self.SetCash(1000000)
        self.SetBenchmark("SPY")
        
        # Warm-up using LEAN's built-in system
        self.SetWarmUp(timedelta(days=60))
        self.Settings.AutomaticIndicatorWarmUp = True
        
        # Initialize components
        self._initialize_configuration()
        self._initialize_universe()
        self._initialize_strategies()
        self._initialize_scheduling()
    
    def OnData(self, slice):
        """Standard OnData pattern"""
        if self.IsWarmingUp:  # LEAN's native property
            return
            
        # Process data using LEAN's slice structure
        self._process_slice_data(slice)
        
    def OnWarmupFinished(self):  # LEAN's native event
        """Post-warmup initialization"""
        self.Log("Warm-up completed - algorithm ready")
```

### **Component Initialization Pattern**
```python
def _initialize_configuration(self):
    """Centralized configuration using LEAN's ObjectStore"""
    # Use LEAN's persistent storage for configuration
    if self.ObjectStore.ContainsKey("config"):
        config = self.ObjectStore.Read("config")
    else:
        config = self._load_default_config()
        self.ObjectStore.Save("config", config)

def _initialize_universe(self):
    """Universe setup using LEAN's AddFuture"""
    # CTA futures universe
    futures_symbols = ["ES", "NQ", "YM", "ZN", "ZB", "6E", "CL", "GC"]
    
    for ticker in futures_symbols:
        future = self.AddFuture(ticker, Resolution.Daily,
            dataMappingMode=DataMappingMode.OpenInterest,
            dataNormalizationMode=DataNormalizationMode.BackwardsRatio)
        
        # Store symbol for later use
        setattr(self, f"{ticker.lower()}_symbol", future.Symbol)

def _initialize_strategies(self):
    """Strategy initialization using LEAN's indicators"""
    # Initialize strategies with LEAN indicators
    self.strategies = {}
    
    for symbol in self.futures_symbols:
        self.strategies[symbol] = {
            'sma_fast': self.SMA(symbol, 20),
            'sma_slow': self.SMA(symbol, 50),
            'rsi': self.RSI(symbol, 14),
            'atr': self.ATR(symbol, 20)
        }

def _initialize_scheduling(self):
    """Scheduling using LEAN's Schedule.On"""
    # Weekly rebalancing
    self.Schedule.On(
        self.DateRules.Every(DayOfWeek.Friday),
        self.TimeRules.BeforeMarketClose("ES", 30),
        self.WeeklyRebalance
    )
    
    # Monthly allocation review
    self.Schedule.On(
        self.DateRules.MonthStart("ES"),
        self.TimeRules.AfterMarketOpen("ES", 60),
        self.MonthlyAllocationReview
    )
```

---

## 📊 **DATA ACCESS PATTERNS**

### **Futures Chain Analysis Pattern**
```python
def analyze_futures_chains(self, slice):
    """Standard pattern for futures chain analysis"""
    liquid_contracts = {}
    
    for symbol in self.futures_symbols:
        if symbol in slice.FuturesChains:  # LEAN's native FuturesChains
            chain = slice.FuturesChains[symbol]
            
            if len(chain) > 0:
                # Find most liquid contract using LEAN's chain data
                liquid_contract = max(chain, key=lambda x: x.Volume)
                
                liquid_contracts[symbol] = {
                    'contract': liquid_contract,
                    'volume': liquid_contract.Volume,
                    'open_interest': liquid_contract.OpenInterest,
                    'expiry': liquid_contract.Expiry
                }
    
    return liquid_contracts

def OnData(self, slice):
    """Process futures data using LEAN's native methods"""
    if self.IsWarmingUp:
        return
    
    # Handle rollover events using LEAN's SymbolChangedEvents
    for changed in slice.SymbolChangedEvents.Values:
        self.handle_rollover(changed.OldSymbol, changed.NewSymbol)
    
    # Analyze chains
    liquid_contracts = self.analyze_futures_chains(slice)
    
    # Generate signals using current contract data
    self.generate_strategy_signals(liquid_contracts)
```

### **Historical Data Pattern**
```python
def get_momentum_data(self, symbol, lookback_days):
    """Standard pattern for historical data retrieval"""
    # Use LEAN's native History method
    history = self.History(symbol, timedelta(days=lookback_days), Resolution.Daily)
    
    if history.empty:
        return None
    
    # Process using pandas (already available in LEAN)
    closes = history['close']
    returns = closes.pct_change().dropna()
    
    return {
        'prices': closes,
        'returns': returns,
        'volatility': returns.std() * np.sqrt(252),
        'momentum': (closes.iloc[-1] / closes.iloc[0]) - 1
    }
```

---

## 🎯 **STRATEGY IMPLEMENTATION PATTERNS**

### **Technical Strategy Pattern**
```python
class MomentumStrategy:
    """Standard momentum strategy using LEAN indicators"""
    
    def __init__(self, algorithm, symbol, config):
        self.algorithm = algorithm
        self.symbol = symbol
        self.config = config
        
        # Use LEAN's built-in indicators - NEVER custom implementations
        self.sma_fast = algorithm.SMA(symbol, config['sma_fast_period'])
        self.sma_slow = algorithm.SMA(symbol, config['sma_slow_period'])
        self.rsi = algorithm.RSI(symbol, config['rsi_period'])
        self.atr = algorithm.ATR(symbol, config['atr_period'])
        
    def generate_signal(self):
        """Generate trading signal using LEAN indicator values"""
        if not (self.sma_fast.IsReady and self.sma_slow.IsReady and self.rsi.IsReady):
            return 0
        
        # Trend signal
        trend_signal = 1 if self.sma_fast.Current.Value > self.sma_slow.Current.Value else -1
        
        # RSI filter
        rsi_value = self.rsi.Current.Value
        rsi_filter = 1 if 30 < rsi_value < 70 else 0
        
        # Position sizing using ATR
        atr_value = self.atr.Current.Value
        volatility_adjustment = self._calculate_position_size(atr_value)
        
        return trend_signal * rsi_filter * volatility_adjustment
    
    def _calculate_position_size(self, atr_value):
        """Position sizing using LEAN's CalculateOrderQuantity"""
        target_risk = self.config['target_risk_per_trade']
        
        # Use LEAN's portfolio methods
        portfolio_value = self.algorithm.Portfolio.TotalPortfolioValue
        risk_amount = portfolio_value * target_risk
        
        # Calculate position size based on ATR
        if atr_value > 0:
            return min(risk_amount / atr_value, self.config['max_position_size'])
        return 0
```

### **Multi-Strategy Combination Pattern**
```python
def combine_strategy_signals(self):
    """Standard pattern for combining multiple strategies"""
    combined_signals = {}
    
    for symbol in self.futures_symbols:
        # Get signals from each strategy
        momentum_signal = self.momentum_strategy.generate_signal(symbol)
        mean_reversion_signal = self.mean_reversion_strategy.generate_signal(symbol)
        breakout_signal = self.breakout_strategy.generate_signal(symbol)
        
        # Weight combination using configuration
        weights = self.config['strategy_weights']
        combined_signal = (
            momentum_signal * weights['momentum'] +
            mean_reversion_signal * weights['mean_reversion'] +
            breakout_signal * weights['breakout']
        )
        
        combined_signals[symbol] = combined_signal
    
    return combined_signals
```

---

## 💰 **PORTFOLIO MANAGEMENT PATTERNS**

### **Position Sizing Pattern**
```python
def calculate_position_sizes(self, signals):
    """Standard position sizing using LEAN's portfolio methods"""
    position_targets = {}
    
    # Use LEAN's native portfolio properties
    total_value = self.Portfolio.TotalPortfolioValue
    available_cash = self.Portfolio.Cash
    
    for symbol, signal in signals.items():
        if signal == 0:
            position_targets[symbol] = 0
            continue
        
        # Risk-based position sizing
        risk_per_trade = self.config['risk_per_trade']
        max_position_size = self.config['max_position_size']
        
        # Get current price using LEAN's Securities
        if symbol in self.Securities and self.Securities[symbol].HasData:
            current_price = self.Securities[symbol].Price
            
            # Calculate target position size
            risk_amount = total_value * risk_per_trade
            position_size = min(risk_amount / current_price, max_position_size)
            
            # Apply signal direction
            position_targets[symbol] = position_size * np.sign(signal)
    
    return position_targets

def execute_portfolio_changes(self, position_targets):
    """Execute trades using LEAN's order methods"""
    for symbol, target_quantity in position_targets.items():
        # Get current position using LEAN's Portfolio
        current_quantity = self.Portfolio[symbol].Quantity
        
        # Calculate required trade
        trade_quantity = target_quantity - current_quantity
        
        if abs(trade_quantity) > self.config['min_trade_size']:
            # Use LEAN's native order methods
            if abs(target_quantity) < self.config['min_position_size']:
                # Close position
                self.Liquidate(symbol, "Position too small")
            else:
                # Adjust position using SetHoldings for percentage-based
                target_percentage = target_quantity / self.Portfolio.TotalPortfolioValue
                self.SetHoldings(symbol, target_percentage)
```

---

## 🛡️ **RISK MANAGEMENT PATTERNS**

### **Portfolio Risk Control Pattern**
```python
def apply_portfolio_risk_controls(self, position_targets):
    """Standard risk control using LEAN's portfolio properties"""
    # Use LEAN's built-in risk properties
    total_value = self.Portfolio.TotalPortfolioValue
    margin_used = self.Portfolio.TotalMarginUsed
    
    # Check leverage limits
    max_leverage = self.config['max_leverage']
    projected_margin = sum(abs(target) * self.Securities[symbol].Price 
                          for symbol, target in position_targets.items()
                          if symbol in self.Securities)
    
    projected_leverage = projected_margin / total_value
    
    if projected_leverage > max_leverage:
        # Scale down positions
        scale_factor = max_leverage / projected_leverage
        position_targets = {symbol: target * scale_factor 
                           for symbol, target in position_targets.items()}
        
        self.Log(f"Scaling positions by {scale_factor:.2f} due to leverage limit")
    
    # Check drawdown limits
    self.check_drawdown_limits()
    
    return position_targets

def check_drawdown_limits(self):
    """Drawdown monitoring using LEAN's performance tracking"""
    # Use LEAN's built-in performance tracking
    current_value = self.Portfolio.TotalPortfolioValue
    
    # Track high water mark
    if not hasattr(self, 'high_water_mark'):
        self.high_water_mark = current_value
    else:
        self.high_water_mark = max(self.high_water_mark, current_value)
    
    # Calculate drawdown
    drawdown = (self.high_water_mark - current_value) / self.high_water_mark
    max_drawdown = self.config['max_drawdown']
    
    if drawdown > max_drawdown:
        # Emergency liquidation using LEAN's Liquidate
        self.Liquidate()
        self.Log(f"Emergency liquidation: Drawdown {drawdown:.2%} exceeds limit {max_drawdown:.2%}")
        
        # Halt trading
        self.Quit("Maximum drawdown exceeded")
```

---

## 📅 **SCHEDULING PATTERNS**

### **Multi-Frequency Rebalancing Pattern**
```python
def setup_rebalancing_schedule(self):
    """Standard scheduling pattern for CTA strategies"""
    
    # Daily signal generation
    self.Schedule.On(
        self.DateRules.EveryDay("ES"),
        self.TimeRules.AfterMarketOpen("ES", 30),
        self.daily_signal_update
    )
    
    # Weekly portfolio rebalancing
    self.Schedule.On(
        self.DateRules.Every(DayOfWeek.Friday),
        self.TimeRules.BeforeMarketClose("ES", 30),
        self.weekly_rebalance
    )
    
    # Monthly allocation review
    self.Schedule.On(
        self.DateRules.MonthStart("ES"),
        self.TimeRules.At(10, 0),
        self.monthly_allocation_review
    )
    
    # Risk monitoring (multiple times per day)
    self.Schedule.On(
        self.DateRules.EveryDay("ES"),
        self.TimeRules.Every(TimeSpan.FromHours(2)),
        self.risk_monitoring_update
    )

def daily_signal_update(self):
    """Daily signal updates using LEAN indicators"""
    # Update signals without trading
    for symbol in self.futures_symbols:
        self.update_strategy_signals(symbol)

def weekly_rebalance(self):
    """Weekly rebalancing using LEAN's order management"""
    if not self.IsMarketOpen("ES"):
        return
    
    # Generate new position targets
    signals = self.generate_combined_signals()
    position_targets = self.calculate_position_sizes(signals)
    
    # Apply risk controls
    risk_adjusted_targets = self.apply_portfolio_risk_controls(position_targets)
    
    # Execute trades
    self.execute_portfolio_changes(risk_adjusted_targets)

def monthly_allocation_review(self):
    """Monthly strategy allocation review"""
    # Analyze strategy performance using LEAN's tracking
    performance_data = self.analyze_strategy_performance()
    
    # Adjust allocations based on performance
    self.update_strategy_allocations(performance_data)
```

---

## 🔍 **MONITORING & LOGGING PATTERNS**

### **Performance Tracking Pattern**
```python
def track_performance_metrics(self):
    """Standard performance tracking using LEAN's built-ins"""
    # Use LEAN's native portfolio tracking
    portfolio_value = self.Portfolio.TotalPortfolioValue
    total_profit = self.Portfolio.TotalProfit
    
    # Custom plotting using LEAN's Plot method
    self.Plot("Performance", "Portfolio Value", portfolio_value)
    self.Plot("Performance", "Total Profit", total_profit)
    
    # Track individual strategy performance
    for strategy_name, strategy in self.strategies.items():
        strategy_pnl = self.calculate_strategy_pnl(strategy_name)
        self.Plot("Strategy Performance", strategy_name, strategy_pnl)

def log_trading_activity(self, symbol, action, quantity, price):
    """Standard logging pattern"""
    # Use LEAN's logging with structured format
    message = f"TRADE: {action} {quantity} {symbol} @ ${price:.2f}"
    
    if self.LiveMode:
        self.Log(message)  # Info level for live trading
    else:
        self.Debug(message)  # Debug level for backtesting
    
    # Additional context for debugging
    portfolio_value = self.Portfolio.TotalPortfolioValue
    position_value = self.Portfolio[symbol].HoldingsValue
    
    self.Debug(f"CONTEXT: Portfolio=${portfolio_value:.0f}, Position=${position_value:.0f}")
```

### **Error Handling Pattern**
```python
def OnData(self, slice):
    """Robust OnData with error handling"""
    try:
        if self.IsWarmingUp:
            return
        
        # Main trading logic
        self.process_trading_signals(slice)
        
    except Exception as e:
        # Use LEAN's error logging
        self.Error(f"OnData error: {str(e)}")
        
        # Don't halt algorithm for recoverable errors
        if "network" in str(e).lower() or "timeout" in str(e).lower():
            self.Debug("Recoverable error, continuing...")
            return
        
        # Re-raise for serious errors
        raise

def safe_order_execution(self, symbol, quantity):
    """Safe order execution with error handling"""
    try:
        # Validate inputs
        if not self.Securities.ContainsKey(symbol):
            self.Error(f"Symbol {symbol} not in securities")
            return None
        
        if not self.Securities[symbol].IsTradable:
            self.Debug(f"Symbol {symbol} not tradeable at this time")
            return None
        
        # Execute order using LEAN's methods
        ticket = self.MarketOrder(symbol, quantity)
        
        # Log successful order
        self.Debug(f"Order submitted: {ticket.OrderId}")
        return ticket
        
    except Exception as e:
        self.Error(f"Order execution failed for {symbol}: {str(e)}")
        return None
```

---

## 🎯 **BEST PRACTICES SUMMARY**

### **✅ DO (Use LEAN's Built-ins):**
- Use `self.Portfolio` for all portfolio tracking
- Use `self.SMA()`, `self.RSI()` etc. for indicators  
- Use `self.Schedule.On()` for all scheduling
- Use `self.MarketOrder()`, `self.SetHoldings()` for trading
- Use `self.History()` for historical data
- Use `self.IsWarmingUp` for warm-up logic
- Use `self.Log()`, `self.Debug()`, `self.Error()` for logging

### **❌ DON'T (Custom Implementations):**
- Custom portfolio value calculations
- Custom technical indicators
- Custom order management systems
- Custom scheduling/timing logic
- Custom data fetching
- Custom market hours logic

### **🔧 Configuration:**
- Use centralized configuration management
- Leverage LEAN's `ObjectStore` for persistence
- Validate all configuration on startup
- Use config-driven parameter values

### **⚡ Performance:**
- Minimize OnData processing time
- Use LEAN's native data structures
- Leverage built-in caching where possible
- Avoid unnecessary historical data requests

This pattern library ensures your CTA framework leverages QuantConnect's full capabilities while maintaining professional trading system standards. 