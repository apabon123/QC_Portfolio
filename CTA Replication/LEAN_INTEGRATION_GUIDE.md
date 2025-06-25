# LEAN Integration Guide for CTA Framework

> **🎯 PRAGMATIC APPROACH: Use the right tool for the right job**

---

## 🎯 **SMART INTEGRATION PHILOSOPHY**

### **Use LEAN For What It's Good At:**
- **📊 Data Access**: Market data, historical data, real-time feeds
- **📈 Basic Indicators**: SMA, EMA, RSI, ROC, STD, ATR, MACD
- **🏗️ Trading Infrastructure**: Order execution, portfolio tracking, universe management
- **⏰ Scheduling**: Market hours, rebalancing timing, event handling

### **Use Numpy/Pandas For Analytics:**
- **📊 Portfolio Metrics**: Sharpe ratios, VaR, correlation matrices, drawdown analysis
- **🧮 Complex Math**: Covariance matrices, optimization, statistical analysis
- **📈 Performance Analytics**: Risk-adjusted returns, factor analysis
- **🔬 Research**: Backtesting analytics, strategy research, data exploration

### **Don't Force Everything Into QC:**
- **❌ Avoid**: Complex indicator warmup gymnastics
- **❌ Avoid**: Fighting QC's framework for simple calculations
- **❌ Avoid**: Over-engineering when numpy/pandas is simpler
- **✅ Keep It Simple**: Let QC handle infrastructure, use Python for analytics

---

## 🏗️ **CTA FRAMEWORK INTEGRATION STRATEGY**

### **✅ LEAN's SWEET SPOT - Use These Always:**

#### **Core Trading Infrastructure:**
```python
# Portfolio Management - LEAN excels here
self.Portfolio.TotalPortfolioValue        # ✅ Real-time portfolio value
self.Portfolio[symbol].Quantity           # ✅ Current positions
self.Portfolio.TotalMarginUsed           # ✅ Margin tracking
self.Portfolio.Cash                       # ✅ Available cash

# Order Execution - LEAN's core strength
self.MarketOrder(symbol, quantity)        # ✅ Market orders
self.SetHoldings(symbol, percentage)      # ✅ Position sizing
self.Liquidate(symbol)                    # ✅ Position closure
self.CalculateOrderQuantity(symbol, target) # ✅ Quantity calculation

# Data Access - LEAN's data engine
self.Securities[symbol].Price            # ✅ Current prices
self.Securities[symbol].HasData          # ✅ Data validation
self.History(symbol, periods, resolution) # ✅ Historical data
slice.FuturesChains[symbol]              # ✅ Futures chain data

# Basic Indicators - LEAN's built-ins
self.ROC(symbol, period)                 # ✅ Rate of change
self.STD(symbol, period)                 # ✅ Standard deviation  
self.SMA(symbol, period)                 # ✅ Simple moving average
self.C(symbol1, symbol2, period)         # ✅ Correlation

# System Management - LEAN's framework
self.IsWarmingUp                         # ✅ Warm-up detection
self.SetWarmUp(timedelta(days=60))       # ✅ Automatic warmup
self.Schedule.On(date_rule, time_rule, action) # ✅ Scheduling
```

### **🧮 NUMPY/PANDAS SWEET SPOT - Use These for Analytics:**

#### **Portfolio Analytics:**
```python
import numpy as np
import pandas as pd

# Portfolio metrics - numpy/pandas excel here
def calculate_portfolio_metrics(returns):
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
    var_95 = np.percentile(returns, 5)
    return {'sharpe': sharpe_ratio, 'max_dd': max_drawdown, 'var_95': var_95}

# Correlation analysis - pandas built-in
correlation_matrix = returns_df.corr()

# Risk calculations - numpy linear algebra
portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)

# Performance analysis - pandas time series
rolling_sharpe = returns.rolling(252).apply(lambda x: x.mean()/x.std()*np.sqrt(252))
```

### **❌ AVOID THESE ANTI-PATTERNS:**

#### **Don't Fight QC's Framework:**
```python
# ❌ AVOID: Complex indicator warmup gymnastics
def complex_indicator_warmup():
    for symbol in symbols:
        history = self.History(symbol, 365, Resolution.Daily)
        for index, row in history.iterrows():
            data_point = IndicatorDataPoint(time, price)  # Complex!
            indicator.Update(data_point)

# ✅ SIMPLE: Let QC handle warmup naturally
self.SetWarmUp(timedelta(days=365))  # QC handles everything
# Indicators warm up automatically as data flows in

# ❌ AVOID: Over-engineering simple calculations
def custom_portfolio_value():
    total = 0
    for symbol in self.Securities:
        total += self.Securities[symbol].Holdings.HoldingsValue
    return total

# ✅ SIMPLE: Use QC's built-in
portfolio_value = self.Portfolio.TotalPortfolioValue

# ❌ AVOID: Reinventing basic indicators
def custom_moving_average(prices, period):
    return sum(prices[-period:]) / period

# ✅ SIMPLE: Use QC's indicator
sma = self.SMA(symbol, period)
```

#### **Don't Force Analytics Into QC:**
```python
# ❌ AVOID: Complex QC-based portfolio analytics
def qc_based_sharpe_calculation():
    # Fighting QC's framework for something pandas does easily

# ✅ SIMPLE: Use pandas for what it's good at
def calculate_sharpe(returns_series):
    return returns_series.mean() / returns_series.std() * np.sqrt(252)

# ❌ AVOID: QC-based correlation matrices when you have the data
def qc_correlation_gymnastics():
    # Complex QC indicator management for simple correlation

# ✅ SIMPLE: Direct pandas correlation
correlation_matrix = returns_df.corr()
```

---

## 🔧 **INTEGRATION ACTION PLAN**

### **Phase 1: Indicator Integration (IMMEDIATE)**

#### **KestnerCTAStrategy Integration:**
```python
# Current custom implementation
class KestnerCTAStrategy:
    def __init__(self, algorithm, config_manager):
        # Replace these custom implementations:
        # self.custom_momentum_calculator = CustomMomentum()
        # self.custom_volatility_tracker = CustomVolatility()
        
        # With LEAN's built-in indicators:
        self.momentum_indicators = {}
        self.volatility_indicators = {}
        
        for symbol in self.symbols:
            # Use LEAN's indicators instead of custom calculations
            self.momentum_indicators[symbol] = {
                'sma_short': algorithm.SMA(symbol, config['sma_short_period']),
                'sma_long': algorithm.SMA(symbol, config['sma_long_period']),
                'rsi': algorithm.RSI(symbol, config['rsi_period'])
            }
            
            self.volatility_indicators[symbol] = {
                'atr': algorithm.ATR(symbol, config['atr_period']),
                'bb': algorithm.BB(symbol, config['bb_period'], config['bb_std'])
            }
    
    def calculate_momentum_signal(self, symbol):
        """Use LEAN indicators instead of custom calculations"""
        indicators = self.momentum_indicators[symbol]
        
        # Check if LEAN indicators are ready
        if not (indicators['sma_short'].IsReady and indicators['sma_long'].IsReady):
            return 0
        
        # Use LEAN indicator values directly
        short_ma = indicators['sma_short'].Current.Value
        long_ma = indicators['sma_long'].Current.Value
        rsi_value = indicators['rsi'].Current.Value
        
        # Simple momentum signal using LEAN data
        trend_signal = 1 if short_ma > long_ma else -1
        rsi_filter = 1 if 30 < rsi_value < 70 else 0
        
        return trend_signal * rsi_filter
```

#### **MTUMCTAStrategy Integration:**
```python
class MTUMCTAStrategy:
    def __init__(self, algorithm, config_manager):
        # Replace custom MTUM calculations with LEAN indicators
        self.mtum_indicators = {}
        
        for symbol in self.symbols:
            # Use LEAN's momentum indicators
            self.mtum_indicators[symbol] = {
                'momentum': algorithm.MOM(symbol, config['momentum_period']),
                'roc': algorithm.ROC(symbol, config['roc_period']),
                'ema': algorithm.EMA(symbol, config['ema_period'])
            }
    
    def calculate_mtum_signal(self, symbol):
        """Use LEAN's momentum indicators"""
        indicators = self.mtum_indicators[symbol]
        
        if not all(ind.IsReady for ind in indicators.values()):
            return 0
        
        # Use LEAN indicator values
        momentum = indicators['momentum'].Current.Value
        roc = indicators['roc'].Current.Value
        ema_value = indicators['ema'].Current.Value
        
        # MTUM logic using LEAN data
        return self._calculate_mtum_score(momentum, roc, ema_value)
```

### **Phase 2: Order Management Integration**

#### **PortfolioExecutionManager Integration:**
```python
class PortfolioExecutionManager:
    def execute_trades(self, position_targets):
        """Use LEAN's order methods instead of custom execution"""
        for symbol, target_quantity in position_targets.items():
            # Use LEAN's Portfolio for current positions
            current_quantity = self.algorithm.Portfolio[symbol].Quantity
            trade_quantity = target_quantity - current_quantity
            
            if abs(trade_quantity) > self.min_trade_size:
                # Use LEAN's order methods
                if abs(target_quantity) < self.min_position_size:
                    # Use LEAN's Liquidate
                    ticket = self.algorithm.Liquidate(symbol, "Position too small")
                else:
                    # Use LEAN's MarketOrder
                    ticket = self.algorithm.MarketOrder(symbol, trade_quantity)
                
                # Track using LEAN's order ticket
                self._track_order(ticket)
    
    def calculate_position_sizes(self, signals):
        """Use LEAN's position sizing methods"""
        position_targets = {}
        
        # Use LEAN's Portfolio properties
        total_value = self.algorithm.Portfolio.TotalPortfolioValue
        
        for symbol, signal in signals.items():
            if signal != 0:
                # Use LEAN's CalculateOrderQuantity for proper sizing
                target_percentage = signal * self.position_size_config[symbol]
                target_quantity = self.algorithm.CalculateOrderQuantity(symbol, target_percentage)
                position_targets[symbol] = target_quantity
        
        return position_targets
```

### **Phase 3: Risk Management Integration**

#### **LayerThreeRiskManager Integration:**
```python
class LayerThreeRiskManager:
    def apply_portfolio_risk_management(self, targets):
        """Use LEAN's risk properties for portfolio management"""
        # Use LEAN's Portfolio properties for risk calculations
        total_value = self.algorithm.Portfolio.TotalPortfolioValue
        margin_used = self.algorithm.Portfolio.TotalMarginUsed
        current_leverage = margin_used / total_value
        
        # Apply leverage limits using LEAN data
        if current_leverage > self.max_leverage:
            scale_factor = self.max_leverage / current_leverage
            targets = {symbol: qty * scale_factor for symbol, qty in targets.items()}
            
            self.algorithm.Log(f"Scaling positions due to leverage: {current_leverage:.2f}")
        
        # Check drawdown using LEAN's portfolio tracking
        self._check_drawdown_limits()
        
        return targets
    
    def _check_drawdown_limits(self):
        """Use LEAN's portfolio value for drawdown calculation"""
        current_value = self.algorithm.Portfolio.TotalPortfolioValue
        
        # Update high water mark
        if not hasattr(self, 'high_water_mark'):
            self.high_water_mark = current_value
        else:
            self.high_water_mark = max(self.high_water_mark, current_value)
        
        # Calculate drawdown
        drawdown = (self.high_water_mark - current_value) / self.high_water_mark
        
        if drawdown > self.max_drawdown:
            # Use LEAN's emergency liquidation
            self.algorithm.Liquidate()
            self.algorithm.Error(f"Emergency stop: Drawdown {drawdown:.2%}")
            self.algorithm.Quit("Maximum drawdown exceeded")
```

---

## ✅ **LEAN INTEGRATION CHECKLIST**

### **Before Committing Any Code:**

#### **Portfolio Management:**
- [ ] Using `self.Portfolio.TotalPortfolioValue` (not custom tracking)
- [ ] Using `self.Portfolio[symbol].Quantity` (not custom positions)
- [ ] Using `self.Portfolio.TotalMarginUsed` (not custom margin calc)
- [ ] Using `self.Portfolio.Cash` (not custom cash tracking)

#### **Technical Indicators:**
- [ ] Using `self.SMA()`, `self.EMA()` (not custom moving averages)
- [ ] Using `self.RSI()`, `self.MACD()` (not custom momentum indicators)
- [ ] Using `self.ATR()`, `self.BB()` (not custom volatility indicators)
- [ ] Using `indicator.IsReady` (not custom readiness checks)

#### **Order Management:**
- [ ] Using `self.MarketOrder()` (not custom order systems)
- [ ] Using `self.SetHoldings()` (not custom position sizing)
- [ ] Using `self.Liquidate()` (not custom position closure)
- [ ] Using `self.CalculateOrderQuantity()` (not custom quantity calc)

#### **Data Access:**
- [ ] Using `self.History()` (not custom data fetching)
- [ ] Using `self.Securities[symbol].Price` (not custom price tracking)
- [ ] Using `self.Securities[symbol].HasData` (not custom data validation)
- [ ] Using `slice.FuturesChains` (not custom chain processing)

#### **Scheduling & Timing:**
- [ ] Using `self.Schedule.On()` (not custom schedulers)
- [ ] Using `self.IsMarketOpen()` (not custom market hours)
- [ ] Using `self.IsWarmingUp` (not custom warm-up logic)
- [ ] Using `self.DateRules`/`TimeRules` (not custom timing)

#### **Risk Management:**
- [ ] Using LEAN's portfolio properties for risk calculations
- [ ] Using `self.Settings.MaximumOrderValue` for limits
- [ ] Using LEAN's built-in liquidation for emergency stops
- [ ] Using LEAN's portfolio tracking for performance metrics

---

## 🚨 **COMMON LEAN VIOLATIONS TO FIX**

### **❌ Custom Implementations Found:**
```python
# ❌ Custom portfolio tracking (FOUND IN: system_reporter.py)
portfolio_value = sum(position_values)

# ✅ Should be LEAN's method:
portfolio_value = self.Portfolio.TotalPortfolioValue

# ❌ Custom indicator calculations (FOUND IN: strategies/*.py)
sma_value = sum(prices[-period:]) / period

# ✅ Should be LEAN's indicator:
sma = self.SMA(symbol, period)
sma_value = sma.Current.Value

# ❌ Custom order tracking (FOUND IN: execution_manager.py)
pending_orders = {}

# ✅ Should be LEAN's order tickets:
ticket = self.MarketOrder(symbol, quantity)
status = ticket.Status
```

### **Priority Fix Order:**
1. **Indicators** (High Impact) - Replace custom calculations with LEAN indicators
2. **Order Management** (Medium Impact) - Use LEAN's order system
3. **Scheduling** (Low Impact) - Use LEAN's scheduling system

---

## 📚 **LEAN DOCUMENTATION RESOURCES**

### **Quick Reference:**
- `docs/lean-cheatsheet.md` - Most common LEAN methods
- `docs/qc-common-patterns.md` - Proven QC patterns for CTA
- `templates/lean_strategy_template.py` - Template showing proper LEAN usage

### **External Documentation:**
- [LEAN API Reference](https://www.lean.io/docs/v2/lean-engine/class-reference)
- [QuantConnect Documentation](https://www.quantconnect.com/docs/v2)
- [Algorithm Framework](https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework)

### **Code Review Process:**
1. Check against LEAN cheat sheet
2. Verify no custom implementations of LEAN features
3. Ensure proper use of LEAN's native properties
4. Test with LEAN's built-in validation

---

## 🎯 **SUCCESS METRICS**

### **Pragmatic Integration Goals:**
- **🏗️ Infrastructure**: 95% LEAN (data, orders, portfolio, scheduling)
- **📊 Analytics**: 80% Numpy/Pandas (metrics, correlations, optimization)
- **🧠 Strategy Logic**: Hybrid approach - use best tool for each task
- **⚡ Performance**: Fast, reliable, maintainable code

### **Current Status:**
- **✅ Core Infrastructure**: Using LEAN for data, orders, portfolio tracking
- **✅ Basic Indicators**: Using LEAN's ROC, STD, SMA, correlation indicators  
- **✅ Analytics**: Using numpy/pandas for portfolio metrics and risk calculations
- **✅ Warmup Strategy**: Simplified - let QC handle indicator warmup naturally
- **✅ Anti-Patterns Avoided**: No complex indicator gymnastics or forced QC analytics

### **Philosophy Wins:**
- **🚀 Faster Development**: Less fighting with QC's framework
- **🔧 Easier Maintenance**: Cleaner separation of concerns
- **📈 Better Analytics**: Numpy/pandas excel at mathematical operations
- **⚡ Reliable Execution**: QC handles what it's designed for

**✅ APPROACH VALIDATED: Right tool for the right job** 🎯 