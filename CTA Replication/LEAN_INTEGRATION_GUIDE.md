# LEAN Integration Guide for CTA Framework

> **🚨 MANDATORY: Check LEAN capabilities before implementing ANY custom solutions**

---

## 🎯 **LEAN-FIRST DEVELOPMENT WORKFLOW**

### **Before Writing ANY Code:**

1. **🔍 CHECK**: `docs/lean-cheatsheet.md` for existing LEAN methods
2. **📚 VERIFY**: [LEAN API Documentation](https://www.lean.io/docs/v2/lean-engine/class-reference)
3. **❓ ASK**: "Does this feel like basic trading infrastructure?" (If yes, LEAN has it)
4. **✅ IMPLEMENT**: Use LEAN's native methods, not custom solutions

---

## 🏗️ **CTA FRAMEWORK LEAN INTEGRATION**

### **Current Integration Status**

#### **✅ PROPERLY USING LEAN:**
```python
# Portfolio Management
self.Portfolio.TotalPortfolioValue        # ✅ Using LEAN's portfolio tracking
self.Portfolio[symbol].Quantity           # ✅ Using LEAN's position tracking
self.Portfolio.TotalMarginUsed           # ✅ Using LEAN's margin tracking

# Futures Management  
self.AddFuture("ES", Resolution.Daily)    # ✅ Using LEAN's continuous contracts
slice.FuturesChains[symbol]              # ✅ Using LEAN's chain data
slice.SymbolChangedEvents                # ✅ Using LEAN's rollover events

# Data Access
self.Securities[symbol].Price            # ✅ Using LEAN's price data
self.Securities[symbol].HasData          # ✅ Using LEAN's data validation
self.History(symbol, periods, resolution) # ✅ Using LEAN's historical data

# Warm-up System
self.IsWarmingUp                         # ✅ Using LEAN's warm-up detection
self.SetWarmUp(timedelta(days=60))       # ✅ Using LEAN's warm-up system

# Universe Management (UPDATED)
self._setup_futures_universe()          # ✅ Simple QC native setup
```

#### **🔧 NEEDS LEAN INTEGRATION:**

**Indicators (High Priority):**
```python
# ❌ Current: Custom indicator implementations
# ✅ Should be: LEAN's built-in indicators

# Replace custom momentum calculations with:
self.SMA(symbol, 20)                     # LEAN's Simple Moving Average
self.RSI(symbol, 14)                     # LEAN's RSI
self.ATR(symbol, 20)                     # LEAN's Average True Range
self.MACD(symbol, 12, 26, 9)             # LEAN's MACD

# Replace custom volatility calculations with:
self.BB(symbol, 20, 2)                   # LEAN's Bollinger Bands
self.STD(symbol, 20)                     # LEAN's Standard Deviation
```

**Order Management (Medium Priority):**
```python
# ❌ Current: Custom order tracking
# ✅ Should be: LEAN's order system

# Replace custom position sizing with:
self.SetHoldings(symbol, percentage)     # LEAN's position sizing
self.CalculateOrderQuantity(symbol, target) # LEAN's quantity calculation

# Replace custom order execution with:
self.MarketOrder(symbol, quantity)       # LEAN's market orders
self.Liquidate(symbol)                   # LEAN's position closure
```

**Scheduling (Low Priority):**
```python
# ❌ Current: Custom timing logic
# ✅ Should be: LEAN's scheduling

# Replace custom rebalancing timing with:
self.Schedule.On(
    self.DateRules.Every(DayOfWeek.Friday),
    self.TimeRules.BeforeMarketClose("ES", 30),
    self.WeeklyRebalance
)
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

### **Target State:**
- **90%+ LEAN Integration** - Most functionality uses LEAN's built-ins
- **Zero Custom Infrastructure** - No custom portfolio/order/indicator systems
- **Full QC Compatibility** - Works seamlessly on QC cloud platform
- **Optimized Performance** - Leverages QC's optimized implementations

### **Current Progress:**
- **Foundation Layer**: 95% LEAN-compliant ✅ (Universe management now QC native)
- **Strategy Layer**: 40% LEAN-compliant ⚠️ (Needs indicator integration)
- **Execution Layer**: 60% LEAN-compliant ⚠️ (Needs order system integration)
- **Risk Layer**: 70% LEAN-compliant ⚠️ (Needs portfolio property usage)

**✅ MAJOR UPDATE: Removed FuturesManager - Now using QC native universe management**

**Next Priority: Integrate LEAN indicators into all strategies** 🎯 