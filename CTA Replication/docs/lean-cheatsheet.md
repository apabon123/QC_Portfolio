# LEAN/QuantConnect Method Cheat Sheet

> **🚨 ALWAYS CHECK THIS FILE BEFORE IMPLEMENTING CUSTOM SOLUTIONS**
> This file contains the most commonly used LEAN methods to avoid reinventing the wheel.

---

## 🎯 **CORE ALGORITHM METHODS**

### **Initialization & Setup**
```python
# Basic algorithm setup
self.SetStartDate(2020, 1, 1)                    # Set backtest start date
self.SetEndDate(2023, 12, 31)                    # Set backtest end date
self.SetCash(1000000)                            # Set starting capital
self.SetBrokerageModel(BrokerageName.Interactive) # Set brokerage model
self.SetBenchmark("SPY")                         # Set benchmark

# Warm-up period
self.SetWarmUp(timedelta(days=30))               # Time-based warm-up
self.SetWarmUp(100)                              # Bar-based warm-up
self.Settings.AutomaticIndicatorWarmUp = True    # Auto-warm indicators

# Algorithm settings
self.Settings.RebalancePortfolioOnSecurityChanges = False
self.Settings.RebalancePortfolioOnInsightChanges = False
self.Settings.MaximumOrderValue = 10000000       # Order value limits
```

### **Universe Selection & Securities**
```python
# Add individual securities
self.AddEquity("SPY", Resolution.Daily)          # Add equity
self.AddForex("EURUSD", Resolution.Hour)         # Add forex
self.AddCrypto("BTCUSD", Resolution.Minute)      # Add crypto
self.AddFuture("ES", Resolution.Daily)           # Add continuous futures
self.AddOption("SPY", Resolution.Daily)          # Add options

# Futures-specific (for CTA strategies)
future = self.AddFuture("ES", Resolution.Daily,
    dataMappingMode=DataMappingMode.OpenInterest,
    dataNormalizationMode=DataNormalizationMode.BackwardsRatio)

# Universe settings
self.UniverseSettings.Resolution = Resolution.Daily
self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Adjusted
```

---

## 📊 **PORTFOLIO MANAGEMENT** (NEVER BUILD CUSTOM)

### **Portfolio Properties**
```python
# Portfolio values
self.Portfolio.TotalPortfolioValue               # Total portfolio value
self.Portfolio.Cash                              # Available cash
self.Portfolio.TotalMarginUsed                   # Used margin
self.Portfolio.TotalFees                         # Total fees paid
self.Portfolio.TotalProfit                       # Total unrealized profit
self.Portfolio.TotalSaleVolume                   # Total sales volume

# Individual position data
self.Portfolio[symbol].Quantity                  # Position quantity
self.Portfolio[symbol].HoldingsValue             # Position value
self.Portfolio[symbol].UnrealizedProfit          # Unrealized P&L
self.Portfolio[symbol].AveragePrice              # Average entry price
self.Portfolio[symbol].MarketPrice               # Current market price
self.Portfolio[symbol].Invested                  # Whether position exists
```

### **Position Management**
```python
# Check positions
self.Portfolio.Invested                          # Any positions open
self.Portfolio[symbol].Invested                  # Symbol position exists
self.Portfolio.Keys                              # All symbols with positions

# Portfolio metrics
holdings_value = sum(self.Portfolio[s].HoldingsValue for s in self.Portfolio.Keys)
```

---

## 📈 **TECHNICAL INDICATORS** (NEVER BUILD CUSTOM)

### **Trend Indicators**
```python
# Moving averages
self.SMA(symbol, period, resolution)             # Simple Moving Average
self.EMA(symbol, period, resolution)             # Exponential Moving Average
self.LWMA(symbol, period, resolution)            # Linear Weighted MA
self.T3(symbol, period, volume_factor)           # T3 Moving Average

# Trend strength
self.AROON(symbol, period)                       # Aroon Oscillator
self.ADX(symbol, period)                         # Average Directional Index
self.ADXR(symbol, period)                        # ADX Rating
```

### **Momentum Indicators**
```python
# Oscillators
self.RSI(symbol, period)                         # Relative Strength Index
self.STOCH(symbol, period, k_period, d_period)   # Stochastic Oscillator
self.WILLR(symbol, period)                       # Williams %R
self.CCI(symbol, period)                         # Commodity Channel Index
self.MOM(symbol, period)                         # Momentum

# Advanced momentum
self.MACD(symbol, fast, slow, signal)            # MACD
self.PPO(symbol, fast, slow, signal)             # Percentage Price Oscillator
```

### **Volatility Indicators**
```python
# Volatility measures
self.ATR(symbol, period)                         # Average True Range
self.TR(symbol)                                  # True Range
self.BB(symbol, period, std_dev)                 # Bollinger Bands
self.KCH(symbol, period, multiplier)             # Keltner Channels

# Market volatility
self.STD(symbol, period)                         # Standard Deviation
self.VAR(symbol, period, ddof)                   # Variance
```

### **Volume Indicators**
```python
# Volume analysis
self.OBV(symbol)                                 # On-Balance Volume
self.AD(symbol)                                  # Accumulation/Distribution
self.ADOSC(symbol, fast, slow)                   # A/D Oscillator
self.NVI(symbol)                                 # Negative Volume Index
```

### **Using Indicators**
```python
# Create indicators
sma = self.SMA("SPY", 20, Resolution.Daily)
rsi = self.RSI("SPY", 14)

# Check if ready
if sma.IsReady and rsi.IsReady:
    current_sma = sma.Current.Value              # Current indicator value
    current_rsi = rsi.Current.Value
    
# Get historical values
sma_history = sma.Window                         # RollingWindow of values
previous_sma = sma[1]                            # Previous value
```

---

## 🛒 **ORDER MANAGEMENT** (NEVER BUILD CUSTOM)

### **Basic Orders**
```python
# Market orders
ticket = self.MarketOrder(symbol, quantity)      # Market order
ticket = self.MarketOnOpenOrder(symbol, quantity) # Market on open
ticket = self.MarketOnCloseOrder(symbol, quantity) # Market on close

# Limit orders
ticket = self.LimitOrder(symbol, quantity, price) # Limit order
ticket = self.StopMarketOrder(symbol, quantity, stop_price) # Stop market
ticket = self.StopLimitOrder(symbol, quantity, stop_price, limit_price) # Stop limit

# Position sizing
self.SetHoldings(symbol, percentage)             # Set position percentage
self.CalculateOrderQuantity(symbol, target)     # Calculate quantity
```

### **Order Management**
```python
# Order operations
ticket.Cancel()                                  # Cancel order
ticket.Update(fields)                            # Update order
ticket.UpdateQuantity(new_quantity)             # Update quantity
ticket.UpdateLimitPrice(new_price)              # Update limit price
ticket.UpdateStopPrice(new_stop)                # Update stop price

# Order status
ticket.Status                                    # Order status
ticket.OrderId                                   # Order ID
ticket.SubmitRequest                             # Submit request details
ticket.Quantity                                  # Order quantity
```

### **Liquidation**
```python
# Close positions
self.Liquidate(symbol)                           # Close specific position
self.Liquidate()                                 # Close all positions
self.Liquidate(symbol, tag="exit_signal")       # Close with tag
```

---

## 📅 **SCHEDULING** (NEVER BUILD CUSTOM)

### **Date Rules**
```python
# Date-based scheduling
self.DateRules.EveryDay(symbol)                  # Every trading day
self.DateRules.Every(DayOfWeek.Monday)           # Every Monday
self.DateRules.MonthStart(symbol)                # Month start
self.DateRules.MonthEnd(symbol)                  # Month end
self.DateRules.WeekStart(symbol)                 # Week start
self.DateRules.WeekEnd(symbol)                   # Week end
```

### **Time Rules**
```python
# Time-based scheduling
self.TimeRules.AfterMarketOpen(symbol, minutes)  # After market open
self.TimeRules.BeforeMarketClose(symbol, minutes) # Before market close
self.TimeRules.At(hour, minute)                  # Specific time
self.TimeRules.Every(TimeSpan.FromMinutes(30))   # Every 30 minutes
```

### **Scheduling Events**
```python
# Schedule rebalancing
self.Schedule.On(
    self.DateRules.EveryDay("SPY"),              # When
    self.TimeRules.AfterMarketOpen("SPY", 30),   # What time
    self.Rebalance                               # What function
)

# Schedule with parameters
self.Schedule.On(
    self.DateRules.MonthStart("SPY"),
    self.TimeRules.At(10, 0),
    lambda: self.MonthlyRebalance("aggressive")
)
```

---

## 📊 **DATA ACCESS** (NEVER BUILD CUSTOM)

### **Current Data**
```python
# Security data access
self.Securities[symbol].Price                    # Current price
self.Securities[symbol].HasData                  # Has current data
self.Securities[symbol].IsTradable               # Can trade now
self.Securities[symbol].Volume                   # Current volume
self.Securities[symbol].Open                     # Open price
self.Securities[symbol].High                     # High price
self.Securities[symbol].Low                      # Low price
self.Securities[symbol].Close                    # Close price

# Futures-specific
self.Securities[symbol].Mapped                   # Currently mapped contract
```

### **Historical Data**
```python
# Get historical data
history = self.History(symbol, bars, resolution) # Get history
history = self.History([symbols], timedelta(days=30), Resolution.Daily)
history = self.History(symbols, start_date, end_date, Resolution.Hour)

# Specific data types
bars = self.History(TradeBar, symbol, 100, Resolution.Daily)
ticks = self.History(Tick, symbol, timedelta(hours=1))
quotes = self.History(QuoteBar, symbol, 50, Resolution.Minute)
```

### **OnData Event**
```python
def OnData(self, slice):
    # Access slice data
    if slice.Bars.ContainsKey(symbol):
        bar = slice.Bars[symbol]
        price = bar.Close
    
    # Futures chains
    if slice.FuturesChains.ContainsKey(symbol):
        chain = slice.FuturesChains[symbol]
        
    # Options chains
    if slice.OptionChains.ContainsKey(symbol):
        chain = slice.OptionChains[symbol]
    
    # Symbol changes (rollovers)
    for changed in slice.SymbolChangedEvents.Values:
        old_symbol = changed.OldSymbol
        new_symbol = changed.NewSymbol
```

---

## ⏰ **TIME & MARKET HOURS** (NEVER BUILD CUSTOM)

### **Time Properties**
```python
# Algorithm time
self.Time                                        # Current algorithm time
self.UtcTime                                     # UTC time
self.StartDate                                   # Backtest start date
self.EndDate                                     # Backtest end date

# Time zones
self.TimeZone                                    # Algorithm timezone
market_time = self.Time.ConvertToUtc(self.TimeZone)
```

### **Market Hours**
```python
# Market status
self.IsMarketOpen(symbol)                        # Is market open now
self.Securities[symbol].Exchange.Hours.IsOpen(time) # Check specific time

# Market hours
hours = self.Securities[symbol].Exchange.Hours   # Exchange hours
next_open = hours.GetNextMarketOpen(self.Time)   # Next market open
next_close = hours.GetNextMarketClose(self.Time) # Next market close
```

### **Algorithm State**
```python
# Algorithm lifecycle
self.IsWarmingUp                                 # Is warming up
self.LiveMode                                    # Is live trading
self.ObjectStore                                 # Persistent storage
```

---

## 🎯 **FUTURES TRADING** (CTA-SPECIFIC)

### **Futures Contracts**
```python
# Add continuous futures
future = self.AddFuture("ES", Resolution.Daily,
    dataMappingMode=DataMappingMode.OpenInterest,
    dataNormalizationMode=DataNormalizationMode.BackwardsRatio,
    contractDepthOffset=0)

# Futures chain analysis
def OnData(self, slice):
    if slice.FuturesChains.ContainsKey(symbol):
        chain = slice.FuturesChains[symbol]
        
        # Get contracts
        contracts = [x for x in chain]
        liquid_contract = sorted(contracts, key=lambda x: x.Volume, reverse=True)[0]
        
        # Contract properties
        expiry = liquid_contract.Expiry
        volume = liquid_contract.Volume
        open_interest = liquid_contract.OpenInterest
```

### **Rollover Handling**
```python
def OnData(self, slice):
    # Handle automatic rollovers
    for changed in slice.SymbolChangedEvents.Values:
        if changed.Symbol in self.futures_symbols:
            old_symbol = changed.OldSymbol
            new_symbol = changed.NewSymbol
            self.Log(f"Rollover: {old_symbol} -> {new_symbol}")
            
            # Positions automatically transfer
            # Just update any tracking
```

---

## 🔍 **DEBUGGING & LOGGING** (BUILT-IN METHODS)

### **Logging**
```python
# Log levels
self.Log(message)                                # Info level
self.Debug(message)                              # Debug level
self.Error(message)                              # Error level

# Conditional logging
if self.LiveMode:
    self.Log("Live trading message")
else:
    self.Debug("Backtest debug message")
```

### **Performance Tracking**
```python
# Built-in performance
self.Portfolio.TotalPerformance.PortfolioValue   # Portfolio performance
benchmark_performance = self.Benchmark          # Benchmark comparison

# Custom plotting
self.Plot("Strategy", "Portfolio Value", self.Portfolio.TotalPortfolioValue)
self.Plot("Indicators", "RSI", rsi.Current.Value)
```

---

## 🚨 **COMMON MISTAKES TO AVOID**

### **❌ DON'T BUILD THESE (LEAN HAS THEM)**
```python
# ❌ Custom portfolio tracking
portfolio_value = 0
for symbol in positions:
    portfolio_value += positions[symbol] * prices[symbol]

# ✅ Use LEAN's built-in
portfolio_value = self.Portfolio.TotalPortfolioValue

# ❌ Custom moving average
prices = []
def calculate_sma(period):
    return sum(prices[-period:]) / period

# ✅ Use LEAN's built-in
sma = self.SMA(symbol, period)

# ❌ Custom order management
pending_orders = {}
def track_order_status():
    # Custom tracking logic

# ✅ Use LEAN's built-in
ticket = self.MarketOrder(symbol, quantity)
status = ticket.Status

# ❌ Custom scheduling
last_rebalance = None
def should_rebalance():
    return (self.Time - last_rebalance).days >= 7

# ✅ Use LEAN's built-in
self.Schedule.On(self.DateRules.Every(DayOfWeek.Monday), 
                 self.TimeRules.AfterMarketOpen("SPY", 30),
                 self.Rebalance)
```

---

## 📚 **QUICK REFERENCE BY USE CASE**

### **For CTA/Momentum Strategies:**
- Use `self.AddFuture()` for continuous contracts
- Use `self.SMA()`, `self.EMA()` for trend following
- Use `self.ATR()` for volatility-based position sizing
- Use `self.RSI()`, `self.MACD()` for momentum
- Use `self.SetHoldings()` for position sizing
- Use `self.Schedule.On()` for periodic rebalancing

### **For Portfolio Management:**
- Use `self.Portfolio.TotalPortfolioValue` for total value
- Use `self.Portfolio[symbol].HoldingsValue` for positions
- Use `self.CalculateOrderQuantity()` for sizing
- Use `self.Portfolio.TotalMarginUsed` for risk

### **For Risk Management:**
- Use `self.Settings.MaximumOrderValue` for limits
- Use `self.Portfolio.TotalMarginUsed` for leverage
- Use `self.Liquidate()` for emergency exits
- Use `self.IsMarketOpen()` for trading hours

---

## 🎯 **LEAN-FIRST DEVELOPMENT PROCESS**

1. **Before writing ANY financial code, ask:**
   - "Does LEAN already have this capability?"
   - "Am I reinventing basic trading infrastructure?"

2. **Check this cheat sheet first**

3. **Search the LEAN documentation:**
   - [LEAN API Documentation](https://www.lean.io/docs/v2/lean-engine/class-reference)
   - [QuantConnect Documentation](https://www.quantconnect.com/docs/v2)

4. **Only implement custom logic for:**
   - Business-specific strategy logic
   - Custom data transformations
   - Unique risk models
   - Specialized analytics

**Remember: If it feels like "basic trading functionality", LEAN probably has it built-in!** 