# QuantConnect Integration Standards - QC "Plumbing" Best Practices

## 🎯 **CRITICAL FOR SCALABILITY: 10-15 STRATEGIES**

This document defines the standardized QuantConnect integration patterns that **ALL** strategies must follow. These patterns have been proven across Kestner, MTUM, and HMM strategies and are essential for scaling to 10-15 strategies without maintenance nightmares.

## 🚨 **CRITICAL: QuantConnect Symbol Object Limitation**

**NEVER pass QuantConnect Symbol objects as constructor parameters** - this causes the infamous "error return without exception set" error.

### **The Problem**
QuantConnect Symbol objects get wrapped in `clr.MetaClass` which cannot be passed to Python constructors, causing complete algorithm initialization failure.

### **Solution Patterns**
```python
# ❌ NEVER DO THIS - Will crash constructors
def __init__(self, shared_symbols):
    self.symbols = shared_symbols  # Contains Symbol objects - CRASH

# ✅ CORRECT - Use string identifiers
def __init__(self, symbol_strings):
    self.symbol_strings = symbol_strings  # ['ES', 'NQ', 'ZN'] - Safe

# ✅ CORRECT - Use QC native methods directly
def _setup_universe(self):
    for symbol_str in ['ES', 'NQ', 'ZN']:
        future = self.AddFuture(symbol_str, Resolution.Daily)
        self.futures_symbols.append(future.Symbol)  # QC handles Symbol creation
```

**Reference**: [QuantConnect Forum - Python Symbol Object Issue](https://www.quantconnect.com/forum/discussion/9331/python-custom-exception-definition-does-not-seem-to-work)

---

## **📋 STANDARDIZED QC INTEGRATION CHECKLIST**

### **✅ 1. History API Usage - STANDARDIZED**
```python
# ✅ CORRECT - Use this pattern ALWAYS
history = self.algorithm.History(symbol, periods, Resolution.Daily)

# ❌ INCORRECT - Never use this
history = self.algorithm.History[TradeBar](symbol, periods, Resolution.Daily)
```

**Key Requirements:**
- Always use `self.algorithm.History(symbol, periods, Resolution.Daily)`
- Check for `history.empty` before processing
- **CRITICAL: NO MOCK DATA** - Log errors and mark symbols as unavailable
- Use pandas `iterrows()` pattern for processing

### **✅ 2. History Data Processing - STANDARDIZED**
```python
# ✅ CORRECT - Pandas iterrows pattern
for index, row in history.iterrows():
    close = row['close'] if 'close' in row else row.get('Close', 0)
    open_price = row['open'] if 'open' in row else row.get('Open', close)
    
    # Validate data
    if close > 0 and not np.isnan(close) and not np.isinf(close):
        # Process data
        pass

# ❌ INCORRECT - TradeBar iteration
for bar in history:
    close = bar.Close  # Don't use this pattern
```

### **✅ 3. Consolidator Management - STANDARDIZED**
```python
# ✅ CORRECT - Store consolidator reference for proper disposal
self.consolidator = None
try:
    self.consolidator = TradeBarConsolidator(timedelta(days=1))
    self.consolidator.DataConsolidated += self.OnDataConsolidated
    algorithm.SubscriptionManager.AddConsolidator(symbol, self.consolidator)
except Exception as e:
    algorithm.Log(f"SymbolData {symbol}: Consolidator setup error: {str(e)}")

# ❌ INCORRECT - Local variable, can't dispose properly
consolidator = TradeBarConsolidator(timedelta(days=1))
```

### **✅ 4. Resource Disposal - STANDARDIZED**
```python
# ✅ CORRECT - Comprehensive disposal
def Dispose(self):
    try:
        if hasattr(self, 'consolidator') and self.consolidator is not None:
            self.consolidator.DataConsolidated -= self.OnDataConsolidated
            if hasattr(self.algorithm, 'SubscriptionManager'):
                try:
                    self.algorithm.SubscriptionManager.RemoveConsolidator(self.symbol, self.consolidator)
                except:
                    pass  # May already be removed
            self.consolidator = None
            
        # Clear data structures
        if hasattr(self, 'price_window'):
            self.price_window.Reset()
            
    except Exception as e:
        self.algorithm.Log(f"SymbolData {self.symbol}: Dispose error: {str(e)}")
```

### **✅ 5. Security Property Access - STANDARDIZED**
```python
# ✅ CORRECT - Safe property access with validation
def get_security_price(self, symbol):
    try:
        if symbol in self.algorithm.Securities:
            price = self.algorithm.Securities[symbol].Price
            return price if price > 0 and not np.isnan(price) and not np.isinf(price) else None
        return None
    except Exception as e:
        self.algorithm.Log(f"Price access error for {symbol}: {str(e)}")
        return None

# ✅ CORRECT - Mapped contract access
def get_mapped_contract(self, symbol):
    try:
        if symbol in self.algorithm.Securities:
            mapped = self.algorithm.Securities[symbol].Mapped
            security = self.algorithm.Securities[mapped] if mapped else None
            return mapped if security and security.HasData else None
        return None
    except Exception as e:
        self.algorithm.Log(f"Mapped contract error for {symbol}: {str(e)}")
        return None
```

### **✅ 6. Order Management - STANDARDIZED**
```python
# ✅ CORRECT - Safe order placement
def place_market_order(self, symbol, quantity, tag=None):
    try:
        if tag is None:
            tag = f"{self.name}_order"
        
        order_ticket = self.algorithm.MarketOrder(symbol, quantity, tag=tag)
        return order_ticket
    except Exception as e:
        self.algorithm.Log(f"Order placement error for {symbol}: {str(e)}")
        return None
```

### **✅ 7. Data Availability Handling - STANDARDIZED**
```python
# ✅ REQUIRED - Handle missing data properly, NO MOCK DATA
def handle_no_data_available(self, required_periods):
    """Handle case when no historical data is available."""
    self.algorithm.Error(f"CRITICAL: SymbolData {self.symbol} - No historical data available for {required_periods} periods")
    self.algorithm.Error(f"TRADING DECISION RISK: Strategy cannot initialize {self.symbol} without real market data")
    
    # Mark this symbol as having insufficient data
    self.has_sufficient_data = False
    self.data_availability_error = f"No historical data available for {required_periods} periods"
    
    # This symbol will not be ready for trading
    self.algorithm.Log(f"SymbolData {self.symbol}: Marked as insufficient data - will not be ready for trading")
```

### **✅ 8. OnSecuritiesChanged - STANDARDIZED**
```python
# ✅ CORRECT - Only track continuous contracts
def OnSecuritiesChanged(self, changes):
    for security in changes.AddedSecurities:
        symbol = security.Symbol
        symbol_str = str(symbol)
        
        # Only track continuous contracts (start with '/') - QC STANDARDIZED
        if symbol_str.startswith('/') or symbol_str.startswith('futures/'):
            if symbol not in self.symbol_data:
                self.algorithm.Log(f"{self.name}: Initializing SymbolData for continuous contract: {symbol}")
                try:
                    self._create_symbol_data(symbol)
                except Exception as e:
                    self.algorithm.Error(f"{self.name}: Failed to create SymbolData for {symbol}: {e}")
        else:
            # Skip rollover contracts - QC STANDARDIZED
            self.algorithm.Log(f"{self.name}: Skipping rollover contract: {symbol_str}")
```

---

## **🏗️ BASE STRATEGY CLASS - MANDATORY INHERITANCE**

All new strategies **MUST** inherit from `BaseStrategy` class:

```python
from src.strategies.base_strategy import BaseStrategy

class MyNewStrategy(BaseStrategy):
    def __init__(self, algorithm, futures_manager, name="MyNewStrategy", config_manager=None):
        super().__init__(algorithm, futures_manager, name, config_manager)
        # Strategy-specific initialization
    
    # Implement required abstract methods
    def _build_config_dict(self, config):
        # Build strategy config
        pass
    
    def _load_fallback_config(self):
        # Load fallback config
        pass
    
    def update(self, slice_data):
        # Update strategy
        pass
    
    def generate_targets(self):
        # Generate targets
        pass
    
    # ... other required methods
```

---

## **📊 SYMBOL DATA STANDARDIZATION**

All SymbolData classes **MUST** inherit from `BaseStrategy.BaseSymbolData`:

```python
class MyStrategySymbolData(BaseStrategy.BaseSymbolData):
    def __init__(self, algorithm, symbol, strategy_params):
        super().__init__(algorithm, symbol)
        
        # Strategy-specific initialization
        self.strategy_params = strategy_params
        
        # Initialize with history using standardized pattern
        self._initialize_with_history()
    
    def _initialize_with_history(self):
        """Use standardized history initialization pattern"""
        history = self.get_qc_history(self.required_periods)
        
                 if history is None:
             self.algorithm.Error(f"CRITICAL: SymbolData {self.symbol} - No historical data available")
             self.algorithm.Error(f"TRADING DECISION RISK: Cannot initialize without real market data")
             self.has_sufficient_data = False
             self.data_availability_error = "No historical data available"
             return
         else:
            processed_data = self.process_qc_history(history)
            self._process_historical_data(processed_data)
    
    def OnDataConsolidated(self, sender, bar):
        """Handle consolidated data"""
        # Strategy-specific implementation
        pass
    
    @property
    def IsReady(self):
        """Check if ready"""
        # Strategy-specific implementation
        return True
```

---

## **🔧 CONFIGURATION STANDARDIZATION**

All strategies must support both config_manager and fallback configurations:

```python
def _load_configuration(self):
    try:
        if self.config_manager:
            # Primary: Load from config_manager
            strategy_config = self.config_manager.get_strategy_config(self.name)
            if strategy_config:
                self._build_config_dict(strategy_config)
                return
        
        # Fallback: Use default configuration
        self._load_fallback_config()
        
    except Exception as e:
        self.algorithm.Error(f"{self.name}: Config loading error: {str(e)}")
        self._load_fallback_config()
```

---

## **📈 PERFORMANCE TRACKING STANDARDIZATION**

All strategies must implement standardized performance tracking:

```python
def get_performance_metrics(self):
    return {
        'strategy_name': self.name,
        'trades_executed': self.trades_executed,
        'total_rebalances': self.total_rebalances,
        'current_positions': len(self.current_targets),
        'ready_symbols': len([sd for sd in self.symbol_data.values() if sd.IsReady]),
        'total_symbols': len(self.symbol_data),
        'last_rebalance': self.last_rebalance_date,
        'last_update': self.last_update_time
    }
```

---

## **🚨 CRITICAL ANTI-PATTERNS TO AVOID**

### **❌ Never Do These:**

1. **History API Misuse:**
   ```python
   # ❌ WRONG
   history = self.algorithm.History[TradeBar](symbol, days, Resolution.Daily)
   ```

2. **Consolidator Memory Leaks:**
   ```python
   # ❌ WRONG - Can't dispose properly
   consolidator = TradeBarConsolidator(timedelta(days=1))
   ```

3. **Unsafe Property Access:**
   ```python
   # ❌ WRONG - No validation
   price = self.algorithm.Securities[symbol].Price
   ```

4. **Missing Data Handling:**
   ```python
   # ❌ WRONG - Silent failure or mock data
   if history.empty:
       return  # Strategy will fail silently
   
   # ❌ WRONG - Mock data for trading decisions
   if history.empty:
       self._create_mock_data()  # DANGEROUS!
   ```

5. **Inconsistent Error Handling:**
   ```python
   # ❌ WRONG - Silent failures
   try:
       # some operation
   except:
       pass  # Don't ignore errors silently
   ```

---

## **✅ IMPLEMENTATION STATUS**

### **Current Strategy Compliance:**

| Strategy | History API | Consolidators | Disposal | Mock Data | Base Class |
|----------|-------------|---------------|----------|-----------|------------|
| **Kestner** | ✅ | ✅ | ✅ | ✅ No Mock | ⏳ Pending |
| **MTUM** | ✅ Fixed | ✅ Fixed | ✅ Fixed | ✅ No Mock | ⏳ Pending |
| **HMM** | ✅ | ✅ | ✅ | ✅ No Mock | ⏳ Pending |

### **Next Steps for Full Standardization:**

1. **Migrate existing strategies to inherit from BaseStrategy**
2. **Update SymbolData classes to inherit from BaseSymbolData**
3. **Test all strategies with standardized patterns**
4. **Create strategy template for new strategies**

---

## **🎯 BENEFITS OF STANDARDIZATION**

1. **Scalability:** Easy to add 10-15 strategies without code duplication
2. **Maintainability:** Single source of truth for QC integration patterns
3. **Reliability:** Proven patterns reduce bugs and memory leaks
4. **Testing:** Robust error handling prevents trading on invalid data
5. **Performance:** Optimized QC API usage patterns
6. **Documentation:** Clear patterns for new team members

---

## **📝 STRATEGY DEVELOPMENT WORKFLOW**

For adding new strategies:

1. **Inherit from BaseStrategy**
2. **Use BaseSymbolData for symbol data**
3. **Follow standardized patterns from this document**
4. **Test with real data only - validate data availability handling**
5. **Validate resource disposal**
6. **Add to dynamic strategy loader configuration**

This standardization ensures that scaling to 10-15 strategies will be manageable and maintainable! 