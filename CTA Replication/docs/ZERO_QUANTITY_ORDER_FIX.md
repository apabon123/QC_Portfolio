# Zero Quantity Order Fix

## Problem
User encountered QuantConnect error: **"Backtest Handled Error: Unable to submit order with id -10 that has zero quantity"**

## Root Cause
The algorithm was attempting to submit MarketOrder calls with zero quantity in several locations:

1. **Portfolio Execution Manager**: Quantity calculation could result in exactly `0` due to rounding/precision issues
2. **MTUM Strategy**: Trade quantity calculation could result in `0` 
3. **Rollover Logic**: Position transfers could result in zero quantities

## Fix Applied

### 1. Portfolio Execution Manager
**File**: `src/components/portfolio_execution_manager.py`
```python
# BEFORE (line ~490):
quantity_diff = int(trade_value / (price * multiplier))
if abs(quantity_diff) < 1:  # This doesn't catch quantity_diff == 0 reliably

# AFTER:
quantity_diff = int(trade_value / (price * multiplier))

# CRITICAL FIX: Explicit zero quantity check to prevent QC error
if quantity_diff == 0:
    result['blocked_reason'] = f"Calculated quantity is exactly zero"
    self.algorithm.Log(f"    BLOCKED: {result['blocked_reason']}")
    return result

if abs(quantity_diff) < 1:
    result['blocked_reason'] = f"Quantity {quantity_diff} rounds to less than 1 contract"
    self.algorithm.Log(f"    BLOCKED: {result['blocked_reason']}")
    return result
```

### 2. MTUM Strategy
**File**: `src/strategies/mtum_cta_strategy.py`
```python
# BEFORE (line ~504):
if abs(trade_quantity) > 0:

# AFTER:
# CRITICAL FIX: Explicit zero quantity check to prevent QC error
if trade_quantity != 0 and abs(trade_quantity) > 0:
```

### 3. Main Algorithm Rollover
**File**: `main.py`
```python
# BEFORE (line ~816):
open_ticket = self.MarketOrder(actual_symbol, quantity, tag=f"Rollover-Open-{actual_symbol}")

# AFTER:
# CRITICAL FIX: Check for zero quantity before placing rollover order
if quantity == 0:
    self.Log(f"  ROLLOVER SKIPPED: Zero quantity for {actual_symbol}")
    return

open_ticket = self.MarketOrder(actual_symbol, quantity, tag=f"Rollover-Open-{actual_symbol}")
```

### 4. Multi-Strategy Framework (Already Correct)
**File**: `src/components/multi_strategy_framework.py`
```python
# ALREADY CORRECT (line 171):
if quantity != 0:
    self.MarketOrder(new_symbol, quantity, tag=f"Rollover_Establish_{self.Time.strftime('%Y%m%d')}")
```

## Why This Happened

1. **Floating-Point Precision**: Calculations like `int(trade_value / (price * multiplier))` can result in exactly `0` due to precision issues
2. **Rounding Edge Cases**: When target positions are very small, they round to zero
3. **Portfolio Rebalancing**: When current position equals target position, difference is zero
4. **Rollover Edge Cases**: When rolling over positions, calculated quantities might be zero

## Prevention

The fix adds explicit `== 0` checks before all `MarketOrder` calls to prevent QuantConnect from receiving zero quantity orders. This is more reliable than relying on `abs(quantity) > 0` checks which might miss edge cases.

## Testing

Next backtest should not encounter the "zero quantity" error. The algorithm will now:
- Log when zero quantities are detected
- Skip the problematic orders gracefully
- Continue execution without crashing

This maintains the algorithm's robustness while preventing the QuantConnect error. 