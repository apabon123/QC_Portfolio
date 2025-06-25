# Warmup Configuration Fix

## Problem Identified

The algorithm was setting inconsistent warmup periods due to multiple conflicting warmup systems:

### 1. **Expected: 756 Days (MTUM_CTA Strategy)**
- MTUM_CTA strategy configured with `'warmup_days': 756` (3 years)
- Required for 3-year volatility calculations in momentum strategy
- This is the CORRECT requirement

### 2. **Problem 1: Hardcoded 80 Days Override**
- `multi_strategy_framework.py` had hardcoded: `self.SetWarmUp(timedelta(days=80), Resolution.Daily)`
- This **overrode** the strategy-specific 756 days requirement
- Resulted in insufficient warmup for MTUM calculations

### 3. **Problem 2: Calculation Bug (16 Days)**
- `AlgorithmConfigManager.calculate_max_warmup_needed()` was only finding 16 days
- Algorithm config had `minimum_days: 15` with `buffer_multiplier: 1.1` = 16.5 days
- **BUG**: Method wasn't reading strategy's `warmup_days: 756` field correctly

## Fixes Applied

### Fix 1: Removed Hardcoded Warmup
**File**: `CTA Replication/src/components/multi_strategy_framework.py`
```python
# REMOVED this hardcoded override:
# self.SetWarmUp(timedelta(days=80), Resolution.Daily)

# REPLACED with comment explaining main.py handles warmup
```

### Fix 2: Fixed Warmup Calculation
**File**: `CTA Replication/src/config/algorithm_config_manager.py`
```python
# ADDED proper reading of strategy 'warmup_days' field:
direct_warmup_days = strategy_config.get('warmup_days', 0)
if direct_warmup_days > 0:
    max_days = max(max_days, direct_warmup_days)
    self.algorithm.Log(f"CONFIG: Strategy {strategy_name} requires {direct_warmup_days} warmup days")
```

## Expected Result

Next algorithm run should show:
- **CONFIG: Strategy MTUM_CTA requires 756 warmup days**
- **CONFIG MANAGER: Calculated warmup period: ~832 days** (756 × 1.1 buffer)
- **CONFIG APPLIED: SetWarmUp(832 days)** (or similar with buffer)

## Verification

Check logs for:
1. No "80 days" warmup anymore
2. Strategy-specific warmup detection: "Strategy MTUM_CTA requires 756 warmup days"
3. Proper buffered calculation: "Calculated warmup period: 832 days" (756 × 1.1 = 831.6)

This ensures MTUM_CTA gets the full 3 years of data needed for proper volatility calculations. 