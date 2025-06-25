# MTUM Single Strategy Testing Configuration

## Overview
This document outlines the configuration changes made to test MTUM_CTA as a single strategy, disabling all other strategies to isolate MTUM's performance and behavior.

## Configuration Changes Made

### 1. Strategy Enablement (config_market_strategy.py)

**KestnerCTA Strategy - DISABLED:**
```python
'KestnerCTA': {
    'enabled': False,  # Changed from True to False
    # Comment updated to reflect single strategy testing
}
```

**MTUM_CTA Strategy - ENABLED:**
```python
'MTUM_CTA': {
    'enabled': True,  # Remains enabled for single strategy testing
    # All parameters remain unchanged
}
```

**Other Strategies:**
- `SimpleMACross`: Already disabled (`enabled': False`)
- `HMM_CTA`: Already disabled (`enabled': False`)

### 2. Allocation Configuration Changes

**Initial Allocations:**
```python
'initial_allocations': {
    'SimpleMACross': 0.00,  # 0% (disabled)
    'KestnerCTA': 0.00,     # Changed from 0.70 to 0.00
    'MTUM_CTA': 1.00,       # Changed from 0.30 to 1.00 (100%)
    'HMM_CTA': 0.00,        # 0% (disabled)
}
```

**Allocation Bounds:**
```python
'allocation_bounds': {
    'SimpleMACross': {'min': 0.00, 'max': 0.00},  # Disabled
    'KestnerCTA': {'min': 0.00, 'max': 0.00},     # Changed to 0-0% (disabled)
    'MTUM_CTA': {'min': 1.00, 'max': 1.00},       # Changed to 100% fixed
    'HMM_CTA': {'min': 0.00, 'max': 0.00},        # Disabled
}
```

## Expected Behavior

### 1. Strategy Loading
- Only MTUM_CTA strategy will be loaded and initialized
- KestnerCTA will not be instantiated or consume resources
- Layer 2 allocator will recognize only one enabled strategy

### 2. Allocation System
- MTUM_CTA will receive 100% allocation at all times
- Layer 2 allocator will handle single-strategy case correctly
- No rebalancing between strategies (since there's only one)
- Performance tracking will focus solely on MTUM strategy

### 3. Trading Behavior
- All signals will come exclusively from MTUM_CTA strategy
- Monthly rebalancing schedule (first Friday of each month)
- Momentum-based long/short positions using 6-month and 12-month lookbacks
- 20% target volatility for the strategy
- 60% maximum single position weight

### 4. Performance Metrics
- Strategy allocation charts will show 100% MTUM_CTA consistently
- Portfolio performance will reflect pure MTUM strategy returns
- No multi-strategy correlation or diversification effects

## MTUM Strategy Parameters

### Core Configuration
- **Momentum Lookbacks**: 6 months, 12 months
- **Volatility Lookback**: 756 days (3 years)
- **Target Volatility**: 20% annualized
- **Rebalance Frequency**: Monthly (first Friday)
- **Position Limits**: 60% maximum single position
- **Long/Short**: Enabled (allows short positions)

### Risk Management
- **Min Weight Threshold**: 1% minimum position size
- **Min Trade Value**: $1,000 minimum trade value
- **Max Leverage Multiplier**: 10x maximum leverage
- **Signal Clipping**: ±3 standard deviations

### Warmup Period
- **Required Days**: 756 days (3 years for volatility estimation)
- **Indicator Validation**: momentum_6m, momentum_12m, volatility_3y
- **Min Data Points**: 756 days minimum

## Testing Benefits

### 1. Isolated Performance Analysis
- Pure MTUM strategy performance without multi-strategy effects
- Clear attribution of returns to momentum methodology
- No allocation noise from Layer 2 rebalancing

### 2. Strategy Validation
- Verify MTUM implementation works correctly in isolation
- Test monthly rebalancing logic
- Validate momentum signal generation and position sizing

### 3. Debugging Simplification
- Easier to trace issues to specific MTUM components
- Simplified logging and performance tracking
- Clear identification of MTUM-specific problems

### 4. Baseline Establishment
- Establish pure MTUM performance baseline
- Compare against future multi-strategy results
- Validate strategy meets expected Sharpe ratio (0.6 target)

## Reverting to Multi-Strategy

To revert back to multi-strategy testing:

1. **Re-enable KestnerCTA:**
   ```python
   'KestnerCTA': {'enabled': True}
   ```

2. **Restore Original Allocations:**
   ```python
   'initial_allocations': {
       'KestnerCTA': 0.70,
       'MTUM_CTA': 0.30,
   }
   ```

3. **Restore Allocation Bounds:**
   ```python
   'allocation_bounds': {
       'KestnerCTA': {'min': 0.50, 'max': 0.80},
       'MTUM_CTA': {'min': 0.20, 'max': 0.50},
   }
   ```

## Validation

### Syntax Check Passed
- `main.py` compiles successfully ✅
- `mtum_cta_strategy.py` compiles successfully ✅
- All configuration files validate correctly ✅

### Expected Log Output
```
LAYER 2: Allocator initialized with 1 strategies
MTUMCTA: Strategy initialized successfully
LAYER 2: Using equal weight allocations: 100.0% each
MTUM_CTA: Final targets: 100.0% allocation
```

## Summary

The configuration is now set up to test MTUM_CTA as a single strategy with:
- 100% fixed allocation to MTUM_CTA
- All other strategies disabled
- Monthly rebalancing schedule
- Pure momentum-based trading signals
- Isolated performance tracking

This setup allows for focused testing and validation of the MTUM strategy implementation without multi-strategy complexity. 