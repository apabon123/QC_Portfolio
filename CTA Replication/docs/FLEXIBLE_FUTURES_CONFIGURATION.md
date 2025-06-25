# Flexible Futures Configuration Guide

This document explains how to easily configure which futures contracts the CTA algorithm trades.

## Two Configuration Methods

### Method 1: Simple Selection (Easiest)

**Use Case**: When you want to specify exactly which futures to trade without worrying about priorities.

**Configuration**: In `src/config/config_market_strategy.py`, find the `UNIVERSE_CONFIG` section:

```python
'simple_selection': {
    'enabled': True,                        # SET TO TRUE to use this method
    'futures_list': [                       # Specify exactly which futures to trade
        'ES',  # E-mini S&P 500
        'NQ',  # Nasdaq 100  
        'ZN',  # 10-Year Treasury Note
        'CL',  # Crude Oil
        'GC',  # Gold
        '6E',  # Euro FX
    ],
    'description': "When enabled, trades exactly these futures regardless of priority settings"
},
```

**How it works**: 
- Set `enabled: True`
- List exactly the futures symbols you want in `futures_list`
- The algorithm will trade only these futures
- Exclusions from `exclude_problematic_symbols` still apply

### Method 2: Priority-Based Filtering (Advanced)

**Use Case**: When you want sophisticated control based on liquidity tiers and categories.

**Configuration**: Keep `simple_selection.enabled: False` and use:

```python
'loading': {
    'max_priority': 2,                      # Load futures with priority 1 and 2
    'exclude_problematic_symbols': ['6J'],  # Exclude specific symbols
},
```

**How it works**:
- Futures are organized by category (equity_index, commodities, etc.)
- Each future has a priority (1 = most liquid, 2 = very liquid, 3+ = less liquid)
- Set `max_priority` to include all futures up to that priority level
- Use exclusions to remove specific problematic symbols

## Available Futures by Category

### Currently Configured Futures:

**Priority 1 (Most Liquid)**:
- `ES`: E-mini S&P 500 (equity_index)
- `CL`: Crude Oil (commodities) 
- `GC`: Gold (commodities)

**Priority 2 (Very Liquid)**:
- `NQ`: Nasdaq 100 (equity_index)
- `ZN`: 10-Year Treasury Note (interest_rates)
- `6E`: Euro FX (foreign_exchange)

### Adding New Futures

To add a new futures contract, add it to the appropriate category in `futures` section:

```python
'futures': {
    'equity_index': {
        'RTY': {                            # NEW: Russell 2000
            'name': "Russell 2000",
            'category': 'futures_equity',
            'priority': 2,
            'min_volume': 30000,
        }
    }
}
```

## Quick Configuration Examples

### Example 1: Trade Only the Big 3 (ES, CL, GC)
```python
'simple_selection': {
    'enabled': True,
    'futures_list': ['ES', 'CL', 'GC'],
}
```

### Example 2: Trade All 6 Current Futures
```python
'simple_selection': {
    'enabled': True,
    'futures_list': ['ES', 'NQ', 'ZN', 'CL', 'GC', '6E'],
}
```

### Example 3: Use Priority System for Only Tier 1
```python
'simple_selection': {
    'enabled': False,  # Use priority system
},
'loading': {
    'max_priority': 1,  # Only most liquid futures
}
```

### Example 4: Use Priority System for Tier 1 + 2
```python
'simple_selection': {
    'enabled': False,  # Use priority system
},
'loading': {
    'max_priority': 2,  # Include very liquid futures too
}
```

## Troubleshooting

### Problem: Algorithm only trades 3 futures instead of 6
**Solution**: Check that either:
- Simple selection is enabled with 6 futures in the list, OR
- Priority system has `max_priority: 2` (not 1)

### Problem: Specific future not trading
**Solution**: Check that the symbol is not in `exclude_problematic_symbols` list

### Problem: No futures trading at all
**Solution**: Verify that:
- Simple selection list is not empty, OR
- Priority system has valid futures at the specified priority level
- No configuration syntax errors

## Current Status

As of the latest update, the algorithm is configured to:
- Use **priority-based filtering** (`simple_selection.enabled: False`)
- Include **priority 1 and 2** futures (`max_priority: 2`)
- Exclude **6J** due to pricing issues
- Result: **6 futures** (ES, NQ, ZN, CL, GC, 6E)

To switch to simple selection mode, just change `enabled: True` in the `simple_selection` section. 