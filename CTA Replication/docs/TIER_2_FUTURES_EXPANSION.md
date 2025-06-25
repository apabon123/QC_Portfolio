# Tier 2 Futures Expansion for MTUM Strategy

## Overview
This document outlines the expansion from tier 1 to tier 2 futures for MTUM strategy testing, increasing the universe from 3 to 6 liquid futures contracts.

## Changes Made

### 1. Universe Configuration (config_market_strategy.py)

**Priority Level Expansion:**
```python
# Universe Loading Configuration
'loading': {
    'max_priority': 2,                      # EXPANDED: Load priority 1 and 2 symbols
    'include_expansion_candidates': False,  # TESTING: Disable expansion candidates  
    'priority_override': 2,                 # EXPANDED: Allow priority 1 and 2
    'exclude_problematic_symbols': ['6J'],  # TESTING: Exclude 6J due to pricing issues
},
```

**Futures Universe Expansion:**
```python
'futures': {
    'equity_index': {
        'ES': {'priority': 1, 'name': "E-mini S&P 500"},      # Tier 1: Most liquid
        'NQ': {'priority': 2, 'name': "Nasdaq 100"},          # Tier 2: Very liquid
    },
    'interest_rates': {
        'ZN': {'priority': 2, 'name': "10-Year Treasury Note"}, # Tier 2: Very liquid
    },
    'commodities': {
        'CL': {'priority': 1, 'name': "Crude Oil"},           # Tier 1: Most liquid
        'GC': {'priority': 1, 'name': "Gold"},                # Tier 1: Most liquid
    },
    'foreign_exchange': {
        '6E': {'priority': 2, 'name': "Euro FX"},             # Tier 2: Very liquid
    }
}
```

### 2. MTUM Strategy Asset Filter Update

**Expanded Allowed Categories:**
```python
'MTUM_CTA': {
    'allowed_categories': [
        'futures_priority_1', 'futures_priority_2',           # Priority-based filtering
        'futures_equity', 'futures_rates', 'futures_fx', 'futures_commodities'
    ],
    'excluded_categories': ['futures_vix', 'options', 'futures_priority_4'],
    'excluded_symbols': ['6J'],  # Exclude 6J due to pricing issues
    'reason': 'Momentum works on liquid futures markets - using priority 1 and 2 futures for expanded universe'
},
```

### 3. Main Algorithm Universe Setup (main.py)

**Priority-Based Universe Loading:**
```python
# Priority-based futures universe
priority_1_futures = ['ES', 'CL', 'GC']  # S&P 500, Crude Oil, Gold - most liquid
priority_2_futures = ['NQ', 'ZN', '6E']  # NASDAQ, 10Y Note, EUR/USD - very liquid

# Build final futures list based on priority and exclusions
futures_to_add = []

if max_priority >= 1:
    for symbol in priority_1_futures:
        if symbol not in excluded_symbols:
            futures_to_add.append(symbol)

if max_priority >= 2:
    for symbol in priority_2_futures:
        if symbol not in excluded_symbols:
            futures_to_add.append(symbol)
```

## Expanded Universe Details

### **Tier 1 Futures (Most Liquid)**
1. **ES** - E-mini S&P 500
   - Category: Equity Index
   - Liquidity: Highest (100,000+ daily volume)
   - Use Case: Core equity exposure

2. **CL** - Crude Oil
   - Category: Commodities
   - Liquidity: Very High (30,000+ daily volume)
   - Use Case: Energy/commodity exposure

3. **GC** - Gold
   - Category: Commodities  
   - Liquidity: Very High (25,000+ daily volume)
   - Use Case: Safe haven/inflation hedge

### **Tier 2 Futures (Very Liquid)**
4. **NQ** - Nasdaq 100
   - Category: Equity Index
   - Liquidity: High (50,000+ daily volume)
   - Use Case: Tech-heavy equity exposure

5. **ZN** - 10-Year Treasury Note
   - Category: Interest Rates
   - Liquidity: High (50,000+ daily volume)
   - Use Case: Interest rate/duration exposure

6. **6E** - Euro FX
   - Category: Foreign Exchange
   - Liquidity: High (20,000+ daily volume)
   - Use Case: EUR/USD currency exposure

### **Excluded Symbols**
- **6J** (Japanese Yen): Excluded due to pricing inconsistencies (0.01 vs 0.00694)

## Expected Impact

### **Diversification Benefits:**
- **Asset Classes**: 3 → 4 (added FX)
- **Geographic Exposure**: US-focused → US + EUR
- **Sector Coverage**: Broader equity (S&P 500 + NASDAQ) + rates + commodities + FX

### **Strategy Performance:**
- **More Signals**: 6 instruments vs 3 for momentum detection
- **Risk Spreading**: Reduced concentration risk across asset classes
- **Rebalancing Efficiency**: More opportunities for monthly rebalancing

### **Risk Considerations:**
- **Correlation Risk**: Monitor cross-asset correlations during stress periods
- **Complexity**: More instruments to monitor and validate
- **Rollover Events**: 6 contracts vs 3 requiring rollover management

## Testing Strategy

### **Phase 1: Validation (Current)**
- Verify all 6 contracts load correctly
- Check data quality and pricing consistency
- Validate MTUM signals across expanded universe

### **Phase 2: Performance Analysis**
- Compare 6-asset vs 3-asset MTUM performance
- Analyze Sharpe ratio and drawdown improvements
- Monitor allocation distribution across assets

### **Phase 3: Optimization**
- Fine-tune momentum parameters for expanded universe
- Adjust position sizing for 6-asset portfolio
- Optimize rebalancing frequency if needed

## Configuration Status

✅ **Universe Loading**: Expanded to priority 2  
✅ **Asset Filters**: Updated for MTUM strategy  
✅ **Main Algorithm**: Updated universe setup  
✅ **Syntax Validation**: All files compile successfully  
✅ **Exclusions**: 6J properly excluded due to pricing issues  

## Next Steps

1. **Run Backtest**: Test MTUM with expanded 6-asset universe
2. **Monitor Logs**: Check for any data quality issues with new assets
3. **Performance Review**: Compare results vs 3-asset universe
4. **Risk Analysis**: Monitor correlations and drawdowns
5. **Consider Tier 3**: If successful, consider adding YM, ZB (priority 3 futures)

The algorithm is now ready to test MTUM strategy with the expanded tier 2 futures universe. 