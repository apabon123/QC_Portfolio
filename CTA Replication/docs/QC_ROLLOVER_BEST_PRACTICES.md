# QuantConnect Futures Rollover - Best Practices Guide

## Problem: May 12 Rollover Warning

### The Issue
```
2017-05-12 00:00:00 CL WLIYRH06USOX: The security does not have an accurate price as it has not yet received a bar of data. Before placing a trade (or using SetHoldings) warm up your algorithm with SetWarmup, or use slice.Contains(symbol) to confirm the Slice object has price before using the data.
```

**Root Cause**: During futures rollover, QC updates the continuous contract's `.Mapped` property before the new contract receives its first data bar, creating a timing gap where:
- New contract exists but `HasData = False`
- New contract may have `Price = $0.00`
- Traditional validation fails even though rollover will succeed

## QuantConnect's Recommended Solution

Based on [LEAN documentation](https://www.lean.io/docs/v2/lean-engine/class-reference/classes.html) and community best practices:

### 1. Use `slice.Contains()` Pattern ✅

**QC's Primary Recommendation**: Always check `slice.ContainsKey(symbol)` before accessing price data.

```python
def validate_symbol_for_trading(self, symbol, slice):
    """QC's recommended validation pattern"""
    if symbol not in self.Securities:
        return False
    
    security = self.Securities[symbol]
    
    # PRIMARY: Check if symbol is in current slice
    if slice and slice.ContainsKey(symbol):
        # Symbol has data in current slice - safe to trade
        return security.IsTradable and security.Price > 0
    
    # FALLBACK: Traditional HasData check
    return security.IsTradable and security.HasData and security.Price > 0
```

### 2. Rollover-Aware Validation ✅

For futures with continuous contracts, implement rollover-aware logic:

```python
def validate_futures_symbol(self, symbol, slice):
    """Rollover-aware futures validation"""
    if symbol not in self.Securities:
        return False
    
    security = self.Securities[symbol]
    
    # Check if this is a mapped futures contract
    if hasattr(security, 'Mapped') and security.Mapped:
        mapped_contract = security.Mapped
        
        # ROLLOVER DETECTION: Continuous has data but mapped doesn't
        if security.HasData and mapped_contract in self.Securities:
            mapped_security = self.Securities[mapped_contract]
            
            if not mapped_security.HasData:
                # ROLLOVER SITUATION: Use slice.Contains() for new contract
                if slice and slice.ContainsKey(mapped_contract):
                    self.Log(f"ROLLOVER: {mapped_contract} ready via slice")
                    return mapped_security.IsTradable
                else:
                    # Check if price is available despite no HasData
                    if hasattr(mapped_security, 'Price') and mapped_security.Price > 0:
                        self.Log(f"ROLLOVER: {mapped_contract} has price ${mapped_security.Price:.2f}")
                        return mapped_security.IsTradable
                    else:
                        self.Log(f"ROLLOVER: {mapped_contract} not ready, price: ${mapped_security.Price}")
                        return False
            
            # Normal case: both contracts have data
            return mapped_security.IsTradable and mapped_security.HasData
    
    # Non-futures or direct contract
    return self._validate_with_slice_pattern(symbol, slice)
```

### 3. Store Current Slice ✅

Make the current slice available to all components:

```python
def OnData(self, slice):
    """Store slice for component access"""
    # Store current slice for QC's recommended validation patterns
    self.current_slice = slice
    
    # Continue with normal processing
    self._handle_trading_logic(slice)
```

## Implementation in Our System

### Files Updated

1. **`main.py`**: Stores `current_slice` for component access
2. **`portfolio_execution_manager.py`**: Implements slice.Contains() validation
3. **Enhanced rollover logging**: Shows actual prices during rollover

### Expected Log Output (After Fix)

```
2017-05-12 00:00:00     ROLLOVER: New contract CL WLIYRH06USOX ready via slice (price $47.86)
2017-05-12 00:00:00 ORDER FILLED: CL WKQESREAMX1D +64 @ $47.84 ($+3,062)
2017-05-12 00:00:00 ROLLOVER SUCCESS: CL WKQESREAMX1D -> CL WLIYRH06USOX, quantity: -64.0
2017-05-12 00:00:00   Rollover prices: CL WKQESREAMX1D @ $47.84 -> CL WLIYRH06USOX @ $47.86 (spread: $0.02)
```

## Key Takeaways

### ✅ Do This
- **Always use `slice.ContainsKey(symbol)`** before accessing price data
- **Store current slice** in `OnData()` for component access
- **Implement rollover-aware validation** for futures
- **Log actual prices** during rollover for debugging
- **Trust QC's rollover system** - positions transfer automatically

### ❌ Don't Do This
- Don't rely solely on `HasData` during rollover periods
- Don't assume `Price > 0` means the contract is ready
- Don't implement custom rollover logic - use QC's native system
- Don't skip validation because "rollover should work"

## Performance Impact

- **Minimal overhead**: `slice.ContainsKey()` is a fast dictionary lookup
- **Better reliability**: Eliminates rollover warnings and failed validations
- **Cleaner logs**: Professional appearance suitable for production
- **Future-proof**: Uses QC's recommended patterns that work across all asset classes

## Testing Validation

To test if the fix works:

1. **Look for eliminated warnings**: No more "security does not have accurate price" messages
2. **Check rollover logs**: Should show actual prices, not $0.00
3. **Verify trades execute**: Rollover should not block legitimate trades
4. **Monitor performance**: No degradation in execution speed

This approach aligns with QuantConnect's best practices and ensures robust futures trading during rollover periods. 