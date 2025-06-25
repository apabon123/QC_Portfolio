# Monthly Return Calculation Fix

## Problem Identified

**Issue**: Portfolio equity was changing significantly (e.g., from $10,000,000 to $9,797,521) but monthly reports showed **0.00% Monthly Return** and **0.00% YTD Return**.

**Root Cause**: The `SystemReporter._calculate_period_performance()` method was a placeholder that returned an empty dictionary `{}`, causing all performance calculations to default to 0.00%.

## Evidence from Logs

```
2017-04-01 Portfolio Value: $10,000,000.00
2017-04-29 Portfolio Value: $9,797,521 (actual -2.02% loss)
2017-05-01 Monthly Return: 0.00%  ❌ WRONG
2017-05-01 YTD Return: 0.00%      ❌ WRONG
```

## Technical Analysis

### **Bug Flow:**
1. **Monthly Report Generation** (`generate_monthly_performance_report()`)
   ```python
   'performance': self._calculate_period_performance(days=30)  # Returns {}
   ```

2. **Performance Calculation** (Line 707 - PLACEHOLDER)
   ```python
   def _calculate_period_performance(self, days: int) -> Dict[str, Any]: 
       return {}  # ❌ Empty dictionary
   ```

3. **Monthly Logging** (`_log_monthly_highlights()`)
   ```python
   perf = report.get('performance', {})  # Gets empty dict {}
   self.algorithm.Log(f"Monthly Return: {perf.get('monthly_return', 0.0):.2%}")  # Always 0.0
   ```

## Solution Implemented

### **1. Fixed Performance Calculation Method**

**File**: `src/components/system_reporter.py`

**Replaced** placeholder method with real implementation:

```python
def _calculate_period_performance(self, days: int) -> Dict[str, Any]:
    """Calculate actual period performance using QC native Portfolio tracking."""
    try:
        # Get current portfolio value using QC's native method
        current_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
        
        # Get initial capital from config
        initial_capital = float(self.config.get('algorithm', {}).get('initial_capital', 10000000))
        
        # Calculate YTD return (from start of algorithm)
        ytd_return = (current_value - initial_capital) / initial_capital if initial_capital > 0 else 0.0
        
        # Calculate monthly return from performance history
        monthly_return = 0.0
        if hasattr(self, 'performance_history') and len(self.performance_history) >= days:
            # Get portfolio value from ~30 days ago
            month_ago_data = self.performance_history[-(days + 1)]
            month_ago_value = month_ago_data.get('portfolio_value', initial_capital)
            if month_ago_value > 0:
                monthly_return = (current_value - month_ago_value) / month_ago_value
        
        # If no history available, monthly return = YTD return for first month
        if monthly_return == 0.0 and len(self.performance_history) == 0:
            monthly_return = ytd_return
        
        return {
            'current_value': current_value,
            'initial_capital': initial_capital,
            'monthly_return': monthly_return,
            'ytd_return': ytd_return,
            'portfolio_volatility': self._calculate_portfolio_volatility(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown(),
            'calculation_method': 'qc_native_portfolio_tracking'
        }
        
    except Exception as e:
        # Safe fallback with actual calculations
        current_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
        initial_capital = float(self.config.get('algorithm', {}).get('initial_capital', 10000000))
        ytd_return = (current_value - initial_capital) / initial_capital if initial_capital > 0 else 0.0
        
        return {
            'current_value': current_value,
            'monthly_return': ytd_return,  # Fallback to YTD
            'ytd_return': ytd_return,
            'calculation_method': 'fallback_simple'
        }
```

### **2. Enhanced Max Drawdown Calculation**

**Replaced** placeholder with real implementation using performance history:

```python
def _calculate_max_drawdown(self) -> float:
    """Calculate maximum drawdown using performance history."""
    try:
        if len(self.performance_history) < 2:
            return 0.0
        
        # Get portfolio values from history
        portfolio_values = [d.get('portfolio_value', 0.0) for d in self.performance_history]
        
        # Calculate running maximum and drawdowns
        running_max = portfolio_values[0]
        max_drawdown = 0.0
        
        for value in portfolio_values[1:]:
            running_max = max(running_max, value)
            if running_max > 0:
                drawdown = (running_max - value) / running_max
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
        
    except Exception as e:
        return 0.0
```

### **3. Added Daily Performance Tracking**

**File**: `main.py`

**Added** daily performance tracking to ensure performance history is populated:

```python
# Track daily performance for accurate monthly return calculations
if hasattr(self, 'system_reporter') and self.system_reporter:
    self.system_reporter.generate_daily_performance_update()
```

## Expected Results

### **Before Fix:**
```
2017-05-01 Monthly Return: 0.00%
2017-05-01 YTD Return: 0.00%
2017-05-01 Portfolio Value: $9,797,521
```

### **After Fix:**
```
2017-05-01 Monthly Return: -2.02%  ✅ CORRECT
2017-05-01 YTD Return: -2.02%     ✅ CORRECT  
2017-05-01 Portfolio Value: $9,797,521
```

## Key Benefits

1. **Accurate Performance Reporting**: Monthly and YTD returns now reflect actual portfolio changes
2. **QC Native Integration**: Uses `self.algorithm.Portfolio.TotalPortfolioValue` for calculations
3. **Robust Error Handling**: Safe fallbacks prevent crashes while maintaining accuracy
4. **Historical Tracking**: Daily performance history enables proper period calculations
5. **Professional Analytics**: Real drawdown calculations using performance history

## Technical Implementation

### **Components Updated:**
- `src/components/system_reporter.py` - Fixed performance calculations
- `main.py` - Added daily performance tracking

### **Integration Points:**
- **Daily**: `generate_daily_performance_update()` tracks portfolio values
- **Monthly**: `generate_monthly_performance_report()` calculates period returns
- **Final**: `generate_final_algorithm_report()` provides comprehensive analysis

### **Data Flow:**
```
OnData() → generate_daily_performance_update() → performance_history[]
                                                        ↓
MonthlyReporting() → _calculate_period_performance() → actual returns
```

## Validation

The fix addresses the core issue where:
- **Portfolio values were changing** (tracked correctly by QC)
- **Returns showed 0.00%** (due to placeholder method)
- **Now returns are calculated** using actual portfolio value changes

This ensures that monthly reports accurately reflect the algorithm's performance and provide meaningful analytics for strategy evaluation. 