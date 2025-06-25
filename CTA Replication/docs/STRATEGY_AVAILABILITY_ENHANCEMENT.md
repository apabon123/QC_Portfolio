# Strategy Availability Enhancement

## 🚨 **Issue Identified**
From log analysis: `MTUM_CTA - NOT_AVAILABLE` occurring frequently without diagnostic information, causing:
1. **No visibility** into WHY strategies become unavailable
2. **Allocation instability** when strategies fail temporarily  
3. **No fallback mechanisms** for temporary failures
4. **Log space waste** with verbose diagnostics

## ✅ **Solution Implemented**

### **1. Enhanced Diagnostic Logging (Compact Format)**

#### **MTUM Strategy Enhancements:**
```python
@property
def IsAvailable(self):
    """Check availability with detailed diagnostics but compact logging."""
    # Collect diagnostic info for not-ready symbols
    if not is_available:
        not_ready_symbols = [detail['symbol'] for detail in not_ready_details]
        self._last_unavailable_reason = f"Ready: {ready_count}/{total_count} (need {required_count}). Not ready: {', '.join(not_ready_symbols[:3])}" + ("..." if len(not_ready_symbols) > 3 else "")
    
    return is_available

def log_availability_status(self, force=False):
    """Log availability status (compact format to save log space)."""
    if not force and self.IsAvailable:
        return  # Don't log when available unless forced
    
    reason = getattr(self, '_last_unavailable_reason', 'Unknown')
    self.algorithm.Log(f"{self.name}: NOT_AVAILABLE - {reason}")
```

**Example Log Output:**
```
MTUM_CTA: NOT_AVAILABLE - Ready: 1/3 (need 2). Not ready: ES, CL...
```

### **2. Allocation Persistence System**

#### **Configuration-Driven Behavior:**
```python
# In config_market_strategy.py
'availability_handling': {
    'mode': 'persistence',                  # 'persistence' or 'reallocate'
    'persistence_threshold': 0.1,           # Min allocation for available strategies (10%)
    'emergency_reallocation_ratio': 0.5,    # Split 50/50 between available/unavailable in emergency
    'log_unavailable_reasons': True,        # Log WHY strategies are unavailable
    'max_consecutive_unavailable_days': 7,  # Alert after N days unavailable
}
```

#### **Two Allocation Modes:**

**A) Persistence Mode (Default):**
- **Maintains allocations** for unavailable strategies
- **Normalizes among active strategies** for actual trading
- **Prevents allocation churn** from temporary failures
- **Emergency reallocation** only if <10% allocation available

**B) Reallocate Mode:**
- **Zeros out unavailable strategies** immediately
- **Reallocates 100%** to available strategies
- **More aggressive** but causes allocation instability

### **3. Smart Combination Logic**

#### **Active vs Inactive Strategy Handling:**
```python
def combine_strategy_targets(self, strategy_targets):
    """Only combine targets from strategies that provided signals"""
    # STEP 1: Identify which strategies provided targets
    active_strategies = set(strategy_targets.keys())
    inactive_strategies = all_strategies - active_strategies
    
    # STEP 2: Normalize allocations among active strategies only
    if total_active_allocation > 0:
        for strategy_name in active_allocations:
            active_allocations[strategy_name] /= total_active_allocation
    
    # STEP 3: Log what happened
    if inactive_strategies:
        self.algorithm.Log(f"ALLOCATOR: Combined {len(active_strategies)} active strategies (inactive: {', '.join(inactive_strategies)})")
```

### **4. Consecutive Day Tracking & Alerts**

#### **Availability Monitoring:**
```python
def _track_strategy_availability(self, available_strategies, unavailable_strategies):
    """Track consecutive unavailable days and generate alerts."""
    for strategy_name in unavailable_strategies:
        self.unavailable_days_counter[strategy_name] += 1
        
        # Alert for extended unavailability
        days_unavailable = self.unavailable_days_counter[strategy_name]
        if days_unavailable == self.max_unavailable_days:
            self.algorithm.Log(f"ALERT: {strategy_name} unavailable for {days_unavailable} consecutive days")
```

### **5. Orchestrator Integration**

#### **Enhanced Layer 1 Execution:**
```python
# Log summary with unavailable details
if unavailable_strategies:
    self.algorithm.Log(f"LAYER 1: {available_count}/{len(loaded_strategies)} available. Unavailable: {', '.join(unavailable_strategies)}")

# Log WHY strategies are not available (compact format)
if not is_available:
    if hasattr(strategy, 'log_availability_status'):
        strategy.log_availability_status()  # Compact diagnostic logging
```

## 📊 **Expected Log Output**

### **Before Enhancement:**
```
2017-02-18 00:00:00 LAYER 1: MTUM_CTA - NOT_AVAILABLE
2017-02-18 00:00:00 LAYER 1: Generating signals from 0/1 available strategies
```

### **After Enhancement:**
```
2017-02-18 00:00:00 MTUM_CTA: NOT_AVAILABLE - Ready: 1/3 (need 2). Not ready: ES, CL
2017-02-18 00:00:00 LAYER 1: 0/1 available. Unavailable: MTUM_CTA
2017-02-18 00:00:00 ALLOCATOR: Persistence mode - 0 available (0.0%), 1 unavailable (100.0%)
2017-02-18 00:00:00 ALLOCATOR: No strategies available - maintaining current allocations
```

### **When Strategy Recovers:**
```
2017-02-25 00:00:00 ALLOCATOR: MTUM_CTA available again after 7 days
2017-02-25 00:00:00 LAYER 1: All 1/1 strategies available
```

## 🎯 **Benefits Achieved**

### **1. Diagnostic Visibility**
- **Compact logging** shows exactly why strategies are unavailable
- **Symbol-level details** (Ready: 1/3, Not ready: ES, CL)
- **Saves log space** by avoiding verbose diagnostics unless needed

### **2. Allocation Stability** 
- **Persistence mode** prevents constant reallocation churn
- **Layer 2 allocations remain stable** even when strategies fail
- **Emergency reallocation** only when absolutely necessary

### **3. Robust Fallback System**
- **Graceful degradation** when strategies become unavailable
- **Automatic recovery** when strategies become available again
- **Configurable behavior** for different operational needs

### **4. Operational Monitoring**
- **Consecutive day tracking** identifies persistent issues
- **Automatic alerts** after configurable thresholds
- **Recovery notifications** when strategies come back online

## 🔧 **Configuration Options**

### **For Conservative Operation (Default):**
```python
'availability_handling': {
    'mode': 'persistence',           # Maintain allocations for unavailable strategies
    'persistence_threshold': 0.1,    # Allow down to 10% for available strategies
    'emergency_reallocation_ratio': 0.5,  # 50/50 split in emergency
}
```

### **For Aggressive Reallocation:**
```python
'availability_handling': {
    'mode': 'reallocate',           # Zero out unavailable strategies immediately
    'persistence_threshold': 0.0,   # No persistence
    'emergency_reallocation_ratio': 1.0,  # 100% to available
}
```

This enhancement addresses all your concerns:
- ✅ **Logs WHY strategies are unavailable** (compact format)
- ✅ **Allocation persistence** prevents rebalancing churn  
- ✅ **Layer 2 stability** even when strategies fail
- ✅ **Minimal log usage** with maximum diagnostic value
- ✅ **Configurable behavior** for different operational needs

## 🔄 **Integration with Bad Data Position Manager**

### **Two-Level Data Quality Architecture**

The enhanced strategy availability system works **in tandem** with the existing bad data position manager to provide comprehensive data quality management:

#### **Level 1: Strategy-Level Availability (New Enhancement)**
- **Scope**: Overall strategy data sufficiency for signal generation
- **Triggers**: 
  - Insufficient historical data for momentum calculations
  - Too few symbols ready (MTUM needs ≥50% of symbols)
  - Data initialization failures during warmup
- **Actions**: 
  - Mark strategy as "NOT_AVAILABLE"
  - Log compact diagnostic reason
  - Maintain allocations in persistence mode
  - Track consecutive unavailable days

#### **Level 2: Position-Level Data Management (Existing)**
- **Scope**: Individual position management when data becomes unreliable
- **Triggers**:
  - Real-time price spikes or outliers
  - Missing data for existing positions
  - Extreme price movements (potential bad data)
- **Actions**:
  - HOLD: Keep position, use last good price
  - LIQUIDATE: Close position immediately
  - HEDGE: Reduce position size gradually  
  - FREEZE: Block new trades but keep existing position

### **Integration Flow & Data Pipeline**

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Data Slice    │───▶│  Strategy Availability │───▶│  Signal Generation  │
│                 │    │       Check            │    │                     │
└─────────────────┘    └──────────────────────┘    └─────────────────────┘
                                │                               │
                                ▼                               ▼
                    ┌─────────────────────┐         ┌─────────────────────┐
                    │ NOT_AVAILABLE:      │         │   Layer 2 Allocation │
                    │ • Log reason        │         │   • Persistence mode │
                    │ • Maintain alloc    │         │   • Normalize active │
                    │ • Track days        │         └─────────────────────┘
                    └─────────────────────┘                     │
                                                                ▼
                                                    ┌─────────────────────┐
                                                    │ Bad Data Position   │
                                                    │      Check          │
                                                    └─────────────────────┘
                                                                │
                                    ┌───────────────────────────┼───────────────────────────┐
                                    ▼                           ▼                           ▼
                        ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
                        │ ALLOW NEW TRADE │         │ BLOCK NEW TRADE │         │ MANAGE EXISTING │
                        │ • Execute order │         │ • Log strategy  │         │ • HOLD/LIQUIDATE│
                        │ • Normal flow   │         │ • Skip symbol   │         │ • HEDGE/FREEZE  │
                        └─────────────────┘         └─────────────────┘         └─────────────────┘
```

### **Integration Points in Code**

#### **1. Strategy Availability → Signal Generation**
```python
# In three_layer_orchestrator.py
if not strategy.IsAvailable:
    strategy.log_availability_status()  # Compact diagnostic
    continue  # Skip signal generation, maintain allocation
```

#### **2. Bad Data Position → Trade Execution**
```python
# In portfolio_execution_manager.py
if self.bad_data_manager and not self.bad_data_manager.should_allow_new_trade(symbol):
    result['blocked_reason'] = f"Bad data manager blocked new trade for {ticker}"
    # Log specific bad data strategy (HOLD/LIQUIDATE/HEDGE/FREEZE)
    return result  # Block execution
```

#### **3. Data Quality Reporting Integration**
```python
# In centralized_data_validator.py
if validation_fails:
    if self.bad_data_manager:
        self.bad_data_manager.report_data_issue(symbol, issue_type, severity)
```

### **Complementary Behavior Examples**

#### **Scenario 1: Insufficient Historical Data**
```
Day 1: MTUM_CTA: NOT_AVAILABLE - Ready: 1/3 (need 2). Not ready: ES, CL
       ALLOCATOR: Persistence mode - 0 available (0.0%), 1 unavailable (100.0%)
       → Strategy allocation maintained, no trades executed
       
Day 7: ALERT: MTUM_CTA unavailable for 7 consecutive days
       → Operator alerted to investigate data issue
       
Day 10: ALLOCATOR: MTUM_CTA available again after 10 days
        → Automatic recovery, normal trading resumes
```

#### **Scenario 2: Bad Real-Time Data During Trading**
```
MTUM_CTA: AVAILABLE (strategy can generate signals)
ES Signal: +60% allocation target
BadDataManager: ES price spike detected ($4000 → $40000)
EXECUTION: BLOCKED - Bad data manager blocked new trade for ES
BAD DATA: Strategy=FREEZE, Issues=1
→ Existing ES position kept, no new trades until data quality improves
```

#### **Scenario 3: Combined Issues**
```
MTUM_CTA: NOT_AVAILABLE - Ready: 2/3 (need 2). Not ready: CL
BadDataManager: GC price outlier detected
EXECUTION: 
  - ES: Executed (strategy available + data good)
  - CL: Skipped (strategy unavailable)
  - GC: Blocked (bad data manager FREEZE)
→ Only ES trades, CL/GC positions managed appropriately
```

### **Configuration Coordination**

#### **Strategy Availability Settings**
```python
# In config_market_strategy.py
'availability_handling': {
    'mode': 'persistence',                  # Maintain allocations when unavailable
    'max_consecutive_unavailable_days': 7,  # Alert threshold
}
```

#### **Bad Data Position Settings**
```python
# In bad_data_position_manager.py  
'default_strategies': {
    'ES': 'HOLD',      # Core positions - conservative
    'CL': 'HEDGE',     # Commodities - reduce gradually
    '6E': 'FREEZE',    # FX - block new trades
}
```

### **Benefits of Integrated System**

1. **Comprehensive Coverage**: Strategy-level + position-level data quality management
2. **No Gaps**: Issues caught at multiple levels prevent trading on bad data
3. **Graceful Degradation**: System continues operating even with partial data issues
4. **Operational Clarity**: Clear logging shows exactly what's happening and why
5. **Configurable Response**: Different strategies for different types of data issues
6. **Automatic Recovery**: Both systems detect when issues resolve and resume normal operation

This integrated approach ensures robust data quality management across all aspects of the trading system while maintaining operational transparency and control. 