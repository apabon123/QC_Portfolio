# ========================================
# LAYER 2: dynamic_strategy_allocator.py
# ========================================

from AlgorithmImports import *
import numpy as np
from collections import deque

class DynamicStrategyAllocator:
    """
    Layer 2: Dynamic Strategy Allocation
    
    Allocates capital between Layer 1 strategies based on Sharpe ratios.
    Keeps strategies "naive" by handling allocation without internal risk management.
    """
    
    def __init__(self, algorithm, config=None):
        """Initialize dynamic strategy allocator with enhanced availability handling."""
        self.algorithm = algorithm
        self.config = config or {}
        
        # Strategy tracking
        self.strategies = {}
        self.strategy_returns = {}
        self.current_allocations = {}
        self.last_allocation_update = None
        
        # Enhanced availability tracking
        self.availability_config = self.config.get('availability_handling', {})
        self.unavailable_days_counter = {}  # Track consecutive unavailable days
        self.last_availability_check = None
        
        # Default availability handling settings
        self.availability_mode = self.availability_config.get('mode', 'persistence')
        self.persistence_threshold = self.availability_config.get('persistence_threshold', 0.1)
        self.emergency_ratio = self.availability_config.get('emergency_reallocation_ratio', 0.5)
        self.log_reasons = self.availability_config.get('log_unavailable_reasons', True)
        self.max_unavailable_days = self.availability_config.get('max_consecutive_unavailable_days', 7)
        
        algorithm.Log(f"Layer2: Initialized with availability mode: {self.availability_mode}")
    
    def register_strategy(self, strategy_name, strategy_instance, initial_allocation=None):
        """
        Register a strategy for dynamic allocation
        
        Args:
            strategy_name (str): Name of the strategy
            strategy_instance: The actual strategy object
            initial_allocation (float): Initial allocation (if None, uses equal weight)
        """
        self.strategies[strategy_name] = strategy_instance
        
        # Initialize return tracking
        self.strategy_returns[strategy_name] = deque(maxlen=self.config['lookback_days'])
        
        # Set initial allocation
        if initial_allocation is None:
            n_strategies = len(self.strategies)
            initial_allocation = 1.0 / n_strategies
        
        self.current_allocations[strategy_name] = initial_allocation
        
        self.algorithm.Log(f"Layer2: Registered {strategy_name} with {initial_allocation:.1%} allocation")
    
    def should_rebalance_allocations(self):
        """Check if allocations should be updated"""
        if self.config['rebalance_frequency'] == 'weekly':
            return (self.algorithm.Time.weekday() == 4 and  # Friday
                   (self.last_allocation_update is None or 
                    (self.algorithm.Time - self.last_allocation_update).days >= 7))
        elif self.config['rebalance_frequency'] == 'monthly':
            return (self.last_allocation_update is None or 
                   (self.algorithm.Time - self.last_allocation_update).days >= 30)
        return False
    
    def update_strategy_performance(self, strategy_name, strategy_return):
        """
        Update performance tracking for a strategy
        
        Args:
            strategy_name (str): Name of the strategy
            strategy_return (float): Daily return of the strategy
        """
        if strategy_name in self.strategy_returns:
            self.strategy_returns[strategy_name].append(strategy_return)
    
    def calculate_sharpe_ratios(self):
        """
        Calculate Sharpe ratios for all strategies
        
        Returns:
            dict: {strategy_name: sharpe_ratio}
        """
        sharpe_ratios = {}
        risk_free_rate = 0.02  # 2% annual risk-free rate
        
        for strategy_name, returns in self.strategy_returns.items():
            if len(returns) < self.config['min_track_record_days']:
                sharpe_ratios[strategy_name] = 0.0
                continue
            
            returns_array = np.array(list(returns))
            
            if len(returns_array) == 0:
                sharpe_ratios[strategy_name] = 0.0
                continue
            
            # Calculate annualized metrics
            mean_return = np.mean(returns_array) * 252
            volatility = np.std(returns_array, ddof=1) * np.sqrt(252)
            
            if volatility > 0:
                sharpe_ratio = (mean_return - risk_free_rate) / volatility
            else:
                sharpe_ratio = 0.0
            
            sharpe_ratios[strategy_name] = sharpe_ratio
        
        return sharpe_ratios
    
    def calculate_optimal_allocations(self):
        """
        Calculate optimal allocations using simple Sharpe ratio proportional method
        ENHANCED: Allocation persistence for unavailable strategies
        
        Returns:
            dict: New allocation weights for each strategy
        """
        # STEP 1: Identify available vs unavailable strategies
        available_strategies = {}
        unavailable_strategies = {}
        
        for strategy_name, strategy in self.strategies.items():
            if hasattr(strategy, 'IsAvailable') and strategy.IsAvailable:
                available_strategies[strategy_name] = strategy
            else:
                unavailable_strategies[strategy_name] = strategy
        
        # Track availability for alerting
        self._track_strategy_availability(available_strategies.keys(), unavailable_strategies.keys())
        
        # STEP 2: Allocation persistence logic based on configuration
        if not available_strategies:
            if self.availability_mode == 'persistence':
                self.algorithm.Log("ALLOCATOR: No strategies available - maintaining current allocations")
                return self.current_allocations.copy()  # Keep current allocations unchanged
            else:
                self.algorithm.Log("ALLOCATOR: No strategies available - returning zero allocations")
                return {strategy: 0.0 for strategy in self.strategies.keys()}
        
        # STEP 3: Calculate new allocations only for available strategies
        sharpe_ratios = self.calculate_sharpe_ratios()
        available_sharpes = {k: v for k, v in sharpe_ratios.items() if k in available_strategies}
        
        # STEP 4: Determine allocation approach based on configuration and unavailable strategies
        if unavailable_strategies and self.availability_mode == 'persistence':
            # PERSISTENCE MODE: Some strategies unavailable
            # Keep allocations for unavailable strategies, reallocate only available portion
            
            # Calculate current allocation to unavailable strategies
            unavailable_allocation = sum(self.current_allocations.get(name, 0.0) 
                                       for name in unavailable_strategies.keys())
            available_allocation = 1.0 - unavailable_allocation
            
            # Ensure we have allocation space for available strategies using config threshold
            if available_allocation <= self.persistence_threshold:
                # Emergency reallocation using configured ratio
                unavailable_allocation = self.emergency_ratio
                available_allocation = 1.0 - self.emergency_ratio
                
                self.algorithm.Log(f"ALLOCATOR: Emergency reallocation - {len(unavailable_strategies)} unavailable strategies scaled to {unavailable_allocation:.1%}")
            
            # Allocate among available strategies using their portion
            positive_sharpes = {k: max(0, v) for k, v in available_sharpes.items()}
            total_positive_sharpe = sum(positive_sharpes.values())
            
            new_allocations = {}
            
            # Maintain allocations for unavailable strategies (scaled if needed)
            if unavailable_allocation > 0:
                current_unavailable_total = sum(self.current_allocations.get(name, 0.0) 
                                              for name in unavailable_strategies.keys())
                if current_unavailable_total > 0:
                    unavailable_scale = unavailable_allocation / current_unavailable_total
                    for name in unavailable_strategies.keys():
                        new_allocations[name] = self.current_allocations.get(name, 0.0) * unavailable_scale
                else:
                    # Equal weight among unavailable if no current allocation
                    equal_weight = unavailable_allocation / len(unavailable_strategies)
                    for name in unavailable_strategies.keys():
                        new_allocations[name] = equal_weight
            else:
                for name in unavailable_strategies.keys():
                    new_allocations[name] = 0.0
            
            # Allocate remaining portion among available strategies
            if total_positive_sharpe > 0:
                for strategy, sharpe in positive_sharpes.items():
                    allocation = (sharpe / total_positive_sharpe) * available_allocation
                    new_allocations[strategy] = allocation
            else:
                # Equal weight among available strategies
                equal_weight = available_allocation / len(available_strategies)
                for strategy in available_strategies.keys():
                    new_allocations[strategy] = equal_weight
            
            # Log persistence decision
            self.algorithm.Log(f"ALLOCATOR: Persistence mode - {len(available_strategies)} available ({available_allocation:.1%}), {len(unavailable_strategies)} unavailable ({unavailable_allocation:.1%})")
            
        else:
            # NORMAL MODE: All strategies available OR reallocate mode
            # Zero out unavailable strategies and reallocate everything to available ones
            positive_sharpes = {k: max(0, v) for k, v in available_sharpes.items()}
            total_positive_sharpe = sum(positive_sharpes.values())
            
            new_allocations = {}
            
            # Zero allocation for unavailable strategies
            for name in unavailable_strategies.keys():
                new_allocations[name] = 0.0
            
            if total_positive_sharpe > 0:
                # Allocate proportionally to positive Sharpe ratios
                for strategy, sharpe in positive_sharpes.items():
                    new_allocations[strategy] = sharpe / total_positive_sharpe
            else:
                # Equal weight if no positive Sharpes
                equal_weight = 1.0 / len(available_strategies)
                for strategy in available_strategies.keys():
                    new_allocations[strategy] = equal_weight
            
            if unavailable_strategies:
                self.algorithm.Log(f"ALLOCATOR: Reallocate mode - {len(available_strategies)} available (100%), {len(unavailable_strategies)} unavailable (0%)")
        
        return new_allocations
    
    def update_allocations(self):
        """
        Update strategy allocations based on recent performance
        
        Returns:
            dict: New allocations if updated, None if no change
        """
        if not self.should_rebalance_allocations():
            return None
        
        # Calculate new optimal allocations
        new_allocations = self.calculate_optimal_allocations()
        
        # Apply smoothing to prevent whipsawing
        smoothed_allocations = {}
        smoothing = self.config['allocation_smoothing']
        
        for strategy_name in self.strategies.keys():
            current_alloc = self.current_allocations.get(strategy_name, 0.0)
            new_alloc = new_allocations.get(strategy_name, 0.0)
            
            # Smooth the allocation change
            smoothed_alloc = smoothing * current_alloc + (1 - smoothing) * new_alloc
            smoothed_allocations[strategy_name] = smoothed_alloc
        
        # Check if allocations changed meaningfully (>5% change)
        meaningful_change = False
        for strategy_name in self.strategies.keys():
            current_alloc = self.current_allocations.get(strategy_name, 0.0)
            new_alloc = smoothed_allocations.get(strategy_name, 0.0)
            
            if abs(new_alloc - current_alloc) > 0.05:
                meaningful_change = True
                break
        
        if meaningful_change:
            old_allocations = self.current_allocations.copy()
            self.current_allocations = smoothed_allocations
            self.last_allocation_update = self.algorithm.Time
            
            # Log changes
            self._log_allocation_update(old_allocations, smoothed_allocations)
            
            return smoothed_allocations
        
        return None
    
    def get_current_allocations(self):
        """Get current strategy allocations"""
        return self.current_allocations.copy()
    
    def combine_strategy_targets(self, strategy_targets):
        """
        Combine strategy targets using current allocations
        ENHANCED: Only combine targets from strategies that provided signals
        
        Args:
            strategy_targets (dict): {strategy_name: {symbol: weight}}
            
        Returns:
            dict: Combined target weights {symbol: total_weight}
        """
        combined_targets = {}
        
        # STEP 1: Identify which strategies provided targets
        active_strategies = set(strategy_targets.keys())
        all_strategies = set(self.current_allocations.keys())
        inactive_strategies = all_strategies - active_strategies
        
        # STEP 2: Calculate effective allocations (normalize among active strategies only)
        active_allocations = {}
        total_active_allocation = 0.0
        
        for strategy_name in active_strategies:
            allocation = self.current_allocations.get(strategy_name, 0.0)
            active_allocations[strategy_name] = allocation
            total_active_allocation += allocation
        
        # STEP 3: Normalize allocations among active strategies
        if total_active_allocation > 0:
            for strategy_name in active_allocations:
                active_allocations[strategy_name] /= total_active_allocation
        else:
            # Equal weight fallback
            equal_weight = 1.0 / len(active_strategies) if active_strategies else 0
            active_allocations = {name: equal_weight for name in active_strategies}
        
        # STEP 4: Combine targets using normalized allocations
        for strategy_name, targets in strategy_targets.items():
            allocation = active_allocations.get(strategy_name, 0.0)
            
            for symbol, weight in targets.items():
                if symbol not in combined_targets:
                    combined_targets[symbol] = 0.0
                
                # Apply normalized allocation to strategy weight
                combined_targets[symbol] += weight * allocation
        
        # STEP 5: Log combination summary
        if inactive_strategies:
            self.algorithm.Log(f"ALLOCATOR: Combined {len(active_strategies)} active strategies (inactive: {', '.join(inactive_strategies)})")
        
        return combined_targets
    
    def _log_allocation_update(self, old_allocations, new_allocations):
        """Log allocation changes"""
        changes = []
        sharpe_ratios = self.calculate_sharpe_ratios()
        
        for strategy_name in self.strategies.keys():
            old_alloc = old_allocations.get(strategy_name, 0.0)
            new_alloc = new_allocations.get(strategy_name, 0.0)
            change = new_alloc - old_alloc
            sharpe = sharpe_ratios.get(strategy_name, 0.0)
            
            changes.append(f"{strategy_name}: {old_alloc:.1%}→{new_alloc:.1%} "
                          f"(Δ{change:+.1%}, Sharpe: {sharpe:.2f})")
        
        self.algorithm.Log(f"Layer2: Updated allocations - {', '.join(changes)}")

    def _track_strategy_availability(self, available_strategies, unavailable_strategies):
        """Track consecutive unavailable days and generate alerts."""
        current_date = self.algorithm.Time.date()
        
        # Update counters
        for strategy_name in available_strategies:
            if strategy_name in self.unavailable_days_counter:
                # Strategy became available again
                days_unavailable = self.unavailable_days_counter[strategy_name]
                if days_unavailable > 0:
                    self.algorithm.Log(f"ALLOCATOR: {strategy_name} available again after {days_unavailable} days")
                del self.unavailable_days_counter[strategy_name]
        
        for strategy_name in unavailable_strategies:
            if strategy_name not in self.unavailable_days_counter:
                self.unavailable_days_counter[strategy_name] = 1
            else:
                self.unavailable_days_counter[strategy_name] += 1
            
            # Alert for extended unavailability
            days_unavailable = self.unavailable_days_counter[strategy_name]
            if days_unavailable == self.max_unavailable_days:
                self.algorithm.Log(f"ALERT: {strategy_name} unavailable for {days_unavailable} consecutive days")
        
        self.last_availability_check = current_date

