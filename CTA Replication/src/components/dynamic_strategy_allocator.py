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
        self.algorithm = algorithm
        
        # Default configuration
        self.config = {
            'lookback_days': 63,              # 3 months for Sharpe calculation
            'min_track_record_days': 21,      # Minimum days before changing allocations
            'rebalance_frequency': 'weekly',  # How often to update allocations
            'allocation_smoothing': 0.7,      # Smoothing factor (0.7 = 70% old, 30% new)
            'use_correlation': True,          # Use correlation in portfolio vol calculation
        }
        
        if config:
            self.config.update(config)
        
        # Strategy tracking
        self.strategies = {}
        self.strategy_returns = {}
        self.current_allocations = {}
        self.last_allocation_update = None
        
        self.algorithm.Log(f"Layer2 Allocator: Initialized with config: {self.config}")
    
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
        ENHANCED: Only allocate to available strategies
        
        Returns:
            dict: New allocation weights for each strategy
        """
        # STEP 1: Filter to only available strategies
        available_strategies = {}
        for strategy_name, strategy in self.strategies.items():
            if hasattr(strategy, 'IsAvailable') and strategy.IsAvailable:
                available_strategies[strategy_name] = strategy
            else:
                self.algorithm.Log(f"ALLOCATOR: Strategy {strategy_name} not available for allocation")
        
        if not available_strategies:
            self.algorithm.Log("ALLOCATOR: No strategies available - returning zero allocations")
            return {strategy: 0.0 for strategy in self.strategies.keys()}
        
        # STEP 2: Calculate Sharpe ratios only for available strategies
        sharpe_ratios = self.calculate_sharpe_ratios()
        available_sharpes = {k: v for k, v in sharpe_ratios.items() if k in available_strategies}
        
        # STEP 3: Allocate only among available strategies
        positive_sharpes = {k: max(0, v) for k, v in available_sharpes.items()}
        total_positive_sharpe = sum(positive_sharpes.values())
        
        # Initialize all allocations to zero
        new_allocations = {strategy: 0.0 for strategy in self.strategies.keys()}
        
        if total_positive_sharpe > 0:
            # Allocate proportionally to positive Sharpe ratios (available strategies only)
            for strategy, sharpe in positive_sharpes.items():
                new_allocations[strategy] = sharpe / total_positive_sharpe
        else:
            # If no positive Sharpes, use equal weight among available strategies
            n_available = len(available_strategies)
            if n_available > 0:
                equal_weight = 1.0 / n_available
                for strategy in available_strategies.keys():
                    new_allocations[strategy] = equal_weight
        
        # Log allocation decisions
        available_count = len(available_strategies)
        total_count = len(self.strategies)
        self.algorithm.Log(f"ALLOCATOR: Allocated to {available_count}/{total_count} available strategies")
        
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
        
        Args:
            strategy_targets (dict): {strategy_name: {symbol: weight}}
            
        Returns:
            dict: Combined target weights {symbol: total_weight}
        """
        combined_targets = {}
        
        for strategy_name, targets in strategy_targets.items():
            allocation = self.current_allocations.get(strategy_name, 0.0)
            
            for symbol, weight in targets.items():
                if symbol not in combined_targets:
                    combined_targets[symbol] = 0.0
                
                # Apply allocation to strategy weight
                combined_targets[symbol] += weight * allocation
        
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

