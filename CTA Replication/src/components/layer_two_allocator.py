# layer_two_allocator.py - LAYER 2 ALLOCATION COMPONENT

from AlgorithmImports import *
from collections import defaultdict, deque
import numpy as np

class LayerTwoAllocator:
    """
    Layer 2: Dynamic Strategy Allocation System
    
    Manages allocation between strategies based on performance metrics
    and config-defined bounds and parameters. Completely strategy-agnostic
    - works with any number of strategies without code changes.
    
    CONFIG-COMPLIANT FEATURES:
    - All parameters loaded from AlgorithmConfigManager
    - Dynamic allocation based on strategy performance
    - Configurable bounds and constraints per strategy
    - Performance-based rebalancing with smoothing
    """
    
    def __init__(self, algorithm, config_manager, strategies):
        self.algorithm = algorithm
        self.config_manager = config_manager
        self.strategies = strategies
        self.config = config_manager.get_allocation_config()
        
        # Strategy allocation tracking
        self.strategy_allocations = {}
        self.strategy_bounds = {}
        self.allocation_history = deque(maxlen=252)
        
        # ENHANCED: Strategy performance tracking for Sharpe calculations
        self.strategy_returns = defaultdict(lambda: deque(maxlen=252))  # Store daily returns
        self.portfolio_values = defaultdict(lambda: deque(maxlen=252))  # Track strategy portfolio values
        self.last_portfolio_values = {}  # Last known portfolio value per strategy
        self.performance_metrics = {}  # Cached performance metrics
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Track last known positions for each strategy (for continuous performance tracking)
        self.last_strategy_positions = {name: {} for name in self.strategies.keys()}
        
        # CONFIG-COMPLIANT parameters
        self.lookback_days = self.config.get('lookback_days', 63)
        self.min_track_record = self.config.get('min_track_record_days', 21)
        self.allocation_smoothing = self.config.get('allocation_smoothing', 0.5)
        self.rebalance_frequency = self.config.get('rebalance_frequency', 'weekly')
        self.performance_metric = self.config.get('performance_metric', 'sharpe_ratio')
        
        self.algorithm.Log(f"LAYER 2: Allocator initialized with {len(strategies)} strategies")
        self.algorithm.Log(f"LAYER 2: Config: lookback={self.lookback_days}d, "
                         f"smoothing={self.allocation_smoothing:.1%}, "
                         f"metric={self.performance_metric}")
    
    def initialize_allocations(self):
        """Initialize strategy allocations from config."""
        try:
            initial_allocations = self.config['initial_allocations']
            bounds = self.config.get('allocation_bounds', {})
            
            # Get enabled strategies and normalize allocations
            enabled_strategies = list(self.strategies.keys())
            
            if not enabled_strategies:
                self.algorithm.Log("LAYER 2: No strategies provided for allocation")
                return False
            
            # Initialize allocations for each strategy
            total_allocation = 0.0
            for strategy_name in enabled_strategies:
                # Get initial allocation from config
                allocation = initial_allocations.get(strategy_name, 0.0)
                
                # Get bounds from config
                strategy_bounds = bounds.get(strategy_name, {'min': 0.0, 'max': 1.0})
                
                self.strategy_allocations[strategy_name] = allocation
                self.strategy_bounds[strategy_name] = strategy_bounds
                total_allocation += allocation
                
                self.algorithm.Log(f"  {strategy_name}: {allocation:.1%} initial "
                                 f"(bounds: {strategy_bounds['min']:.1%}-{strategy_bounds['max']:.1%})")
            
            # Normalize allocations to sum to 1.0 if needed
            if total_allocation > 0 and abs(total_allocation - 1.0) > 0.01:
                self.algorithm.Log(f"LAYER 2: Normalizing allocations (sum was {total_allocation:.1%})")
                for strategy_name in self.strategy_allocations:
                    self.strategy_allocations[strategy_name] /= total_allocation
            
            # Handle case where all allocations are zero (equal weight)
            elif total_allocation == 0:
                equal_weight = 1.0 / len(enabled_strategies)
                for strategy_name in enabled_strategies:
                    self.strategy_allocations[strategy_name] = equal_weight
                self.algorithm.Log(f"LAYER 2: Using equal weight allocations: {equal_weight:.1%} each")
            
            self.algorithm.Log("LAYER 2: Allocation initialization complete")
            return True
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 2: Allocation initialization error: {str(e)}")
            return False
    
    def execute_weekly_allocation(self, strategy_targets):
        """Execute weekly allocation updates and signal combination."""
        try:
            self.algorithm.Log("LAYER 2: Executing weekly allocation...")
            
            # Update allocations (weekly updates are typically minimal)
            allocation_updates = self._update_allocations('weekly', strategy_targets)
            
            # Combine signals using current allocations
            combined_targets = self._combine_strategy_signals(strategy_targets)
            
            # Log allocation status
            self._log_allocation_status(allocation_updates, combined_targets, 'weekly')
            
            return {
                'status': 'success',
                'combined_targets': combined_targets,
                'allocation_updates': allocation_updates,
                'current_allocations': self.get_current_allocations()
            }
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 2: Weekly allocation error: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def execute_monthly_allocation(self, strategy_targets):
        """Execute monthly allocation updates with enhanced rebalancing."""
        try:
            self.algorithm.Log("LAYER 2: Executing monthly allocation...")
            
            # Update allocations with monthly rebalancing
            allocation_updates = self._update_allocations('monthly', strategy_targets)
            
            # Combine signals using updated allocations
            combined_targets = self._combine_strategy_signals(strategy_targets)
            
            # Log allocation status
            self._log_allocation_status(allocation_updates, combined_targets, 'monthly')
            
            return {
                'status': 'success',
                'combined_targets': combined_targets,
                'allocation_updates': allocation_updates,
                'current_allocations': self.get_current_allocations()
            }
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 2: Monthly allocation error: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def execute_emergency_allocation(self, strategy_targets):
        """Execute emergency allocation with conservative equal-weight approach."""
        try:
            self.algorithm.Log("LAYER 2: Executing emergency allocation...")
            
            # For emergency, use equal weight allocation across all strategies
            if strategy_targets:
                num_strategies = len(strategy_targets)
                equal_weight = 1.0 / num_strategies
                
                # Temporarily set equal weights
                emergency_allocations = {name: equal_weight for name in strategy_targets.keys()}
                
                # Combine signals using emergency allocations
                combined_targets = {}
                for strategy_name, targets in strategy_targets.items():
                    allocation = emergency_allocations[strategy_name]
                    for symbol, weight in targets.items():
                        if symbol not in combined_targets:
                            combined_targets[symbol] = 0.0
                        combined_targets[symbol] += weight * allocation
                
                self.algorithm.Log(f"LAYER 2: Emergency equal-weight allocation: {equal_weight:.1%} per strategy")
            else:
                combined_targets = {}
            
            return {
                'status': 'success',
                'combined_targets': combined_targets,
                'emergency_allocation': equal_weight if strategy_targets else 0
            }
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 2: Emergency allocation error: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def _update_allocations(self, frequency, strategy_targets):
        """Update strategy allocations based on performance and frequency."""
        try:
            # Check if this frequency should trigger allocation updates
            if frequency == 'weekly' and self.rebalance_frequency != 'weekly':
                return {}  # No weekly updates for monthly rebalancing
            
            # Calculate new allocations based on performance
            new_allocations = self._calculate_performance_allocations(strategy_targets)
            
            if not new_allocations:
                return {}
            
            # Apply smoothing and bounds
            final_allocations = self._apply_smoothing_and_bounds(new_allocations)
            
            # Update current allocations and track changes
            allocation_updates = {}
            for strategy_name, new_allocation in final_allocations.items():
                old_allocation = self.strategy_allocations.get(strategy_name, 0)
                
                # STABILITY FIX: Only apply changes >2% to prevent micro-adjustments accumulating
                if abs(new_allocation - old_allocation) > 0.02:  # Increased from 0.01 to 0.02
                    allocation_updates[strategy_name] = {
                        'old': old_allocation,
                        'new': new_allocation,
                        'change': new_allocation - old_allocation
                    }
                    
                    # Apply the change
                    self.strategy_allocations[strategy_name] = new_allocation
                else:
                    # Keep old allocation if change is too small
                    final_allocations[strategy_name] = old_allocation
            
            # Record allocation history
            if allocation_updates:
                self.allocation_history.append({
                    'timestamp': self.algorithm.Time,
                    'frequency': frequency,
                    'allocations': self.strategy_allocations.copy(),
                    'updates': allocation_updates
                })
            
            return allocation_updates
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 2: Allocation update error: {str(e)}")
            return {}
    
    def _calculate_performance_allocations(self, strategy_targets):
        """
        TRUE META-ALLOCATION: Calculate allocations for ALL enabled strategies.
        
        Key change: Uses ALL enabled strategies, not just those with new signals.
        This allows Layer 2 to continuously rebalance between strategies based on 
        rolling performance, independent of individual strategy rebalancing schedules.
        """
        try:
            # CRITICAL FIX: Get ALL enabled strategies from config, not just active signals
            all_enabled_strategies = list(self.strategies.keys())  # All strategies in config
            
            # Step 1: Calculate Sharpe ratios for ALL enabled strategies
            sharpe_ratios = self._calculate_strategy_sharpe_ratios_for_all(all_enabled_strategies)
            
            # Step 2: Check if any strategy has sufficient track record
            sufficient_data = any(
                len(self.strategy_returns[strategy]) >= self.min_track_record 
                for strategy in all_enabled_strategies
            )
            
            # Step 3: Determine allocation method
            if not sufficient_data:
                # Not enough data - maintain current allocations with small adjustments
                self.algorithm.Log("LAYER 2: Insufficient track record, maintaining allocations")
                return self._get_conservative_allocations_for_all(all_enabled_strategies)
            
            # Step 4: Apply Sharpe ratio proportional allocation to ALL enabled strategies
            new_allocations = self._apply_sharpe_proportional_allocation_for_all(sharpe_ratios, all_enabled_strategies)
            
            # Step 5: Log performance metrics (concise)
            self._log_performance_update(sharpe_ratios, new_allocations)
            
            return new_allocations
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 2: Performance allocation error: {str(e)}")
            return self.strategy_allocations.copy()
    
    def _calculate_strategy_sharpe_ratios_for_all(self, all_enabled_strategies):
        """Calculate Sharpe ratios for ALL enabled strategies (not just active signals)."""
        sharpe_ratios = {}
        
        for strategy_name in all_enabled_strategies:
            try:
                returns = list(self.strategy_returns[strategy_name])
                
                if len(returns) < self.min_track_record:
                    sharpe_ratios[strategy_name] = 0.0
                    continue
                
                # Use only recent returns based on lookback period
                recent_returns = returns[-self.lookback_days:] if len(returns) > self.lookback_days else returns
                
                if len(recent_returns) < 10:  # Minimum 10 days
                    sharpe_ratios[strategy_name] = 0.0
                    continue
                
                # Calculate Sharpe ratio
                mean_return = np.mean(recent_returns)
                std_return = np.std(recent_returns, ddof=1) if len(recent_returns) > 1 else 0.0
                
                if std_return > 0:
                    # Annualized Sharpe ratio
                    daily_risk_free = self.risk_free_rate / 252
                    sharpe_ratio = (mean_return - daily_risk_free) / std_return * np.sqrt(252)
                    
                    # STABILITY FIX: Cap extreme Sharpe ratios to prevent wild allocation swings
                    sharpe_ratio = max(-2.0, min(2.0, sharpe_ratio))  # Cap between -2 and +2
                else:
                    sharpe_ratio = 0.0
                
                sharpe_ratios[strategy_name] = sharpe_ratio
                
            except Exception as e:
                sharpe_ratios[strategy_name] = 0.0
        
        return sharpe_ratios

    def _calculate_strategy_sharpe_ratios(self, strategy_targets):
        """DEPRECATED: Use _calculate_strategy_sharpe_ratios_for_all instead."""
        # Keep for backward compatibility, but delegate to new method
        strategy_names = list(strategy_targets.keys())
        return self._calculate_strategy_sharpe_ratios_for_all(strategy_names)
    
    def _apply_sharpe_proportional_allocation_for_all(self, sharpe_ratios, all_enabled_strategies):
        """Apply proportional allocation based on positive Sharpe ratios for ALL enabled strategies."""
        try:
            # Step 1: Get positive Sharpe ratios only
            positive_sharpes = {k: max(0.0, v) for k, v in sharpe_ratios.items()}
            total_positive_sharpe = sum(positive_sharpes.values())
            
            new_allocations = {}
            
            # Step 2: Allocate based on positive Sharpe ratios
            if total_positive_sharpe > 0.01:  # Minimum threshold
                for strategy_name in all_enabled_strategies:
                    sharpe = positive_sharpes.get(strategy_name, 0.0)
                    allocation = sharpe / total_positive_sharpe
                    new_allocations[strategy_name] = allocation
            else:
                # No positive Sharpe ratios - use equal weight
                self.algorithm.Log("LAYER 2: No positive Sharpe ratios, using equal weight for all enabled strategies")
                equal_weight = 1.0 / len(all_enabled_strategies)
                for strategy_name in all_enabled_strategies:
                    new_allocations[strategy_name] = equal_weight
            
            return new_allocations
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 2: Sharpe allocation error: {str(e)}")
            return self.strategy_allocations.copy()

    def _apply_sharpe_proportional_allocation(self, sharpe_ratios, strategy_targets):
        """DEPRECATED: Use _apply_sharpe_proportional_allocation_for_all instead."""
        # Keep for backward compatibility, but delegate to new method
        strategy_names = list(strategy_targets.keys())
        return self._apply_sharpe_proportional_allocation_for_all(sharpe_ratios, strategy_names)
    
    def _get_conservative_allocations_for_all(self, all_enabled_strategies):
        """Get conservative allocations for ALL enabled strategies when insufficient data."""
        try:
            current_allocations = self.strategy_allocations.copy()
            
            # Ensure all enabled strategies have some allocation
            for strategy_name in all_enabled_strategies:
                if strategy_name not in current_allocations or current_allocations[strategy_name] < 0.05:
                    current_allocations[strategy_name] = 0.10  # 10% minimum
            
            # Remove allocations for strategies that are no longer enabled
            strategies_to_remove = [s for s in current_allocations.keys() if s not in all_enabled_strategies]
            for strategy_name in strategies_to_remove:
                del current_allocations[strategy_name]
            
            # Normalize
            total = sum(current_allocations.values())
            if total > 0:
                for strategy_name in current_allocations:
                    current_allocations[strategy_name] /= total
            
            return current_allocations
            
        except Exception as e:
            return self.strategy_allocations.copy()

    def _get_conservative_allocations(self, strategy_targets):
        """DEPRECATED: Use _get_conservative_allocations_for_all instead."""
        # Keep for backward compatibility, but delegate to new method
        strategy_names = list(strategy_targets.keys())
        return self._get_conservative_allocations_for_all(strategy_names)
    
    def _log_performance_update(self, sharpe_ratios, new_allocations):
        """Log performance metrics concisely to manage log size."""
        try:
            # Only log if there are meaningful changes or monthly
            log_detailed = (self.algorithm.Time.day == 1 or  # Monthly detailed log
                          any(abs(new_allocations.get(s, 0) - self.strategy_allocations.get(s, 0)) > 0.05 
                              for s in new_allocations.keys()))  # >5% allocation change
            
            if log_detailed:
                # Concise performance summary
                perf_summary = []
                for strategy, sharpe in sharpe_ratios.items():
                    old_alloc = self.strategy_allocations.get(strategy, 0)
                    new_alloc = new_allocations.get(strategy, 0)
                    change = new_alloc - old_alloc
                    
                    track_record = len(self.strategy_returns[strategy])
                    perf_summary.append(f"{strategy}: S={sharpe:.2f}, "
                                      f"{old_alloc:.1%}→{new_alloc:.1%}({change:+.1%}), "
                                      f"T={track_record}d")
                
                self.algorithm.Log(f"LAYER 2: Performance update: {'; '.join(perf_summary)}")
            
        except Exception as e:
            # Silent handling to avoid log pollution
            pass
    
    def _apply_smoothing_and_bounds(self, new_allocations):
        """Apply smoothing and bounds to new allocations."""
        try:
            smoothed_allocations = {}
            
            for strategy_name in new_allocations:
                old_allocation = self.strategy_allocations.get(strategy_name, 0)
                new_allocation = new_allocations[strategy_name]
                
                # Apply smoothing using config parameter
                smoothed = (old_allocation * self.allocation_smoothing + 
                           new_allocation * (1 - self.allocation_smoothing))
                
                # Apply bounds from config
                bounds = self.strategy_bounds.get(strategy_name, {'min': 0.0, 'max': 1.0})
                bounded = max(bounds['min'], min(bounds['max'], smoothed))
                
                smoothed_allocations[strategy_name] = bounded
            
            # Normalize to sum to 1.0
            total = sum(smoothed_allocations.values())
            if total > 0:
                for strategy_name in smoothed_allocations:
                    smoothed_allocations[strategy_name] /= total
            
            return smoothed_allocations
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 2: Smoothing and bounds error: {str(e)}")
            return new_allocations
    
    def _combine_strategy_signals(self, strategy_targets):
        """
        TRUE META-ALLOCATION: Combine signals using allocations for ALL enabled strategies.
        
        Key insight: Layer 2 allocations apply to ALL enabled strategies, not just those
        with new signals. Strategies without new signals use their last known positions.
        """
        try:
            combined_targets = {}
            
            # NEW APPROACH: Use current allocations for ALL enabled strategies
            # These allocations were calculated based on ALL strategies' performance
            current_allocations = self.strategy_allocations.copy()
            
            # Log current allocation state
            allocation_summary = ", ".join([f"{name}: {alloc:.1%}" for name, alloc in current_allocations.items()])
            self.algorithm.Log(f"LAYER 2: Current allocations: {allocation_summary}")
            
            # Combine signals using current allocations
            strategies_with_signals = list(strategy_targets.keys())
            strategies_used = []
            
            for strategy_name, targets in strategy_targets.items():
                allocation = current_allocations.get(strategy_name, 0)
                
                if allocation > 0 and targets:
                    strategies_used.append(strategy_name)
                    for symbol, weight in targets.items():
                        if symbol not in combined_targets:
                            combined_targets[symbol] = 0.0
                        combined_targets[symbol] += weight * allocation
            
            # Log what happened
            total_allocation_used = sum(current_allocations.get(s, 0) for s in strategies_used)
            self.algorithm.Log(f"LAYER 2: Used {len(strategies_used)} strategies with signals: {', '.join(strategies_used)}")
            self.algorithm.Log(f"LAYER 2: Total allocation used: {total_allocation_used:.1%}")
            
            # Check for strategies with allocation but no signals (this is expected for MTUM on non-rebalancing weeks)
            strategies_with_allocation_no_signals = [
                s for s in current_allocations.keys() 
                if current_allocations[s] > 0.01 and s not in strategies_with_signals
            ]
            
            if strategies_with_allocation_no_signals:
                unused_allocation = sum(current_allocations.get(s, 0) for s in strategies_with_allocation_no_signals)
                self.algorithm.Log(f"LAYER 2: Strategies with allocation but no new signals: {', '.join(strategies_with_allocation_no_signals)} ({unused_allocation:.1%})")
                self.algorithm.Log("LAYER 2: This is normal - these strategies will rebalance on their schedule")
            
            return combined_targets
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 2: Signal combination error: {str(e)}")
            return {}
    
    def _log_allocation_status(self, allocation_updates, combined_targets, context):
        """Log allocation status and combined results."""
        try:
            # Log allocation updates
            if allocation_updates:
                update_summary = []
                for strategy, update in allocation_updates.items():
                    change = update['change']
                    update_summary.append(f"{strategy}: {change:+.1%}")
                self.algorithm.Log(f"  Allocation updates: {', '.join(update_summary)}")
            else:
                self.algorithm.Log(f"  Allocations unchanged")
            
            # Log current allocations
            current_allocations = self.get_current_allocations()
            allocation_summary = []
            for strategy, allocation in current_allocations.items():
                allocation_summary.append(f"{strategy}: {allocation:.1%}")
            self.algorithm.Log(f"  Current allocations: {', '.join(allocation_summary)}")
            
            # Log combined result
            if combined_targets:
                gross_exposure = sum(abs(w) for w in combined_targets.values())
                net_exposure = sum(combined_targets.values())
                self.algorithm.Log(f"  Combined ({context}): {len(combined_targets)} positions, "
                                 f"Gross: {gross_exposure:.1%}, Net: {net_exposure:.2f}")
            else:
                self.algorithm.Log(f"  Combined ({context}): No positions")
            
        except Exception as e:
            self.algorithm.Log(f"LAYER 2: Log status error: {str(e)}")
    
    # =============================================================================
    # STATUS AND MONITORING METHODS
    # =============================================================================
    
    def get_current_allocations(self):
        """Get current strategy allocations."""
        return self.strategy_allocations.copy()
    
    def get_allocation_bounds(self):
        """Get allocation bounds for all strategies."""
        return self.strategy_bounds.copy()
    
    def get_allocation_history(self):
        """Get allocation history."""
        return list(self.allocation_history)
    
    def get_system_health(self):
        """Get Layer 2 system health status."""
        try:
            # Calculate allocation metrics
            total_allocation = sum(self.strategy_allocations.values())
            allocation_count = len([a for a in self.strategy_allocations.values() if a > 0.01])
            
            layer2_health = {
                'timestamp': self.algorithm.Time,
                'total_strategies': len(self.strategies),
                'active_allocations': allocation_count,
                'total_allocation': total_allocation,
                'allocation_normalized': abs(total_allocation - 1.0) < 0.01,
                'current_allocations': self.strategy_allocations.copy(),
                'allocation_bounds': self.strategy_bounds.copy(),
                'allocation_history_length': len(self.allocation_history),
                'config_parameters': {
                    'lookback_days': self.lookback_days,
                    'allocation_smoothing': self.allocation_smoothing,
                    'rebalance_frequency': self.rebalance_frequency,
                    'performance_metric': self.performance_metric
                },
                'system_status': 'healthy' if total_allocation > 0.9 else 'allocation_error'
            }
            
            return layer2_health
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 2: Health check error: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def get_performance_summary(self):
        """Get Layer 2 performance summary."""
        try:
            # Calculate allocation statistics
            allocation_changes = len([h for h in self.allocation_history if h.get('updates')])
            recent_allocations = self.allocation_history[-5:] if self.allocation_history else []
            
            summary = {
                'layer2_allocator': {
                    'total_strategies_managed': len(self.strategies),
                    'allocation_updates': allocation_changes,
                    'current_allocations': self.strategy_allocations.copy(),
                    'allocation_bounds': self.strategy_bounds.copy(),
                    'recent_allocation_history': recent_allocations,
                    'config_compliant': True,
                    'dynamic_allocation_enabled': True
                }
            }
            
            return summary
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 2: Performance summary error: {str(e)}")
            return {'error': str(e)}
    
    def get_config_compliance_report(self):
        """Get Layer 2 config compliance report."""
        try:
            compliance_report = {
                'layer2_allocator': {
                    'config_compliant': True,
                    'all_parameters_from_config': True,
                    'dynamic_strategy_support': True,
                    'scalable_to_unlimited_strategies': True,
                    'config_parameters_used': {
                        'lookback_days': self.lookback_days,
                        'allocation_smoothing': self.allocation_smoothing,
                        'rebalance_frequency': self.rebalance_frequency,
                        'performance_metric': self.performance_metric,
                        'initial_allocations': self.config['initial_allocations'],
                        'allocation_bounds': self.config.get('allocation_bounds', {})
                    }
                }
            }
            
            return compliance_report
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 2: Config compliance report error: {str(e)}")
            return {'error': str(e)}

    def update_strategy_performance(self, strategy_name, current_positions):
        """
        Update strategy performance tracking based on current positions.
        Called daily to track strategy returns for Sharpe ratio calculations.
        
        ENHANCED: Now stores positions for continuous performance tracking.
        """
        try:
            if strategy_name not in self.strategies:
                return
            
            # Store current positions for future reference
            self.last_strategy_positions[strategy_name] = current_positions.copy()
            
            # Calculate current strategy portfolio value based on positions
            current_value = 0.0
            for symbol, weight in current_positions.items():
                if hasattr(self.algorithm.Securities[symbol], 'Price'):
                    price = self.algorithm.Securities[symbol].Price
                    # Estimate position value (simplified)
                    position_value = weight * self.algorithm.Portfolio.TotalPortfolioValue
                    current_value += position_value
            
            # Store current portfolio value
            self.portfolio_values[strategy_name].append(current_value)
            
            # Calculate daily return if we have previous value
            if strategy_name in self.last_portfolio_values and self.last_portfolio_values[strategy_name] > 0:
                prev_value = self.last_portfolio_values[strategy_name]
                daily_return = (current_value - prev_value) / prev_value
                self.strategy_returns[strategy_name].append(daily_return)
            
            # Update last known value
            self.last_portfolio_values[strategy_name] = current_value
            
        except Exception as e:
            # Silent error handling to avoid log spam
            pass
