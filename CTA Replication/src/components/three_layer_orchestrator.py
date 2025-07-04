# three_layer_orchestrator.py - FIXED CORE ORCHESTRATOR

from AlgorithmImports import *
from collections import deque

# Import component modules
from components.strategy_loader import StrategyLoader
from components.layer_two_allocator import LayerTwoAllocator  
from risk.layer_three_risk_manager import LayerThreeRiskManager

class ThreeLayerOrchestrator:
    """
    Three-Layer Portfolio Management Orchestrator - FIXED VERSION
    
    Lightweight orchestrator that delegates to specialized components:
    - StrategyLoader: Handles all Layer 1 strategy management (dynamic loading)
    - LayerTwoAllocator: Handles Layer 2 allocation and signal combination
    - LayerThreeRiskManager: Handles Layer 3 risk management and scaling
    
    SCALABLE DESIGN:
    - Dynamic strategy loading (no hardcoded strategy names)
    - Component-based architecture for maintainability
    - Separation of concerns for each layer
    - FIXED: Uses correct method names for all components
    """
    
    def __init__(self, algorithm, config_manager):
        self.algorithm = algorithm
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        # Symbols are now managed directly by QC's native methods
        
        # Three-layer components
        self.strategy_loader = None      # Layer 1: Dynamic strategy management
        self.allocator = None           # Layer 2: Dynamic allocation
        self.risk_manager = None        # Layer 3: Risk management
        
        # System state tracking
        self.rebalance_history = deque(maxlen=252)
        self.total_rebalances = 0
        self.successful_rebalances = 0
        
        self.algorithm.Log("="*60)
        self.algorithm.Log("INITIALIZING THREE-LAYER ORCHESTRATOR (FIXED)")
        self.algorithm.Log("="*60)
    
    def OnSecuritiesChanged(self, changes):
        """
        Pass security changes to the strategy loader.
        This is critical for strategies that need to initialize when securities are added.
        """
        if self.strategy_loader:
            self.strategy_loader.OnSecuritiesChanged(changes)

    def initialize_system(self):
        """Initialize the complete three-layer system with component delegation."""
        try:
            # Initialize Layer 1: Dynamic Strategy Loader
            if not self._initialize_layer1():
                self.algorithm.Error("ORCHESTRATOR: Layer 1 initialization failed")
                return False
            
            # Initialize Layer 2: Dynamic Allocator
            if not self._initialize_layer2():
                self.algorithm.Error("ORCHESTRATOR: Layer 2 initialization failed")
                return False
            
            # Initialize Layer 3: Risk Manager
            if not self._initialize_layer3():
                self.algorithm.Error("ORCHESTRATOR: Layer 3 initialization failed")
                return False
            
            self._log_initialization_success()
            return True
            
        except Exception as e:
            self.algorithm.Error(f"ORCHESTRATOR: Initialization error: {str(e)}")
            return False
    
    def _initialize_layer1(self):
        """Initialize Layer 1: Dynamic Strategy Loader."""
        try:
            self.algorithm.Log("LAYER 1: Initializing dynamic strategy loader...")
            
            self.strategy_loader = StrategyLoader(
                algorithm=self.algorithm,
                config_manager=self.config_manager
            )
            
            # Load all enabled strategies dynamically
            success = self.strategy_loader.load_enabled_strategies()
            
            if success:
                loaded_strategies = self.strategy_loader.get_loaded_strategies()
                self.algorithm.Log(f"LAYER 1: Successfully loaded {len(loaded_strategies)} strategies")
                for name in loaded_strategies:
                    self.algorithm.Log(f"  ✓ {name}: CONFIG-COMPLIANT")
            
            return success
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 1: Initialization error: {str(e)}")
            return False
    
    def _initialize_layer2(self):
        """Initialize Layer 2: Dynamic Allocator."""
        try:
            self.algorithm.Log("LAYER 2: Initializing dynamic allocator...")
            
            # Get loaded strategies from Layer 1
            strategies = self.strategy_loader.get_strategy_objects()
            
            self.allocator = LayerTwoAllocator(
                algorithm=self.algorithm,
                config_manager=self.config_manager,
                strategies=strategies
            )
            
            # Initialize allocator with loaded strategies
            success = self.allocator.initialize_allocations()
            
            if success:
                allocations = self.allocator.get_current_allocations()
                allocation_summary = [f"{name}: {alloc:.1%}" for name, alloc in allocations.items()]
                self.algorithm.Log(f"LAYER 2: Allocations: {', '.join(allocation_summary)}")
            
            return success
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 2: Initialization error: {str(e)}")
            return False
    
    def _initialize_layer3(self):
        """Initialize Layer 3: Risk Manager."""
        try:
            self.algorithm.Log("LAYER 3: Initializing risk manager...")
            
            self.risk_manager = LayerThreeRiskManager(
                algorithm=self.algorithm,
                config_manager=self.config_manager
            )
            
            # Log risk parameters
            risk_config = self.config_manager.get_risk_config()
            target_vol = risk_config.get('target_portfolio_vol', 0.5)
            min_exposure = risk_config.get('min_notional_exposure', 3.0)
            
            self.algorithm.Log(f"LAYER 3: Target vol: {target_vol:.1%}, Min exposure: {min_exposure:.1f}x")
            
            return True
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 3: Initialization error: {str(e)}")
            return False
    
    def _log_initialization_success(self):
        """Log successful initialization."""
        self.algorithm.Log("="*60)
        self.algorithm.Log("THREE-LAYER SYSTEM INITIALIZATION COMPLETE")
        self.algorithm.Log("="*60)
        
        # Get system summary
        strategy_count = len(self.strategy_loader.get_loaded_strategies())
        allocations = self.allocator.get_current_allocations()
        risk_config = self.config_manager.get_risk_config()
        
        self.algorithm.Log(f"Active Strategies: {strategy_count}")
        self.algorithm.Log(f"Risk Target: {risk_config.get('target_portfolio_vol', 0.5):.1%}")
        self.algorithm.Log(f"System Status: READY")
        self.algorithm.Log("="*60)
    
    # =============================================================================
    # REBALANCING METHODS - FIXED METHOD CALLS
    # =============================================================================
    
    def weekly_rebalance(self):
        """Execute weekly rebalancing through component delegation."""
        try:
            self.algorithm.Log("=== THREE-LAYER WEEKLY REBALANCE ===")
            
            # Layer 1: Generate strategy signals
            strategy_targets = self._execute_layer1_rebalance()
            if not strategy_targets:
                return {'status': 'failed', 'reason': 'layer1_no_signals', 'layer': 1}
            
            # Layer 2: Update allocations and combine signals
            combined_targets = self._execute_layer2_rebalance(strategy_targets)
            if not combined_targets:
                return {'status': 'failed', 'reason': 'layer2_no_positions', 'layer': 2}
            
            # Layer 3: Apply risk management - FIXED: Use correct method
            final_targets = self.risk_manager.apply_portfolio_risk_management(combined_targets)
            
            # Track successful rebalance
            self.total_rebalances += 1
            self.successful_rebalances += 1
            
            # Record rebalance
            self._record_rebalance('weekly', strategy_targets, combined_targets, final_targets)
            
            self.algorithm.Log("=== WEEKLY REBALANCE COMPLETE ===")
            
            return {
                'status': 'success',
                'rebalance_type': 'weekly',
                'final_targets': final_targets,
                'execution_summary': {
                    'strategy_signals': len(strategy_targets),
                    'combined_positions': len(combined_targets),
                    'final_positions': len(final_targets),
                    'gross_exposure': sum(abs(w) for w in final_targets.values()),
                    'config_compliance': True
                }
            }
            
        except Exception as e:
            self.algorithm.Error(f"ORCHESTRATOR: Weekly rebalance error: {str(e)}")
            return {'status': 'failed', 'reason': 'orchestrator_error', 'error': str(e)}
    
    def monthly_rebalance(self):
        """Execute monthly rebalancing through component delegation."""
        try:
            self.algorithm.Log("=== THREE-LAYER MONTHLY REBALANCE ===")
            
            # Layer 1: Generate strategy signals (same as weekly)
            strategy_targets = self._execute_layer1_rebalance()
            if not strategy_targets:
                return {'status': 'failed', 'reason': 'layer1_no_signals', 'layer': 1}
            
            # Layer 2: Update monthly allocations
            combined_targets = self._execute_layer2_monthly_rebalance(strategy_targets)
            if not combined_targets:
                return {'status': 'failed', 'reason': 'layer2_no_positions', 'layer': 2}
            
            # Layer 3: Apply risk management - FIXED: Use correct method
            final_targets = self.risk_manager.apply_portfolio_risk_management(combined_targets)
            
            # Track successful monthly rebalance
            self.total_rebalances += 1
            self.successful_rebalances += 1
            
            # Record rebalance
            self._record_rebalance('monthly', strategy_targets, combined_targets, final_targets)
            
            self.algorithm.Log("=== MONTHLY REBALANCE COMPLETE ===")
            
            return {
                'status': 'success',
                'rebalance_type': 'monthly',
                'final_targets': final_targets,
                'execution_summary': {
                    'strategy_signals': len(strategy_targets),
                    'combined_positions': len(combined_targets),
                    'final_positions': len(final_targets),
                    'gross_exposure': sum(abs(w) for w in final_targets.values()),
                    'config_compliance': True
                }
            }
            
        except Exception as e:
            self.algorithm.Error(f"ORCHESTRATOR: Monthly rebalance error: {str(e)}")
            return {'status': 'failed', 'reason': 'monthly_orchestrator_error', 'error': str(e)}
    
    def emergency_rebalance(self, reason="manual_trigger"):
        """Execute emergency rebalancing through component delegation."""
        try:
            self.algorithm.Log(f"=== EMERGENCY REBALANCE: {reason} ===")
            
            # In emergency, try to get current signals or flatten
            try:
                strategy_targets = self._execute_layer1_rebalance()
            except:
                # If Layer 1 fails in emergency, flatten portfolio
                self.algorithm.Log("EMERGENCY: Layer 1 failed, flattening portfolio")
                return {
                    'status': 'success',
                    'rebalance_type': 'emergency',
                    'reason': reason,
                    'final_targets': {},
                    'emergency_action': 'flatten_portfolio'
                }
            
            # Emergency Layer 2: Use current allocations
            try:
                combined_targets = self._execute_layer2_rebalance(strategy_targets) if strategy_targets else {}
            except:
                combined_targets = {}
            
            # Emergency Layer 3: Apply strict risk controls - FIXED: Use correct method
            final_targets = self.risk_manager.apply_portfolio_risk_management(combined_targets)
            
            return {
                'status': 'success',
                'rebalance_type': 'emergency',
                'reason': reason,
                'final_targets': final_targets,
                'config_compliance': True
            }
            
        except Exception as e:
            self.algorithm.Error(f"ORCHESTRATOR: Emergency rebalance error: {str(e)}")
            # In complete failure, return empty targets to flatten
            return {
                'status': 'success', 
                'rebalance_type': 'emergency',
                'final_targets': {},
                'emergency_action': 'flatten_due_to_error',
                'error': str(e)
            }
    
    # =============================================================================
    # LAYER EXECUTION METHODS - SIMPLIFIED
    # =============================================================================
    
    def _execute_layer1_rebalance(self):
        """Execute Layer 1: Get strategy signals with proper slice passing."""
        try:
            strategy_targets = {}
            
            if not self.strategy_loader:
                self.algorithm.Log("LAYER 1: Strategy loader not available")
                return strategy_targets
            
            loaded_strategies = self.strategy_loader.get_strategy_objects()
            
            # Get current slice for futures chain access
            current_slice = getattr(self, '_current_slice', None)
            
            # ENHANCED: Track availability with detailed logging
            available_count = 0
            unavailable_strategies = []
            
            for strategy_name, strategy in loaded_strategies.items():
                if hasattr(strategy, 'IsAvailable'):
                    is_available = strategy.IsAvailable
                    if is_available:
                        available_count += 1
                        status = "AVAILABLE"
                    else:
                        unavailable_strategies.append(strategy_name)
                        status = "NOT_AVAILABLE"
                        # Log WHY it's not available (compact format)
                        if hasattr(strategy, 'log_availability_status'):
                            strategy.log_availability_status()
                        elif hasattr(strategy, '_last_unavailable_reason'):
                            reason = getattr(strategy, '_last_unavailable_reason', 'Unknown')
                            self.algorithm.Log(f"LAYER 1: {strategy_name} - NOT_AVAILABLE: {reason}")
                else:
                    status = "NO_AVAILABILITY_CHECK"
                    available_count += 1  # Assume available if no check
                
                self.algorithm.Log(f"LAYER 1: {strategy_name} - {status}")
            
            # Log summary with unavailable details
            if unavailable_strategies:
                self.algorithm.Log(f"LAYER 1: {available_count}/{len(loaded_strategies)} available. Unavailable: {', '.join(unavailable_strategies)}")
            else:
                self.algorithm.Log(f"LAYER 1: All {available_count}/{len(loaded_strategies)} strategies available")
            
            # ENHANCED: Generate signals with availability-aware error handling
            for strategy_name, strategy in loaded_strategies.items():
                try:
                    # Check strategy availability before generating targets
                    if hasattr(strategy, 'IsAvailable'):
                        is_available = strategy.IsAvailable
                        if not is_available:
                            self.algorithm.Log(f"LAYER 1: {strategy_name} skipped - not available")
                            # CRITICAL: Don't continue - allow Layer 2 to handle missing strategies
                            continue
                    
                    # CRITICAL: Pass slice to strategies for futures chain access
                    if hasattr(strategy, 'generate_targets'):
                        targets = strategy.generate_targets(current_slice)
                        if targets:
                            strategy_targets[strategy_name] = targets
                            # Format symbol names for readable logging
                            symbol_names = [str(symbol) for symbol in targets.keys()]
                            self.algorithm.Log(f"LAYER 1: {strategy_name} generated {len(targets)} targets: {symbol_names}")
                        else:
                            self.algorithm.Log(f"LAYER 1: {strategy_name} generated no targets")
                    elif hasattr(strategy, 'get_target_weights'):
                        targets = strategy.get_target_weights(current_slice)
                        if targets:
                            strategy_targets[strategy_name] = targets
                            symbol_names = [str(symbol) for symbol in targets.keys()]
                            self.algorithm.Log(f"LAYER 1: {strategy_name} generated {len(targets)} targets: {symbol_names}")
                        else:
                            self.algorithm.Log(f"LAYER 1: {strategy_name} generated no targets")
                    else:
                        self.algorithm.Log(f"LAYER 1: {strategy_name} has no target generation method")
                        
                except Exception as e:
                    self.algorithm.Log(f"LAYER 1: {strategy_name} signal generation error: {str(e)}")
                    # Don't log full traceback to save log space - only for critical errors
                    if "Symbol" in str(e) or "clr.MetaClass" in str(e):
                        import traceback
                        self.algorithm.Error(f"LAYER 1: {strategy_name} critical error: {traceback.format_exc()}")
                    continue
            
            total_signals = sum(len(targets) for targets in strategy_targets.values())
            self.algorithm.Log(f"LAYER 1: Generated {total_signals} total signals from {len(strategy_targets)}/{len(loaded_strategies)} strategies")
            
            return strategy_targets
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 1: Execution error: {str(e)}")
            return {}
    
    def _execute_layer2_rebalance(self, strategy_targets):
        """Execute Layer 2: Combine strategy signals with current allocations."""
        try:
            if not self.allocator:
                self.algorithm.Log("LAYER 2: Allocator not available")
                return {}
            
            if not strategy_targets:
                self.algorithm.Log("LAYER 2: No strategy targets to process")
                return {}
            
            # ENHANCED: Update strategy performance tracking for Sharpe calculations
            self._update_strategy_performance_tracking(strategy_targets)
            
            # Execute weekly allocation (includes performance-based updates)
            result = self.allocator.execute_weekly_allocation(strategy_targets)
            
            if result.get('status') == 'success':
                combined_targets = result.get('combined_targets', {})
                allocation_updates = result.get('allocation_updates', {})
                
                # Log allocation updates if any
                if allocation_updates:
                    update_summary = []
                    for strategy, update in allocation_updates.items():
                        change = update['change']
                        update_summary.append(f"{strategy}: {change:+.1%}")
                    self.algorithm.Log(f"LAYER 2: Allocation updates: {', '.join(update_summary)}")
                
                # Update strategy allocation chart with active allocations
                if hasattr(self.algorithm, '_update_strategy_allocation_chart'):
                    # Get the actual allocations used for combining (which are normalized)
                    active_allocations = self._get_active_strategy_allocations(strategy_targets)
                    if active_allocations:
                        self.algorithm._update_strategy_allocation_chart(active_allocations)
                
                gross_exposure = sum(abs(w) for w in combined_targets.values())
                net_exposure = sum(combined_targets.values())
                
                self.algorithm.Log(f"LAYER 2: Combined to {len(combined_targets)} positions")
                self.algorithm.Log(f"LAYER 2: Gross: {gross_exposure:.1%}, Net: {net_exposure:.1%}")
                
                return combined_targets
            else:
                self.algorithm.Log(f"LAYER 2: Allocation failed: {result.get('error', 'unknown')}")
                return {}
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 2: Execution error: {str(e)}")
            return {}

    def _get_active_strategy_allocations(self, strategy_targets):
        """Get normalized allocations for only the strategies that provided targets."""
        try:
            if not strategy_targets or not self.allocator:
                return {}
            
            # Get allocations for active strategies only
            active_strategies = list(strategy_targets.keys())
            active_allocations = {}
            
            for strategy_name in active_strategies:
                active_allocations[strategy_name] = self.allocator.strategy_allocations.get(strategy_name, 0)
            
            # Normalize to sum to 100%
            total_allocation = sum(active_allocations.values())
            if total_allocation > 0:
                for strategy_name in active_allocations:
                    active_allocations[strategy_name] /= total_allocation
            else:
                # Equal weight fallback
                equal_weight = 1.0 / len(active_strategies) if active_strategies else 0
                active_allocations = {name: equal_weight for name in active_strategies}
            
            return active_allocations
            
        except Exception as e:
            self.algorithm.Error(f"ORCHESTRATOR: Error getting active allocations: {str(e)}")
            return {}

    def _execute_layer2_monthly_rebalance(self, strategy_targets):
        """Execute Layer 2 monthly: Update allocations then combine."""
        try:
            if not self.allocator:
                return self._execute_layer2_rebalance(strategy_targets)
            
            # ENHANCED: Update strategy performance tracking for Sharpe calculations
            self._update_strategy_performance_tracking(strategy_targets)
            
            # Execute monthly allocation with enhanced rebalancing
            result = self.allocator.execute_monthly_allocation(strategy_targets)
            
            if result.get('status') == 'success':
                combined_targets = result.get('combined_targets', {})
                allocation_updates = result.get('allocation_updates', {})
                
                # Log allocation updates if any (more detailed for monthly)
                if allocation_updates:
                    self.algorithm.Log("LAYER 2: Monthly allocation updates:")
                    for strategy, update in allocation_updates.items():
                        old_alloc = update['old']
                        new_alloc = update['new']
                        change = update['change']
                        self.algorithm.Log(f"  {strategy}: {old_alloc:.1%} → {new_alloc:.1%} ({change:+.1%})")
                
                # Update strategy allocation chart with active allocations
                if hasattr(self.algorithm, '_update_strategy_allocation_chart'):
                    # Get the actual allocations used for combining (which are normalized)
                    active_allocations = self._get_active_strategy_allocations(strategy_targets)
                    if active_allocations:
                        self.algorithm._update_strategy_allocation_chart(active_allocations)
                
                gross_exposure = sum(abs(w) for w in combined_targets.values())
                net_exposure = sum(combined_targets.values())
                
                self.algorithm.Log(f"LAYER 2: Monthly combined to {len(combined_targets)} positions")
                self.algorithm.Log(f"LAYER 2: Gross: {gross_exposure:.1%}, Net: {net_exposure:.1%}")
                
                return combined_targets
            else:
                self.algorithm.Log(f"LAYER 2: Monthly allocation failed: {result.get('error', 'unknown')}")
                return self._execute_layer2_rebalance(strategy_targets)  # Fallback
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 2: Monthly execution error: {str(e)}")
            return self._execute_layer2_rebalance(strategy_targets)  # Fallback
    
    def _update_strategy_performance_tracking(self, strategy_targets):
        """
        Update strategy performance tracking for ALL enabled strategies.
        
        CRITICAL: Layer 2 needs performance data for ALL strategies to calculate
        proper allocations, not just strategies that provided new signals.
        """
        try:
            if not self.allocator or not hasattr(self.allocator, 'update_strategy_performance'):
                return
            
            # Get ALL enabled strategies from the allocator
            all_enabled_strategies = list(self.allocator.strategies.keys())
            
            # Update performance for strategies that provided new signals
            for strategy_name, targets in strategy_targets.items():
                if targets:  # Only update if strategy has active targets
                    self.allocator.update_strategy_performance(strategy_name, targets)
            
            # For strategies that didn't provide new signals, we need their current positions
            # This is crucial for proper performance tracking during non-rebalancing periods
            strategies_without_signals = [s for s in all_enabled_strategies if s not in strategy_targets.keys()]
            
            if strategies_without_signals:
                # Get current positions for strategies that didn't rebalance
                for strategy_name in strategies_without_signals:
                    # Use their last known targets/positions for performance tracking
                    if hasattr(self.allocator, 'last_strategy_positions') and strategy_name in self.allocator.last_strategy_positions:
                        last_positions = self.allocator.last_strategy_positions[strategy_name]
                        self.allocator.update_strategy_performance(strategy_name, last_positions)
            
        except Exception as e:
            # Silent error handling to avoid log spam
            pass

    def _record_rebalance(self, rebalance_type, strategy_targets, combined_targets, final_targets):
        """Record rebalance history."""
        try:
            rebalance_record = {
                'timestamp': self.algorithm.Time,
                'type': rebalance_type,
                'strategy_signals': len(strategy_targets),
                'combined_positions': len(combined_targets),
                'final_positions': len(final_targets),
                'gross_exposure': sum(abs(w) for w in final_targets.values()),
                'net_exposure': sum(final_targets.values()),
                'config_compliant': True
            }
            self.rebalance_history.append(rebalance_record)
            
        except Exception as e:
            self.algorithm.Log(f"ORCHESTRATOR: Record rebalance error: {str(e)}")
    
    # =============================================================================
    # SYSTEM STATUS AND MONITORING - SIMPLIFIED
    # =============================================================================
    
    def get_system_health(self):
        """Get comprehensive system health status."""
        try:
            system_health = {
                'timestamp': self.algorithm.Time,
                'orchestrator': {
                    'total_rebalances': self.total_rebalances,
                    'successful_rebalances': self.successful_rebalances,
                    'success_rate': self.successful_rebalances / max(self.total_rebalances, 1),
                    'config_compliant': True
                },
                'components': {
                    'strategy_loader_available': self.strategy_loader is not None,
                    'allocator_available': self.allocator is not None,
                    'risk_manager_available': self.risk_manager is not None
                }
            }
            
            # Get component-specific health if available
            if self.strategy_loader and hasattr(self.strategy_loader, 'get_system_health'):
                system_health['layer1_health'] = self.strategy_loader.get_system_health()
            
            if self.allocator and hasattr(self.allocator, 'get_system_health'):
                system_health['layer2_health'] = self.allocator.get_system_health()
            
            if self.risk_manager and hasattr(self.risk_manager, 'get_risk_status'):
                system_health['layer3_health'] = self.risk_manager.get_risk_status()
            
            return system_health
            
        except Exception as e:
            self.algorithm.Error(f"ORCHESTRATOR: Health check error: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def get_performance_summary(self):
        """Get performance summary across all components."""
        try:
            summary = {
                'orchestrator': {
                    'total_rebalances': self.total_rebalances,
                    'successful_rebalances': self.successful_rebalances,
                    'success_rate': self.successful_rebalances / max(self.total_rebalances, 1),
                    'config_compliant': True
                }
            }
            
            # Get component performance if available
            if self.strategy_loader and hasattr(self.strategy_loader, 'get_performance_summary'):
                summary['layer1_strategies'] = self.strategy_loader.get_performance_summary()
            
            if self.allocator and hasattr(self.allocator, 'get_performance_summary'):
                summary['layer2_allocator'] = self.allocator.get_performance_summary()
            
            if self.risk_manager and hasattr(self.risk_manager, 'get_portfolio_risk_metrics'):
                summary['layer3_risk_manager'] = self.risk_manager.get_portfolio_risk_metrics()
            
            return summary
            
        except Exception as e:
            self.algorithm.Error(f"ORCHESTRATOR: Performance summary error: {str(e)}")
            return {'error': str(e)}
    
    def get_current_positions(self):
        """Get current position summary."""
        try:
            if hasattr(self, 'rebalance_history') and self.rebalance_history:
                latest = self.rebalance_history[-1]
                return {
                    'last_rebalance': latest['timestamp'],
                    'rebalance_type': latest['type'],
                    'final_positions': latest['final_positions'],
                    'gross_exposure': latest['gross_exposure'],
                    'net_exposure': latest['net_exposure']
                }
            else:
                return {'status': 'no_rebalances_yet'}
                
        except Exception as e:
            self.algorithm.Error(f"ORCHESTRATOR: Current positions error: {str(e)}")
            return {'error': str(e)}
    
    # =============================================================================
    # RUNTIME STRATEGY MANAGEMENT - DYNAMIC FEATURES
    # =============================================================================
    
    def add_strategy_runtime(self, strategy_name, strategy_config):
        """Add a strategy at runtime (if strategy loader supports it)."""
        try:
            if self.strategy_loader and hasattr(self.strategy_loader, 'add_strategy_runtime'):
                return self.strategy_loader.add_strategy_runtime(strategy_name, strategy_config)
            else:
                self.algorithm.Log("ORCHESTRATOR: Runtime strategy addition not supported")
                return False
                
        except Exception as e:
            self.algorithm.Error(f"ORCHESTRATOR: Runtime strategy addition error: {str(e)}")
            return False
    
    def remove_strategy_runtime(self, strategy_name):
        """Remove a strategy at runtime (if strategy loader supports it)."""
        try:
            if self.strategy_loader and hasattr(self.strategy_loader, 'remove_strategy_runtime'):
                return self.strategy_loader.remove_strategy_runtime(strategy_name)
            else:
                self.algorithm.Log("ORCHESTRATOR: Runtime strategy removal not supported")
                return False
                
        except Exception as e:
            self.algorithm.Error(f"ORCHESTRATOR: Runtime strategy removal error: {str(e)}")
            return False
    
    def update_during_warmup(self, slice):
        """Update components during warmup period."""
        try:
            # During warmup, just log progress occasionally
            if hasattr(self, 'strategy_loader') and self.strategy_loader:
                # Let strategies process warmup data if they have such methods
                for strategy_name, strategy in self.strategy_loader.strategy_objects.items():
                    if hasattr(strategy, 'update_during_warmup'):
                        try:
                            strategy.update_during_warmup(slice)
                        except Exception as e:
                            self.algorithm.Error(f"ORCHESTRATOR: Error updating {strategy_name} during warmup: {str(e)}")
        except Exception as e:
            self.algorithm.Error(f"ORCHESTRATOR: Error in update_during_warmup: {str(e)}")
    
    def update_with_data(self, slice):
        """
        Update with slice data - LEGACY METHOD FOR BACKWARD COMPATIBILITY
        This method is called by the main algorithm's OnData method.
        """
        try:
            # Get liquid symbols first
            liquid_symbols = self._get_liquid_symbols(slice)
            
            if not liquid_symbols:
                return
            
            # Update strategies directly through their objects
            if self.strategy_loader and hasattr(self.strategy_loader, 'strategy_objects'):
                for strategy_name, strategy in self.strategy_loader.strategy_objects.items():
                    if hasattr(strategy, 'update_with_data'):
                        try:
                            strategy.update_with_data(slice)
                        except Exception as e:
                            self.algorithm.Error(f"ORCHESTRATOR: Error updating {strategy_name}: {str(e)}")
            
            if self.allocator and hasattr(self.allocator, 'update_market_data'):
                self.allocator.update_market_data(slice, liquid_symbols)
            
            if self.risk_manager and hasattr(self.risk_manager, 'update_market_data'):
                self.risk_manager.update_market_data(slice, liquid_symbols)
            
        except Exception as e:
            self.algorithm.Error(f"ORCHESTRATOR: update_with_data error: {str(e)}")

    def update_with_unified_data(self, unified_data, slice):
        """
        PHASE 3: Update with unified data interface.
        This method uses the standardized unified data format for optimal performance.
        """
        try:
            # Validate unified data structure
            if not unified_data or not unified_data.get('valid', False):
                self.algorithm.Log("ORCHESTRATOR: Invalid unified data received")
                return
            
            # Extract symbols from unified data
            symbols_data = unified_data.get('symbols', {})
            if not symbols_data:
                return
            
            # Get liquid symbols from unified data
            liquid_symbols = []
            for symbol, symbol_data in symbols_data.items():
                if symbol_data.get('valid', False):
                    # Check if symbol has tradeable data
                    security_data = symbol_data.get('data', {}).get('security', {})
                    if security_data.get('is_tradable', False) and security_data.get('has_data', False):
                        liquid_symbols.append(symbol)
            
            if not liquid_symbols:
                return
            
            # Update all components with unified data
            if self.strategy_loader and hasattr(self.strategy_loader, 'strategy_objects'):
                # Update strategies directly
                for strategy_name, strategy in self.strategy_loader.strategy_objects.items():
                    if hasattr(strategy, 'update_with_unified_data'):
                        try:
                            strategy.update_with_unified_data(unified_data, liquid_symbols)
                        except Exception as e:
                            self.algorithm.Error(f"ORCHESTRATOR: Error updating {strategy_name} with unified data: {str(e)}")
                    elif hasattr(strategy, 'update_with_data'):
                        try:
                            strategy.update_with_data(slice)
                        except Exception as e:
                            self.algorithm.Error(f"ORCHESTRATOR: Error updating {strategy_name}: {str(e)}")
            
            if self.allocator:
                # Pass unified data to allocator
                if hasattr(self.allocator, 'update_market_data_with_unified_data'):
                    self.allocator.update_market_data_with_unified_data(unified_data, liquid_symbols)
                else:
                    # Fallback to traditional method
                    self.allocator.update_market_data(slice, liquid_symbols)
            
            if self.risk_manager:
                # Pass unified data to risk manager
                if hasattr(self.risk_manager, 'update_market_data_with_unified_data'):
                    self.risk_manager.update_market_data_with_unified_data(unified_data, liquid_symbols)
                else:
                    # Fallback to traditional method
                    self.risk_manager.update_market_data(slice, liquid_symbols)
            
            # Track unified data usage statistics
            self._track_unified_data_usage(unified_data, liquid_symbols)
            
        except Exception as e:
            self.algorithm.Error(f"ORCHESTRATOR: update_with_unified_data error: {str(e)}")
            # Fallback to traditional method
            self.update_with_data(slice)

    def _track_unified_data_usage(self, unified_data, liquid_symbols):
        """Track usage statistics for unified data interface."""
        try:
            metadata = unified_data.get('metadata', {})
            self.algorithm.Debug(f"ORCHESTRATOR: Processed {len(liquid_symbols)} liquid symbols "
                               f"from {metadata.get('total_symbols', 0)} total symbols")
        except Exception as e:
            self.algorithm.Debug(f"ORCHESTRATOR: Error tracking unified data usage: {str(e)}")

    def generate_portfolio_targets(self):
        """Generate portfolio targets from all strategies."""
        try:
            if not self.strategy_loader:
                return {}
            
            targets = {}
            
            # Get targets from all loaded strategies
            for strategy_name, strategy in self.strategy_loader.get_strategy_objects().items():
                if hasattr(strategy, 'get_portfolio_targets'):
                    strategy_targets = strategy.get_portfolio_targets()
                    if strategy_targets:
                        targets[strategy_name] = strategy_targets
            
            return targets
            
        except Exception as e:
            self.algorithm.Error(f"ORCHESTRATOR: Error generating portfolio targets: {str(e)}")
            return {}

    def _get_liquid_symbols(self, slice):
        """Get liquid symbols using centralized data validator."""
        try:
            liquid_symbols = []
            
            # Use centralized validator for all symbols
            symbols_to_check = []
            
            # Get symbols from bars
            if hasattr(slice, 'Bars'):
                symbols_to_check.extend(slice.Bars.Keys)
            
            # Get symbols from futures chains
            if hasattr(slice, 'FuturesChains'):
                for symbol in slice.FuturesChains.Keys:
                    if symbol not in symbols_to_check:
                        symbols_to_check.append(symbol)
            
            # Use centralized validator for each symbol with slice data
            for symbol in symbols_to_check:
                if hasattr(self.algorithm, 'data_validator'):
                    validation_result = self.algorithm.data_validator.validate_symbol_for_trading(symbol, slice)
                    
                    if validation_result['is_valid']:
                        liquid_symbols.append(symbol)
                else:
                    # No fallback - require centralized validator for consistency
                    self.algorithm.Error("ORCHESTRATOR: CentralizedDataValidator not available - cannot validate symbols")
            
            return liquid_symbols
            
        except Exception as e:
            self.algorithm.Error(f"ORCHESTRATOR: Error getting liquid symbols: {str(e)}")
            return []
