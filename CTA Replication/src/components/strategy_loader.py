# strategy_loader.py - DYNAMIC STRATEGY LOADER (Layer 1)

from AlgorithmImports import *
from collections import deque
import importlib

class StrategyLoader:
    """
    Layer 1: Dynamic Strategy Management System
    
    SCALABLE DESIGN:
    - Dynamically loads strategies based on config (no hardcoded strategy names)
    - Supports unlimited strategies without code changes
    - Config-driven strategy registration and management
    - Automatic strategy discovery and initialization
    
    STRATEGY REGISTRATION:
    Strategies are registered in config.py with their module and class names:
    
    'strategies': {
        'MTUM_CTA': {
            'enabled': True,
            'module': 'mtum_cta_strategy',
            'class': 'MTUMCTAStrategy',
            # ... other config parameters
        },
        'MyNewStrategy': {
            'enabled': True,
            'module': 'my_new_strategy',
            'class': 'MyNewStrategy',
            # ... strategy-specific config
        }
    }
    
    Adding new strategies requires ZERO code changes - just config updates!
    """
    
    def __init__(self, algorithm, config_manager):
        self.algorithm = algorithm
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        # Symbols are now managed directly by QC's native methods
        
        # Strategy management
        self.strategy_objects = {}        # Loaded strategy instances
        self.strategy_metadata = {}       # Strategy metadata and status
        self.loading_errors = {}          # Track loading failures
        
        # Performance tracking
        self.rebalance_history = deque(maxlen=252)
        self.strategy_performance = {}
        
        self.algorithm.Log("LAYER 1: Dynamic Strategy Loader initialized")
    
    def OnSecuritiesChanged(self, changes):
        """
        Distribute security change events to all loaded strategies.
        """
        if not self.strategy_objects:
            return

        for strategy_name, strategy in self.strategy_objects.items():
            if hasattr(strategy, 'OnSecuritiesChanged'):
                try:
                    strategy.OnSecuritiesChanged(changes)
                except Exception as e:
                    self.algorithm.Error(f"Error in {strategy_name}.OnSecuritiesChanged: {e}")
    
    def load_enabled_strategies(self):
        """
        Dynamically load all enabled strategies from config.
        
        Returns:
            bool: True if at least one strategy loaded successfully
        """
        try:
            enabled_strategies = self.config_manager.get_enabled_strategies()
            
            if not enabled_strategies:
                self.algorithm.Log("LAYER 1: No strategies enabled in configuration")
                return False
            
            successful_loads = 0
            total_strategies = len(enabled_strategies)
            
            self.algorithm.Log(f"LAYER 1: Loading {total_strategies} enabled strategies...")
            
            # Load each enabled strategy dynamically
            for strategy_name, strategy_config in enabled_strategies.items():
                try:
                    if self._load_single_strategy(strategy_name, strategy_config):
                        successful_loads += 1
                        self.algorithm.Log(f"  ✓ {strategy_name}: Loaded successfully")
                    else:
                        self.algorithm.Log(f"  ✗ {strategy_name}: Load failed")
                        
                except Exception as e:
                    self.loading_errors[strategy_name] = str(e)
                    self.algorithm.Error(f"LAYER 1: {strategy_name} load error: {str(e)}")
            
            # Log loading summary
            self.algorithm.Log(f"LAYER 1: Loaded {successful_loads}/{total_strategies} strategies")
            
            if self.loading_errors:
                self.algorithm.Log(f"LAYER 1: {len(self.loading_errors)} strategies failed to load")
            
            return successful_loads > 0
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 1: Strategy loading error: {str(e)}")
            return False
    
    def _load_single_strategy(self, strategy_name, strategy_config):
        """
        Load a single strategy dynamically using config metadata.
        
        Args:
            strategy_name (str): Name of the strategy
            strategy_config (dict): Strategy configuration
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            # Get module and class information from config
            module_name = strategy_config.get('module')
            class_name = strategy_config.get('class')
            
            if not module_name or not class_name:
                self.algorithm.Error(f"LAYER 1: {strategy_name} missing 'module' or 'class' in config")
                return False
            
            # QC handles symbol management natively - no custom managers needed
            
            # Dynamic import and instantiation
            strategy_module = importlib.import_module(module_name)
            strategy_class = getattr(strategy_module, class_name)
            
            # Create strategy instance with CONFIG-COMPLIANT approach
            # BaseStrategy constructor expects: (algorithm, config_manager, strategy_name)
            strategy_instance = strategy_class(
                algorithm=self.algorithm,
                config_manager=self.config_manager,
                strategy_name=strategy_name
            )
            
            # Store strategy and metadata
            self.strategy_objects[strategy_name] = strategy_instance
            self.strategy_metadata[strategy_name] = {
                'module': module_name,
                'class': class_name,
                'config_compliant': True,
                'loaded_at': self.algorithm.Time,
                'rebalance_frequency': strategy_config.get('rebalance_frequency', 'weekly'),
                'target_volatility': strategy_config.get('target_volatility', 0.2),
                'max_position_weight': strategy_config.get('max_position_weight', 0.5)
            }
            
            # Log strategy details
            rebalance_freq = strategy_config.get('rebalance_frequency', 'weekly')
            target_vol = strategy_config.get('target_volatility', 0.2)
            max_weight = strategy_config.get('max_position_weight', 0.5)
            
            self.algorithm.Log(f"    {strategy_name}: {target_vol:.1%} vol, {rebalance_freq}, "
                             f"{max_weight:.1%} max weight, module: {module_name}")
            
            return True
            
        except ImportError as e:
            self.algorithm.Error(f"LAYER 1: {strategy_name} import error - module '{module_name}' not found: {str(e)}")
            return False
        except AttributeError as e:
            self.algorithm.Error(f"LAYER 1: {strategy_name} class error - class '{class_name}' not found: {str(e)}")
            return False
        except Exception as e:
            self.algorithm.Error(f"LAYER 1: {strategy_name} instantiation error: {str(e)}")
            return False
    
    def execute_weekly_rebalance(self):
        """Execute weekly rebalancing for all appropriate strategies."""
        try:
            self.algorithm.Log("LAYER 1: Executing weekly strategy rebalance...")
            
            strategy_targets = {}
            strategies_executed = 0
            
            for strategy_name, strategy in self.strategy_objects.items():
                try:
                    # Check if strategy should rebalance this week
                    should_rebalance = self._should_strategy_rebalance(strategy_name, strategy, 'weekly')
                    
                    if should_rebalance:
                        # Apply asset filtering
                        self._apply_asset_filtering(strategy_name)
                        
                        # Generate signals
                        targets = strategy.generate_signals()
                        
                        if targets:
                            strategy_targets[strategy_name] = targets
                            strategies_executed += 1
                            
                            # Log strategy summary
                            self._log_strategy_summary(strategy_name, targets, 'weekly')
                        else:
                            self.algorithm.Log(f"  {strategy_name}: No signals generated")
                    else:
                        # Use existing targets if not rebalancing
                        if hasattr(strategy, 'current_targets') and strategy.current_targets:
                            strategy_targets[strategy_name] = strategy.current_targets.copy()
                
                except Exception as e:
                    self.algorithm.Error(f"LAYER 1: {strategy_name} weekly error: {str(e)}")
                    continue
            
            return {
                'status': 'success',
                'strategy_targets': strategy_targets,
                'strategies_executed': strategies_executed,
                'total_strategies': len(self.strategy_objects)
            }
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 1: Weekly execution error: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def execute_monthly_rebalance(self):
        """Execute monthly rebalancing for all appropriate strategies."""
        try:
            self.algorithm.Log("LAYER 1: Executing monthly strategy rebalance...")
            
            strategy_targets = {}
            strategies_executed = 0
            
            for strategy_name, strategy in self.strategy_objects.items():
                try:
                    # Check if strategy should rebalance this month
                    should_rebalance = self._should_strategy_rebalance(strategy_name, strategy, 'monthly')
                    
                    if should_rebalance:
                        # Apply asset filtering
                        self._apply_asset_filtering(strategy_name)
                        
                        # Generate signals
                        targets = strategy.generate_signals()
                        
                        if targets:
                            strategy_targets[strategy_name] = targets
                            strategies_executed += 1
                            
                            # Log strategy summary
                            self._log_strategy_summary(strategy_name, targets, 'monthly')
                        else:
                            self.algorithm.Log(f"  {strategy_name}: No monthly signals generated")
                    
                except Exception as e:
                    self.algorithm.Error(f"LAYER 1: {strategy_name} monthly error: {str(e)}")
                    continue
            
            return {
                'status': 'success',
                'strategy_targets': strategy_targets,
                'strategies_executed': strategies_executed,
                'total_strategies': len(self.strategy_objects)
            }
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 1: Monthly execution error: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def execute_emergency_rebalance(self):
        """Execute emergency rebalancing with conservative scaling."""
        try:
            self.algorithm.Log("LAYER 1: Executing emergency strategy rebalance...")
            
            strategy_targets = {}
            
            for strategy_name, strategy in self.strategy_objects.items():
                try:
                    # For emergency, use current targets with conservative scaling
                    if hasattr(strategy, 'current_targets') and strategy.current_targets:
                        # Scale down positions by 50% for emergency
                        emergency_targets = {
                            symbol: weight * 0.5 
                            for symbol, weight in strategy.current_targets.items()
                        }
                        strategy_targets[strategy_name] = emergency_targets
                        
                        self.algorithm.Log(f"  {strategy_name}: Emergency scaling (50% reduction)")
                
                except Exception as e:
                    self.algorithm.Error(f"LAYER 1: {strategy_name} emergency error: {str(e)}")
                    continue
            
            return {
                'status': 'success',
                'strategy_targets': strategy_targets,
                'emergency_scaling': 0.5
            }
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 1: Emergency execution error: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def _should_strategy_rebalance(self, strategy_name, strategy, frequency_context):
        """Determine if a strategy should rebalance based on its frequency and context."""
        try:
            # Get strategy metadata
            metadata = self.strategy_metadata.get(strategy_name, {})
            strategy_frequency = metadata.get('rebalance_frequency', 'weekly')
            
            # Check frequency match
            if frequency_context == 'weekly':
                # Weekly context: rebalance weekly strategies and some monthly strategies
                if strategy_frequency == 'weekly':
                    return hasattr(strategy, 'should_rebalance') and strategy.should_rebalance(self.algorithm.Time)
                elif strategy_frequency == 'monthly':
                    # Monthly strategies might also rebalance weekly
                    return hasattr(strategy, 'should_rebalance') and strategy.should_rebalance(self.algorithm.Time)
                
            elif frequency_context == 'monthly':
                # Monthly context: primarily monthly strategies, but allow weekly strategies too
                if strategy_frequency in ['monthly', 'weekly']:
                    return True  # Force rebalance for monthly context
            
            return False
            
        except Exception as e:
            self.algorithm.Log(f"LAYER 1: Rebalance check error for {strategy_name}: {str(e)}")
            return False
    
    def _apply_asset_filtering(self, strategy_name):
        """Apply asset filtering for the strategy (delegated from config)."""
        try:
            config = self.config_manager.get_config()
            asset_filtering = config.get('asset_filtering', {})
            
            if asset_filtering.get('enabled', False):
                strategy_filters = asset_filtering.get('strategy_filters', {})
                
                if strategy_name in strategy_filters:
                    allowed_categories = strategy_filters[strategy_name]
                    # Asset filtering is applied by the strategy itself based on futures_manager
                    # This is just for logging
                    pass
                
        except Exception as e:
            self.algorithm.Log(f"LAYER 1: Asset filtering error for {strategy_name}: {str(e)}")
    
    def _log_strategy_summary(self, strategy_name, targets, context):
        """Log strategy signal summary with config compliance info."""
        try:
            if not targets:
                return
            
            gross_exposure = sum(abs(w) for w in targets.values())
            net_exposure = sum(targets.values())
            max_position = max((abs(w) for w in targets.values()), default=0)
            
            # Get config source
            config_source = "CONFIG-COMPLIANT"
            if strategy_name in self.strategy_objects:
                strategy = self.strategy_objects[strategy_name]
                if hasattr(strategy, 'get_config_status'):
                    config_status = strategy.get_config_status()
                    config_source = config_status.get('config_source', 'unknown')
            
            self.algorithm.Log(f"  {strategy_name} ({context.upper()}): {len(targets)} positions, "
                             f"Gross: {gross_exposure:.1%}, Net: {net_exposure:.2f}, "
                             f"Max: {max_position:.1%} ({config_source})")
            
        except Exception as e:
            self.algorithm.Log(f"LAYER 1: Log summary error for {strategy_name}: {str(e)}")
    
    # =============================================================================
    # STRATEGY MANAGEMENT AND STATUS METHODS
    # =============================================================================
    
    def get_loaded_strategies(self):
        """Get list of successfully loaded strategy names."""
        return list(self.strategy_objects.keys())
    
    def get_strategy_objects(self):
        """Get dictionary of loaded strategy objects."""
        return self.strategy_objects.copy()
    
    def get_strategy_metadata(self):
        """Get metadata for all loaded strategies."""
        return self.strategy_metadata.copy()
    
    def get_loading_errors(self):
        """Get dictionary of strategies that failed to load."""
        return self.loading_errors.copy()
    
    def add_new_strategy_dynamically(self, strategy_name, strategy_config):
        """
        Dynamically add a new strategy at runtime.
        
        Args:
            strategy_name (str): Name of the new strategy
            strategy_config (dict): Strategy configuration including 'module' and 'class'
            
        Returns:
            bool: True if added successfully
        """
        try:
            if strategy_name in self.strategy_objects:
                self.algorithm.Log(f"LAYER 1: Strategy {strategy_name} already loaded")
                return False
            
            if self._load_single_strategy(strategy_name, strategy_config):
                self.algorithm.Log(f"LAYER 1: Dynamically added strategy {strategy_name}")
                return True
            else:
                self.algorithm.Log(f"LAYER 1: Failed to dynamically add strategy {strategy_name}")
                return False
                
        except Exception as e:
            self.algorithm.Error(f"LAYER 1: Dynamic strategy addition error: {str(e)}")
            return False
    
    def remove_strategy_dynamically(self, strategy_name):
        """
        Dynamically remove a strategy at runtime.
        
        Args:
            strategy_name (str): Name of the strategy to remove
            
        Returns:
            bool: True if removed successfully
        """
        try:
            if strategy_name not in self.strategy_objects:
                self.algorithm.Log(f"LAYER 1: Strategy {strategy_name} not found")
                return False
            
            # Clean up strategy resources
            strategy = self.strategy_objects[strategy_name]
            if hasattr(strategy, 'Dispose'):
                strategy.Dispose()
            
            # Remove from tracking
            del self.strategy_objects[strategy_name]
            if strategy_name in self.strategy_metadata:
                del self.strategy_metadata[strategy_name]
            
            self.algorithm.Log(f"LAYER 1: Dynamically removed strategy {strategy_name}")
            return True
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 1: Dynamic strategy removal error: {str(e)}")
            return False
    
    def get_system_health(self):
        """Get Layer 1 system health status."""
        try:
            # Strategy health
            strategy_health = {}
            config_compliant_count = 0
            
            for name, strategy in self.strategy_objects.items():
                strategy_status = {
                    'loaded': True,
                    'config_compliant': False,
                    'ready_symbols': 0,
                    'current_positions': 0
                }
                
                # Get strategy performance metrics if available
                if hasattr(strategy, 'get_performance_metrics'):
                    metrics = strategy.get_performance_metrics()
                    strategy_status.update({
                        'config_compliant': metrics.get('config_compliant', False),
                        'ready_symbols': metrics.get('ready_symbols', 0),
                        'current_positions': metrics.get('current_positions', 0),
                        'config_source': metrics.get('config_source', 'unknown')
                    })
                    
                    if strategy_status['config_compliant']:
                        config_compliant_count += 1
                
                strategy_health[name] = strategy_status
            
            # Overall health
            layer1_health = {
                'timestamp': self.algorithm.Time,
                'total_strategies_loaded': len(self.strategy_objects),
                'loading_errors': len(self.loading_errors),
                'config_compliant_strategies': config_compliant_count,
                'compliance_rate': config_compliant_count / max(len(self.strategy_objects), 1),
                'strategies': strategy_health,
                'loading_errors_detail': self.loading_errors,
                'dynamic_loading_enabled': True,
                'system_status': 'healthy' if self.strategy_objects else 'no_strategies'
            }
            
            return layer1_health
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 1: Health check error: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def get_performance_summary(self):
        """Get Layer 1 performance summary."""
        try:
            summary = {
                'layer1_strategy_loader': {
                    'total_strategies': len(self.strategy_objects),
                    'loading_errors': len(self.loading_errors),
                    'dynamic_loading_enabled': True,
                    'config_compliant': True
                },
                'individual_strategies': {}
            }
            
            # Individual strategy performance
            for name, strategy in self.strategy_objects.items():
                if hasattr(strategy, 'get_performance_metrics'):
                    summary['individual_strategies'][name] = strategy.get_performance_metrics()
                else:
                    summary['individual_strategies'][name] = {
                        'name': name,
                        'status': 'loaded',
                        'performance_metrics_available': False
                    }
            
            return summary
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 1: Performance summary error: {str(e)}")
            return {'error': str(e)}
    
    def get_config_compliance_report(self):
        """Get Layer 1 config compliance report."""
        try:
            compliance_report = {
                'layer1_strategy_loader': {
                    'config_compliant': True,
                    'dynamic_loading': True,
                    'zero_hardcoded_strategies': True,
                    'config_driven_registration': True
                },
                'individual_strategies': {}
            }
            
            # Individual strategy compliance
            compliant_count = 0
            for name, strategy in self.strategy_objects.items():
                strategy_compliance = {
                    'loaded_dynamically': True,
                    'config_compliant': False,
                    'config_source': 'unknown'
                }
                
                if hasattr(strategy, 'get_config_status'):
                    config_status = strategy.get_config_status()
                    strategy_compliance.update({
                        'config_compliant': config_status.get('config_manager_available', False),
                        'config_source': config_status.get('config_source', 'unknown'),
                        'critical_parameters': config_status.get('critical_parameters', {})
                    })
                    
                    if strategy_compliance['config_compliant']:
                        compliant_count += 1
                
                compliance_report['individual_strategies'][name] = strategy_compliance
            
            # Summary
            compliance_report['summary'] = {
                'total_strategies': len(self.strategy_objects),
                'config_compliant_strategies': compliant_count,
                'compliance_rate': compliant_count / max(len(self.strategy_objects), 1),
                'dynamic_loading_success': len(self.loading_errors) == 0
            }
            
            return compliance_report
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 1: Config compliance report error: {str(e)}")
            return {'error': str(e)}
