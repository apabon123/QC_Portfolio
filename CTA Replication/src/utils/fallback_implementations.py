"""
Fallback implementations for when main components fail to load.
These provide minimal functionality to keep the system running.
"""

class FallbackComponents:
    """Factory for creating fallback implementations."""
    
    def create_fallback_config(self):
        """Create fallback configuration."""
        return {
            'initial_cash': 10000000,
            'target_volatility': 0.50,
            'min_leverage_multiplier': 3.0,
            'rebalance_frequency': 'weekly',
            'futures_universe': ['ES', 'NQ', 'ZN'],
            'strategy_allocations': {
                'KestnerCTA': 0.50,
                'MTUM_CTA': 0.30, 
                'HMM_CTA': 0.20
            },
            'strategies': {
                'KestnerCTA': {
                    'enabled': True,
                    'momentum_lookbacks': [16, 32, 52],
                    'volatility_lookback_days': 90,
                    'signal_cap': 1.0,
                    'target_volatility': 0.15,
                    'max_position_weight': 0.5
                }
            }
        }
    
    def create_fallback_config_manager(self, config):
        """Create fallback config manager."""
        class FallbackConfigManager:
            def __init__(self, config):
                self.config = config
            
            def get_enabled_strategies(self):
                return ['KestnerCTA']
            
            def load_and_validate_config(self, variant="full"):
                return self.config
        
        return FallbackConfigManager(config)
    
    def create_minimal_orchestrator(self, algorithm):
        """Create minimal orchestrator."""
        class MinimalOrchestrator:
            def __init__(self, algorithm):
                self.algorithm = algorithm
            
            def initialize_system(self):
                self.algorithm.Log("Minimal orchestrator initialized")
            
            def generate_portfolio_targets(self):
                return {}
            
            def update_during_warmup(self, slice):
                pass
            
            def update_with_data(self, slice):
                pass
        
        return MinimalOrchestrator(algorithm)
    
    def create_minimal_execution_manager(self, algorithm):
        """Create minimal execution manager."""
        class MinimalExecutionManager:
            def __init__(self, algorithm):
                self.algorithm = algorithm
            
            def execute_portfolio_rebalance(self, targets):
                self.algorithm.Log(f"Minimal execution: {len(targets)} targets")
                return {}
            
            def handle_rollover_events(self, events):
                self.algorithm.Log(f"Minimal rollover handling: {len(events)} events")
        
        return MinimalExecutionManager(algorithm)
    
    def create_minimal_system_reporter(self, algorithm):
        """Create minimal system reporter."""
        class MinimalSystemReporter:
            def __init__(self, algorithm):
                self.algorithm = algorithm
            
            def track_rebalance_performance(self, result):
                self.algorithm.Log("Minimal performance tracking")
            
            def generate_monthly_performance_report(self):
                self.algorithm.Log("Minimal monthly report")
            
            def generate_final_algorithm_report(self):
                self.algorithm.Log("Minimal final report")
        
        return MinimalSystemReporter(algorithm) 