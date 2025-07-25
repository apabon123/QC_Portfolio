# config.py - Unified Configuration Orchestrator
"""
Unified Configuration Module - Clean Architecture

This module orchestrates configuration from specialized modules:
- config_market_strategy.py: Market and strategy configurations
- config_execution_plumbing.py: Execution and technical infrastructure

Benefits of this split:
- Market/strategy logic separated from execution/plumbing
- Easier to maintain and modify
- Clear separation of concerns
- Smaller, focused files

This is the main entry point that assembles the complete configuration.
"""

from AlgorithmImports import *

# Import specialized configuration modules (QuantConnect compatible imports)
try:
    # Try relative imports first (QuantConnect cloud)
    from .config_market_strategy import (
        ALGORITHM_CONFIG,
        FUTURES_TIERS,
        STRATEGY_CONFIGS,
        ALLOCATION_CONFIG,
        RISK_CONFIG,
        UNIVERSE_CONFIG,
        LOGGING_CONFIG,
        get_active_futures,
        get_futures_by_category,
        get_enabled_strategies,
        add_future_to_tier
    )
    
    from .config_execution_plumbing import (
        QC_NATIVE_CONFIG,
        FUTURES_CONFIG,
        EXECUTION_CONFIG,
        MONITORING_CONFIG,
        CONSTRAINTS_CONFIG,
        CONTINUOUS_CONTRACTS_CONFIG,
        get_futures_rollover_config,
        validate_futures_config
    )
except ImportError:
    # Fallback to direct imports (local development)
    from config_market_strategy import (
        ALGORITHM_CONFIG,
        FUTURES_TIERS,
        STRATEGY_CONFIGS,
        ALLOCATION_CONFIG,
        RISK_CONFIG,
        UNIVERSE_CONFIG,
        LOGGING_CONFIG,
        get_active_futures,
        get_futures_by_category,
        get_enabled_strategies,
        add_future_to_tier
    )
    
    from config_execution_plumbing import (
        QC_NATIVE_CONFIG,
        FUTURES_CONFIG,
        EXECUTION_CONFIG,
        MONITORING_CONFIG,
        CONSTRAINTS_CONFIG,
        CONTINUOUS_CONTRACTS_CONFIG,
        get_futures_rollover_config,
        validate_futures_config
    )

# =============================================================================
# BACKWARD COMPATIBILITY HELPERS
# =============================================================================
def _generate_asset_categories_from_tiers():
    """Generate old asset categories structure from new tiers for backward compatibility."""
    categories = {}
    
    # Generate priority-based categories
    for tier_name, tier_futures in FUTURES_TIERS.items():
        priority_num = {'tier_1': 1, 'tier_2': 2, 'tier_3': 3}.get(tier_name, 3)
        priority_key = f'futures_priority_{priority_num}'
        categories[priority_key] = list(tier_futures.keys())
    
    # Generate category-based groupings
    by_category = get_futures_by_category()
    for category, symbols in by_category.items():
        categories[f'futures_{category}'] = symbols
    
    return categories

def _generate_strategy_filters_from_tiers():
    """Generate strategy asset filters from tiers for backward compatibility."""
    active_futures = get_active_futures()
    
    return {
        'MTUM_CTA': {
            'allowed_categories': ['futures_priority_1', 'futures_priority_2'],
            'excluded_categories': [],
            'excluded_symbols': ['6J'],
            'reason': 'Clean tier-based filtering'
        },
        'KestnerCTA': {
            'allowed_categories': ['futures_priority_1', 'futures_priority_2'],
            'excluded_categories': [],
            'excluded_symbols': [],
            'reason': 'Clean tier-based filtering'
        }
    }

# =============================================================================
# MASTER CONFIGURATION ASSEMBLY
# =============================================================================
def get_full_config():
    """
    Assemble the complete configuration dictionary from specialized modules
    
    Returns:
        dict: Complete configuration for the three-layer system
    """
    return {
        # Market and Strategy Configuration (Clean Structure)
        'algorithm': ALGORITHM_CONFIG,
        'futures_tiers': FUTURES_TIERS,
        'strategies': STRATEGY_CONFIGS,
        'strategy_allocation': ALLOCATION_CONFIG,
        'portfolio_risk': RISK_CONFIG,
        'universe': {
            **UNIVERSE_CONFIG,
            'futures_config': FUTURES_CONFIG  # Add technical futures config
        },
        'logging': LOGGING_CONFIG,
        
        # Execution and Plumbing Configuration
        'qc_native': QC_NATIVE_CONFIG,
        'execution': {
            **EXECUTION_CONFIG,
            'futures_config': FUTURES_CONFIG  # Add futures config to execution for easier access
        },
        'monitoring': MONITORING_CONFIG,
        'constraints': CONSTRAINTS_CONFIG,
        'continuous_contracts': CONTINUOUS_CONTRACTS_CONFIG,
        
        # Backward compatibility fields (generate from new structure)
        'system': {
            'name': "Three-Layer CTA Portfolio System",
            'version': "3.0",
            'description': "Clean tier-based configuration system",
        },
        'asset_categories': _generate_asset_categories_from_tiers(),
        'strategy_asset_filters': _generate_strategy_filters_from_tiers(),
    }

# =============================================================================
# CONFIGURATION VARIANTS (SIMPLIFIED)
# =============================================================================

def get_conservative_config():
    """Get a conservative version of the configuration"""
    config = get_full_config()
    
    # Lower overall risk
    config['portfolio_risk']['target_portfolio_vol'] = 0.15
    config['portfolio_risk']['max_leverage_multiplier'] = 2.0
    config['portfolio_risk']['min_notional_exposure'] = 1.5
    
    # More conservative strategy allocations
    config['strategy_allocation']['initial_allocations'] = {
        'KestnerCTA': 0.50,
        'MTUM_CTA': 0.30,
        'HMM_CTA': 0.20
    }
    
    return config

def get_aggressive_config():
    """Get an aggressive version of the configuration"""
    config = get_full_config()
    
    # Higher overall risk
    config['portfolio_risk']['target_portfolio_vol'] = 0.60
    config['portfolio_risk']['max_leverage_multiplier'] = 150
    config['portfolio_risk']['min_notional_exposure'] = 5.0
    
    # Enable all strategies with more aggressive allocation
    config['strategies']['MTUM_CTA']['enabled'] = True
    config['strategies']['HMM_CTA']['enabled'] = True
    config['strategy_allocation']['initial_allocations'] = {
        'KestnerCTA': 0.50,
        'MTUM_CTA': 0.30,
        'HMM_CTA': 0.20
    }
    
    return config

def get_single_strategy_test_config(strategy_name):
    """Get config for testing a single strategy"""
    config = get_full_config()
    
    # Disable all strategies except the one being tested
    for name in config['strategies']:
        config['strategies'][name]['enabled'] = (name == strategy_name)
    
    # Set 100% allocation to the enabled strategy
    config['strategy_allocation']['initial_allocations'] = {
        name: (1.0 if name == strategy_name else 0.0) 
        for name in config['strategies']
    }
    
    return config

def get_development_config():
    """Get config optimized for development and quick testing"""
    config = get_full_config()
    
    # Very short timeframe for quick testing
    config['algorithm']['start_date'] = {'year': 2015, 'month': 1, 'day': 1}
    config['algorithm']['end_date'] = {'year': 2015, 'month': 6, 'day': 1}
    config['algorithm']['warmup_period_days'] = 50
    
    # More conservative risk for testing
    config['portfolio_risk']['target_portfolio_vol'] = 0.20
    config['portfolio_risk']['max_leverage_multiplier'] = 3
    
    # Enable only KestnerCTA for faster testing
    config['strategies']['MTUM_CTA']['enabled'] = False
    config['strategies']['HMM_CTA']['enabled'] = False
    config['strategy_allocation']['initial_allocations'] = {
        'KestnerCTA': 1.00,
        'MTUM_CTA': 0.00,
        'HMM_CTA': 0.00
    }
    
    return config

def get_rollover_test_config():
    """Get config optimized for testing futures rollover behavior"""
    config = get_full_config()
    
    # Short timeframe but spanning multiple contract expirations
    config['algorithm']['start_date'] = {'year': 2019, 'month': 1, 'day': 1}
    config['algorithm']['end_date'] = {'year': 2019, 'month': 12, 'day': 31}
    config['algorithm']['warmup_period_days'] = 30
    
    # Enhanced rollover logging
    config['execution']['rollover_config']['log_rollover_events'] = True
    config['monitoring']['rollover_logging'] = "detailed"
    config['monitoring']['rollover_monitoring']['rollover_alerts'] = True
    
    # Single strategy for cleaner rollover testing
    config['strategies']['MTUM_CTA']['enabled'] = False
    config['strategies']['HMM_CTA']['enabled'] = False
    config['strategy_allocation']['initial_allocations'] = {
        'KestnerCTA': 1.00,
        'MTUM_CTA': 0.00,
        'HMM_CTA': 0.00
    }
    
    # Conservative risk for rollover testing
    config['portfolio_risk']['target_portfolio_vol'] = 0.15
    config['portfolio_risk']['max_leverage_multiplier'] = 3
    
    return config

# =============================================================================
# CONFIGURATION UTILITIES
# =============================================================================

def enable_strategy(strategy_name):
    """Enable a specific strategy in the configuration"""
    config = get_full_config()
    if strategy_name in config['strategies']:
        config['strategies'][strategy_name]['enabled'] = True
    return config

def disable_strategy(strategy_name):
    """Disable a specific strategy in the configuration"""
    config = get_full_config()
    if strategy_name in config['strategies']:
        config['strategies'][strategy_name]['enabled'] = False
    return config

def get_strategy_info():
    """Get information about all available strategies"""
    config = get_full_config()
    return {
        name: {
            'enabled': strategy_config['enabled'],
            'description': strategy_config['description'],
            'rebalance_frequency': strategy_config['rebalance_frequency'],
            'target_volatility': strategy_config['target_volatility'],
            'module': strategy_config['module'],
            'class': strategy_config['class']
        }
        for name, strategy_config in config['strategies'].items()
    }

# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

def test_configuration_system():
    """Test the split configuration system"""
    print("Configuration System Test")
    print("=" * 50)
    
    # Test main config assembly
    try:
        config = get_full_config()
        print("Main configuration assembly: SUCCESS")
        print(f"   - Algorithm config: {bool(config.get('algorithm'))}")
        print(f"   - Strategy config: {len(config.get('strategies', {}))}")
        print(f"   - QC native config: {bool(config.get('qc_native'))}")
        print(f"   - Execution config: {bool(config.get('execution'))}")
    except Exception as e:
        print(f"Main configuration assembly: FAILED - {e}")
    
    # Test strategy utilities
    try:
        enabled = get_enabled_strategies()
        modules = get_strategy_modules()
        info = get_strategy_info()
        print("Strategy utilities: SUCCESS")
        print(f"   - Enabled strategies: {enabled}")
        print(f"   - Strategy modules: {len(modules)}")
        print(f"   - Strategy info: {len(info)}")
    except Exception as e:
        print(f"Strategy utilities: FAILED - {e}")
    
    # Test configuration variants
    variants = ['conservative', 'aggressive', 'development', 'rollover_test']
    for variant in variants:
        try:
            if variant == 'conservative':
                test_config = get_conservative_config()
            elif variant == 'aggressive':
                test_config = get_aggressive_config()
            elif variant == 'development':
                test_config = get_development_config()
            elif variant == 'rollover_test':
                test_config = get_rollover_test_config()
            
            print(f"{variant} config: SUCCESS")
        except Exception as e:
            print(f"{variant} config: FAILED - {e}")
    
    # Test futures validation
    try:
        issues = validate_futures_config()
        if not issues:
            print("Futures configuration validation: SUCCESS")
        else:
            print(f"WARNING: Futures configuration issues: {issues}")
    except Exception as e:
        print(f"Futures validation: FAILED - {e}")
    
    print("\nConfiguration system test complete!")

# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

# For backward compatibility with existing code
def get_config():
    """Legacy compatibility function"""
    return get_full_config()

# =============================================================================
# USAGE EXAMPLES AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Uncomment to test the split configuration system
    # test_configuration_system()
    pass
