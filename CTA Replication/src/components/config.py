# config.py - UPDATED FOR DYNAMIC STRATEGY LOADING SYSTEM WITH FUTURES ROLLOVER
# region imports
from AlgorithmImports import *
# endregion
"""
QuantConnect-Compatible Configuration Module - SINGLE SOURCE OF TRUTH

This module contains ALL configuration parameters as Python dictionaries.
CLEAN ARCHITECTURE: Everything configured in ONE place only.
DYNAMIC STRATEGY LOADING: Supports unlimited strategies without code changes.
FUTURES ROLLOVER: Proper configuration for QuantConnect's built-in rollover system.

To modify parameters:
1. Edit the values in this file
2. Save and run your backtest
3. No external file dependencies
4. No duplicate settings anywhere

To add new strategies:
1. Create your strategy file (e.g., my_new_strategy.py)
2. Add strategy config below in STRATEGY_CONFIGS
3. ZERO code changes needed in orchestrator!
"""

# =============================================================================
# ALGORITHM SETTINGS CONFIGURATION - SINGLE SOURCE
# =============================================================================
ALGORITHM_CONFIG = {
    'start_date': {'year': 2015, 'month': 1, 'day': 1},
    'end_date': {'year': 2020, 'month': 1, 'day': 1},
    'initial_cash': 10000000,               # $10M initial capital
    'resolution': 'Daily',                  # Daily resolution
    'timezone': 'America/New_York',         # Eastern time
    'benchmark': 'SPY',                     # Benchmark for comparison
    'brokerage_model': 'InteractiveBrokers', # Brokerage model
    'warmup_period_days': 80,               # SINGLE SOURCE: Algorithm warmup period
}

# =============================================================================
# ASSET CATEGORIES AND FILTERING CONFIGURATION
# =============================================================================
ASSET_CATEGORIES = {
    'futures_equity': ['ES', 'NQ', 'YM', 'RTY'],              # Equity index futures
    'futures_rates': ['ZN', 'ZB', 'ZF', 'ZT'],                # Interest rate futures  
    'futures_fx': ['6E', '6J', '6B', '6A'],                   # Currency futures
    'futures_commodities': ['CL', 'GC', 'SI', 'HG'],          # Commodity futures
    'futures_vix': ['VX'],                                     # VIX futures (special case)
    'equities': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],  # Individual stocks
    'options': ['SPY', 'QQQ', 'IWM'],                         # Options (underlying symbols)
}

STRATEGY_ASSET_FILTERS = {
    'KestnerCTA': {
        'allowed_categories': ['futures_equity', 'futures_rates', 'futures_fx', 'futures_commodities'],
        'excluded_categories': ['futures_vix', 'equities', 'options'],
        'excluded_symbols': [],
        'reason': 'Trend following works best on liquid, trending futures markets'
    },
    
    'MTUM_CTA': {
        'allowed_categories': ['futures_equity', 'futures_rates', 'equities'],
        'excluded_categories': ['futures_vix', 'options'],
        'excluded_symbols': [],
        'reason': 'Momentum works on equity futures, rates, and individual stocks'
    },
    
    'HMM_CTA': {
        'allowed_categories': ['futures_equity', 'futures_rates'],
        'excluded_categories': ['futures_vix', 'equities', 'options', 'futures_fx'],
        'excluded_symbols': [],
        'reason': 'Regime detection needs stable, liquid markets with clean data'
    },
    
    # Future strategy examples (ready for when you implement them):
    'VixContango': {
        'allowed_categories': ['futures_vix'],
        'excluded_categories': ['futures_equity', 'futures_rates', 'equities', 'options'],
        'excluded_symbols': [],
        'reason': 'VIX rolldown strategy specifically for VX futures'
    },
    
    'TechMomentum': {
        'allowed_categories': ['equities'],
        'excluded_categories': ['futures_*'],
        'allowed_symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
        'reason': 'Tech sector momentum on individual names'
    },
    
    'CarryStrategy': {
        'allowed_categories': ['futures_rates', 'futures_fx'],
        'excluded_categories': ['futures_equity', 'futures_vix', 'equities'],
        'excluded_symbols': [],
        'reason': 'Carry strategies need yield-bearing assets'
    }
}

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================
SYSTEM_CONFIG = {
    'name': "Three-Layer CTA Portfolio System",
    'version': "2.5",
    'description': "Multi-strategy futures trading with dynamic allocation, centralized risk management, dynamic strategy loading, and proper futures rollover handling",
    'last_updated': "2025-01-20"
}

# =============================================================================
# DYNAMIC STRATEGY REGISTRATION - NEW SYSTEM
# THIS IS THE KEY INNOVATION: Add strategies here without touching orchestrator code!
# =============================================================================
STRATEGY_CONFIGS = {
    # Strategy 1: Kestner CTA (Academic trend following replication)
    'KestnerCTA': {
        'enabled': True,                        # Set to False to disable without code changes
        'module': 'kestner_cta_strategy',       # File name (without .py)
        'class': 'KestnerCTAStrategy',          # Class name in the file
        'name': 'KestnerCTA',                   # Display name
        'description': 'Academic replication of Lars Kestner methodology',
        
        # Strategy-specific parameters
        'momentum_lookbacks': [16, 32, 52],     # Weekly lookback periods for ensemble
        'volatility_lookback_days': 63,         # Volatility calculation window (matches paper)
        'signal_cap': 1.0,                      # Maximum raw signal strength
        'target_volatility': 0.2,              # 20% individual strategy vol target
        'rebalance_frequency': 'weekly',        # Weekly rebalancing on Fridays
        'max_position_weight': 0.6,             # 60% maximum single position
        'warmup_days': 400,                     # Strategy-specific warmup
        'expected_sharpe': 0.8,                 # Historical Sharpe expectation
        'correlation_with_trend': 0.75,         # Expected correlation with SG Trend Index
    },
    
    # Strategy 2: MTUM CTA (Momentum futures adaptation)
    'MTUM_CTA': {
        'enabled': False,                       # Set to False to disable without code changes
        'module': 'mtum_cta_strategy',          # File name (without .py)
        'class': 'MTUMCTAStrategy',             # Class name in the file
        'name': 'MTUM_CTA',                     # Display name
        'description': 'Futures adaptation of MSCI USA Momentum methodology',
        
        # Strategy-specific parameters
        'momentum_lookbacks_months': [6, 12],   # 6-month and 12-month periods
        'volatility_lookback_days': 252,        # 1 year for volatility estimation
        'signal_standardization_clip': 3.0,     # ±3 standard deviation clipping
        'target_volatility': 0.2,               # 20% individual strategy vol target
        'rebalance_frequency': 'monthly',       # Monthly rebalancing (first Friday)
        'max_position_weight': 0.6,             # 60% maximum single position
        'risk_free_rate': 0.02,                 # 2% annual risk-free rate assumption
        'long_short_enabled': True,             # Allow short positions (vs equity MTUM)
        'warmup_days': 400,                     # Strategy-specific warmup
        'expected_sharpe': 0.6,                 # Conservative Sharpe expectation
        'correlation_with_momentum': 0.7,       # Expected momentum factor correlation
    },
    
    # Strategy 3: HMM CTA (Hidden Markov Model regime detection)
    'HMM_CTA': {
        'enabled': False,                       # Currently disabled for testing
        'module': 'hmm_cta_strategy',           # File name (without .py)
        'class': 'HMMCTAStrategy',              # Class name in the file
        'name': 'HMM_CTA',                      # Display name
        'description': 'Hidden Markov Model regime detection for futures',
        
        # Strategy-specific parameters
        'n_components': 3,                      # 3 regimes: down, ranging, up
        'n_iter': 100,                          # HMM training iterations
        'random_state': 42,                     # Reproducible results
        'returns_window': 60,                   # Rolling window for regime detection
        'target_volatility': 0.20,             # 20% individual strategy vol target
        'rebalance_frequency': 'weekly',        # Weekly rebalancing for regime responsiveness
        'max_position_weight': 0.6,             # 60% maximum single position
        'regime_threshold': 0.70,               # 70% confidence threshold for regime identification
        'regime_persistence_days': 3,           # Require 3-day regime confirmation to avoid whipsawing
        'regime_smoothing_alpha': 0.3,          # Exponential smoothing factor for regime probabilities
        'retrain_frequency': 'monthly',         # Monthly model retraining for market adaptation
        'warmup_days': 80,                      # Strategy-specific warmup
        'expected_sharpe': 0.5,                 # Conservative expectation (regime strategy)
        'correlation_with_regime': 0.6,         # Expected regime factor correlation
    },
    
    # EXAMPLES: Future strategies (ready to implement)
    # Uncomment and implement when ready:
    
    # 'VixContango': {
    #     'enabled': False,
    #     'module': 'vix_contango_strategy',
    #     'class': 'VixContangoStrategy',
    #     'name': 'VixContango',
    #     'description': 'VIX futures contango and backwardation strategy',
    #     'contango_threshold': 0.05,
    #     'lookback_days': 20,
    #     'target_volatility': 0.25,
    #     'rebalance_frequency': 'weekly',
    #     'max_position_weight': 0.3,
    #     'warmup_days': 100,
    #     'expected_sharpe': 0.4,
    # },
    
    # 'TechMomentum': {
    #     'enabled': False,
    #     'module': 'tech_momentum_strategy',
    #     'class': 'TechMomentumStrategy', 
    #     'name': 'TechMomentum',
    #     'description': 'Technology sector momentum on individual stocks',
    #     'momentum_window': 126,
    #     'rebalance_frequency': 'monthly',
    #     'target_volatility': 0.18,
    #     'max_position_weight': 0.15,
    #     'sector_filter': ['Technology'],
    #     'warmup_days': 200,
    #     'expected_sharpe': 0.7,
    # },
    
    # 'CarryStrategy': {
    #     'enabled': False,
    #     'module': 'carry_strategy',
    #     'class': 'CarryStrategy',
    #     'name': 'CarryStrategy', 
    #     'description': 'Interest rate and FX carry strategy',
    #     'yield_lookback': 60,
    #     'rebalance_frequency': 'monthly',
    #     'target_volatility': 0.12,
    #     'max_position_weight': 0.4,
    #     'warmup_days': 150,
    #     'expected_sharpe': 0.5,
    # }
}

# =============================================================================
# LAYER 2: DYNAMIC STRATEGY ALLOCATION CONFIGURATION
# =============================================================================
ALLOCATION_CONFIG = {
    'enabled': True,
    'name': "Dynamic Sharpe Allocation",
    'description': "Performance-based allocation using risk-adjusted returns",
    
    # Allocation Method
    'allocation_method': "sharpe_proportional",  # Sharpe ratio proportional allocation
    'lookback_days': 63,                         # 3 months for performance evaluation
    'min_track_record_days': 21,                 # 3 weeks minimum before allocation changes
    'rebalance_frequency': "weekly",             # Weekly allocation updates
    
    # Smoothing and Stability
    'allocation_smoothing': 0.5,                 # 50% old allocation, 50% new (balanced)
    'min_allocation': 0.05,                      # 5% minimum allocation per strategy
    'max_allocation': 0.70,                      # 70% maximum allocation per strategy
    
    # Portfolio Construction
    'use_correlation': True,                     # Consider correlations in portfolio vol
    'correlation_lookback_days': 126,            # 6 months for correlation estimation
    
    # Initial Allocations (using strategy names from STRATEGY_CONFIGS)
    'initial_allocations': {
        'KestnerCTA': 1.00,                      # 100% to trend following (single strategy test)
        'MTUM_CTA': 0.00,                        # 0% (disabled in STRATEGY_CONFIGS) 
        'HMM_CTA': 0.00,                         # 0% (disabled in STRATEGY_CONFIGS)
    },
    
    # Allocation Bounds (using strategy names from STRATEGY_CONFIGS)
    'allocation_bounds': {
        'KestnerCTA': {'min': 0.30, 'max': 1.00},   # Minimum 30%, Maximum 100%
        'MTUM_CTA': {'min': 0.00, 'max': 0.70},     # Minimum 0%, Maximum 70%
        'HMM_CTA': {'min': 0.00, 'max': 0.25},      # Minimum 0%, Maximum 25%
    }
}

# =============================================================================
# LAYER 3: PORTFOLIO RISK MANAGEMENT CONFIGURATION  
# =============================================================================
RISK_CONFIG = {
    'enabled': True,
    'name': "Centralized Risk Management",
    'description': "Portfolio-level volatility targeting and risk controls",
    
    # Volatility Targeting
    'target_portfolio_vol': 0.6,               # 60% annualized portfolio volatility
    'vol_estimation_days': 63,                 # 3 months for volatility estimation
    'vol_targeting_method': "exponential",     # Exponential weighting for recent data
    
    # Exposure Management
    'min_notional_exposure': 0.25,             # Minimum 25% of capital deployed
    'max_gross_exposure': 10.0,                # Maximum 1000% gross exposure
    'max_leverage_multiplier': 100,            # Maximum 100x leverage scalar
    
    # Stop Loss Controls
    'daily_stop_loss': 0.2,                    # 20% daily portfolio stop loss
    'max_drawdown_stop': 0.75,                 # 75% maximum drawdown stop
    
    # Position Limits
    'max_single_position': 10.00,              # 1000% maximum single futures position
    'max_sector_exposure': 5.00,               # 500% maximum sector exposure
    
    # Risk Monitoring
    'risk_check_frequency': "daily",           # Daily risk monitoring
    'correlation_monitoring': True,            # Monitor strategy correlations
    'volatility_regime_detection': True,       # Adjust targets based on vol regime
}

# =============================================================================
# UNIVERSE AND EXECUTION CONFIGURATION WITH FUTURES ROLLOVER
# =============================================================================
UNIVERSE_CONFIG = {
    # FUTURES CONFIGURATION - CRITICAL FOR ROLLOVER BEHAVIOR
    'futures_config': {
        'data_mapping_mode': 'DataMappingMode.OpenInterest',      # WHEN to rollover: based on open interest
        'data_normalization_mode': 'DataNormalizationMode.BackwardsRatio',  # HOW to adjust prices
        'contract_depth_offset': 0,                               # 0 = front month, 1 = back month
        'extended_market_hours': True,                            # Include extended hours data
        'resolution': 'Resolution.Daily',                         # Data resolution
        'fill_forward': True,                                     # Fill missing data points
        'contract_filter_days': 182,                             # Include contracts expiring within 182 days
    },
    
    # Futures Universe (simplified - QC provides contract specs automatically)
    'futures': {
        'equity_index': {
            'ES': {
                'name': "E-mini S&P 500",
                'category': 'futures_equity',
                'priority': 1,
                'min_volume': 100000,  # Our trading requirements
            },
            'NQ': {
                'name': "Nasdaq 100",
                'category': 'futures_equity',
                'priority': 1,
                'min_volume': 50000,
            }
        },
        'interest_rates': {
            'ZN': {
                'name': "10-Year Treasury Note",
                'category': 'futures_rates',
                'priority': 1,
                'min_volume': 50000,
            }
        }
    },
    
    # Future Expansion Ready (only our business logic, not contract specs)
    'expansion_candidates': {
        "6E": {
            'category': 'futures_fx', 
            'name': "Euro FX",
            'priority': 2,
            'min_volume': 20000,
        },
        "6J": {
            'category': 'futures_fx', 
            'name': "Japanese Yen",
            'priority': 2,
            'min_volume': 15000,
        },
        "CL": {
            'category': 'futures_commodities', 
            'name': "Crude Oil",
            'priority': 2,
            'min_volume': 30000,
        },
        "GC": {
            'category': 'futures_commodities', 
            'name': "Gold",
            'priority': 2,
            'min_volume': 25000,
        },
        "VX": {
            'category': 'futures_vix', 
            'name': "VIX Futures",
            'priority': 3,
            'min_volume': 10000,
        },
        "ZB": {
            'category': 'futures_rates', 
            'name': "30-Year Treasury Bond",
            'priority': 2,
            'min_volume': 30000,
        },
        "YM": {
            'category': 'futures_equity', 
            'name': "Dow Jones Industrial Average",
            'priority': 2,
            'min_volume': 40000,
        }
    }
}

EXECUTION_CONFIG = {
    'order_type': "market",                     # Market orders for immediacy
    'min_trade_value': 1000,                    # Minimum $1,000 trade size
    'min_weight_change': 0.01,                  # 1% minimum weight change to trade
    'max_single_order_value': 50000000,         # $50M maximum single order
    'max_single_position': 10.0,                # 1000% maximum single position
    'max_portfolio_turnover': 2.0,              # 200% maximum daily turnover
    
    # FUTURES ROLLOVER CONFIGURATION - CRITICAL FOR PROPER ROLLOVER HANDLING
    'rollover_config': {
        'enabled': True,                        # Enable automatic rollover handling
        'rollover_method': 'OnSymbolChangedEvents',  # Use QuantConnect's built-in system
        'order_type': 'market',                 # Market orders for rollovers (immediate execution)
        'max_rollover_slippage': 0.001,         # 0.1% maximum acceptable slippage
        'emergency_liquidation': True,          # Liquidate if rollover fails
        'rollover_tag_prefix': 'ROLLOVER',      # Tag prefix for rollover trades
        'position_tolerance': 0.01,             # 1% tolerance for position matching
        'retry_attempts': 3,                    # Number of retry attempts for failed rollovers
        'retry_delay_seconds': 5,               # Delay between retry attempts
        'log_rollover_events': True,            # Log all rollover events for debugging
        'validate_rollover_contracts': True,    # Validate new contracts before trading
    },
    
    # FUTURES-SPECIFIC EXECUTION PARAMETERS
    'futures_execution': {
        'use_continuous_contracts': True,        # Trade using continuous contract symbols
        'validate_expiry_dates': True,          # Check contract expiry before trading
        'min_days_to_expiry': 7,                # Minimum days to expiry for new positions
        'rollover_notification': True,          # Log rollover events
        'track_rollover_costs': True,           # Track transaction costs from rollovers
        'rollover_performance_attribution': True, # Attribute performance impact of rollovers
        'use_mapped_contracts': True,           # Use the .Mapped property for actual trading
        'continuous_contract_for_data': True,   # Use continuous contracts for price data/indicators
    },
    
    # Transaction Cost Assumptions
    'transaction_costs': {
        'futures_commission': 1.50,             # $1.50 per futures contract
        'slippage_bps': 1.0,                    # 1 basis point estimated slippage
        'rollover_slippage_bps': 2.0,           # Higher slippage for rollovers
        'rollover_commission_multiplier': 2.0,  # 2x commission for rollover (close + open)
    }
}

# =============================================================================
# MONITORING AND REPORTING CONFIGURATION
# =============================================================================
MONITORING_CONFIG = {
    # Performance Tracking
    'performance_frequency': "daily",          # Daily performance calculation
    'benchmark_tracking': True,                # Track vs SG Trend Index
    'attribution_analysis': True,              # Strategy performance attribution
    
    # Rollover Monitoring
    'rollover_monitoring': {
        'track_rollover_frequency': True,       # Monitor rollover frequency by contract
        'rollover_cost_analysis': True,         # Analyze rollover transaction costs
        'rollover_timing_analysis': True,       # Analyze rollover timing vs expiry
        'rollover_performance_impact': True,    # Track performance impact of rollovers
        'rollover_alerts': True,                # Alert on rollover events
    },
    
    # Risk Monitoring
    'risk_alerts': {
        'volatility_breach': 0.25,             # Alert if vol exceeds 25%
        'drawdown_warning': 0.10,              # Warning at 10% drawdown
        'correlation_spike': 0.90,             # Alert if strategy correlation > 90%
        'rollover_failure': True,              # Alert on rollover failures
        'position_mismatch': 0.05,             # Alert if position mismatch > 5%
    },
    
    # Logging Levels
    'strategy_logging': "detailed",            # Detailed strategy logging
    'execution_logging': "summary",            # Summary execution logging
    'risk_logging': "detailed",                # Detailed risk logging
    'rollover_logging': "detailed",            # Detailed rollover logging
    
    # Reporting
    'daily_reports': True,                     # Generate daily performance reports
    'weekly_reports': True,                    # Generate weekly allocation reports
    'monthly_reports': True,                   # Generate monthly risk reports
    'rollover_reports': True,                  # Generate rollover analysis reports
}

# =============================================================================
# SYSTEM CONSTRAINTS AND LIMITS
# =============================================================================
CONSTRAINTS_CONFIG = {
    # Capital Constraints
    'initial_capital': 10000000,               # $10M initial capital (matches ALGORITHM_CONFIG)
    'min_capital': 5000000,                    # $5M minimum to continue trading
    
    # Position Constraints
    'max_positions_per_strategy': 10,          # Maximum 10 positions per strategy
    'max_total_positions': 20,                 # Maximum 20 total positions
    
    # Futures-Specific Constraints
    'futures_constraints': {
        'max_contracts_per_symbol': 1000,      # Maximum contracts per symbol
        'min_margin_buffer': 0.20,             # 20% margin buffer
        'max_notional_per_contract': 100000000, # $100M max notional per contract type
        'rollover_blackout_hours': 2,          # Hours before/after rollover to avoid new positions
    },
    
    # Data Quality
    'min_data_quality': 0.95,                  # 95% minimum data completeness
    'max_price_gap': 0.10,                     # 10% maximum price gap filter
    
    # System Limits
    'max_memory_usage': "2GB",                 # Maximum memory usage
    'max_execution_time': "30min",             # Maximum daily execution time
}

# =============================================================================
# MASTER CONFIGURATION ASSEMBLY
# =============================================================================
def get_full_config():
    """
    Assemble the complete configuration dictionary
    
    Returns:
        dict: Complete configuration for the three-layer system with futures rollover
    """
    return {
        'algorithm': ALGORITHM_CONFIG,              # SINGLE SOURCE: Algorithm settings
        'system': SYSTEM_CONFIG,
        'asset_categories': ASSET_CATEGORIES,
        'strategy_asset_filters': STRATEGY_ASSET_FILTERS,
        'strategies': STRATEGY_CONFIGS,             # DYNAMIC STRATEGY CONFIGS
        'strategy_allocation': ALLOCATION_CONFIG,
        'portfolio_risk': RISK_CONFIG,
        'universe': UNIVERSE_CONFIG,                # NOW INCLUDES FUTURES ROLLOVER CONFIG
        'execution': EXECUTION_CONFIG,              # NOW INCLUDES ROLLOVER EXECUTION CONFIG
        'monitoring': MONITORING_CONFIG,            # NOW INCLUDES ROLLOVER MONITORING
        'constraints': CONSTRAINTS_CONFIG
    }

# =============================================================================
# QUICK CONFIGURATION MODIFICATIONS
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
        'KestnerCTA': 0.70,
        'MTUM_CTA': 0.30,
        'HMM_CTA': 0.00
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
    """Get config optimized for testing rollover functionality"""
    config = get_full_config()
    
    # Medium timeframe to catch multiple rollovers
    config['algorithm']['start_date'] = {'year': 2018, 'month': 1, 'day': 1}
    config['algorithm']['end_date'] = {'year': 2020, 'month': 1, 'day': 1}
    config['algorithm']['warmup_period_days'] = 80
    
    # Enhanced rollover monitoring for testing
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
    
    return config

# =============================================================================
# STRATEGY MANAGEMENT HELPER FUNCTIONS  
# =============================================================================

def get_enabled_strategies():
    """Get list of currently enabled strategies"""
    return [name for name, config in STRATEGY_CONFIGS.items() if config.get('enabled', False)]

def get_strategy_modules():
    """Get mapping of strategy names to their module/class info"""
    return {
        name: {
            'module': config['module'],
            'class': config['class'],
            'name': config['name']
        }
        for name, config in STRATEGY_CONFIGS.items() 
        if config.get('enabled', False)
    }

def enable_strategy(strategy_name):
    """Enable a specific strategy"""
    if strategy_name in STRATEGY_CONFIGS:
        STRATEGY_CONFIGS[strategy_name]['enabled'] = True
        return True
    return False

def disable_strategy(strategy_name):
    """Disable a specific strategy"""
    if strategy_name in STRATEGY_CONFIGS:
        STRATEGY_CONFIGS[strategy_name]['enabled'] = False
        return True
    return False

def get_strategy_info():
    """Get summary info about all strategies"""
    info = {}
    for name, config in STRATEGY_CONFIGS.items():
        info[name] = {
            'enabled': config.get('enabled', False),
            'description': config.get('description', 'No description'),
            'rebalance_frequency': config.get('rebalance_frequency', 'Unknown'),
            'target_volatility': config.get('target_volatility', 'Unknown'),
            'max_position_weight': config.get('max_position_weight', 'Unknown')
        }
    return info

# =============================================================================
# ASSET FILTERING HELPER FUNCTIONS
# =============================================================================

def get_strategy_allowed_symbols(strategy_name, all_symbols):
    """
    Get the symbols a strategy is allowed to trade.
    
    Args:
        strategy_name: Name of the strategy (e.g., 'KestnerCTA')
        all_symbols: List of all available symbols
        
    Returns:
        List of symbols this strategy should trade
    """
    if strategy_name not in STRATEGY_ASSET_FILTERS:
        return all_symbols
    
    filter_config = STRATEGY_ASSET_FILTERS[strategy_name]
    allowed_symbols = set()
    
    # Add symbols from allowed categories
    for category in filter_config.get('allowed_categories', []):
        if category in ASSET_CATEGORIES:
            allowed_symbols.update(ASSET_CATEGORIES[category])
    
    # Add specifically allowed symbols
    allowed_symbols.update(filter_config.get('allowed_symbols', []))
    
    # Remove symbols from excluded categories
    for category in filter_config.get('excluded_categories', []):
        if category.endswith('*'):
            # Handle wildcard exclusions like 'futures_*'
            prefix = category[:-1]
            for cat_name, symbols in ASSET_CATEGORIES.items():
                if cat_name.startswith(prefix):
                    allowed_symbols -= set(symbols)
        elif category in ASSET_CATEGORIES:
            allowed_symbols -= set(ASSET_CATEGORIES[category])
    
    # Remove specifically excluded symbols
    allowed_symbols -= set(filter_config.get('excluded_symbols', []))
    
    # Only return symbols that actually exist in the universe
    return [sym for sym in all_symbols if sym in allowed_symbols]

# =============================================================================
# FUTURES ROLLOVER HELPER FUNCTIONS
# =============================================================================

def get_futures_rollover_config():
    """Get futures-specific rollover configuration"""
    config = get_full_config()
    return {
        'data_mapping_mode': config['universe']['futures_config']['data_mapping_mode'],
        'data_normalization_mode': config['universe']['futures_config']['data_normalization_mode'],
        'rollover_method': config['execution']['rollover_config']['rollover_method'],
        'rollover_enabled': config['execution']['rollover_config']['enabled'],
        'log_rollovers': config['execution']['rollover_config']['log_rollover_events'],
        'use_continuous_contracts': config['execution']['futures_execution']['use_continuous_contracts'],
    }

def get_contract_specifications(symbol):
    """Get contract specifications for a specific futures symbol"""
    config = get_full_config()
    
    # Check in main futures config
    for category, contracts in config['universe']['futures'].items():
        if symbol in contracts:
            return contracts[symbol]
    
    # Check in expansion candidates
    if symbol in config['universe']['expansion_candidates']:
        return config['universe']['expansion_candidates'][symbol]
    
    return None

def validate_futures_config():
    """Validate that futures configuration is properly set up"""
    config = get_full_config()
    issues = []
    
    # Check required futures config
    futures_config = config['universe']['futures_config']
    required_fields = ['data_mapping_mode', 'data_normalization_mode', 'contract_depth_offset']
    
    for field in required_fields:
        if field not in futures_config:
            issues.append(f"Missing required futures config: {field}")
    
    # Check rollover config
    rollover_config = config['execution']['rollover_config']
    if not rollover_config.get('enabled', False):
        issues.append("Rollover handling is disabled - this will cause position loss on expiry!")
    
    # Check if we have any futures contracts defined
    if not config['universe']['futures']:
        issues.append("No futures contracts defined in universe config")
    
    return issues

def test_dynamic_strategy_system():
    """Test the dynamic strategy loading system"""
    print("Dynamic Strategy Loading System Test")
    print("=" * 50)
    
    enabled = get_enabled_strategies()
    print(f"Enabled Strategies: {enabled}")
    
    modules = get_strategy_modules()
    print(f"Strategy Modules: {modules}")
    
    info = get_strategy_info()
    for name, details in info.items():
        status = "✅ ENABLED" if details['enabled'] else "❌ DISABLED"
        print(f"{name}: {status}")
        print(f"  Description: {details['description']}")
        print(f"  Rebalance: {details['rebalance_frequency']}")
        print(f"  Target Vol: {details['target_volatility']}")
        print()

def test_futures_rollover_config():
    """Test the futures rollover configuration"""
    print("Futures Rollover Configuration Test")
    print("=" * 50)
    
    # Validate configuration
    issues = validate_futures_config()
    if issues:
        print("❌ CONFIGURATION ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ CONFIGURATION VALID")
    
    # Show rollover settings
    rollover_config = get_futures_rollover_config()
    print("\nRollover Configuration:")
    for key, value in rollover_config.items():
        print(f"  {key}: {value}")
    
    # Show contract specs
    print("\nContract Specifications:")
    config = get_full_config()
    for category, contracts in config['universe']['futures'].items():
        print(f"  {category.upper()}:")
        for symbol, spec in contracts.items():
            typical_rollover = spec.get('typical_rollover_days_before_expiry', 'Unknown')
            print(f"    {symbol}: {spec['name']} (typical rollover: {typical_rollover} days before expiry)")

# =============================================================================
# IMPORT VERIFICATION
# =============================================================================

def verify_all_imports():
    """Verify all functions expected by algorithm_config_manager are available"""
    functions_to_test = [
        'get_full_config',
        'get_conservative_config', 
        'get_aggressive_config',
        'get_single_strategy_test_config',
        'get_development_config',
        'get_rollover_test_config'
    ]
    
    print("Verifying config functions:")
    for func_name in functions_to_test:
        try:
            func = globals()[func_name]
            result = func()
            print(f"  ✓ {func_name}: OK")
        except Exception as e:
            print(f"  ✗ {func_name}: ERROR - {e}")
    
    print("\nVerifying futures config:")
    try:
        issues = validate_futures_config()
        if not issues:
            print("  ✓ Futures configuration: VALID")
        else:
            print("  ✗ Futures configuration: ISSUES FOUND")
            for issue in issues:
                print(f"    - {issue}")
    except Exception as e:
        print(f"  ✗ Futures validation: ERROR - {e}")
    
    print("Config verification complete!")

# =============================================================================
# USAGE EXAMPLES AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Uncomment to test configuration
    # test_dynamic_strategy_system()
    # test_futures_rollover_config()
    # verify_all_imports()
    pass
