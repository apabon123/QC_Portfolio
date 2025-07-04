# config_execution_plumbing.py - Execution and Plumbing Configuration
"""
Execution and Plumbing Configuration Module

This module contains all execution-related and technical infrastructure configurations:
- QuantConnect native features settings
- Order execution and transaction management
- Futures rollover and data management
- Monitoring and reporting settings
- System constraints and limits

Separated from market/strategy config for better organization.
"""

from AlgorithmImports import *

# =============================================================================
# QUANTCONNECT NATIVE FEATURES CONFIGURATION
# =============================================================================
QC_NATIVE_CONFIG = {
    # Portfolio Performance Tracking (Use QC's built-in instead of custom)
    'portfolio_tracking': {
        'use_qc_statistics': True,                  # Use QC's built-in Statistics object
        'use_qc_portfolio': True,                   # Use QC's Portfolio object for holdings
        'custom_performance_tracking': False,       # Disable our custom performance tracking
        'track_benchmark': True,                    # Track performance vs benchmark
        'enable_runtime_statistics': True,         # Enable runtime statistics
        'log_portfolio_value': True,               # Log portfolio value changes
        'log_holdings': False,                     # Don't spam logs with holdings (available via QC)
        'performance_frequency': 'daily',          # How often to log performance
    },
    
    # Order and Transaction Management (Use QC's built-in)
    'order_management': {
        'use_qc_transactions': True,               # Use QC's Transactions object
        'use_qc_order_events': True,              # Use QC's OrderEvent handling
        'custom_fill_tracking': False,            # Disable our custom fill tracking
        'custom_trade_history': False,            # Disable our custom trade history
        'log_order_events': True,                 # Log important order events
        'log_filled_orders': True,                # Log filled orders
        'log_rejected_orders': True,              # Log rejected orders
        'track_order_tags': True,                 # Track order tags for attribution
    },
    
    # Data Management (Use QC's built-in)
    'data_management': {
        'use_qc_history': True,                   # Use QC's History() method
        'use_qc_consolidators': True,             # Use QC's data consolidators
        'custom_data_validation': True,          # Keep our business logic validation
        'enable_data_normalization': True,       # Use QC's data normalization
        'cache_historical_data': False,          # Let QC handle caching
    },
    
    # Futures Management (Use QC's built-in rollover)
    'futures_management': {
        'use_qc_rollover': True,                  # Use QC's automatic rollover
        'use_qc_mapping': True,                   # Use QC's data mapping
        'custom_rollover_tracking': False,       # Disable our custom rollover logic
        'log_rollover_events': True,             # Log when QC rolls contracts
        'track_rollover_costs': True,            # Track rollover transaction costs
        'validate_rollover_symbols': True,       # Validate symbols after rollover
    },
    
    # Algorithm Framework Integration
    'algorithm_framework': {
        'use_portfolio_construction_model': False,  # We have custom portfolio construction
        'use_risk_management_model': False,         # We have custom risk management
        'use_execution_model': False,               # We have custom execution
        'use_alpha_model': False,                   # We have custom signal generation
        'enable_insights': False,                   # Don't use Insights framework
    }
}

# =============================================================================
# FUTURES CONFIGURATION - CRITICAL FOR ROLLOVER BEHAVIOR
# =============================================================================
FUTURES_CONFIG = {
    # AddFuture Parameters - Now configurable instead of hardcoded
    'add_future_params': {
        'resolution': 'Daily',                              # Resolution.Daily
        'fill_forward': True,                               # Fill missing data points
        'leverage': 1.0,                                    # Conservative leverage
        'extended_market_hours': False,                     # Standard market hours for most futures
        'data_mapping_mode': 'OpenInterest',                # DataMappingMode.OpenInterest - WHEN to rollover
        'data_normalization_mode': 'BackwardsRatio',        # DataNormalizationMode.BackwardsRatio - HOW to adjust prices
        'contract_depth_offset': 0,                         # 0 = front month, 1 = back month
    },
    
    # TWO-CONTRACT ARCHITECTURE - Configurable contract depth per futures type
    'contract_depth_config': {
        # Equity Index Futures (simple term structure, 2 contracts sufficient)
        'equity_index': {
            'contracts': ['ES', 'NQ', 'YM', 'RTY'],
            'depth': 2,  # Front month + Second month
            'description': 'Equity index futures with simple term structure'
        },
        
        # Energy Futures (moderate term structure, 2-3 contracts)
        'energy': {
            'contracts': ['CL', 'NG', 'RB', 'HO'],
            'depth': 2,  # Front month + Second month
            'description': 'Energy futures with moderate term structure'
        },
        
        # Precious Metals (simple term structure, 2 contracts)
        'metals': {
            'contracts': ['GC', 'SI', 'PL', 'PA'],
            'depth': 2,  # Front month + Second month
            'description': 'Precious metals futures with simple term structure'
        },
        
        # Agricultural Futures (seasonal patterns, 3 contracts may be useful)
        'agricultural': {
            'contracts': ['ZC', 'ZS', 'ZW', 'ZL', 'ZM'],
            'depth': 3,  # Front + Second + Third month
            'description': 'Agricultural futures with seasonal patterns'
        },
        
        # Interest Rate Futures (complex term structure, 2 contracts sufficient for most)
        'rates': {
            'contracts': ['ZN', 'ZB', 'ZF', 'ZT'],
            'depth': 2,  # Front month + Second month
            'description': 'Interest rate futures with yield curve structure'
        },
        
        # Currency Futures (simple term structure, 2 contracts)
        'currency': {
            'contracts': ['6E', '6J', '6B', '6A', '6C', '6S'],
            'depth': 2,  # Front month + Second month
            'description': 'Currency futures with simple term structure'
        },
        
        # Volatility Futures (COMPLEX TERM STRUCTURE - needs multiple contracts!)
        'volatility': {
            'contracts': ['VX'],
            'depth': 6,  # Front + 5 additional months for term structure
            'description': 'Volatility futures with complex contango/backwardation term structure'
        },
        
        # Default configuration for unspecified contracts
        'default': {
            'depth': 2,  # Conservative default: front + second month
            'description': 'Default configuration for unspecified futures contracts'
        }
    },
    
    # Contract Filter Parameters
    'contract_filter': {
        'min_days_out': 0,                                  # Include contracts expiring in 0+ days
        'max_days_out': 182,                                # Include contracts expiring within 182 days (6 months)
    },
    
    # Legacy string format (for backward compatibility)
    'data_mapping_mode': 'DataMappingMode.OpenInterest',      # WHEN to rollover: based on open interest
    'data_normalization_mode': 'DataNormalizationMode.BackwardsRatio',  # HOW to adjust prices
    'contract_depth_offset': 0,                               # 0 = front month, 1 = back month
    'extended_market_hours': True,                            # Include extended hours data
    'resolution': 'Resolution.Daily',                         # Data resolution
    'fill_forward': True,                                     # Fill missing data points
    'contract_filter_days': 182,                             # Include contracts expiring within 182 days
}

# =============================================================================
# EXECUTION CONFIGURATION
# =============================================================================
EXECUTION_CONFIG = {
    'order_type': "market",                     # Market orders for immediacy
    'min_trade_value': 1000,                    # Minimum $1,000 trade size
    'min_weight_change': 0.01,                  # 1% minimum weight change to trade
    'max_single_order_value': 50000000,         # $50M maximum single order
    'max_single_position': 10.0,                # 1000% maximum single position
    'max_portfolio_turnover': 2.0,              # 200% maximum daily turnover
    
    # FUTURES ROLLOVER CONFIGURATION - CRITICAL FOR PROPER ROLLOVER HANDLING
    'rollover': {
        'enabled': True,                        # Enable automatic rollover handling
        'rollover_method': 'OnSymbolChangedEvents',  # Use QuantConnect's built-in system
        'immediate_reopen': True,               # KEY: Force immediate reopen - NO MARKET GAPS
        'order_type': 'market',                 # Market orders for rollovers (immediate execution)
        'max_rollover_slippage': 0.001,         # 0.1% maximum acceptable slippage
        'emergency_liquidation': True,          # Liquidate if rollover fails
        'rollover_tag_prefix': 'ROLLOVER',      # Tag prefix for rollover trades
        'position_tolerance': 0.01,             # 1% tolerance for position matching
        'retry_attempts': 3,                    # Number of retry attempts for failed rollovers
        'retry_delay_seconds': 5,               # Delay between retry attempts
        'log_rollover_events': True,            # Log all rollover events for debugging
        'log_rollover_details': True,           # Log detailed rollover execution
        'log_rollover_prices': True,            # Log rollover contract prices
        'validate_rollover_contracts': True,    # Validate new contracts before trading
        'fallback_to_rebalance': False,         # KEY: Don't wait for rebalance - execute immediately
        'max_rollover_history': 100,            # Maximum rollover events to track
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
            'rollover_logging': "summary",             # Summary rollover logging (not detailed)
    
    # Reporting
    'daily_reports': True,                     # Generate daily performance reports
    'weekly_reports': True,                    # Generate weekly allocation reports
    'monthly_reports': True,                   # Generate monthly risk reports
    'rollover_reports': True,                  # Generate rollover analysis reports
    'skip_reports_during_warmup': True,        # Skip monthly reports during warmup to reduce log noise
    'warmup_progress_frequency_days': 90,      # Log warmup progress every N days (0 = never)
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
# CONTINUOUS CONTRACT MANAGEMENT CONFIGURATION
# =============================================================================
CONTINUOUS_CONTRACTS_CONFIG = {
    # Fallback Mapping Configuration
    'enable_fallback_mapping': True,           # Enable fallback when continuous contracts fail
    'max_days_without_data': 3,                # Maximum days to wait before activating fallback
    'prefer_front_month': True,                # Prefer front month contracts as fallback
    'validate_underlying_data': True,          # Validate underlying contract data quality
    'log_mapping_changes': True,               # Log when mappings change
    'retry_failed_continuous': True,           # Retry failed continuous contracts
    
    # Symbol Resolution
    'auto_discover_underlying': True,          # Automatically discover underlying contracts
    'update_mappings_on_securities_change': True,  # Update mappings when securities change
    'validate_mappings_frequency': 'daily',    # How often to validate mappings
    
    # Fallback Strategy
    'fallback_strategy': 'best_data_quality',  # Strategy for selecting fallback contracts
    'min_underlying_data_history': 30,         # Minimum days of data for underlying contracts
    'max_underlying_contracts_per_continuous': 5,  # Maximum underlying contracts to track
    
    # Monitoring and Alerting
    'alert_on_continuous_failure': True,       # Alert when continuous contracts fail
    'alert_on_fallback_activation': True,      # Alert when fallback is activated
    'track_fallback_performance': True,        # Track performance impact of fallbacks
    'log_level': 'INFO'                        # Logging level for continuous contract manager
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_futures_rollover_config():
    """Get futures-specific rollover configuration"""
    return {
        'data_mapping_mode': FUTURES_CONFIG['data_mapping_mode'],
        'data_normalization_mode': FUTURES_CONFIG['data_normalization_mode'],
        'rollover_method': EXECUTION_CONFIG['rollover_config']['rollover_method'],
        'rollover_enabled': EXECUTION_CONFIG['rollover_config']['enabled'],
        'log_rollovers': EXECUTION_CONFIG['rollover_config']['log_rollover_events'],
        'use_continuous_contracts': EXECUTION_CONFIG['futures_execution']['use_continuous_contracts'],
    }

def validate_futures_config():
    """Validate that futures configuration is properly set up"""
    issues = []
    
    # Check required futures config
    required_fields = ['data_mapping_mode', 'data_normalization_mode', 'contract_depth_offset']
    
    for field in required_fields:
        if field not in FUTURES_CONFIG:
            issues.append(f"Missing required futures config: {field}")
    
    # Check rollover config
    rollover_config = EXECUTION_CONFIG['rollover_config']
    if not rollover_config.get('enabled', False):
        issues.append("Rollover handling is disabled - this will cause position loss on expiry!")
    
    return issues 