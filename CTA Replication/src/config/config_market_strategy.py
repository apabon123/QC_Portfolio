# config_market_strategy.py - Market and Strategy Configuration
"""
Market and Strategy Configuration Module

This module contains all market-related and strategy-specific configurations:
- Asset categories and universe definitions
- Strategy parameters and settings
- Market-specific constraints
- Strategy allocation rules

Separated from execution/plumbing config for better organization.
"""

from AlgorithmImports import *

# =============================================================================
# ALGORITHM CORE SETTINGS
# =============================================================================
ALGORITHM_CONFIG = {
    'start_date': {'year': 2015, 'month': 1, 'day': 1},
    'end_date': {'year': 2020, 'month': 1, 'day': 1},
    'initial_cash': 10000000,               # $10M initial capital
    'resolution': 'Daily',                  # Daily resolution
    'timezone': 'America/New_York',         # Eastern time
    'benchmark': 'SPY',                     # Benchmark for comparison
    'brokerage_model': 'InteractiveBrokers', # Brokerage model
    'warmup_period_days': 100,              # Reduced warmup to get trading started (was 756)
}

# =============================================================================
# ASSET CATEGORIES AND UNIVERSE DEFINITION
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

# =============================================================================
# STRATEGY ASSET FILTERING
# =============================================================================
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
}

# =============================================================================
# STRATEGY DEFINITIONS AND PARAMETERS
# =============================================================================
STRATEGY_CONFIGS = {
    # Strategy 1: Kestner CTA (Academic trend following replication)
    'KestnerCTA': {
        'enabled': True,                        # Set to False to disable without code changes
        'module': 'src.strategies.kestner_cta_strategy',       # File name (without .py)
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
        'warmup_days': 400,                     # Strategy-specific warmup (based on max lookback of 52 weeks + buffer)
        'expected_sharpe': 0.8,                 # Historical Sharpe expectation
        'correlation_with_trend': 0.75,         # Expected correlation with SG Trend Index
    },
    
    # Strategy 2: MTUM CTA (Momentum futures adaptation)
    'MTUM_CTA': {
        'enabled': True,                       # Set to False to disable without code changes
        'module': 'src.strategies.mtum_cta_strategy',          # File name (without .py)
        'class': 'MTUMCTAStrategy',             # Class name in the file
        'name': 'MTUM_CTA',                     # Display name
        'description': 'Futures adaptation of MSCI USA Momentum methodology',
        
        # Strategy-specific parameters
        'momentum_lookbacks_months': [6, 12],   # 6-month and 12-month periods
        'volatility_lookback_days': 252 * 3,    # 3 years for volatility estimation (MSCI standard)
        'recent_exclusion_days': 22,         # 1 month exclusion period (MSCI standard)
        'signal_standardization_clip': 3.0,     # ±3 standard deviation clipping
        'target_volatility': 0.2,               # 20% individual strategy vol target
        'rebalance_frequency': 'monthly',       # Monthly rebalancing (first Friday)
        'max_position_weight': 0.6,             # 60% maximum single position
        'risk_free_rate': 0.02,                 # 2% annual risk-free rate assumption
        'long_short_enabled': True,             # Allow short positions (vs equity MTUM)
        'warmup_days': 252 * 3,                 # Strategy-specific warmup (3 years for volatility estimation)
        'expected_sharpe': 0.6,                 # Conservative Sharpe expectation
        'correlation_with_momentum': 0.7,       # Expected momentum factor correlation
    },
    
    # Strategy 3: HMM CTA (Hidden Markov Model regime detection)
    'HMM_CTA': {
        'enabled': True,                        # Enabled with QC plumbing fixes
        'module': 'src.strategies.hmm_cta_strategy',           # File name (without .py)
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
        'max_position_weight': 0.75,             # 60% maximum single position
        'regime_threshold': 0.45,               # 70% confidence threshold for regime identification
        'regime_persistence_days': 1,           # Require 3-day regime confirmation to avoid whipsawing
        'regime_smoothing_alpha': 0.3,          # Exponential smoothing factor for regime probabilities
        'retrain_frequency': 'monthly',         # Monthly model retraining for market adaptation
        'warmup_days': 80,                      # Strategy-specific warmup
        'expected_sharpe': 0.5,                 # Conservative expectation (regime strategy)
        'correlation_with_regime': 0.6,         # Expected regime factor correlation
    },
}

# =============================================================================
# STRATEGY ALLOCATION CONFIGURATION
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
        'KestnerCTA': 0.50,                      # 100% to trend following (single strategy test)
        'MTUM_CTA': 0.30,                        # 0% (disabled in STRATEGY_CONFIGS) 
        'HMM_CTA': 0.20,                         # 0% (disabled in STRATEGY_CONFIGS)
    },
    
    # Allocation Bounds (using strategy names from STRATEGY_CONFIGS)
    'allocation_bounds': {
        'KestnerCTA': {'min': 0.05, 'max': 1.00},   # Minimum 30%, Maximum 100%
        'MTUM_CTA': {'min': 0.00, 'max': 0.70},     # Minimum 0%, Maximum 70%
        'HMM_CTA': {'min': 0.00, 'max': 0.25},      # Minimum 0%, Maximum 25%
    }
}

# =============================================================================
# PORTFOLIO RISK MANAGEMENT CONFIGURATION  
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
# UNIVERSE CONFIGURATION (MARKET-SPECIFIC)
# =============================================================================
UNIVERSE_CONFIG = {
    # Universe Loading Configuration
    'loading': {
        'max_priority': 2,                      # Only load priority 1 and 2 symbols by default
        'include_expansion_candidates': True,   # Include expansion candidates in loading
        'priority_override': None,              # Set to specific priority to load only that priority
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

# =============================================================================
# SYSTEM INFORMATION
# =============================================================================
SYSTEM_CONFIG = {
    'name': "Three-Layer CTA Portfolio System",
    'version': "2.5",
    'description': "Multi-strategy futures trading with dynamic allocation, centralized risk management, dynamic strategy loading, and proper futures rollover handling",
    'last_updated': "2025-01-20"
}

# =============================================================================
# HELPER FUNCTIONS
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

def get_enabled_strategies():
    """Get list of enabled strategy names."""
    return [name for name, config in STRATEGY_CONFIGS.items() if config.get('enabled', False)]

def get_strategy_modules():
    """Get dictionary mapping strategy names to their module paths."""
    return {name: config['module'] for name, config in STRATEGY_CONFIGS.items() if config.get('enabled', False)} 