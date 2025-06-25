# config_market_strategy.py - Clean Market and Strategy Configuration
"""
Clean Market and Strategy Configuration Module

Simplified, well-structured configuration with:
- Clear futures tiers (Tier 1, 2, 3)
- Easy future addition
- No repetition
- Simple strategy setup
"""

from AlgorithmImports import *

# =============================================================================
# ALGORITHM SETTINGS
# =============================================================================
ALGORITHM_CONFIG = {
    'start_date': {'year': 2017, 'month': 1, 'day': 1},
    'end_date': {'year': 2020, 'month': 1, 'day': 1},
    'initial_cash': 10000000,               # $10M initial capital
    'resolution': 'Daily',                  # Daily resolution
    'timezone': 'America/New_York',         # Eastern time
    'benchmark': 'SPY',                     # Benchmark for comparison
    'brokerage_model': 'InteractiveBrokers', # Brokerage model
    
    # Warm-up Configuration
    'warmup': {
        'enabled': True,
        'method': 'time_based',
        'auto_calculate_period': True,
        'minimum_days': 15,
        'resolution': 'Daily',
        'buffer_multiplier': 1.1,           # 10% buffer
        'validate_indicators_ready': True,
        'log_warmup_progress': True,
    },
}

# =============================================================================
# FUTURES UNIVERSE - CLEAN TIER SYSTEM
# =============================================================================
FUTURES_TIERS = {
    # Tier 1: Most Liquid (Core positions)
    'tier_1': {
        'ES': {'name': 'E-mini S&P 500', 'category': 'equity'},
        'CL': {'name': 'Crude Oil', 'category': 'commodity'},
        'GC': {'name': 'Gold', 'category': 'commodity'},
    },
    
    # Tier 2: Very Liquid (Expansion)
    'tier_2': {
        'NQ': {'name': 'Nasdaq 100', 'category': 'equity'},
        'ZN': {'name': '10-Year Treasury', 'category': 'rates'},
        '6E': {'name': 'Euro FX', 'category': 'fx'},
    },
    
    # Tier 3: Liquid (Additional diversification)
    'tier_3': {
        'YM': {'name': 'Dow Jones', 'category': 'equity'},
        'RTY': {'name': 'Russell 2000', 'category': 'equity'},
        'ZB': {'name': '30-Year Treasury', 'category': 'rates'},
        'ZF': {'name': '5-Year Treasury', 'category': 'rates'},
        '6J': {'name': 'Japanese Yen', 'category': 'fx'},
        '6B': {'name': 'British Pound', 'category': 'fx'},
        'SI': {'name': 'Silver', 'category': 'commodity'},
        'HG': {'name': 'Copper', 'category': 'commodity'},
    }
}

# =============================================================================
# UNIVERSE CONFIGURATION - BRIDGING NEW TIER SYSTEM WITH OLD STRUCTURE
# =============================================================================

# New Clean Configuration (Easy to modify)
UNIVERSE_SELECTION = {
    'active_tiers': ['tier_1'],           # Which tiers to trade
    'additional_futures': [],                       # Add specific futures: ['RTY', 'ZF']
    'excluded_futures': ['6J'],                     # Exclude specific futures
    'description': 'Currently trading Tier 1 futures only (ES, CL, GC) - 3 futures total'
}

# Helper function (defined early to avoid circular dependency)
def get_active_futures():
    """Get list of futures to trade based on configuration."""
    active_futures = []
    
    # Add futures from active tiers
    for tier in UNIVERSE_SELECTION['active_tiers']:
        if tier in FUTURES_TIERS:
            active_futures.extend(FUTURES_TIERS[tier].keys())
    
    # Add additional futures
    active_futures.extend(UNIVERSE_SELECTION['additional_futures'])
    
    # Remove excluded futures
    active_futures = [f for f in active_futures if f not in UNIVERSE_SELECTION['excluded_futures']]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_futures = []
    for future in active_futures:
        if future not in seen:
            seen.add(future)
            unique_futures.append(future)
    
    return unique_futures

# Auto-generate old structure for compatibility
def _generate_universe_config():
    """Generate the old universe config structure from the new tier system."""
    active_futures = get_active_futures()
    
    # Create the old nested structure that the system expects
    universe_config = {
        'loading': {
            'max_priority': 2,
            'include_expansion_candidates': False,
            'exclude_problematic_symbols': UNIVERSE_SELECTION['excluded_futures'],
        },
        
        'simple_selection': {
            'enabled': False,
            'futures_list': active_futures,
            'description': "Auto-generated from tier system"
        },
        
        'futures': {},
        'expansion_candidates': {}
    }
    
    # Build futures config by category
    for tier_name, tier_futures in FUTURES_TIERS.items():
        for symbol, config in tier_futures.items():
            if symbol in active_futures:
                category = config['category']
                
                # Create category if it doesn't exist
                if category not in universe_config['futures']:
                    universe_config['futures'][category] = {}
                
                # Add future to category
                universe_config['futures'][category][symbol] = {
                    'name': config['name'],
                    'category': f'futures_{category}',
                    'priority': 1 if tier_name == 'tier_1' else 2,
                    'min_volume': 50000,
                }
    
    return universe_config

# Generate the old structure automatically
UNIVERSE_CONFIG = _generate_universe_config()

# =============================================================================
# STRATEGY CONFIGURATIONS - SIMPLIFIED
# =============================================================================
STRATEGY_CONFIGS = {
    'MTUM_CTA': {
        'enabled': True,
        'module': 'src.strategies.mtum_cta_strategy',
        'class': 'MTUMCTAStrategy',
        'name': 'MTUM_CTA',
        'description': 'MTUM Futures Adaptation - Preserves core MTUM methodology with futures market innovations',
        
        # Core MTUM Parameters (Preserved from Equity Version)
        'target_volatility': 0.2,              # 20% vol target
        'rebalance_frequency': 'monthly',      # Monthly rebalancing
        'max_position_weight': 0.6,            # 60% max position
        'warmup_days': 1092,                   # 3 years for volatility calculation (156 weeks * 7 days)
        
        # MTUM Methodology (Unchanged)
        'momentum_lookbacks_months': [6, 12],  # Dual-period analysis (6m & 12m)
        'volatility_lookback_days': 1092,      # 3-year weekly volatility (156 weeks * 7 days)
        'signal_standardization_clip': 3.0,    # ±3 std dev clipping (MTUM standard)
        'risk_free_rate': 0.02,                # 2% risk-free rate adjustment
        
        # FUTURES MARKET ADAPTATIONS (New)
        'momentum_threshold': 0.2,              # Absolute momentum threshold (futures innovation)
        'long_short_enabled': True,             # Enable short selling (futures advantage)
        'signal_strength_weighting': True,      # Signal-based position sizing (not market cap)
        
        # Legacy Parameters (Kept for Compatibility)
        'recent_exclusion_days': 22,           # 1 month recent exclusion
    },
    
    'KestnerCTA': {
        'enabled': False,
        'module': 'src.strategies.kestner_cta_strategy',
        'class': 'KestnerCTAStrategy',
        'name': 'KestnerCTA',
        'description': 'Academic trend following',
        
        # Core Parameters
        'target_volatility': 0.2,              # 20% vol target
        'rebalance_frequency': 'weekly',       # Weekly rebalancing
        'max_position_weight': 0.6,            # 60% max position
        'warmup_days': 100,                    # Reduced for testing
        
        # Strategy-specific
        'momentum_lookbacks': [16, 32, 52],    # Weekly periods
        'volatility_lookback_days': 63,        # Volatility window
        'signal_cap': 1.0,
    },
    
    'HMM_CTA': {
        'enabled': False,
        'module': 'src.strategies.hmm_cta_strategy',
        'class': 'HMMCTAStrategy',
        'name': 'HMM_CTA',
        'description': 'Hidden Markov Model regime-based CTA',
        
        # Core Parameters
        'target_volatility': 0.2,              # 20% vol target
        'rebalance_frequency': 'weekly',       # Weekly rebalancing
        'max_position_weight': 0.6,            # 60% max position
        'warmup_days': 200,                    # Need good history for regime detection
        
        # HMM-specific parameters
        'n_components': 3,                     # Number of regimes (trending up, ranging, trending down)
        'returns_window': 252,                 # 1 year return window for regime analysis
        'volatility_lookback_days': 252,       # 1 year volatility/correlation window
        'n_iter': 100,                         # Max iterations for model fitting
        'random_state': 42,                    # For reproducibility
        'regime_threshold': 0.60,              # Minimum confidence for regime signal
        'regime_persistence_days': 3,          # Days regime must persist
        'regime_smoothing_alpha': 0.3,         # Exponential smoothing factor
    },
    
    'SimpleMA': {
        'enabled': False,
        'module': 'src.strategies.simple_ma_cross_strategy',
        'class': 'SimpleMACrossStrategy',
        'name': 'SimpleMA',
        'description': 'Simple moving average crossover for testing',
        
        # Core Parameters
        'target_volatility': 0.15,             # 15% vol target
        'rebalance_frequency': 'daily',        # Daily rebalancing
        'max_position_weight': 0.5,            # 50% max position
        'warmup_days': 50,                     # Simple MA warmup
        
        # MA-specific parameters
        'fast_ma_period': 10,                  # Fast MA period
        'slow_ma_period': 30,                  # Slow MA period
        'volatility_lookback_days': 50,        # Short volatility/correlation window for daily strategy
        'signal_threshold': 0.01,              # Min signal threshold
    }
}

# =============================================================================
# LAYER 2: STRATEGY ALLOCATION CONFIGURATION
# =============================================================================
ALLOCATION_CONFIG = {
    # Basic Allocation Settings
    'method': 'fixed',                          # 'fixed' or 'dynamic'
    'allocations': {
        'MTUM_CTA': 1.0,                        # 100% to MTUM for testing
        'KestnerCTA': 0.0,                      # 0% (disabled)
    },
    
    # Dynamic Allocation Parameters (Layer 2 - Sharpe-based allocation only)
    'lookback_days': 63,                        # 3 months for Sharpe calculation
    'min_track_record_days': 21,                # Minimum days before changing allocations
    'rebalance_frequency': 'weekly',            # How often to update allocations
    'allocation_smoothing': 0.7,                # Smoothing factor (70% old, 30% new)
    
    # Strategy Availability Handling
    'availability_handling': {
        'mode': 'persistence',                  # 'persistence' or 'reallocate'
        'persistence_threshold': 0.1,           # Min allocation for available strategies (10%)
        'emergency_reallocation_ratio': 0.5,    # Split 50/50 between available/unavailable in emergency
        'log_unavailable_reasons': True,        # Log WHY strategies are unavailable
        'max_consecutive_unavailable_days': 7,  # Alert after N days unavailable
    }
}

# =============================================================================
# ASSET DEFAULTS: FALLBACK VOLATILITIES & CORRELATIONS
# =============================================================================
ASSET_DEFAULTS = {
    # Asset Volatility Fallbacks (annualized)
    'volatilities': {
        'equities': 0.20,        # 20% - ES, NQ, YM
        'bonds': 0.08,           # 8%  - ZN, ZB  
        'fx': 0.12,              # 12% - 6E, 6J, 6B
        'commodities': 0.3,     # 25% - CL, GC (This drives the 14.9% calculation)
        'default': 0.20,         # 20% - Unknown assets
    },
    
    # Asset Correlation Fallbacks (for portfolio construction)
    'correlations': {
        'within_commodity': 0.30,     # CL vs GC correlation
        'within_equity': 0.85,        # ES vs NQ correlation  
        'within_fixed_income': 0.70,  # ZN vs ZB correlation
        'cross_asset_default': 0.15,  # Default cross-asset correlation
        'equity_commodity': 0.20,     # Equity vs Commodity
        'bond_commodity': -0.05,      # Bond vs Commodity (slight negative)
        'equity_bond': -0.10,         # Equity vs Bond (slight negative)
        'fx_commodity': 0.25,         # FX vs Commodity
        'fx_equity': 0.30,            # FX vs Equity
        'fx_bond': 0.05,              # FX vs Bond
    },
    
    # Data Quality Minimums
    'min_data_points': 30,           # Minimum data points for reliable correlation
    'correlation_lookback_days': 126, # 1 year for asset correlations
}

# =============================================================================
# LAYER 3: PORTFOLIO RISK MANAGEMENT CONFIGURATION  
# =============================================================================
RISK_CONFIG = {
    # Portfolio Volatility Targeting (This drives the 14.9% -> 60.0% scaling)
    # Formula: scaling_factor = target_portfolio_vol / estimated_portfolio_vol
    # Example: 60.0% / 14.9% = 4.02x scaling factor
    'target_portfolio_vol': 0.4,               # 40% portfolio volatility
    'max_leverage_multiplier': 8.0,           # 8x max leverage for futures
    'min_notional_exposure': 3.0,              # 3x minimum notional exposure
    
    # Position Limits
    'max_single_position': 3.0,                # 300% max position (30% real with 10x leverage)
    
    # Emergency Stops
    'max_drawdown_stop': 0.75,                 # 75% max drawdown
    'daily_stop_loss': 0.2,                    # 20% daily stop
    
    # Strategy Correlation Parameters (Layer 3 - for scaling portfolio vol)
    'use_correlation': True,                    # Use strategy correlations for portfolio vol calculation
    'correlation_lookback_days': 126,           # 6 months for strategy correlation calculation
    'vol_estimation_days': 126,                # 1 year for volatility estimation
}

# =============================================================================
# LOGGING - SIMPLIFIED
# =============================================================================
LOGGING_CONFIG = {
    'global_level': 'INFO',
    'components': {
        'main_algorithm': 'INFO',
        'mtum_cta': 'INFO',
        'kestner_cta': 'DEBUG',
        'execution': 'DEBUG',
        'risk': 'INFO',
        'universe': 'WARNING',
        'config': 'ERROR',
    }
}

# =============================================================================
# ADDITIONAL HELPER FUNCTIONS
# =============================================================================

def get_futures_by_category():
    """Get futures organized by category."""
    by_category = {}
    active_futures = get_active_futures()
    
    for tier_name, tier_futures in FUTURES_TIERS.items():
        for symbol, config in tier_futures.items():
            if symbol in active_futures:
                category = config['category']
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(symbol)
    
    return by_category

def get_enabled_strategies():
    """Get list of enabled strategy names."""
    return [name for name, config in STRATEGY_CONFIGS.items() if config.get('enabled', False)]

def add_future_to_tier(symbol, name, category, tier='tier_2'):
    """Helper to add a new future to a tier."""
    if tier not in FUTURES_TIERS:
        FUTURES_TIERS[tier] = {}
    
    FUTURES_TIERS[tier][symbol] = {
        'name': name,
        'category': category
    }
    
    print(f"Added {symbol} ({name}) to {tier}")

# =============================================================================
# EASY CONFIGURATION EXAMPLES
# =============================================================================
"""
EASY CONFIGURATION EXAMPLES:

1. Trade only Tier 1 futures:
   UNIVERSE_SELECTION['active_tiers'] = ['tier_1']

2. Trade Tier 1 + specific futures:
   UNIVERSE_SELECTION['active_tiers'] = ['tier_1']
   UNIVERSE_SELECTION['additional_futures'] = ['NQ', 'ZN']

3. Trade all tiers except specific futures:
   UNIVERSE_SELECTION['active_tiers'] = ['tier_1', 'tier_2', 'tier_3']
   UNIVERSE_SELECTION['excluded_futures'] = ['6J', 'SI']

4. Add a new future:
   add_future_to_tier('RTY', 'Russell 2000', 'equity', 'tier_2')

5. Enable multiple strategies:
   STRATEGY_CONFIGS['KestnerCTA']['enabled'] = True
   ALLOCATION_CONFIG['allocations'] = {'MTUM_CTA': 0.7, 'KestnerCTA': 0.3}

CURRENT ACTIVE FUTURES: {get_active_futures()}
""" 