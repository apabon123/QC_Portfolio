# Data Integrity Configuration for Futures Trading
# This configuration helps prevent backtest failures from bad data in priority 2 futures

DATA_INTEGRITY_CONFIG = {
    # AGGRESSIVE thresholds to prevent 'security does not have accurate price' errors
    'max_zero_price_streak': 2,      # Max consecutive zero prices before quarantine (reduced)
    'max_no_data_streak': 2,         # Max consecutive no-data before quarantine (NEW)
    'max_extreme_change': 0.30,      # Max price change (30%) before flagging (reduced)
    'quarantine_duration_days': 7,   # Days to keep symbol quarantined (increased)
    'extreme_change_threshold': 3,   # Number of extreme changes before quarantine (reduced)
    
    # NEW: CENTRALIZED DATA CACHE CONFIGURATION (solves concurrency issues)
    'cache_max_age_hours': 24,       # How long to keep cached data (hours)
    'cache_cleanup_frequency_hours': 6,  # How often to clean up old cache entries
    'max_cache_entries': 1000,       # Maximum number of cache entries before cleanup
    'cache_enabled': True,           # Enable/disable centralized caching
    'cache_debug_logging': False,    # Enable detailed cache debug logging
    
    # Price validation ranges for each futures contract
    # Format: ticker: (min_price, max_price)
    'price_ranges': {
        # Equity Index Futures
        'ES': (500, 10000),      # S&P 500 E-mini - wide range for market moves
        'NQ': (1000, 30000),     # Nasdaq-100 E-mini - wider for tech volatility
        'YM': (10000, 50000),    # Dow Jones E-mini - higher price range
        'RTY': (500, 5000),      # Russell 2000 E-mini
        
        # Interest Rate Futures
        'ZN': (50, 200),         # 10-Year Treasury Note - normal range
        'ZB': (70, 200),         # 30-Year Treasury Bond
        'ZF': (80, 150),         # 5-Year Treasury Note
        'ZT': (95, 110),         # 2-Year Treasury Note
        
        # Currency Futures
        '6E': (0.80, 1.50),      # Euro FX - EUR/USD range
        '6J': (0.004, 0.020),    # Japanese Yen - JPY/USD range
        '6B': (1.00, 2.00),      # British Pound - GBP/USD range
        '6A': (0.50, 1.20),      # Australian Dollar - AUD/USD range
        '6S': (0.50, 1.20),      # Swiss Franc - CHF/USD range
        '6C': (0.60, 1.20),      # Canadian Dollar - CAD/USD range
        
        # Commodity Futures
        'CL': (10, 200),         # Crude Oil WTI - wide range for oil volatility
        'GC': (800, 3000),       # Gold - precious metal range
        'SI': (10, 100),         # Silver - industrial/precious metal
        'HG': (1.50, 8.00),      # Copper - industrial metal
        'NG': (1.00, 20.00),     # Natural Gas - energy commodity
        
        # Agricultural Futures
        'ZC': (200, 1000),       # Corn
        'ZS': (600, 2000),       # Soybeans
        'ZW': (300, 1500),       # Wheat
        'ZL': (20, 80),          # Soybean Oil
        'ZM': (200, 600),        # Soybean Meal
        
        # Volatility Futures
        'VX': (10, 100),         # VIX Futures - volatility index
    },
    
    # Special handling for specific problematic contracts
    'special_handling': {
        # Priority 2 futures that commonly have data issues
        '6E': {
            'max_extreme_change': 0.30,  # More restrictive for FX
            'quarantine_duration_days': 3,  # Shorter quarantine
        },
        '6J': {
            'max_extreme_change': 0.30,  # More restrictive for FX
            'quarantine_duration_days': 3,  # Shorter quarantine
        },
        'CL': {
            'max_extreme_change': 0.60,  # Oil can be more volatile
            'quarantine_duration_days': 7,  # Longer quarantine for oil
        },
        'GC': {
            'max_extreme_change': 0.40,  # Gold can move significantly
            'quarantine_duration_days': 5,
        },
        'YM': {
            'max_extreme_change': 0.40,  # Dow can be volatile
            'quarantine_duration_days': 5,
        },
        'ZB': {
            'max_extreme_change': 0.25,  # Bonds are generally less volatile
            'quarantine_duration_days': 3,
        }
    },
    
    # Volume validation (minimum daily volume)
    'min_volume_thresholds': {
        'ES': 50000,    # High volume requirement
        'NQ': 30000,    # High volume requirement
        'ZN': 20000,    # Medium volume requirement
        'CL': 25000,    # Medium volume requirement
        'GC': 15000,    # Medium volume requirement
        '6E': 10000,    # Lower volume for FX
        '6J': 8000,     # Lower volume for FX
        'YM': 15000,    # Medium volume requirement
        'ZB': 15000,    # Medium volume requirement
    },
    
    # Logging configuration
    'logging': {
        'log_quarantine_actions': True,
        'log_invalid_data_frequency': 7,  # Log every 7 days
        'log_price_violations': True,
        'log_volume_violations': False,   # Don't spam volume logs
    }
}

def get_data_integrity_config():
    """Get the complete data integrity configuration."""
    return DATA_INTEGRITY_CONFIG

def get_symbol_config(ticker, config_key):
    """Get specific configuration for a symbol, with fallbacks."""
    config = DATA_INTEGRITY_CONFIG
    
    # Check for special handling first
    if ticker in config['special_handling']:
        special_config = config['special_handling'][ticker]
        if config_key in special_config:
            return special_config[config_key]
    
    # Fall back to general config
    if config_key in config:
        return config[config_key]
    
    # Default fallbacks
    defaults = {
        'max_extreme_change': 0.50,
        'quarantine_duration_days': 5,
        'max_zero_price_streak': 3,
    }
    
    return defaults.get(config_key, None)

def get_price_range(ticker):
    """Get price range for a specific ticker."""
    ranges = DATA_INTEGRITY_CONFIG['price_ranges']
    return ranges.get(ticker, (0.001, 1000000))  # Very wide default range

def get_min_volume(ticker):
    """Get minimum volume requirement for a ticker."""
    volumes = DATA_INTEGRITY_CONFIG['min_volume_thresholds']
    return volumes.get(ticker, 1000)  # Default minimum volume 