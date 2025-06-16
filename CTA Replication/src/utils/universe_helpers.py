"""
Helper utilities for universe management and initialization.
"""

class UniverseHelpers:
    """
    Universe management utilities with centralized configuration.
    CRITICAL: All configuration comes through centralized config manager only.
    """
    
    def __init__(self, algorithm, config_manager):
        """
        Initialize Universe Helpers with centralized configuration.
        CRITICAL: NO direct config access - all through config manager.
        """
        self.algorithm = algorithm
        self.config_manager = config_manager
        
        try:
            # Get universe configuration through centralized manager ONLY
            self.universe_config = self.config_manager.get_universe_config()
            
            # Extract futures tickers from validated configuration
            self.futures_tickers = []
            for priority_group in self.universe_config.values():
                for symbol_config in priority_group:
                    self.futures_tickers.append(symbol_config['ticker'])
            
            if not self.futures_tickers:
                error_msg = "No futures tickers found in universe configuration"
                self.algorithm.Error(f"CONFIG ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            self.algorithm.Log(f"UniverseHelpers: Initialized with {len(self.futures_tickers)} futures tickers")
            
        except Exception as e:
            error_msg = f"CRITICAL ERROR initializing UniverseHelpers: {str(e)}"
            self.algorithm.Error(error_msg)
            raise ValueError(error_msg)
    
    def initialize_universe(self):
        """
        Initialize universe data using centralized configuration.
        CRITICAL: Uses only validated configuration from centralized manager.
        """
        try:
            # Use futures tickers from validated configuration
            universe_data = {
                'tickers': self.futures_tickers,
                'symbols': [],  # Will be populated by futures helper
                'status': 'initialized',
                'is_ready': True,
                'config': {
                    'filter_days_out': 182,
                    'data_mapping_mode': 'OpenInterest',
                    'data_normalization_mode': 'BackwardsPanamaCanal'
                }
            }
            
            self.algorithm.Log(f"UniverseHelper: Initialized universe with {len(self.futures_tickers)} tickers")
            return universe_data
            
        except Exception as e:
            error_msg = f"CRITICAL ERROR initializing universe: {str(e)}"
            self.algorithm.Error(error_msg)
            raise ValueError(error_msg)
    
    def get_universe_summary(self, universe_data):
        """Get universe summary for logging."""
        try:
            return {
                'total_tickers': len(universe_data.get('tickers', [])),
                'active_symbols': len(universe_data.get('symbols', [])),
                'status': universe_data.get('status', 'unknown'),
                'is_ready': universe_data.get('is_ready', False)
            }
        except Exception as e:
            self.algorithm.Error(f"UniverseHelper: Error getting summary: {str(e)}")
            return {'error': str(e)}
    
    def validate_universe_data(self, universe_data):
        """Validate universe data structure."""
        required_keys = ['tickers', 'symbols', 'status', 'is_ready']
        
        for key in required_keys:
            if key not in universe_data:
                self.algorithm.Error(f"UniverseHelper: Missing required key: {key}")
                return False
        
        if not isinstance(universe_data['tickers'], list):
            self.algorithm.Error("UniverseHelper: 'tickers' must be a list")
            return False
        
        if len(universe_data['tickers']) == 0:
            self.algorithm.Error("UniverseHelper: No tickers specified")
            return False
        
        self.algorithm.Log("UniverseHelper: Universe data validation passed")
        return True 