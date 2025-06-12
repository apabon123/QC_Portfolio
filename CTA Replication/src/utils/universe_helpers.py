"""
Helper utilities for universe management and initialization.
"""

class UniverseHelpers:
    """Helper class for universe operations."""
    
    def __init__(self, algorithm, config_manager):
        self.algorithm = algorithm
        self.config_manager = config_manager
    
    def initialize_universe(self):
        """Initialize universe data structure without complex objects."""
        try:
            # Get basic configuration
            config = self.config_manager.load_and_validate_config()
            futures_tickers = config.get('futures_universe', ['ES', 'NQ', 'ZN'])
            
            # Create simple universe data structure
            universe_data = {
                'tickers': futures_tickers,
                'symbols': [],  # Will be populated by futures helper
                'status': 'initialized',
                'is_ready': True,
                'config': {
                    'filter_days_out': 182,
                    'data_mapping_mode': 'OpenInterest',
                    'data_normalization_mode': 'BackwardsPanamaCanal'
                }
            }
            
            self.algorithm.Log(f"UniverseHelper: Initialized universe with {len(futures_tickers)} tickers")
            return universe_data
            
        except Exception as e:
            self.algorithm.Error(f"UniverseHelper: Failed to initialize universe: {str(e)}")
            # Return minimal fallback
            return {
                'tickers': ['ES', 'NQ', 'ZN'],
                'symbols': [],
                'status': 'fallback',
                'is_ready': True
            }
    
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