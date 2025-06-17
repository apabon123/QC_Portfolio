# asset_filter_manager.py - Asset Filtering for Multi-Strategy Systems
from AlgorithmImports import *

class AssetFilterManager:
    """
    Asset filtering manager for multi-strategy systems.
    Provides clean separation between strategy logic and asset universe management.
    """
    
    def __init__(self, algorithm, config_manager):
        self.algorithm = algorithm
        self.config_manager = config_manager
        
        # Load asset categories and filters from config
        self.asset_categories = config_manager.config.get('asset_categories', {})
        self.strategy_filters = config_manager.config.get('strategy_asset_filters', {})
        
        self.algorithm.Log(f"AssetFilterManager: Initialized with {len(self.asset_categories)} categories, {len(self.strategy_filters)} strategy filters")
    
    def get_symbols_for_strategy(self, strategy_name, all_symbols):
        """Filter symbols for a specific strategy using config-defined rules."""
        if strategy_name not in self.strategy_filters:
            self.algorithm.Log(f"ASSET FILTER: {strategy_name} - no filtering applied")
            return all_symbols
        
        filter_config = self.strategy_filters[strategy_name]
        allowed_symbols = set()
        
        # Add symbols from allowed categories
        for category in filter_config.get('allowed_categories', []):
            if category in self.asset_categories:
                allowed_symbols.update(self.asset_categories[category])
        
        # Add specifically allowed symbols
        allowed_symbols.update(filter_config.get('allowed_symbols', []))
        
        # Remove symbols from excluded categories
        for category in filter_config.get('excluded_categories', []):
            if category.endswith('*'):
                prefix = category[:-1]
                for cat_name, symbols in self.asset_categories.items():
                    if cat_name.startswith(prefix):
                        allowed_symbols -= set(symbols)
            elif category in self.asset_categories:
                allowed_symbols -= set(self.asset_categories[category])
        
        # Remove specifically excluded symbols
        allowed_symbols -= set(filter_config.get('excluded_symbols', []))
        
        # Convert symbols to tickers for comparison (handle Symbol objects)
        all_tickers = []
        for sym in all_symbols:
            if hasattr(sym, 'Value'):
                ticker = self._extract_ticker_from_symbol(sym)
                all_tickers.append((ticker, sym))
            else:
                all_tickers.append((sym, sym))
        
        # Only return symbols that exist in universe and are allowed
        final_symbols = []
        for ticker, original_symbol in all_tickers:
            if ticker in allowed_symbols:
                final_symbols.append(original_symbol)
        
        return final_symbols
    
    def _extract_ticker_from_symbol(self, symbol):
        """Extract ticker from a QuantConnect Symbol object."""
        try:
            if hasattr(symbol, 'ID') and hasattr(symbol.ID, 'Symbol'):
                return symbol.ID.Symbol
            return str(symbol).split(' ')[0]
        except:
            return str(symbol)
    
    def get_filter_reason(self, strategy_name):
        """Get the reasoning behind a strategy's asset filter."""
        if strategy_name in self.strategy_filters:
            return self.strategy_filters[strategy_name].get('reason', 'No reason specified')
        return 'No filter applied'
    
    def log_filtering_results(self, strategy_name, all_symbols, filtered_symbols):
        """Log filtering results for debugging."""
        if len(filtered_symbols) < len(all_symbols):
            all_tickers = [self._extract_ticker_from_symbol(s) for s in all_symbols]
            filtered_tickers = [self._extract_ticker_from_symbol(s) for s in filtered_symbols]
            removed_tickers = set(all_tickers) - set(filtered_tickers)
            
            self.algorithm.Log(f"ASSET FILTER: {strategy_name} excluded {removed_tickers}")
            self.algorithm.Log(f"ASSET FILTER: {strategy_name} will trade {filtered_tickers}")
            self.algorithm.Log(f"ASSET FILTER: Reason - {self.get_filter_reason(strategy_name)}")
        else:
            self.algorithm.Log(f"ASSET FILTER: {strategy_name} - no filtering applied") 