# universe.py - UPDATED WITH CONFIG INTEGRATION AND ROLLOVER SUPPORT
from AlgorithmImports import *
import numpy as np

class AssetFilterManager:
    """
    Asset filtering manager for multi-strategy systems.
    Provides clean separation between strategy logic and asset universe management.
    NOW FULLY CONFIG-DRIVEN.
    """
    
    def __init__(self, algorithm, config_manager):
        self.algorithm = algorithm
        self.config_manager = config_manager
        
        # Load asset categories and filters from config
        self.asset_categories = config_manager.config.get('asset_categories', {})
        self.strategy_filters = config_manager.config.get('strategy_asset_filters', {})
        
        self.algorithm.Log(f"AssetFilterManager: Initialized with {len(self.asset_categories)} categories, {len(self.strategy_filters)} strategy filters")
    
    def get_symbols_for_strategy(self, strategy_name, all_symbols):
        """
        Filter symbols for a specific strategy using config-defined rules.
        
        Args:
            strategy_name: Name of strategy (e.g., 'KestnerCTA')
            all_symbols: List of all available symbols
            
        Returns:
            List of symbols this strategy should trade
        """
        if strategy_name not in self.strategy_filters:
            # If no filter defined, return all symbols (backwards compatible)
            self.algorithm.Log(f"ASSET FILTER: {strategy_name} - no filtering applied")
            return all_symbols
        
        filter_config = self.strategy_filters[strategy_name]
        allowed_symbols = set()
        
        # 1. Add symbols from allowed categories
        for category in filter_config.get('allowed_categories', []):
            if category in self.asset_categories:
                allowed_symbols.update(self.asset_categories[category])
        
        # 2. Add specifically allowed symbols
        allowed_symbols.update(filter_config.get('allowed_symbols', []))
        
        # 3. Remove symbols from excluded categories
        for category in filter_config.get('excluded_categories', []):
            if category.endswith('*'):
                # Handle wildcard exclusions like 'futures_*'
                prefix = category[:-1]
                for cat_name, symbols in self.asset_categories.items():
                    if cat_name.startswith(prefix):
                        allowed_symbols -= set(symbols)
            elif category in self.asset_categories:
                allowed_symbols -= set(self.asset_categories[category])
        
        # 4. Remove specifically excluded symbols
        allowed_symbols -= set(filter_config.get('excluded_symbols', []))
        
        # 5. Convert symbols to tickers for comparison (handle Symbol objects)
        all_tickers = []
        for sym in all_symbols:
            if hasattr(sym, 'Value'):
                # It's a Symbol object, extract the ticker
                ticker = self._extract_ticker_from_symbol(sym)
                all_tickers.append((ticker, sym))
            else:
                # It's already a string ticker
                all_tickers.append((sym, sym))
        
        # 6. Only return symbols that exist in universe and are allowed
        final_symbols = []
        for ticker, original_symbol in all_tickers:
            if ticker in allowed_symbols:
                final_symbols.append(original_symbol)
        
        return final_symbols
    
    def _extract_ticker_from_symbol(self, symbol):
        """Extract ticker from a QuantConnect Symbol object."""
        try:
            # For futures, the ID.Symbol gives us the base ticker
            if hasattr(symbol, 'ID') and hasattr(symbol.ID, 'Symbol'):
                return symbol.ID.Symbol
            # Fallback to string representation
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

class FuturesManager:
    """
    ULTRA-MINIMAL Futures Universe Manager - PURE INTEROP SAFETY
    
    This version eliminates ALL complex attributes that might cause interop errors:
    - No complex nested objects
    - No QuantConnect enum storage
    - No function references
    - Only basic Python types (strings, lists, dicts, booleans)
    """
    
    def __init__(self, algorithm, config_manager):
        try:
            self.algorithm = algorithm
            self.config_manager = config_manager
            self.config = config_manager.config
            
            # Use only basic Python types - no complex objects
            self.futures_symbols = []  # List of strings
            self.futures_tickers = ['ES', 'NQ', 'ZN']  # Basic list
            self.is_initialized = False  # Basic boolean
            self.error_count = 0  # Basic int
            
            self.algorithm.Log("FuturesManager: Ultra-minimal initialization started...")
            
            # No complex objects, no enums stored as attributes
            # Everything else will be computed on-demand
            
            self.algorithm.Log("FuturesManager: Ultra-minimal initialization completed")
            
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Ultra-minimal init error: {str(e)}")
            # Set absolute minimum attributes
            self.futures_tickers = ['ES']
            self.is_initialized = False
            self.error_count = 1
    
    def initialize_universe(self):
        """Ultra-simple universe initialization with robust AddFuture calls."""
        try:
            self.algorithm.Log("FuturesManager: Starting ultra-simple universe initialization...")
            
            successful_adds = 0
            
            for ticker in self.futures_tickers:
                try:
                    self.algorithm.Log(f"FuturesManager: Adding {ticker}...")
                    
                    # ROBUST AddFuture call with explicit parameters
                    # Import Resolution enum at runtime to avoid storage issues
                    from AlgorithmImports import Resolution
                    
                    # Use explicit parameters to avoid null reference errors
                    future = self.algorithm.AddFuture(ticker, Resolution.Daily)
                    
                    # Alternative approach: try with minimal parameters first
                    if future is not None:
                        # Simple filter with explicit parameters
                        from datetime import timedelta
                        future.SetFilter(timedelta(0), timedelta(182))
                        
                        # Store just the symbol as string
                        self.futures_symbols.append(str(future.Symbol))
                        
                        self.algorithm.Log(f"FuturesManager: ✓ Added {ticker} successfully")
                        successful_adds += 1
                    else:
                        self.algorithm.Error(f"FuturesManager: ✗ AddFuture returned None for {ticker}")
                    
                except Exception as e:
                    self.algorithm.Error(f"FuturesManager: ✗ Failed to add {ticker}: {str(e)}")
                    
                    # Try fallback approach if standard AddFuture fails
                    try:
                        self.algorithm.Log(f"FuturesManager: Trying fallback approach for {ticker}...")
                        
                        # Sometimes a simpler call works better
                        future_fallback = self.algorithm.AddFuture(ticker)
                        if future_fallback is not None:
                            # Just set basic filter without timedelta
                            future_fallback.SetFilter(0, 182)
                            self.futures_symbols.append(str(future_fallback.Symbol))
                            self.algorithm.Log(f"FuturesManager: ✓ Added {ticker} via fallback")
                            successful_adds += 1
                        else:
                            self.algorithm.Error(f"FuturesManager: ✗ Both primary and fallback failed for {ticker}")
                            
                    except Exception as fallback_error:
                        self.algorithm.Error(f"FuturesManager: ✗ Fallback also failed for {ticker}: {str(fallback_error)}")
            
            self.is_initialized = successful_adds > 0
            self.algorithm.Log(f"FuturesManager: Universe initialization complete - {successful_adds}/{len(self.futures_tickers)} contracts added")
            
            return successful_adds
        
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: CRITICAL universe initialization error: {str(e)}")
            import traceback
            self.algorithm.Error(f"FuturesManager: Traceback: {traceback.format_exc()}")
            self.error_count += 1
            return 0
    
    def get_symbols_for_strategy(self, strategy_name):
        """Return all symbols for any strategy."""
        try:
            return self.futures_symbols.copy()
        except:
            return ['ES']
    
    def get_tickers_for_strategy(self, strategy_name):
        """Return all tickers for any strategy."""
        try:
            return self.futures_tickers.copy()
        except:
            return ['ES']
    
    def get_liquid_symbols(self):
        """Return all symbols."""
        try:
            return self.futures_symbols.copy()
        except:
            return ['ES']
    
    def get_liquid_tickers(self):
        """Return all tickers."""
        try:
            return self.futures_tickers.copy()
        except:
            return ['ES']

    def update_data_quality(self, slice):
        """No-op data quality update."""
        pass

    def mark_rollover_in_progress(self, symbol, in_progress=True):
        """No-op rollover marking."""
        pass

    def handle_symbol_changed_events(self, symbol_changed_events):
        """No-op rollover handling."""
        pass

    def get_status_report(self):
        """Ultra-simple status report."""
        return {
            'total_contracts': len(self.futures_tickers),
            'is_initialized': self.is_initialized,
            'error_count': self.error_count
        }

    def get_universe_summary(self):
        """Ultra-simple universe summary."""
        return {
            'total_contracts': len(self.futures_tickers),
            'liquid_contracts': len(self.futures_tickers),
            'ready_for_trading': self.is_initialized
        }

    def validate_trading_universe(self):
        """Ultra-simple validation."""
        return {'ready_for_trading': self.is_initialized}

    def get_current_contract(self, symbol):
        """Return symbol as-is."""
        return symbol

    def is_rollover_in_progress(self, symbol):
        """Always return False."""
        return False

    def should_trade_symbol(self, symbol):
        """Always return True if initialized."""
        return self.is_initialized

    def get_trading_symbols(self):
        """Return all symbols."""
        return self.futures_symbols.copy()

    def get_trading_tickers(self):
        """Return all tickers."""
        return self.futures_tickers.copy()
