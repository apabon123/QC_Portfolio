"""
Helper utilities for futures contract management.
"""

from AlgorithmImports import *
from datetime import timedelta

class FuturesHelpers:
    """Helper class for futures contract operations."""
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
    
    def add_core_futures(self, tickers):
        """Add core futures contracts with error handling."""
        symbols = []
        
        ticker_markets = {
            'ES': Market.CME,
            'NQ': Market.CME,  
            'ZN': Market.CBOT,
            'ZC': Market.CBOT,
            'ZS': Market.CBOT,
            'ZW': Market.CBOT,
            'GC': Market.COMEX,
            'SI': Market.COMEX,
            'CL': Market.NYMEX,
            'NG': Market.NYMEX
        }
        
        for ticker in tickers:
            try:
                # Try basic AddFuture first
                future = self.algorithm.AddFuture(ticker)
                symbol_str = str(future.Symbol)
                symbols.append(symbol_str)
                self.algorithm.Log(f"FuturesHelper: ✓ Added {ticker} -> {symbol_str}")
                
                # Set filter to avoid Universe.cs issues
                try:
                    future.SetFilter(timedelta(0), timedelta(182))  # 6 months filter
                    self.algorithm.Log(f"FuturesHelper: ✓ Set filter for {ticker}")
                except Exception as filter_e:
                    self.algorithm.Log(f"FuturesHelper: ⚠ Could not set filter for {ticker}: {str(filter_e)}")
                    
            except Exception as e:
                self.algorithm.Log(f"FuturesHelper: ✗ Failed basic AddFuture for {ticker}: {str(e)}")
                # Try with market specification
                try:
                    market = ticker_markets.get(ticker, Market.CME)
                    future = self.algorithm.AddFuture(ticker, Resolution.Daily, market)
                    symbol_str = str(future.Symbol)
                    symbols.append(symbol_str)
                    self.algorithm.Log(f"FuturesHelper: ✓ Added with market {ticker} -> {symbol_str}")
                    
                    # Set filter
                    try:
                        future.SetFilter(timedelta(0), timedelta(182))
                        self.algorithm.Log(f"FuturesHelper: ✓ Set filter for {ticker}")
                    except Exception as filter_e:
                        self.algorithm.Log(f"FuturesHelper: ⚠ Could not set filter for {ticker}: {str(filter_e)}")
                        
                except Exception as e2:
                    self.algorithm.Log(f"FuturesHelper: ✗ Failed with market for {ticker}: {str(e2)}")
        
        self.algorithm.Log(f"FuturesHelper: Successfully added {len(symbols)}/{len(tickers)} futures")
        return symbols
    
    def configure_futures_settings(self, symbol, settings=None):
        """Configure futures-specific settings."""
        if settings is None:
            settings = {
                'filter_days_out': 182,
                'data_mapping_mode': DataMappingMode.OpenInterest,
                'data_normalization_mode': DataNormalizationMode.BackwardsPanamaCanal
            }
        
        try:
            if hasattr(symbol, 'SetFilter'):
                symbol.SetFilter(timedelta(0), timedelta(settings['filter_days_out']))
            
            if hasattr(symbol, 'SetDataMappingMode'):
                symbol.SetDataMappingMode(settings['data_mapping_mode'])
            
            if hasattr(symbol, 'SetDataNormalizationMode'):
                symbol.SetDataNormalizationMode(settings['data_normalization_mode'])
            
            self.algorithm.Log(f"FuturesHelper: Configured settings for {symbol}")
            return True
            
        except Exception as e:
            self.algorithm.Error(f"FuturesHelper: Failed to configure {symbol}: {str(e)}")
            return False 