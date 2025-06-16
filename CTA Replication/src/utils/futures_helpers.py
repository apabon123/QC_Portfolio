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
                # Try modern AddFuture with proper parameters first
                market = ticker_markets.get(ticker, Market.CME)
                future = self.algorithm.AddFuture(
                    ticker=ticker,
                    resolution=Resolution.Daily,
                    market=market,
                    fillForward=True,
                    leverage=1.0,
                    extendedMarketHours=False,
                    dataMappingMode=DataMappingMode.OpenInterest,
                    dataNormalizationMode=DataNormalizationMode.BackwardsPanamaCanal,
                    contractDepthOffset=0
                )
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
                self.algorithm.Log(f"FuturesHelper: ✗ Failed modern AddFuture for {ticker}: {str(e)}")
                # Try basic AddFuture as fallback
                try:
                    future = self.algorithm.AddFuture(ticker)
                    symbol_str = str(future.Symbol)
                    symbols.append(symbol_str)
                    self.algorithm.Log(f"FuturesHelper: ✓ Added basic {ticker} -> {symbol_str}")
                    
                    # Set filter
                    try:
                        future.SetFilter(timedelta(0), timedelta(182))
                        self.algorithm.Log(f"FuturesHelper: ✓ Set filter for {ticker}")
                    except Exception as filter_e:
                        self.algorithm.Log(f"FuturesHelper: ⚠ Could not set filter for {ticker}: {str(filter_e)}")
                        
                except Exception as e2:
                    self.algorithm.Log(f"FuturesHelper: ✗ Failed basic AddFuture for {ticker}: {str(e2)}")
        
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
            # Only SetFilter is still available on the symbol object
            if hasattr(symbol, 'SetFilter'):
                symbol.SetFilter(timedelta(0), timedelta(settings['filter_days_out']))
                self.algorithm.Log(f"FuturesHelper: Set filter for {symbol}")
            
            # Note: SetDataMappingMode and SetDataNormalizationMode are deprecated
            # These settings must be passed to AddFuture() directly during symbol creation
            # This method now only handles post-creation configuration
            
            self.algorithm.Log(f"FuturesHelper: Configured available settings for {symbol}")
            self.algorithm.Log(f"FuturesHelper: Note - Data mapping/normalization modes must be set during AddFuture()")
            return True
            
        except Exception as e:
            self.algorithm.Error(f"FuturesHelper: Failed to configure {symbol}: {str(e)}")
            return False 