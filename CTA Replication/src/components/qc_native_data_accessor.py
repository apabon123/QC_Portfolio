"""
QC Native Data Accessor for QuantConnect Three-Layer CTA Framework

This module provides access to QuantConnect's native data caching and sharing
capabilities, replacing custom caching logic with QC's optimized systems.

Key Benefits:
- Leverages QC's built-in Securities[symbol].Cache
- Uses QC's automatic data sharing within algorithms
- Eliminates redundant custom caching logic
- Provides clean interface for strategy data access
"""

from AlgorithmImports import *
from typing import Dict, List, Optional, Any
import pandas as pd

class QCNativeDataAccessor:
    """
    Access QuantConnect's native cached data instead of custom caching.
    
    This class provides a clean interface to QC's built-in data caching
    and sharing capabilities, eliminating the need for custom caching logic.
    """
    
    def __init__(self, algorithm):
        """
        Initialize the QC native data accessor.
        
        Args:
            algorithm: QCAlgorithm instance
        """
        self.algorithm = algorithm
        self.access_stats = {
            'history_calls': 0,
            'cache_hits': 0,
            'security_access': 0,
            'price_requests': 0
        }
        
        self.algorithm.Log("QCNativeDataAccessor: Initialized - leveraging QC's native caching")
    
    def get_cached_price_data(self, symbol: object) -> Optional[Dict[str, Any]]:
        """
        Access QC's native cached price data for a symbol.
        
        Args:
            symbol: QC Symbol object
            
        Returns:
            Dict containing cached price data or None if unavailable
        """
        try:
            self.access_stats['price_requests'] += 1
            
            if symbol not in self.algorithm.Securities:
                return None
                
            security = self.algorithm.Securities[symbol]
            self.access_stats['security_access'] += 1
            
            # Use QC's native properties - these are already cached
            price_data = {
                'symbol': symbol,
                'current_price': security.Price,
                'has_data': security.HasData,
                'is_tradable': security.IsTradable,
                'volume': getattr(security, 'Volume', 0),
                'bid_price': getattr(security, 'BidPrice', 0),
                'ask_price': getattr(security, 'AskPrice', 0),
                'open': getattr(security, 'Open', 0),
                'high': getattr(security, 'High', 0),
                'low': getattr(security, 'Low', 0),
                'close': getattr(security, 'Close', 0),
                'last_update': self.algorithm.Time
            }
            
            # Try to access QC's cache if available
            if hasattr(security, 'Cache'):
                try:
                    # Access cached trade bar if available
                    cached_bar = security.Cache.GetData()
                    if cached_bar:
                        price_data['cached_bar'] = cached_bar
                        self.access_stats['cache_hits'] += 1
                except:
                    pass  # Cache access failed, use basic data
            
            return price_data
            
        except Exception as e:
            self.algorithm.Debug(f"QCNativeDataAccessor: Error getting cached price data for {symbol}: {str(e)}")
            return None
    
    def get_qc_native_history(self, symbol: object, periods: int, resolution = None) -> Optional[pd.DataFrame]:
        """
        Use QC's History API directly - let QC handle caching.
        
        Args:
            symbol: QC Symbol object
            periods: Number of periods to retrieve
            resolution: Data resolution
            
        Returns:
            pandas.DataFrame with historical data or None if unavailable
        """
        try:
            self.access_stats['history_calls'] += 1
            
            # Default to daily resolution if not specified
            if resolution is None:
                resolution = Resolution.Daily
            
            # Use QC's native History API - QC handles all caching internally
            history = self.algorithm.History(symbol, periods, resolution)
            
            if history is not None and not history.empty:
                return history
            else:
                return None
                
        except Exception as e:
            self.algorithm.Debug(f"QCNativeDataAccessor: Error getting history for {symbol}: {str(e)}")
            return None
    
    def get_current_mapped_contract(self, continuous_symbol: object) -> Optional[object]:
        """
        Get the currently mapped contract for a continuous futures symbol.
        
        Args:
            continuous_symbol: Continuous futures symbol
            
        Returns:
            Currently mapped contract symbol or None
        """
        try:
            if continuous_symbol not in self.algorithm.Securities:
                return None
            
            security = self.algorithm.Securities[continuous_symbol]
            
            # Get the currently mapped contract from QC
            if hasattr(security, 'Mapped'):
                return security.Mapped
            
            return None
            
        except Exception as e:
            self.algorithm.Debug(f"QCNativeDataAccessor: Error getting mapped contract for {continuous_symbol}: {str(e)}")
            return None
    
    def validate_symbol_data_quality(self, symbol: object) -> Dict[str, Any]:
        """
        Validate data quality using QC's native properties.
        
        Args:
            symbol: QC Symbol object
            
        Returns:
            Dict containing data quality metrics
        """
        try:
            if symbol not in self.algorithm.Securities:
                return {
                    'valid': False,
                    'reason': 'symbol_not_found',
                    'has_data': False,
                    'is_tradable': False,
                    'price_valid': False
                }
            
            security = self.algorithm.Securities[symbol]
            
            # Use QC's native validation properties
            has_data = security.HasData
            is_tradable = security.IsTradable
            price = security.Price
            price_valid = price > 0
            
            # Overall validation
            valid = has_data and is_tradable and price_valid
            
            return {
                'valid': valid,
                'reason': 'valid' if valid else self._get_validation_failure_reason(has_data, is_tradable, price_valid),
                'has_data': has_data,
                'is_tradable': is_tradable,
                'price_valid': price_valid,
                'current_price': price,
                'validation_time': self.algorithm.Time
            }
            
        except Exception as e:
            return {
                'valid': False,
                'reason': f'validation_error: {str(e)}',
                'has_data': False,
                'is_tradable': False,
                'price_valid': False
            }
    
    def _get_validation_failure_reason(self, has_data: bool, is_tradable: bool, price_valid: bool) -> str:
        """Get specific reason for validation failure."""
        reasons = []
        
        if not has_data:
            reasons.append('no_data')
        if not is_tradable:
            reasons.append('not_tradable')
        if not price_valid:
            reasons.append('invalid_price')
        
        return '_'.join(reasons) if reasons else 'unknown_failure'
    
    def get_futures_chain_data(self, slice: object, symbol: object) -> Optional[Any]:
        """
        Get futures chain data from slice - leverages QC's chain caching.
        
        Args:
            slice: Current data slice
            symbol: Continuous futures symbol
            
        Returns:
            FuturesChain object or None
        """
        try:
            if not slice or not hasattr(slice, 'FuturesChains'):
                return None
            
            if symbol in slice.FuturesChains:
                return slice.FuturesChains[symbol]
            
            return None
            
        except Exception as e:
            self.algorithm.Debug(f"QCNativeDataAccessor: Error getting futures chain for {symbol}: {str(e)}")
            return None
    
    def get_slice_data(self, slice: object, symbol: object) -> Optional[Dict[str, Any]]:
        """
        Extract data for a specific symbol from the current slice.
        
        Args:
            slice: Current data slice
            symbol: Symbol to extract data for
            
        Returns:
            Dict containing slice data for the symbol
        """
        try:
            slice_data = {
                'symbol': symbol,
                'timestamp': slice.Time,
                'has_data': False
            }
            
            # Check for trade bars
            if hasattr(slice, 'Bars') and symbol in slice.Bars:
                bar = slice.Bars[symbol]
                slice_data.update({
                    'has_data': True,
                    'open': bar.Open,
                    'high': bar.High,
                    'low': bar.Low,
                    'close': bar.Close,
                    'volume': bar.Volume,
                    'data_type': 'trade_bar'
                })
            
            # Check for quote bars
            elif hasattr(slice, 'QuoteBars') and symbol in slice.QuoteBars:
                quote_bar = slice.QuoteBars[symbol]
                slice_data.update({
                    'has_data': True,
                    'bid_open': quote_bar.Bid.Open,
                    'bid_high': quote_bar.Bid.High,
                    'bid_low': quote_bar.Bid.Low,
                    'bid_close': quote_bar.Bid.Close,
                    'ask_open': quote_bar.Ask.Open,
                    'ask_high': quote_bar.Ask.High,
                    'ask_low': quote_bar.Ask.Low,
                    'ask_close': quote_bar.Ask.Close,
                    'data_type': 'quote_bar'
                })
            
            # Check for ticks
            elif hasattr(slice, 'Ticks') and symbol in slice.Ticks:
                ticks = slice.Ticks[symbol]
                if ticks:
                    latest_tick = ticks[-1]
                    slice_data.update({
                        'has_data': True,
                        'price': latest_tick.Price,
                        'quantity': latest_tick.Quantity,
                        'tick_count': len(ticks),
                        'data_type': 'tick'
                    })
            
            return slice_data
            
        except Exception as e:
            self.algorithm.Debug(f"QCNativeDataAccessor: Error getting slice data for {symbol}: {str(e)}")
            return None
    
    def get_access_statistics(self) -> Dict[str, int]:
        """Get data access statistics for performance monitoring."""
        return self.access_stats.copy()
    
    def reset_statistics(self):
        """Reset access statistics."""
        self.access_stats = {
            'history_calls': 0,
            'cache_hits': 0,
            'security_access': 0,
            'price_requests': 0
        }
    
    def log_performance_summary(self):
        """Log performance summary of data access patterns."""
        stats = self.access_stats
        
        self.algorithm.Log("=" * 50)
        self.algorithm.Log("QC NATIVE DATA ACCESS PERFORMANCE")
        self.algorithm.Log("=" * 50)
        self.algorithm.Log(f"History API Calls: {stats['history_calls']}")
        self.algorithm.Log(f"Cache Hits: {stats['cache_hits']}")
        self.algorithm.Log(f"Security Access: {stats['security_access']}")
        self.algorithm.Log(f"Price Requests: {stats['price_requests']}")
        
        if stats['security_access'] > 0:
            cache_hit_rate = (stats['cache_hits'] / stats['security_access']) * 100
            self.algorithm.Log(f"Cache Hit Rate: {cache_hit_rate:.1f}%")
        
        self.algorithm.Log("=" * 50) 