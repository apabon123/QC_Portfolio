"""
Unified Data Access Interface for QuantConnect Three-Layer CTA Framework

This module provides a single, standardized interface for all data access patterns,
eliminating direct slice manipulation and leveraging QC's native capabilities.

Phase 3 Objectives:
- Unified data access interface for all components
- Standardized data retrieval patterns
- Enhanced QC native integration
- Streamlined futures chain handling
- Performance monitoring and optimization
"""

from AlgorithmImports import *
from typing import Dict, List, Optional, Any, Union
import pandas as pd

class UnifiedDataInterface:
    """
    Single point of data access for all algorithm components.
    Phase 3: Streamlined data access patterns.
    """
    
    def __init__(self, algorithm, config_manager, data_accessor=None, data_validator=None):
        """
        Initialize unified data interface.
        
        Args:
            algorithm: QC Algorithm instance
            config_manager: Centralized configuration manager
            data_accessor: QC Native Data Accessor (optional)
            data_validator: Data integrity checker (optional)
        """
        self.algorithm = algorithm
        self.config_manager = config_manager
        self.data_accessor = data_accessor
        self.data_validator = data_validator
        
        # Performance tracking
        self.access_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'validation_failures': 0,
            'slice_accesses': 0,
            'history_requests': 0,
            'chain_analyses': 0
        }
        
        # Data interface configuration
        self.config = self.config_manager.get_data_interface_config() if config_manager else {}
        
        self.algorithm.Log("UnifiedDataInterface: Initialized - Single point of data access")
    
    # ============================================================================
    # UNIFIED SLICE DATA ACCESS (Eliminates direct slice manipulation)
    # ============================================================================
    
    def get_slice_data(self, slice, symbols: List = None, data_types: List[str] = None) -> Dict[str, Any]:
        """
        Unified slice data access - replaces all direct slice manipulation.
        
        Args:
            slice: QC Slice object
            symbols: List of symbols to retrieve (None = all available)
            data_types: Data types to retrieve ['bars', 'chains', 'ticks', 'quotes']
            
        Returns:
            Dict with standardized data structure
        """
        try:
            self.access_stats['slice_accesses'] += 1
            self.access_stats['total_requests'] += 1
            
            # Validate slice
            if not slice or not hasattr(slice, 'Keys'):
                return {'valid': False, 'reason': 'invalid_slice', 'data': {}}
            
            # Determine symbols to process
            target_symbols = symbols if symbols else list(slice.Keys)
            if not target_symbols:
                return {'valid': False, 'reason': 'no_symbols', 'data': {}}
            
            # Determine data types to retrieve
            if not data_types:
                data_types = ['bars', 'chains']  # Default types for CTA strategies
            
            # Build unified data structure
            unified_data = {
                'timestamp': slice.Time if hasattr(slice, 'Time') else self.algorithm.Time,
                'symbols': {},
                'metadata': {
                    'total_symbols': len(target_symbols),
                    'data_types_requested': data_types,
                    'slice_has_data': True
                }
            }
            
            # Process each symbol
            valid_symbols = 0
            for symbol in target_symbols:
                symbol_data = self._extract_symbol_data(slice, symbol, data_types)
                if symbol_data['valid']:
                    unified_data['symbols'][symbol] = symbol_data
                    valid_symbols += 1
            
            # Update metadata
            unified_data['metadata']['valid_symbols'] = valid_symbols
            unified_data['valid'] = valid_symbols > 0
            
            # Validate data quality if validator available
            if self.data_validator:
                validated_slice = self.data_validator.validate_slice(slice)
                unified_data['metadata']['validation_passed'] = validated_slice is not None
                if not unified_data['metadata']['validation_passed']:
                    self.access_stats['validation_failures'] += 1
            
            return unified_data
            
        except Exception as e:
            self.algorithm.Error(f"UnifiedDataInterface: Error in get_slice_data: {str(e)}")
            return {'valid': False, 'reason': f'error: {str(e)}', 'data': {}}
    
    def _extract_symbol_data(self, slice, symbol, data_types: List[str]) -> Dict[str, Any]:
        """
        Extract all requested data types for a specific symbol.
        
        Args:
            slice: QC Slice object
            symbol: Symbol to extract data for
            data_types: List of data types to extract
            
        Returns:
            Dict with symbol-specific data
        """
        try:
            symbol_data = {
                'valid': False,
                'symbol': symbol,
                'timestamp': slice.Time if hasattr(slice, 'Time') else self.algorithm.Time,
                'data': {}
            }
            
            # Extract bar data
            if 'bars' in data_types and hasattr(slice, 'Bars') and symbol in slice.Bars:
                bar = slice.Bars[symbol]
                symbol_data['data']['bar'] = {
                    'open': float(bar.Open),
                    'high': float(bar.High),
                    'low': float(bar.Low),
                    'close': float(bar.Close),
                    'volume': int(bar.Volume),
                    'time': bar.Time,
                    'end_time': bar.EndTime
                }
                symbol_data['valid'] = True
            
            # Extract futures chain data
            if 'chains' in data_types and hasattr(slice, 'FuturesChains') and symbol in slice.FuturesChains:
                chain = slice.FuturesChains[symbol]
                symbol_data['data']['chain'] = self._extract_chain_data(chain)
                symbol_data['valid'] = True
            
            # Extract tick data (if requested)
            if 'ticks' in data_types and hasattr(slice, 'Ticks') and symbol in slice.Ticks:
                ticks = slice.Ticks[symbol]
                symbol_data['data']['ticks'] = [
                    {
                        'time': tick.Time,
                        'price': float(tick.Value),
                        'quantity': int(tick.Quantity) if hasattr(tick, 'Quantity') else 0
                    }
                    for tick in ticks
                ]
                symbol_data['valid'] = True
            
            # Extract quote data (if requested)
            if 'quotes' in data_types and hasattr(slice, 'QuoteBars') and symbol in slice.QuoteBars:
                quote = slice.QuoteBars[symbol]
                symbol_data['data']['quote'] = {
                    'bid': float(quote.Bid.Close),
                    'ask': float(quote.Ask.Close),
                    'bid_size': int(quote.LastBidSize),
                    'ask_size': int(quote.LastAskSize),
                    'time': quote.Time
                }
                symbol_data['valid'] = True
            
            # Add QC native security data
            if symbol in self.algorithm.Securities:
                security = self.algorithm.Securities[symbol]
                symbol_data['data']['security'] = {
                    'price': float(security.Price),
                    'has_data': security.HasData,
                    'is_tradable': security.IsTradable,
                    'market_hours': security.Exchange.Hours.IsOpen(self.algorithm.Time),
                    'mapped_symbol': str(security.Mapped) if hasattr(security, 'Mapped') else None
                }
                symbol_data['valid'] = True
            
            return symbol_data
            
        except Exception as e:
            self.algorithm.Error(f"UnifiedDataInterface: Error extracting data for {symbol}: {str(e)}")
            return {'valid': False, 'symbol': symbol, 'error': str(e), 'data': {}}
    
    def _extract_chain_data(self, chain) -> Dict[str, Any]:
        """
        Extract standardized futures chain data.
        
        Args:
            chain: QC FuturesChain object
            
        Returns:
            Dict with chain analysis data
        """
        try:
            if not chain or len(chain) == 0:
                return {'valid': False, 'reason': 'empty_chain', 'contracts': []}
            
            contracts = []
            total_volume = 0
            total_open_interest = 0
            
            for contract in chain:
                contract_data = {
                    'symbol': contract.Symbol,
                    'expiry': contract.Expiry,
                    'price': float(contract.LastPrice),
                    'volume': int(contract.Volume),
                    'open_interest': int(contract.OpenInterest),
                    'bid': float(contract.BidPrice),
                    'ask': float(contract.AskPrice),
                    'bid_size': int(contract.BidSize),
                    'ask_size': int(contract.AskSize),
                    'time': contract.Time
                }
                contracts.append(contract_data)
                total_volume += contract_data['volume']
                total_open_interest += contract_data['open_interest']
            
            # Sort by volume (most liquid first)
            contracts.sort(key=lambda x: x['volume'], reverse=True)
            
            return {
                'valid': True,
                'total_contracts': len(contracts),
                'total_volume': total_volume,
                'total_open_interest': total_open_interest,
                'most_liquid_contract': contracts[0] if contracts else None,
                'contracts': contracts
            }
            
        except Exception as e:
            self.algorithm.Error(f"UnifiedDataInterface: Error extracting chain data: {str(e)}")
            return {'valid': False, 'error': str(e), 'contracts': []}
    
    # ============================================================================
    # UNIFIED HISTORICAL DATA ACCESS
    # ============================================================================
    
    def get_history(self, symbol, periods: int, resolution=None, 
                   data_type: str = 'TradeBar') -> Optional[pd.DataFrame]:
        """
        Unified historical data access - standardized across all components.
        
        Args:
            symbol: Symbol to get history for
            periods: Number of periods
            resolution: Data resolution
            data_type: Type of historical data
            
        Returns:
            pandas.DataFrame or None
        """
        try:
            self.access_stats['history_requests'] += 1
            self.access_stats['total_requests'] += 1
            
            # Default to daily resolution if not specified
            if resolution is None:
                resolution = Resolution.Daily
            
            # Use data accessor if available (leverages QC native caching)
            if self.data_accessor:
                history = self.data_accessor.get_qc_native_history(symbol, periods, resolution)
                if history is not None and not history.empty:
                    self.access_stats['cache_hits'] += 1
                    return history
            
            # Fallback to direct QC History API
            history = self.algorithm.History(symbol, periods, resolution)
            if history is not None and not history.empty:
                return history
            
            return None
            
        except Exception as e:
            self.algorithm.Error(f"UnifiedDataInterface: History error for {symbol}: {str(e)}")
            return None
    
    def get_multiple_history(self, symbols: List, periods: int, resolution=None) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols efficiently.
        
        Args:
            symbols: List of symbols
            periods: Number of periods
            resolution: Data resolution
            
        Returns:
            Dict mapping symbols to their historical data
        """
        try:
            history_data = {}
            
            # Default to daily resolution if not specified
            if resolution is None:
                resolution = Resolution.Daily
            
            for symbol in symbols:
                history = self.get_history(symbol, periods, resolution)
                if history is not None:
                    history_data[symbol] = history
            
            return history_data
            
        except Exception as e:
            self.algorithm.Error(f"UnifiedDataInterface: Multiple history error: {str(e)}")
            return {}
    
    # ============================================================================
    # UNIFIED CURRENT PRICE ACCESS
    # ============================================================================
    
    def get_current_prices(self, symbols: List) -> Dict[str, float]:
        """
        Get current prices for multiple symbols using QC native properties.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Dict mapping symbols to current prices
        """
        try:
            prices = {}
            
            for symbol in symbols:
                if symbol in self.algorithm.Securities:
                    security = self.algorithm.Securities[symbol]
                    if security.HasData and security.Price > 0:
                        prices[symbol] = float(security.Price)
            
            return prices
            
        except Exception as e:
            self.algorithm.Error(f"UnifiedDataInterface: Error getting current prices: {str(e)}")
            return {}
    
    def get_symbol_info(self, symbol) -> Dict[str, Any]:
        """
        Get comprehensive symbol information using QC native properties.
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Dict with symbol information
        """
        try:
            if symbol not in self.algorithm.Securities:
                return {'valid': False, 'reason': 'symbol_not_found'}
            
            security = self.algorithm.Securities[symbol]
            
            return {
                'valid': True,
                'symbol': symbol,
                'price': float(security.Price),
                'has_data': security.HasData,
                'is_tradable': security.IsTradable,
                'market_hours_open': security.Exchange.Hours.IsOpen(self.algorithm.Time),
                'mapped_symbol': str(security.Mapped) if hasattr(security, 'Mapped') else None,
                'security_type': str(security.Type),
                'exchange': str(security.Exchange.Name),
                'last_update': self.algorithm.Time
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    # ============================================================================
    # UNIFIED FUTURES CHAIN ANALYSIS
    # ============================================================================
    
    def analyze_futures_chains(self, slice, symbols: List = None) -> Dict[str, Any]:
        """
        Unified futures chain analysis for all components.
        
        Args:
            slice: QC Slice object
            symbols: Symbols to analyze (None = all available)
            
        Returns:
            Dict with chain analysis results
        """
        try:
            self.access_stats['chain_analyses'] += 1
            
            if not hasattr(slice, 'FuturesChains') or not slice.FuturesChains:
                return {'valid': False, 'reason': 'no_chains_available', 'chains': {}}
            
            # Determine symbols to analyze
            target_symbols = symbols if symbols else list(slice.FuturesChains.keys())
            
            chain_analysis = {
                'valid': True,
                'timestamp': slice.Time if hasattr(slice, 'Time') else self.algorithm.Time,
                'total_chains': len(target_symbols),
                'chains': {},
                'summary': {
                    'total_contracts': 0,
                    'total_volume': 0,
                    'total_open_interest': 0,
                    'liquid_chains': 0
                }
            }
            
            for symbol in target_symbols:
                if symbol in slice.FuturesChains:
                    chain_data = self._extract_chain_data(slice.FuturesChains[symbol])
                    chain_analysis['chains'][symbol] = chain_data
                    
                    if chain_data['valid']:
                        chain_analysis['summary']['total_contracts'] += chain_data['total_contracts']
                        chain_analysis['summary']['total_volume'] += chain_data['total_volume']
                        chain_analysis['summary']['total_open_interest'] += chain_data['total_open_interest']
                        
                        # Consider liquid if has significant volume/OI
                        if (chain_data['total_volume'] > 100 or 
                            chain_data['total_open_interest'] > 1000):
                            chain_analysis['summary']['liquid_chains'] += 1
            
            return chain_analysis
            
        except Exception as e:
            self.algorithm.Error(f"UnifiedDataInterface: Chain analysis error: {str(e)}")
            return {'valid': False, 'error': str(e), 'chains': {}}
    
    # ============================================================================
    # PERFORMANCE MONITORING AND STATISTICS
    # ============================================================================
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get data access performance statistics."""
        try:
            total_requests = self.access_stats['total_requests']
            cache_hit_rate = (self.access_stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'total_requests': total_requests,
                'cache_hits': self.access_stats['cache_hits'],
                'cache_hit_rate_percent': round(cache_hit_rate, 1),
                'slice_accesses': self.access_stats['slice_accesses'],
                'history_requests': self.access_stats['history_requests'],
                'chain_analyses': self.access_stats['chain_analyses'],
                'validation_failures': self.access_stats['validation_failures'],
                'efficiency_rating': 'excellent' if cache_hit_rate > 80 else
                                   'good' if cache_hit_rate > 60 else
                                   'needs_improvement' if cache_hit_rate > 30 else
                                   'poor'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def log_performance_summary(self):
        """Log performance summary."""
        try:
            stats = self.get_performance_stats()
            
            self.algorithm.Log("=" * 60)
            self.algorithm.Log("UNIFIED DATA INTERFACE PERFORMANCE")
            self.algorithm.Log("=" * 60)
            self.algorithm.Log(f"Total Requests: {stats.get('total_requests', 0)}")
            self.algorithm.Log(f"Cache Hit Rate: {stats.get('cache_hit_rate_percent', 0):.1f}%")
            self.algorithm.Log(f"Slice Accesses: {stats.get('slice_accesses', 0)}")
            self.algorithm.Log(f"History Requests: {stats.get('history_requests', 0)}")
            self.algorithm.Log(f"Chain Analyses: {stats.get('chain_analyses', 0)}")
            self.algorithm.Log(f"Efficiency Rating: {stats.get('efficiency_rating', 'unknown')}")
            self.algorithm.Log("=" * 60)
            
        except Exception as e:
            self.algorithm.Error(f"UnifiedDataInterface: Error logging performance: {str(e)}")
    
    # ============================================================================
    # CONFIGURATION AND UTILITIES
    # ============================================================================
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update data interface configuration."""
        try:
            self.config.update(new_config)
            self.algorithm.Log("UnifiedDataInterface: Configuration updated")
        except Exception as e:
            self.algorithm.Error(f"UnifiedDataInterface: Config update error: {str(e)}")
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.access_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'validation_failures': 0,
            'slice_accesses': 0,
            'history_requests': 0,
            'chain_analyses': 0
        }
        self.algorithm.Log("UnifiedDataInterface: Performance statistics reset") 