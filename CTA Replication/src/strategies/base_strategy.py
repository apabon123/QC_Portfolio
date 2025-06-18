# base_strategy.py - ENHANCED BASE STRATEGY CLASS WITH COMMON FUNCTIONALITY

from AlgorithmImports import *
import numpy as np
from abc import ABC, abstractmethod
from collections import deque

class BaseStrategy(ABC):
    """
    ENHANCED Base Strategy Class - Captures ALL common functionality
    
    This eliminates massive code duplication found across Kestner, MTUM, and HMM strategies.
    All strategies should inherit from this class for consistency and maintainability.
    """
    
    def __init__(self, algorithm, config_manager, strategy_name):
        """
        Initialize base strategy with centralized configuration.
        CRITICAL: All configuration MUST come through config_manager.
        NO fallback logic allowed.
        """
        self.algorithm = algorithm
        self.config_manager = config_manager
        self.name = strategy_name
        
        # Set up optimized data access
        self.data_accessor = getattr(algorithm, 'data_accessor', None)
        # Symbols are now managed directly by QC's native methods
        
        try:
            # Get strategy configuration through centralized manager ONLY
            self.config = self.config_manager.get_strategy_config(strategy_name)
            self.algorithm.Log(f"{self.name}: Configuration loaded successfully")
            
            # Validate strategy is enabled
            if not self.config.get('enabled', False):
                error_msg = f"Strategy {self.name} is not enabled in configuration"
                self.algorithm.Error(f"STRATEGY ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            # Initialize strategy-specific components
            self._initialize_strategy_components()
            
        except Exception as e:
            error_msg = f"CRITICAL ERROR initializing strategy {self.name}: {str(e)}"
            self.algorithm.Error(error_msg)
            # Do not use fallback - fail fast
            raise ValueError(error_msg)
    
    def _initialize_strategy_components(self):
        """Initialize strategy-specific components. Override in subclasses."""
        pass
    
    # ============================================================================
    # COMMON CONFIGURATION LOADING (identical across all strategies)
    # ============================================================================
    
    # SECURITY: No fallback configuration allowed
    # All configuration MUST come through centralized config manager
    
    # No _build_config_dict or _load_fallback_config methods allowed
    # All configuration access through self.config only
    # Configuration validation happens in __init__ method, not at class level
    
    def _log_initialization_summary(self):
        """Common initialization logging."""
        if hasattr(self, 'config'):
            target_vol = self.config.get('target_volatility', 0)
            max_pos = self.config.get('max_position_weight', 0)
            self.algorithm.Log(f"{self.name}: Initialized - Target Vol: {target_vol:.1%}, Max Position: {max_pos:.1%}")
    
    # ============================================================================
    # COMMON INTERFACE METHODS (identical implementations across strategies)
    # ============================================================================
    
    def update(self, slice_data):
        """Common update method found in all strategies."""
        try:
            if not self._validate_slice_data_centralized(slice_data):
                return
            
            self.last_update_time = slice_data.Time if hasattr(slice_data, 'Time') else self.algorithm.Time
            
            # Update symbol data - common pattern across strategies
            for symbol, bar in slice_data.Bars.items():
                if symbol in self.symbol_data:
                    symbol_data = self.symbol_data[symbol]
                    if hasattr(symbol_data, 'OnDataConsolidated'):
                        symbol_data.OnDataConsolidated(None, bar)
                    elif hasattr(symbol_data, 'OnConsolidated'):
                        symbol_data.OnConsolidated(None, bar)
                
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in update: {str(e)}")
    
    def generate_targets(self, slice=None):
        """Common target generation pattern found in all strategies."""
        try:
            if not self.should_rebalance(self.algorithm.Time):
                return self.current_targets
            
            signals = self.generate_signals(slice)
            if not signals:
                return {}
            
            # Common processing pipeline found in all strategies
            targets = self._apply_position_limits(signals)
            targets = self._apply_volatility_targeting(targets)
            targets = self._validate_trade_sizes(targets)
            
            self.current_targets = targets
            self.last_rebalance_date = self.algorithm.Time.date()
            self.total_rebalances += 1
            
            return targets
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error generating targets: {str(e)}")
            return {}
    
    def get_exposure(self):
        """Common exposure calculation found in all strategies."""
        try:
            if not self.current_targets:
                return {'gross_exposure': 0.0, 'net_exposure': 0.0, 'long_exposure': 0.0,
                       'short_exposure': 0.0, 'num_positions': 0}
            
            long_exp = sum(max(0, w) for w in self.current_targets.values())
            short_exp = sum(min(0, w) for w in self.current_targets.values())
            
            return {
                'gross_exposure': long_exp + abs(short_exp),
                'net_exposure': long_exp + short_exp,
                'long_exposure': long_exp,
                'short_exposure': abs(short_exp),
                'num_positions': len([w for w in self.current_targets.values() if abs(w) > 0.01])
            }
        except Exception as e:
            return {'gross_exposure': 0.0, 'net_exposure': 0.0, 'long_exposure': 0.0,
                   'short_exposure': 0.0, 'num_positions': 0}
    
    @property
    def IsAvailable(self):
        """Common availability check found in all strategies."""
        if not hasattr(self, 'config') or not self.config.get('enabled', True):
            return False
        
        if not self.symbol_data:
            return False
        
        ready_symbols = sum(1 for sd in self.symbol_data.values() 
                           if hasattr(sd, 'IsReady') and sd.IsReady)
        return ready_symbols >= max(1, len(self.symbol_data) * 0.5)
    
    # ============================================================================
    # COMMON PROCESSING METHODS (found in all strategies)
    # ============================================================================
    
    def _validate_slice_data_centralized(self, slice_data):
        """Common validation using DataIntegrityChecker - found in all strategies."""
        try:
            if hasattr(self.algorithm, 'data_integrity_checker') and self.algorithm.data_integrity_checker:
                return self.algorithm.data_integrity_checker.validate_slice(slice_data) is not None
            
            if not slice_data or not slice_data.Bars:
                return False
            
            for symbol, bar in slice_data.Bars.items():
                if symbol in self.algorithm.Securities:
                    security = self.algorithm.Securities[symbol]
                    if security.HasData and security.IsTradable and security.Price > 0:
                        return True
            return False
            
        except Exception as e:
            return False
    
    def _apply_position_limits(self, signals):
        """Common position limiting found in all strategies."""
        try:
            max_weight = self.config.get('max_position_weight', 0.5)
            return {symbol: max(-max_weight, min(max_weight, signal)) 
                   for symbol, signal in signals.items()}
        except Exception as e:
            return signals
    
    def _apply_volatility_targeting(self, signals):
        """Common volatility targeting found in all strategies."""
        try:
            if not signals:
                return signals
            
            target_vol = self.config.get('target_volatility', 0.15)
            portfolio_vol = self._calculate_portfolio_volatility(signals)
            
            if portfolio_vol > 0:
                vol_scalar = max(0.1, min(10.0, target_vol / portfolio_vol))
                return {symbol: signal * vol_scalar for symbol, signal in signals.items()}
            
            return signals
        except Exception as e:
            return signals
    
    def _calculate_portfolio_volatility(self, weights):
        """Common portfolio volatility calculation."""
        try:
            if not weights:
                return 0.0
            
            total_vol = 0.0
            count = 0
            
            for symbol in weights.keys():
                if symbol in self.symbol_data:
                    symbol_data = self.symbol_data[symbol]
                    if hasattr(symbol_data, 'GetVolatility'):
                        vol = symbol_data.GetVolatility()
                        if vol and vol > 0:
                            total_vol += vol
                            count += 1
            
            if count > 0:
                avg_vol = total_vol / count
                return avg_vol * np.sqrt(len(weights)) * 0.7  # Diversification benefit
            
            return 0.20  # Default assumption
        except Exception as e:
            return 0.20
    
    def _validate_trade_sizes(self, targets):
        """Common trade size validation found in all strategies."""
        try:
            if not targets:
                return targets
            
            min_threshold = self.config.get('min_weight_threshold', 0.01)
            min_trade_value = self.config.get('min_trade_value', 1000)
            portfolio_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
            
            validated = {}
            for symbol, weight in targets.items():
                if abs(weight) >= min_threshold and abs(weight) * portfolio_value >= min_trade_value:
                    validated[symbol] = weight
            
            return validated
        except Exception as e:
            return targets
    
    # ============================================================================
    # COMMON ONSECURITIESCHANGED (identical across strategies)
    # ============================================================================
    
    def OnSecuritiesChanged(self, changes):
        """Common OnSecuritiesChanged handling found in all strategies."""
        try:
            for security in changes.AddedSecurities:
                symbol = security.Symbol
                symbol_str = str(symbol)
                
                if symbol_str.startswith('/') or symbol_str.startswith('futures/'):
                    if symbol not in self.symbol_data:
                        self.algorithm.Log(f"{self.name}: Initializing SymbolData for: {symbol}")
                        try:
                            self.symbol_data[symbol] = self._create_symbol_data(symbol)
                        except Exception as e:
                            self.algorithm.Error(f"{self.name}: Failed to create SymbolData: {e}")
                else:
                    self.algorithm.Log(f"{self.name}: Skipping rollover contract: {symbol_str}")
            
            for security in changes.RemovedSecurities:
                symbol = security.Symbol
                if symbol in self.symbol_data:
                    try:
                        self.symbol_data[symbol].Dispose()
                        del self.symbol_data[symbol]
                        self.algorithm.Log(f"{self.name}: Removed SymbolData for: {symbol}")
                    except Exception as e:
                        self.algorithm.Error(f"{self.name}: Error removing SymbolData: {e}")
                        
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in OnSecuritiesChanged: {str(e)}")
    
    # ============================================================================
    # CENTRALIZED HISTORY API ACCESS (solves concurrency issues)
    # ============================================================================
    
    def get_qc_history(self, symbol, periods, resolution=Resolution.Daily):
        """OPTIMIZED History API usage via QC Native Data Accessor."""
        try:
            # PRIORITY 1: Use QC native data accessor (leverages QC's built-in caching)
            if self.data_accessor:
                history = self.data_accessor.get_qc_native_history(symbol, periods, resolution)
                return history if history is not None and not history.empty else None
            
            # FALLBACK: Direct QC History method (QC handles caching internally)
            history = self.algorithm.History(symbol, periods, resolution)
            return history if history is not None and not history.empty else None
                
        except Exception as e:
            self.algorithm.Log(f"{self.name}: History API error for {symbol}: {str(e)}")
            return None
    
    # ============================================================================
    # COMMON PERFORMANCE METHODS
    # ============================================================================
    
    def get_performance_metrics(self):
        """Common performance metrics calculation."""
        try:
            exposure = self.get_exposure()
            return {
                'total_rebalances': self.total_rebalances,
                'trades_executed': self.trades_executed,
                'current_exposure': exposure,
                'is_available': self.IsAvailable,
                'symbols_ready': len([sd for sd in self.symbol_data.values() 
                                    if hasattr(sd, 'IsReady') and sd.IsReady]),
                'total_symbols': len(self.symbol_data),
                'last_rebalance': self.last_rebalance_date
            }
        except Exception as e:
            return {'error': str(e)}
    
    def log_status(self):
        """Common status logging."""
        try:
            metrics = self.get_performance_metrics()
            exposure = metrics.get('current_exposure', {})
            
            self.algorithm.Log(f"{self.name} STATUS: Available={metrics.get('is_available', False)}, "
                              f"Symbols={metrics.get('symbols_ready', 0)}/{metrics.get('total_symbols', 0)}, "
                              f"Gross={exposure.get('gross_exposure', 0):.1%}, "
                              f"Positions={exposure.get('num_positions', 0)}")
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error logging status: {str(e)}")
    
    # ============================================================================
    # ABSTRACT METHODS - STRATEGY-SPECIFIC IMPLEMENTATIONS
    # ============================================================================
    
    @abstractmethod
    def generate_signals(self, slice=None):
        """Generate raw trading signals - core strategy logic."""
        pass
    
    @abstractmethod
    def should_rebalance(self, current_time):
        """Determine if strategy should rebalance."""
        pass
    
    @abstractmethod
    def _create_symbol_data(self, symbol):
        """Create strategy-specific symbol data object."""
        pass
    
    # ============================================================================
    # BASE SYMBOL DATA CLASS (common patterns)
    # ============================================================================
    
    class BaseSymbolData:
        """Base SymbolData with common patterns."""
        
        def __init__(self, algorithm, symbol):
            self.algorithm = algorithm
            self.symbol = symbol
            self.data_points_received = 0
            self.last_update_time = None
            self.has_sufficient_data = True
            self.data_availability_error = None
        
        def get_qc_history(self, periods, resolution=Resolution.Daily):
            """OPTIMIZED history access for SymbolData via QC Native Data Accessor."""
            try:
                # Use QC native data accessor if available
                if hasattr(self.algorithm, 'data_accessor') and self.algorithm.data_accessor:
                    history = self.algorithm.data_accessor.get_qc_native_history(self.symbol, periods, resolution)
                    return history if history is not None and not history.empty else None
                
                # Fallback to direct QC History API (QC handles caching)
                history = self.algorithm.History(self.symbol, periods, resolution)
                return history if not history.empty else None
            except Exception as e:
                self.algorithm.Log(f"SymbolData {self.symbol}: History error: {str(e)}")
                return None
        
        @abstractmethod
        def IsReady(self):
            """Check if symbol data is ready."""
            pass
        
        def Dispose(self):
            """Base cleanup."""
            pass

    def update_with_data(self, slice):
        """
        Update strategy with new market data - LEGACY METHOD.
        This method should be overridden by derived strategies for custom logic.
        """
        try:
            self.current_slice = slice
            
            # Update indicators if available
            if hasattr(self, 'indicators') and self.indicators:
                for indicator in self.indicators.values():
                    if hasattr(indicator, 'Update'):
                        # Update with appropriate data
                        pass
                        
            # Update any internal state
            self._update_internal_state(slice)
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in update_with_data: {str(e)}")

    def update_with_unified_data(self, unified_data, slice):
        """
        PHASE 3: Update strategy with unified data interface.
        This provides optimized data access for all derived strategies.
        """
        try:
            # Store current slice for any legacy methods that might need it
            self.current_slice = slice
            
            # Validate unified data structure
            if not unified_data or not unified_data.get('valid', False):
                self.algorithm.Debug(f"{self.name}: Invalid unified data received")
                return
            
            # Extract symbols data from unified interface
            symbols_data = unified_data.get('symbols', {})
            if not symbols_data:
                return
            
            # Process unified data for strategy use
            processed_data = self._process_unified_data_for_strategy(symbols_data, unified_data)
            
            # Update strategy with processed unified data
            self._update_strategy_with_unified_data(processed_data, unified_data)
            
            # Update indicators with unified data if available
            if hasattr(self, 'indicators') and self.indicators:
                self._update_indicators_with_unified_data(processed_data)
            
            # Update internal state using unified data
            self._update_internal_state_with_unified_data(processed_data, unified_data)
            
            # Track unified data usage for this strategy
            self._track_unified_data_usage(unified_data, processed_data)
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in update_with_unified_data: {str(e)}")
            # Fallback to traditional method
            self.update_with_data(slice)

    def _process_unified_data_for_strategy(self, symbols_data, unified_data):
        """Process unified data into strategy-friendly format."""
        try:
            processed_data = {
                'timestamp': unified_data.get('timestamp', self.algorithm.Time),
                'bars': {},
                'chains': {},
                'security_info': {},
                'liquid_symbols': []
            }
            
            # Process each symbol's data
            for symbol, symbol_data in symbols_data.items():
                if not symbol_data.get('valid', False):
                    continue
                
                data = symbol_data.get('data', {})
                
                # Extract bar data
                if 'bar' in data:
                    bar_data = data['bar']
                    processed_data['bars'][symbol] = {
                        'open': bar_data.get('open', 0),
                        'high': bar_data.get('high', 0),
                        'low': bar_data.get('low', 0),
                        'close': bar_data.get('close', 0),
                        'volume': bar_data.get('volume', 0),
                        'time': bar_data.get('time', self.algorithm.Time)
                    }
                
                # Extract chain data
                if 'chain' in data and data['chain'].get('valid', False):
                    processed_data['chains'][symbol] = data['chain']
                
                # Extract security information
                if 'security' in data:
                    security_data = data['security']
                    processed_data['security_info'][symbol] = {
                        'price': security_data.get('price', 0),
                        'has_data': security_data.get('has_data', False),
                        'is_tradable': security_data.get('is_tradable', False),
                        'mapped_symbol': security_data.get('mapped_symbol'),
                        'market_hours_open': security_data.get('market_hours', False)
                    }
                    
                    # Add to liquid symbols if tradable
                    if (security_data.get('is_tradable', False) and 
                        security_data.get('has_data', False)):
                        processed_data['liquid_symbols'].append(symbol)
            
            return processed_data
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error processing unified data: {str(e)}")
            return {'timestamp': self.algorithm.Time, 'bars': {}, 'chains': {}, 'security_info': {}, 'liquid_symbols': []}

    def _update_strategy_with_unified_data(self, processed_data, unified_data):
        """Update strategy logic with processed unified data - override in derived classes."""
        try:
            # Base implementation - derived strategies should override this
            # Update any base strategy state with processed data
            
            # Store processed data for access by derived strategies
            self.current_processed_data = processed_data
            self.current_unified_metadata = unified_data.get('metadata', {})
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error updating strategy with unified data: {str(e)}")

    def _update_indicators_with_unified_data(self, processed_data):
        """Update indicators using unified data."""
        try:
            bars_data = processed_data.get('bars', {})
            
            for symbol, bar_data in bars_data.items():
                if symbol in self.indicators:
                    indicator = self.indicators[symbol]
                    
                    # Create IndicatorDataPoint for QC indicators
                    try:
                        data_point = IndicatorDataPoint(
                            bar_data['time'],
                            bar_data['close']
                        )
                        
                        if hasattr(indicator, 'Update'):
                            indicator.Update(data_point)
                            
                    except Exception as e:
                        self.algorithm.Debug(f"{self.name}: Error updating indicator for {symbol}: {str(e)}")
                        
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error updating indicators with unified data: {str(e)}")

    def _update_internal_state_with_unified_data(self, processed_data, unified_data):
        """Update internal strategy state with unified data - override in derived classes."""
        try:
            # Base implementation for common state updates
            
            # Update liquid symbols list
            self.current_liquid_symbols = processed_data.get('liquid_symbols', [])
            
            # Update last update timestamp
            self.last_unified_update = unified_data.get('timestamp', self.algorithm.Time)
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error updating internal state with unified data: {str(e)}")

    def _track_unified_data_usage(self, unified_data, processed_data):
        """Track unified data usage statistics for strategy performance."""
        try:
            # Initialize tracking if not exists
            if not hasattr(self, 'unified_data_stats'):
                self.unified_data_stats = {
                    'total_updates': 0,
                    'symbols_processed': 0,
                    'data_efficiency': 0.0
                }
            
            # Update statistics
            self.unified_data_stats['total_updates'] += 1
            self.unified_data_stats['symbols_processed'] += len(processed_data.get('liquid_symbols', []))
            
            # Calculate running efficiency
            metadata = unified_data.get('metadata', {})
            current_efficiency = len(processed_data.get('liquid_symbols', [])) / max(metadata.get('total_symbols', 1), 1)
            
            updates = self.unified_data_stats['total_updates']
            self.unified_data_stats['data_efficiency'] = (
                (self.unified_data_stats['data_efficiency'] * (updates - 1) + current_efficiency) / updates
            )
            
        except Exception as e:
            self.algorithm.Debug(f"{self.name}: Error tracking unified data usage: {str(e)}")

    def get_unified_data_performance(self):
        """Get unified data performance statistics for this strategy."""
        try:
            if hasattr(self, 'unified_data_stats'):
                return {
                    'strategy_name': self.name,
                    'total_unified_updates': self.unified_data_stats.get('total_updates', 0),
                    'avg_symbols_processed': self.unified_data_stats.get('symbols_processed', 0) / max(self.unified_data_stats.get('total_updates', 1), 1),
                    'data_efficiency': round(self.unified_data_stats.get('data_efficiency', 0.0) * 100, 1),
                    'integration_status': 'active'
                }
            else:
                return {
                    'strategy_name': self.name,
                    'integration_status': 'legacy_mode'
                }
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error getting unified data performance: {str(e)}")
            return {'strategy_name': self.name, 'integration_status': 'error'}

    def _update_internal_state(self, slice):
        """Update internal state - can be overridden by derived strategies."""
        pass 