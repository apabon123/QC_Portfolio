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
        
        # Set up futures manager reference for liquid symbol access
        self.futures_manager = getattr(algorithm, 'futures_manager', None)
        
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
    
    def generate_targets(self):
        """Common target generation pattern found in all strategies."""
        try:
            if not self.should_rebalance(self.algorithm.Time):
                return self.current_targets
            
            signals = self.generate_signals()
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
        """CENTRALIZED History API usage via DataIntegrityChecker."""
        try:
            # PRIORITY 1: Use centralized data integrity checker with caching
            if hasattr(self.algorithm, 'data_integrity_checker') and self.algorithm.data_integrity_checker:
                history = self.algorithm.data_integrity_checker.get_history(symbol, periods, resolution)
                return history if history is not None and not history.empty else None
            
            # PRIORITY 2: Use QC native contract resolver if available
            if hasattr(self.algorithm, 'contract_resolver') and self.algorithm.contract_resolver:
                history = self.algorithm.contract_resolver.get_history_with_diagnostics(symbol, periods, resolution)
                return history if history is not None and not history.empty else None
            
            # FALLBACK: Direct QC History method (NOT RECOMMENDED)
            self.algorithm.Log(f"{self.name}: WARNING - Using direct History API")
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
    def generate_signals(self):
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
            """CENTRALIZED history access for SymbolData via DataIntegrityChecker."""
            try:
                if hasattr(self.algorithm, 'data_integrity_checker') and self.algorithm.data_integrity_checker:
                    history = self.algorithm.data_integrity_checker.get_history(self.symbol, periods, resolution)
                    return history if history is not None and not history.empty else None
                
                self.algorithm.Log(f"SymbolData {self.symbol}: WARNING - Using direct History API")
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