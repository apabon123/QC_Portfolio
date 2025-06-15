# base_strategy.py - STANDARDIZED BASE STRATEGY CLASS

from AlgorithmImports import *
import numpy as np
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """
    Standardized Base Strategy Class - QC PLUMBING BEST PRACTICES
    
    This class encapsulates all QuantConnect integration patterns that have been
    proven to work across Kestner, MTUM, and HMM strategies. All future strategies
    should inherit from this class to ensure consistency and scalability.
    
    KEY QC INTEGRATIONS STANDARDIZED:
    - History API usage patterns
    - Consolidator management
    - Security property access
    - Order management
    - Resource disposal
    - Error handling
    - Configuration loading
    - Performance tracking
    """
    
    def __init__(self, algorithm, futures_manager, name, config_manager=None):
        """
        Standardized strategy initialization.
        
        Args:
            algorithm: QuantConnect algorithm instance
            futures_manager: Futures manager instance
            name: Strategy name
            config_manager: Config manager instance (required for dynamic loading)
        """
        self.algorithm = algorithm
        self.futures_manager = futures_manager
        self.name = name
        self.config_manager = config_manager
        
        # Standardized state tracking
        self.symbol_data = {}
        self.current_targets = {}
        self.last_rebalance_date = None
        self.last_update_time = None
        
        # Standardized performance tracking
        self.trades_executed = 0
        self.total_rebalances = 0
        self.strategy_returns = []
        self.portfolio_values = []
        self.gross_exposure_history = []
        
        # Load configuration using standardized pattern
        self._load_configuration()
        
        # Log successful initialization
        self._log_initialization_summary()
    
    def _load_configuration(self):
        """
        Standardized configuration loading with proper fallback handling.
        """
        try:
            if self.config_manager:
                # Primary: Load from config_manager (dynamic loading)
                strategy_config = self.config_manager.get_strategy_config(self.name)
                if strategy_config:
                    self._build_config_dict(strategy_config)
                    self.algorithm.Log(f"{self.name}: Config loaded from config_manager")
                    return
            
            # Fallback: Use default configuration
            self.algorithm.Log(f"{self.name}: Using fallback configuration")
            self._load_fallback_config()
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Config loading error: {str(e)}")
            self._load_fallback_config()
    
    @abstractmethod
    def _build_config_dict(self, config):
        """Build strategy-specific config dictionary from provided config."""
        pass
    
    @abstractmethod
    def _load_fallback_config(self):
        """Load fallback configuration if config_manager is unavailable."""
        pass
    
    def _log_initialization_summary(self):
        """Log standardized initialization summary."""
        self.algorithm.Log(f"{self.name}: Initialized with CONFIG-COMPLIANT parameters")
        # Strategy-specific logging should be implemented in subclasses
    
    # ============================================================================
    # STANDARDIZED QC INTEGRATION METHODS
    # ============================================================================
    
    def get_qc_history(self, symbol, periods, resolution=Resolution.Daily):
        """
        Standardized History API usage.
        
        Returns:
            pandas.DataFrame: Historical data or empty DataFrame if failed
        """
        try:
            history = self.algorithm.History(symbol, periods, resolution)
            return history if not history.empty else None
        except Exception as e:
            self.algorithm.Log(f"{self.name}: History API error for {symbol}: {str(e)}")
            return None
    
    def get_qc_security_price(self, symbol):
        """
        Standardized security price access.
        
        Returns:
            float: Current price or None if invalid
        """
        try:
            if symbol in self.algorithm.Securities:
                price = self.algorithm.Securities[symbol].Price
                return price if price > 0 and not np.isnan(price) and not np.isinf(price) else None
            return None
        except Exception as e:
            self.algorithm.Log(f"{self.name}: Price access error for {symbol}: {str(e)}")
            return None
    
    def get_qc_mapped_contract(self, symbol):
        """
        Standardized mapped contract access.
        
        Returns:
            Symbol: Mapped contract or None if invalid
        """
        try:
            if symbol in self.algorithm.Securities:
                mapped = self.algorithm.Securities[symbol].Mapped
                security = self.algorithm.Securities[mapped] if mapped else None
                return mapped if security and security.HasData else None
            return None
        except Exception as e:
            self.algorithm.Log(f"{self.name}: Mapped contract error for {symbol}: {str(e)}")
            return None
    
    def place_qc_market_order(self, symbol, quantity, tag=None):
        """
        Standardized market order placement.
        
        Returns:
            OrderTicket: Order ticket or None if failed
        """
        try:
            if tag is None:
                tag = f"{self.name}_order"
            
            order_ticket = self.algorithm.MarketOrder(symbol, quantity, tag=tag)
            return order_ticket
        except Exception as e:
            self.algorithm.Log(f"{self.name}: Order placement error for {symbol}: {str(e)}")
            return None
    
    def liquidate_qc_position(self, symbol, tag=None):
        """
        Standardized position liquidation.
        
        Returns:
            OrderTicket: Liquidation ticket or None if failed
        """
        try:
            if tag is None:
                tag = f"{self.name}_liquidate"
            
            order_ticket = self.algorithm.Liquidate(symbol, tag=tag)
            return order_ticket
        except Exception as e:
            self.algorithm.Log(f"{self.name}: Liquidation error for {symbol}: {str(e)}")
            return None
    
    # ============================================================================
    # STANDARDIZED SYMBOL DATA BASE CLASS
    # ============================================================================
    
    class BaseSymbolData:
        """
        Standardized SymbolData base class with QC best practices.
        
        All strategy-specific SymbolData classes should inherit from this.
        """
        
        def __init__(self, algorithm, symbol):
            self.algorithm = algorithm
            self.symbol = symbol
            
            # Standardized tracking
            self.data_points_received = 0
            self.last_update_time = None
            self.consolidator = None
            
            # Data availability tracking - CRITICAL FOR TRADING DECISIONS
            self.has_sufficient_data = True  # Assume true until proven otherwise
            self.data_availability_error = None
            
            # Initialize consolidator using standardized pattern
            self._setup_consolidator()
        
        def _setup_consolidator(self):
            """Standardized consolidator setup."""
            try:
                self.consolidator = TradeBarConsolidator(timedelta(days=1))
                self.consolidator.DataConsolidated += self.OnDataConsolidated
                self.algorithm.SubscriptionManager.AddConsolidator(self.symbol, self.consolidator)
            except Exception as e:
                self.algorithm.Log(f"SymbolData {self.symbol}: Consolidator setup error: {str(e)}")
        
        def get_qc_history(self, periods, resolution=Resolution.Daily):
            """Standardized history access for SymbolData."""
            try:
                history = self.algorithm.History(self.symbol, periods, resolution)
                return history if not history.empty else None
            except Exception as e:
                self.algorithm.Log(f"SymbolData {self.symbol}: History error: {str(e)}")
                return None
        
        def process_qc_history(self, history):
            """
            Standardized history processing using pandas iterrows pattern.
            
            Args:
                history: pandas.DataFrame from QC History API
                
            Returns:
                list: Processed data points
            """
            processed_data = []
            
            try:
                for index, row in history.iterrows():
                    # Standardized column access
                    close = row['close'] if 'close' in row else row.get('Close', 0)
                    open_price = row['open'] if 'open' in row else row.get('Open', close)
                    
                    # Validate data
                    if close > 0 and not np.isnan(close) and not np.isinf(close):
                        processed_data.append({
                            'close': close,
                            'open': open_price,
                            'time': index
                        })
                        self.data_points_received += 1
                
                return processed_data
                
            except Exception as e:
                self.algorithm.Log(f"SymbolData {self.symbol}: History processing error: {str(e)}")
                return []
        
        def handle_no_data_available(self, required_periods):
            """
            Handle case when no historical data is available.
            
            CRITICAL: No mock data for trading decisions!
            This method logs the issue and marks the symbol as not ready.
            
            Args:
                required_periods: Number of periods that were requested
            """
            self.algorithm.Error(f"CRITICAL: SymbolData {self.symbol} - No historical data available for {required_periods} periods")
            self.algorithm.Error(f"TRADING DECISION RISK: Strategy cannot initialize {self.symbol} without real market data")
            
            # Mark this symbol as having insufficient data
            self.has_sufficient_data = False
            self.data_availability_error = f"No historical data available for {required_periods} periods"
            
            # Log to help with debugging
            self.algorithm.Log(f"SymbolData {self.symbol}: Marked as insufficient data - will not be ready for trading")
            
            return None
        
        @abstractmethod
        def OnDataConsolidated(self, sender, bar):
            """Handle consolidated data - must be implemented by subclasses."""
            pass
        
        @property
        @abstractmethod
        def IsReady(self):
            """Check if symbol data is ready - must be implemented by subclasses."""
            pass
        
        def Dispose(self):
            """Standardized resource disposal."""
            try:
                if hasattr(self, 'consolidator') and self.consolidator is not None:
                    self.consolidator.DataConsolidated -= self.OnDataConsolidated
                    # Properly dispose of consolidator
                    if hasattr(self.algorithm, 'SubscriptionManager'):
                        try:
                            self.algorithm.SubscriptionManager.RemoveConsolidator(self.symbol, self.consolidator)
                        except:
                            pass  # May already be removed
                    self.consolidator = None
                    
            except Exception as e:
                self.algorithm.Log(f"SymbolData {self.symbol}: Dispose error: {str(e)}")
    
    # ============================================================================
    # ABSTRACT METHODS - MUST BE IMPLEMENTED BY SUBCLASSES
    # ============================================================================
    
    @abstractmethod
    def update(self, slice_data):
        """Update strategy with new data."""
        pass
    
    @abstractmethod
    def generate_targets(self):
        """Generate target positions."""
        pass
    
    @abstractmethod
    def get_exposure(self):
        """Get current exposure metrics."""
        pass
    
    @abstractmethod
    def should_rebalance(self, current_time):
        """Check if strategy should rebalance."""
        pass
    
    # ============================================================================
    # STANDARDIZED SECURITIES CHANGED HANDLING
    # ============================================================================
    
    def OnSecuritiesChanged(self, changes):
        """
        Standardized security change handling.
        Only track continuous contracts, ignore rollover-specific contracts.
        """
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            symbol_str = str(symbol)
            
            # Only track continuous contracts (start with '/') - QC STANDARDIZED
            if symbol_str.startswith('/') or symbol_str.startswith('futures/'):
                if symbol not in self.symbol_data:
                    self.algorithm.Log(f"{self.name}: Initializing SymbolData for continuous contract: {symbol}")
                    try:
                        # Subclasses should override this method to create their specific SymbolData
                        self._create_symbol_data(symbol)
                    except Exception as e:
                        self.algorithm.Error(f"{self.name}: Failed to create SymbolData for {symbol}: {e}")
            else:
                # Skip rollover contracts - QC STANDARDIZED
                self.algorithm.Log(f"{self.name}: Skipping rollover contract in OnSecuritiesChanged: {symbol_str}")
                
        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.symbol_data:
                self.algorithm.Log(f"{self.name}: Removing SymbolData for security: {symbol}")
                try:
                    self.symbol_data[symbol].Dispose()
                    del self.symbol_data[symbol]
                except Exception as e:
                    self.algorithm.Log(f"{self.name}: Error disposing SymbolData for {symbol}: {e}")
    
    @abstractmethod
    def _create_symbol_data(self, symbol):
        """Create strategy-specific SymbolData - must be implemented by subclasses."""
        pass
    
    # ============================================================================
    # STANDARDIZED PERFORMANCE TRACKING
    # ============================================================================
    
    def get_performance_metrics(self):
        """Get standardized performance metrics."""
        return {
            'strategy_name': self.name,
            'trades_executed': self.trades_executed,
            'total_rebalances': self.total_rebalances,
            'current_positions': len(self.current_targets),
            'ready_symbols': len([sd for sd in self.symbol_data.values() if sd.IsReady]),
            'total_symbols': len(self.symbol_data),
            'last_rebalance': self.last_rebalance_date,
            'last_update': self.last_update_time
        }
    
    def log_status(self):
        """Log standardized strategy status."""
        metrics = self.get_performance_metrics()
        self.algorithm.Log(f"{self.name} STATUS: "
                         f"Trades: {metrics['trades_executed']}, "
                         f"Rebalances: {metrics['total_rebalances']}, "
                         f"Ready: {metrics['ready_symbols']}/{metrics['total_symbols']}")
    
    def _validate_slice_data_centralized(self, slice_data):
        """
        CENTRALIZED validation using DataIntegrityChecker and QC built-ins
        All strategies should use this method for consistency
        """
        try:
            # Use centralized data integrity checker if available
            if hasattr(self.algorithm, 'data_integrity_checker') and self.algorithm.data_integrity_checker:
                # Use the centralized, QC-optimized validation
                return self.algorithm.data_integrity_checker.validate_slice(slice_data) is not None
            
            # Fallback: Use QC's built-in validation methods
            if not slice_data or not hasattr(slice_data, 'Bars'):
                return False
            
            # Check if any of our symbols have valid data using QC's built-in properties
            valid_symbols = 0
            for symbol in self.symbol_data.keys():
                if slice_data.Contains(symbol) and symbol in slice_data.Bars:
                    # LEVERAGE QC'S BUILT-IN VALIDATION:
                    if symbol in self.algorithm.Securities:
                        security = self.algorithm.Securities[symbol]
                        
                        # Use QC's HasData and IsTradable properties
                        if security.HasData and security.IsTradable:
                            bar = slice_data.Bars[symbol]
                            if bar and bar.Close > 0:
                                valid_symbols += 1
            
            return valid_symbols > 0
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in centralized validation: {str(e)}")
            return False
    
    def _get_safe_symbols_from_futures_manager(self):
        """
        Get safe symbols using centralized FuturesManager
        Leverage QC's built-in validation + centralized quarantine logic
        """
        try:
            if self.futures_manager:
                # Use centralized FuturesManager to get safe symbols
                all_symbols = self.futures_manager.get_liquid_symbols()
                
                # Apply data integrity filtering if available
                if hasattr(self.algorithm, 'data_integrity_checker') and self.algorithm.data_integrity_checker:
                    safe_symbols = self.algorithm.data_integrity_checker.get_safe_symbols(all_symbols)
                    return safe_symbols
                
                return all_symbols
            
            # Fallback: Use QC's Securities directly with basic validation
            safe_symbols = []
            for symbol in self.algorithm.Securities.Keys:
                security = self.algorithm.Securities[symbol]
                
                # Use QC's built-in validation
                if (security.Type == SecurityType.Future and 
                    security.HasData and 
                    security.IsTradable and 
                    security.Price > 0):
                    safe_symbols.append(symbol)
            
            return safe_symbols
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error getting safe symbols: {str(e)}")
            return []
    
    def _validate_symbol_for_trading_qc_native(self, symbol):
        """
        CENTRALIZED symbol validation using QC's built-in properties
        All strategies should use this for consistency
        """
        try:
            # Check if symbol exists in securities (QC managed)
            if symbol not in self.algorithm.Securities:
                return False
            
            security = self.algorithm.Securities[symbol]
            
            # LEVERAGE QC'S BUILT-IN VALIDATION:
            
            # 1. HasData property (QC built-in)
            if not security.HasData:
                return False
            
            # 2. IsTradable property (QC built-in)
            if not security.IsTradable:
                return False
            
            # 3. Price validation (QC built-in)
            if not hasattr(security, 'Price') or security.Price is None or security.Price <= 0:
                return False
            
            # 4. Check with centralized data integrity checker if available
            if hasattr(self.algorithm, 'data_integrity_checker') and self.algorithm.data_integrity_checker:
                if symbol in self.algorithm.data_integrity_checker.quarantined_symbols:
                    return False
            
            # 5. Use centralized FuturesManager validation if available
            if self.futures_manager:
                if not self.futures_manager.validate_price(symbol, security.Price):
                    return False
            
            return True
            
        except Exception as e:
            return False 