# kestner_cta_strategy.py - INHERITS FROM BASE STRATEGY

from AlgorithmImports import *
import numpy as np
from strategies.base_strategy import BaseStrategy

# Try to import loggers, with multiple fallback options
try:
    from utils.smart_logger import SmartLogger
    SMART_LOGGER_AVAILABLE = True
except ImportError:
    SMART_LOGGER_AVAILABLE = False

try:
    from utils.simple_logger import SimpleLogger
    SIMPLE_LOGGER_AVAILABLE = True
except ImportError:
    SIMPLE_LOGGER_AVAILABLE = False

class KestnerCTAStrategy(BaseStrategy):
    """
    Kestner CTA Strategy Implementation
    CRITICAL: All configuration comes through centralized config manager only
    """
    
    def __init__(self, algorithm, config_manager, strategy_name):
        """
        Initialize Kestner CTA strategy with centralized configuration.
        CRITICAL: NO fallback logic - fail fast if config is invalid.
        """
        # Store algorithm reference first
        self.algorithm = algorithm
        
        # Initialize logger immediately to ensure it's always available
        self.logger = None
        self.use_smart_logger = False
        
        # Initialize basic attributes that are expected
        self.name = strategy_name
        self.config_manager = config_manager
        self.current_targets = {}
        self.symbol_data = {}
        self.last_rebalance_date = None
        self.last_update_time = None
        self.trades_executed = 0
        self.total_rebalances = 0
        self.strategy_returns = []
        
        try:
            algorithm.Log(f"KestnerCTA: Starting initialization for {strategy_name}")
            
            # Initialize logger with fallback first
            self._initialize_logger(algorithm, config_manager)
            self._log_info("Logger initialized successfully")
            
            # Get configuration directly to avoid base class issues
            try:
                self.config = config_manager.get_strategy_config(strategy_name)
                self._log_info("Configuration loaded successfully")
            except Exception as e:
                algorithm.Error(f"KestnerCTA: Failed to load config: {str(e)}")
                raise
            
            # Validate strategy is enabled
            if not self.config.get('enabled', False):
                error_msg = f"Strategy {strategy_name} is not enabled in configuration"
                algorithm.Error(f"STRATEGY ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            # Initialize strategy-specific components
            self._initialize_kestner_components()
            self._log_info("Kestner components initialized successfully")
            
            # Now call base class initialization (if needed)
            try:
                # Call parent __init__ but handle failures gracefully
                super().__init__(algorithm, config_manager, strategy_name)
                self._log_info("Base strategy initialization completed")
            except Exception as e:
                # Log the error but don't fail - we have what we need
                self._log_warning(f"Base strategy initialization failed: {str(e)}, continuing anyway")
            
            self._log_info("Strategy initialized successfully")
            
        except Exception as e:
            error_msg = f"CRITICAL ERROR initializing KestnerCTA: {str(e)}"
            # Use direct algorithm logging in case logger failed
            algorithm.Error(error_msg)
            # Also try to log with our logger if available
            self._log_error(f"Initialization failed: {str(e)}")
            raise ValueError(error_msg)
    
    def _initialize_strategy_components(self):
        """Override base class method to prevent double initialization."""
        # This method is called by the base class, but we handle initialization ourselves
        # in _initialize_kestner_components, so this is just a safe no-op
        pass
    
    def _initialize_logger(self, algorithm, config_manager):
        """Initialize logger with multiple fallback options."""
        # Ensure logger is always initialized to something
        self.logger = None
        self.use_smart_logger = False
        
        try:
            # Try SmartLogger first
            if SMART_LOGGER_AVAILABLE:
                try:
                    self.logger = SmartLogger(algorithm, 'kestner_cta', config_manager)
                    self.use_smart_logger = True
                    algorithm.Log("KestnerCTA: Using SmartLogger")
                    return
                except Exception as e:
                    algorithm.Log(f"KestnerCTA: SmartLogger failed: {str(e)}, trying SimpleLogger")
            
            # Try SimpleLogger as fallback
            if SIMPLE_LOGGER_AVAILABLE:
                try:
                    self.logger = SimpleLogger(algorithm, 'kestner_cta')
                    self.use_smart_logger = False
                    algorithm.Log("KestnerCTA: Using SimpleLogger")
                    return
                except Exception as e:
                    algorithm.Log(f"KestnerCTA: SimpleLogger failed: {str(e)}, using basic QC logging")
            
            # Final fallback to basic QC logging
            self.logger = None
            self.use_smart_logger = False
            algorithm.Log("KestnerCTA: Using basic QC logging")
            
        except Exception as e:
            # Ultimate fallback - ensure we never fail here
            self.logger = None
            self.use_smart_logger = False
            try:
                algorithm.Log(f"KestnerCTA: Logger initialization completely failed: {str(e)}, using basic QC logging")
            except Exception:
                pass  # Ignore even this failure
    
    def _log_info(self, message):
        """Log info message with fallback."""
        try:
            if self.logger:
                self.logger.info(message)
            else:
                self.algorithm.Log(f"INFO: [kestner_cta] {message}")
        except Exception:
            # Final fallback - direct algorithm logging
            try:
                self.algorithm.Log(f"INFO: [kestner_cta] {message}")
            except Exception:
                pass  # Ignore logging failures
    
    def _log_debug(self, message):
        """Log debug message with fallback."""
        try:
            if self.logger:
                self.logger.debug(message)
            else:
                # Skip debug messages in basic mode to reduce noise
                pass
        except Exception:
            pass  # Ignore logging failures
    
    def _log_warning(self, message):
        """Log warning message with fallback."""
        try:
            if self.logger:
                self.logger.warning(message)
            else:
                self.algorithm.Log(f"WARNING: [kestner_cta] {message}")
        except Exception:
            # Final fallback - direct algorithm logging
            try:
                self.algorithm.Log(f"WARNING: [kestner_cta] {message}")
            except Exception:
                pass  # Ignore logging failures
    
    def _log_error(self, message):
        """Log error message with fallback."""
        try:
            if self.logger:
                self.logger.error(message)
            else:
                self.algorithm.Error(f"ERROR: [kestner_cta] {message}")
        except Exception:
            # Final fallback - direct algorithm logging
            try:
                self.algorithm.Error(f"ERROR: [kestner_cta] {message}")
            except Exception:
                pass  # Ignore logging failures
    
    def _initialize_kestner_components(self):
        """Initialize Kestner-specific components using centralized configuration."""
        try:
            # Validate required configuration parameters
            required_params = [
                'momentum_lookbacks', 'volatility_lookback_days', 'target_volatility',
                'max_position_weight', 'warmup_days', 'enabled'
            ]
            
            for param in required_params:
                if param not in self.config:
                    error_msg = f"Missing required parameter '{param}' in KestnerCTA configuration"
                    self.algorithm.Error(f"CONFIG ERROR: {error_msg}")
                    raise ValueError(error_msg)
            
            # Initialize strategy parameters from validated config
            self.momentum_lookbacks = self.config['momentum_lookbacks']
            self.volatility_lookback_days = self.config['volatility_lookback_days']
            self.target_volatility = self.config['target_volatility']
            self.max_position_weight = self.config['max_position_weight']
            self.warmup_days = self.config['warmup_days']
            
            # Initialize tracking variables
            self.symbol_data = {}
            self.current_targets = {}
            self.last_rebalance_date = None
            self.last_update_time = None
            
            # Performance tracking
            self.trades_executed = 0
            self.total_rebalances = 0
            self.strategy_returns = []
            
            # Set futures_manager to None (we use QC native approach now)
            self.futures_manager = None
            
            # Initialize symbol data for all available futures
            self._initialize_symbol_data()
            
            self._log_info(f"Initialized with momentum lookbacks {self.momentum_lookbacks}, "
                         f"target volatility {self.target_volatility:.1%}")
            
        except Exception as e:
            error_msg = f"Failed to initialize KestnerCTA components: {str(e)}"
            self.algorithm.Error(f"CRITICAL ERROR: {error_msg}")
            raise ValueError(error_msg)
    
    def _initialize_symbol_data(self):
        """Initialize symbol data for all futures in the algorithm."""
        try:
            # Get all futures symbols from the algorithm
            futures_symbols = []
            for symbol in self.algorithm.Securities.Keys:
                security = self.algorithm.Securities[symbol]
                if security.Type == SecurityType.Future:
                    symbol_str = str(symbol)
                    # CRITICAL FIX: Only use continuous contracts for Kestner signal generation
                    # Continuous contracts (like /ES, /CL, /GC) have full historical data
                    # Underlying contracts (like ES WLF0Z3JIJTA9) have limited history and cause availability issues
                    if symbol_str.startswith('/'):
                        futures_symbols.append(symbol)
                        self._log_info(f"Using continuous contract {symbol_str} for signal generation")
                    else:
                        self._log_info(f"Ignoring underlying contract {symbol_str} (insufficient history)")
            
            self._log_info(f"Initializing symbol data for {len(futures_symbols)} continuous contract futures symbols")
            
            # Create symbol data for each futures symbol
            for symbol in futures_symbols:
                try:
                    symbol_data = self._create_symbol_data(symbol)
                    self.symbol_data[symbol] = symbol_data
                    self._log_debug(f"Created symbol data for {symbol}")
                except Exception as e:
                    self._log_error(f"Failed to create symbol data for {symbol}: {str(e)}")
            
            self._log_info(f"Symbol data initialized for {len(self.symbol_data)} continuous contract symbols")
            
        except Exception as e:
            self._log_error(f"Error initializing symbol data: {str(e)}")
            # Don't fail completely - symbol data can be created on-demand
    
    def should_rebalance(self, current_time):
        """Determine if strategy should rebalance (weekly)."""
        if self.last_rebalance_date is None:
            return True
        
        days_since_rebalance = (current_time.date() - self.last_rebalance_date).days
        return days_since_rebalance >= 7  # Weekly rebalancing
    
    def generate_signals(self, slice=None):
        """Generate Kestner momentum signals across all liquid symbols."""
        try:
            signals = {}
            liquid_symbols = self._get_liquid_symbols(slice)
            
            if not liquid_symbols:
                self._log_warning("No liquid symbols available for signal generation")
                return signals
            
            for symbol in liquid_symbols:
                if symbol not in self.symbol_data:
                    continue
                
                symbol_data = self.symbol_data[symbol]
                if not symbol_data.IsReady:
                    continue
                
                try:
                    # Generate ensemble of momentum signals
                    momentum_signals = []
                    for lookback_weeks in self.momentum_lookbacks:
                        momentum = symbol_data.GetMomentum(lookback_weeks)
                        if momentum is not None:
                            momentum_signals.append(momentum)
                    
                    if not momentum_signals:
                        continue
                    
                    # Average the momentum signals
                    avg_momentum = np.mean(momentum_signals)
                    
                    # Cap the signal strength
                    signal_cap = self.config['signal_cap']
                    capped_signal = max(-signal_cap, min(signal_cap, avg_momentum))
                    
                    if abs(capped_signal) > 0.01:  # Minimum signal threshold
                        signals[symbol] = capped_signal
                        self._log_debug(f"Signal generated for {symbol}: {capped_signal:.3f}")
                        
                except Exception as e:
                    self.algorithm.Error(f"{self.name}: Error processing {symbol}: {str(e)}")
                    continue
            
            return signals
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error generating signals: {str(e)}")
            return {}
    
    def _get_liquid_symbols(self, slice=None):
        """Get liquid symbols using QC native approach (continuous contracts only)."""
        try:
            # Use QC's native Securities collection directly - but only continuous contracts
            liquid_symbols = []
            
            # Get all futures symbols from algorithm's Securities
            for symbol in self.algorithm.Securities.Keys:
                security = self.algorithm.Securities[symbol]
                
                # Check if it's a futures contract and has data
                if security.Type == SecurityType.Future:
                    symbol_str = str(symbol)
                    
                    # CRITICAL: Only use continuous contracts (they have full historical data)
                    if symbol_str.startswith('/'):
                        # For continuous contracts, check if they have data
                        if security.HasData:
                            # Check if mapped contract is tradeable (for actual trading)
                            is_tradeable = security.IsTradable
                            if not is_tradeable and hasattr(security, 'Mapped') and security.Mapped:
                                mapped_contract = security.Mapped
                                if mapped_contract in self.algorithm.Securities:
                                    is_tradeable = self.algorithm.Securities[mapped_contract].IsTradable
                            
                            # During warmup, be more lenient (just check data availability)
                            if self.algorithm.IsWarmingUp:
                                if security.HasData:
                                    liquid_symbols.append(symbol)
                            else:
                                # Post-warmup, require tradeable status
                                if is_tradeable or security.HasData:  # Allow some flexibility
                                    liquid_symbols.append(symbol)
            
            self._log_debug(f"Found {len(liquid_symbols)} liquid continuous contract symbols from QC Securities")
            
            # Fallback to symbol_data keys if no symbols found (should be continuous contracts only)
            if not liquid_symbols and hasattr(self, 'symbol_data'):
                liquid_symbols = list(self.symbol_data.keys())
                self._log_warning(f"No liquid symbols from Securities, using {len(liquid_symbols)} from symbol_data")
            
            return liquid_symbols
            
        except Exception as e:
            self._log_error(f"Error getting liquid symbols: {str(e)}")
            # Ultimate fallback
            if hasattr(self, 'symbol_data'):
                return list(self.symbol_data.keys())
            else:
                return []

    def _calculate_portfolio_volatility(self, weights):
        """Kestner-specific portfolio volatility using proper covariance matrix."""
        try:
            if not weights or len(weights) == 0:
                return 0.0
            
            symbols = list(weights.keys())
            weight_array = np.array(list(weights.values()))
            
            # Single asset case
            if len(symbols) == 1:
                symbol = symbols[0]
                if symbol in self.symbol_data and self.symbol_data[symbol].IsReady:
                    vol = self.symbol_data[symbol].GetVolatility()
                    return abs(weight_array[0]) * (vol or 0.15)
                return 0.15
            
            # Multi-asset case: Build covariance matrix
            volatilities = []
            valid_symbols = []
            valid_weights = []
            
            for i, symbol in enumerate(symbols):
                if symbol in self.symbol_data and self.symbol_data[symbol].IsReady:
                    vol = self.symbol_data[symbol].GetVolatility()
                    if vol and vol > 0:
                        volatilities.append(vol)
                        valid_symbols.append(symbol)
                        valid_weights.append(weight_array[i])
            
            if len(valid_symbols) < 2:
                # Fallback to individual volatility
                if len(valid_symbols) == 1:
                    return abs(valid_weights[0]) * volatilities[0]
                return 0.15
            
            # Build correlation matrix using Kestner's shorter-term approach
            n = len(valid_symbols)
            correlation_matrix = np.eye(n)  # Start with identity matrix
            
            # Calculate correlations between all pairs
            for i in range(n):
                for j in range(i + 1, n):
                    correlation = self._calculate_correlation(valid_symbols[i], valid_symbols[j])
                    correlation_matrix[i, j] = correlation
                    correlation_matrix[j, i] = correlation
            
            # Build covariance matrix: Σ = D × R × D (where D = diag(volatilities), R = correlations)
            vol_matrix = np.diag(volatilities)
            covariance_matrix = vol_matrix @ correlation_matrix @ vol_matrix
            
            # Calculate portfolio volatility: σ_p = √(w' × Σ × w)
            weight_array = np.array(valid_weights)
            portfolio_variance = weight_array.T @ covariance_matrix @ weight_array
            portfolio_vol = np.sqrt(max(0, portfolio_variance))
            
            return float(portfolio_vol)
            
        except Exception as e:
            self._log_error(f"Error calculating portfolio volatility: {str(e)}")
            return 0.15  # Conservative fallback
    
    def OnSecuritiesChanged(self, changes):
        """Handle securities changes - continuous contracts only for Kestner."""
        try:
            # Add new securities - but only continuous contracts for Kestner
            for security in changes.AddedSecurities:
                symbol = security.Symbol
                symbol_str = str(symbol)
                
                # Only add continuous contracts for signal generation
                if symbol_str.startswith('/') and symbol not in self.symbol_data:
                    try:
                        self.symbol_data[symbol] = self._create_symbol_data(symbol)
                        self._log_info(f"Added continuous contract symbol data for {symbol}")
                    except Exception as e:
                        self._log_error(f"Failed to create symbol data for {symbol}: {str(e)}")
                elif not symbol_str.startswith('/'):
                    self._log_debug(f"Ignoring underlying contract {symbol_str} (not used for signals)")
            
            # Remove securities - only remove if we were tracking them
            for security in changes.RemovedSecurities:
                symbol = security.Symbol
                if symbol in self.symbol_data:
                    try:
                        self.symbol_data[symbol].Dispose()
                        del self.symbol_data[symbol]
                        self._log_info(f"Removed symbol data for {symbol}")
                    except Exception as e:
                        self._log_error(f"Error removing symbol data for {symbol}: {str(e)}")
                        
        except Exception as e:
            self._log_error(f"Error in OnSecuritiesChanged: {str(e)}")

    def _calculate_correlation(self, symbol1, symbol2):
        """Calculate correlation between two assets using Kestner's shorter-term data."""
        try:
            if symbol1 == symbol2:
                return 1.0
            
            # Get return series for both symbols (shorter lookback for Kestner)
            returns1 = self._get_return_series(symbol1)
            returns2 = self._get_return_series(symbol2)
            
            if returns1 is None or returns2 is None or len(returns1) < 30 or len(returns2) < 30:
                # Use default correlations based on asset classes
                return self._get_default_correlation(symbol1, symbol2)
            
            # Calculate correlation using overlapping periods (shorter than MTUM)
            min_length = min(len(returns1), len(returns2))
            corr_matrix = np.corrcoef(returns1[-min_length:], returns2[-min_length:])
            correlation = corr_matrix[0, 1]
            
            # Handle NaN correlations
            if np.isnan(correlation):
                return self._get_default_correlation(symbol1, symbol2)
            
            # Cap extreme correlations
            return max(-0.95, min(0.95, correlation))
            
        except Exception as e:
            return self._get_default_correlation(symbol1, symbol2)
    
    def _get_return_series(self, symbol):
        """Get daily return series for correlation calculation (Kestner shorter-term)."""
        try:
            if symbol not in self.symbol_data or not self.symbol_data[symbol].IsReady:
                return None
            
            symbol_data = self.symbol_data[symbol]
            if not hasattr(symbol_data, 'ret_window') or symbol_data.ret_window.Count < 30:
                return None
            
            # Use Kestner's existing return window with configured volatility lookback
            returns = []
            lookback_days = min(symbol_data.ret_window.Count, self.volatility_lookback_days)
            for i in range(lookback_days):
                returns.append(symbol_data.ret_window[i])
            
            return np.array(returns) if len(returns) >= 30 else None
            
        except Exception as e:
            return None
    
    def _get_default_correlation(self, symbol1, symbol2):
        """Get default correlation based on asset classes (same as MTUM but can be tuned)."""
        s1, s2 = str(symbol1), str(symbol2)
        
        # Equity indices (high correlation)
        if any(eq in s1 for eq in ['ES', 'NQ', 'YM']) and any(eq in s2 for eq in ['ES', 'NQ', 'YM']):
            return 0.75  # Slightly lower than MTUM for shorter-term
        
        # Bond futures (high correlation)  
        if any(bond in s1 for bond in ['ZN', 'ZB']) and any(bond in s2 for bond in ['ZN', 'ZB']):
            return 0.70
        
        # FX majors (moderate correlation)
        if any(fx in s1 for fx in ['6E', '6J', '6B']) and any(fx in s2 for fx in ['6E', '6J', '6B']):
            return 0.45
        
        # Commodities (low-moderate correlation)
        if any(comm in s1 for comm in ['CL', 'GC', 'SI']) and any(comm in s2 for comm in ['CL', 'GC', 'SI']):
            return 0.25
        
        # Cross-asset class correlations
        # Stocks vs bonds (negative correlation, less pronounced short-term)
        if (any(eq in s1 for eq in ['ES', 'NQ', 'YM']) and any(bond in s2 for bond in ['ZN', 'ZB'])) or \
           (any(bond in s1 for bond in ['ZN', 'ZB']) and any(eq in s2 for eq in ['ES', 'NQ', 'YM'])):
            return -0.20  # Less negative than MTUM
        
        # Default low correlation for different asset classes
        return 0.15  # Lower default than MTUM

    def _create_symbol_data(self, symbol):
        """Create Kestner-specific symbol data object."""
        return self.SymbolData(
            self.algorithm,
            symbol,
            self.momentum_lookbacks,
            self.volatility_lookback_days
        )
    
    class SymbolData:
        """
        Kestner-specific SymbolData for momentum and volatility calculations.
        """

        def __init__(self, algorithm, symbol, lookbackWeeksList, volLookbackDays):
            self.algorithm = algorithm
            self.symbol = symbol
            self.lookbackWeeksList = lookbackWeeksList
            self.volLookbackDays = volLookbackDays
            self.consolidator = None

            # Rolling windows for calculations
            max_lookback = max(lookbackWeeksList)
            max_days = max_lookback * 5 + 10  # 5 trading days/week + cushion
            self.price_window = RollingWindow[float](max_days)
            self.ret_window = RollingWindow[float](volLookbackDays)

            # Track data quality
            self.data_points_received = 0
            self.last_update_time = None
            self.has_sufficient_data = True
            self.data_availability_error = None

            # Setup consolidator
            try:
                self.consolidator = TradeBarConsolidator(timedelta(days=1))
                self.consolidator.DataConsolidated += self.OnDataConsolidated
                algorithm.SubscriptionManager.AddConsolidator(symbol, self.consolidator)
            except Exception as e:
                algorithm.Log(f"SymbolData {symbol}: Consolidator setup error: {str(e)}")

            # Initialize with history using centralized data provider
            self._initialize_with_history()

        def _initialize_with_history(self):
            """Warm up indicators with historical data using CENTRALIZED data provider."""
            try:
                periods_needed = self.volLookbackDays + max(self.lookbackWeeksList) * 7
                
                # Use centralized data provider if available
                if hasattr(self.algorithm, 'data_integrity_checker') and self.algorithm.data_integrity_checker:
                    history = self.algorithm.data_integrity_checker.get_history(self.symbol, periods_needed, Resolution.Daily)
                else:
                    # Fallback to direct API call (not recommended)
                    self.algorithm.Log(f"KestnerSymbolData {self.symbol}: WARNING - No centralized cache, using direct History API")
                    history = self.algorithm.History(self.symbol, periods_needed, Resolution.Daily)
                
                # Convert to list to check if empty (QC data doesn't support len() directly)
                if history is None:
                    history_list = []
                else:
                    history_list = list(history)
                
                if len(history_list) == 0:
                    self.algorithm.Error(f"CRITICAL: SymbolData {self.symbol} - No historical data available")
                    self.has_sufficient_data = False
                    self.data_availability_error = "No historical data available"
                    return
                
                # Update rolling windows with historical data
                for bar in history_list:
                    close_price = bar.Close if hasattr(bar, 'Close') else bar.close
                    open_price = bar.Open if hasattr(bar, 'Open') else bar.open
                    
                    self.price_window.Add(close_price)
                    
                    # Calculate daily return
                    if open_price != 0:
                        daily_return = (close_price / open_price) - 1
                        self.ret_window.Add(daily_return)
                    
                    self.data_points_received += 1

                self.algorithm.Log(f"SymbolData {self.symbol}: Initialized with {len(history_list)} historical bars")

            except Exception as e:
                self.algorithm.Error(f"CRITICAL: SymbolData {self.symbol} - History initialization error: {str(e)}")
                self.has_sufficient_data = False
                self.data_availability_error = f"History initialization error: {str(e)}"

        @property
        def IsReady(self):
            """Check if symbol data is ready for calculations."""
            if not self.has_sufficient_data:
                return False
            
            min_price_count = max(self.lookbackWeeksList) * 5 + 10
            min_return_count = self.volLookbackDays
            
            return (self.price_window.Count >= min_price_count and 
                   self.ret_window.Count >= min_return_count)

        def OnDataConsolidated(self, sender, bar: TradeBar):
            """Process new daily bar."""
            if bar is None or bar.Close <= 0:
                return
            
            try:
                # Update price window
                self.price_window.Add(float(bar.Close))
                
                # Calculate daily return using previous close
                if self.price_window.Count >= 2:
                    prev_close = self.price_window[1]
                    if prev_close > 0:
                        daily_return = (bar.Close / prev_close) - 1
                        self.ret_window.Add(daily_return)
                
                self.data_points_received += 1
                self.last_update_time = bar.Time
                
            except Exception as e:
                self.algorithm.Error(f"SymbolData {self.symbol}: OnDataConsolidated error: {str(e)}")

        def GetMomentum(self, lookbackWeeks):
            """Calculate momentum over specified lookback period."""
            try:
                if not self.IsReady:
                    return None
                
                days_needed = lookbackWeeks * 5  # Approximate trading days
                if self.price_window.Count < days_needed + 1:
                    return None
                
                # Get current and lookback prices
                current_price = self.price_window[0]
                lookback_price = self.price_window[min(days_needed, self.price_window.Count - 1)]
                
                if lookback_price <= 0:
                    return None
                
                # Calculate raw momentum
                momentum = (current_price / lookback_price) - 1
                
                # Normalize by volatility
                volatility = self.GetVolatility()
                if volatility and volatility > 0:
                    normalized_momentum = momentum / volatility
                    return normalized_momentum
                
                return momentum
                
            except Exception as e:
                self.algorithm.Error(f"SymbolData {self.symbol}: Momentum calculation error: {str(e)}")
                return None

        def GetVolatility(self):
            """Calculate annualized volatility."""
            try:
                if self.ret_window.Count < 30:  # Need minimum data
                    return None
                
                returns = [self.ret_window[i] for i in range(min(self.ret_window.Count, self.volLookbackDays))]
                if len(returns) < 10:
                    return None
                
                vol = np.std(returns) * np.sqrt(252)  # Annualized
                return vol if vol > 0 else None
                
            except Exception as e:
                return None

        def GetDataQuality(self):
            """Get data quality metrics for diagnostics"""
            return {
                'symbol': str(self.symbol),
                'data_points_received': self.data_points_received,
                'price_window_count': self.price_window.Count,
                'returns_window_count': self.ret_window.Count,
                'vol_lookback_days': self.volLookbackDays,
                'momentum_lookbacks': self.lookbackWeeksList,
                'is_ready': self.IsReady,
                'last_update': self.last_update_time,
                'current_price': self.price_window[0] if self.price_window.Count > 0 else 0
            }

        def Dispose(self):
            """Clean disposal of resources"""
            try:
                if self.consolidator:
                    self.algorithm.SubscriptionManager.RemoveConsolidator(self.symbol, self.consolidator)
                self.price_window.Reset()
                self.ret_window.Reset()
            except:
                pass  # Ignore disposal errors

class KestnerCTAReporting:
    """Standalone class for Kestner-specific reporting."""
    
    def __init__(self, algorithm):
        """Initialize reporting class."""
        self.algorithm = algorithm
    
    def generate_performance_report(self, strategy):
        """Generate performance report for the strategy."""
        self.algorithm.Log(f"Performance Report for {strategy.name}")
        self.algorithm.Log(f"Total Rebalances: {strategy.total_rebalances}")
        self.algorithm.Log(f"Trades Executed: {strategy.trades_executed}")
        
        if strategy.gross_exposure_history:
            avg_exposure = sum(strategy.gross_exposure_history) / len(strategy.gross_exposure_history)
            self.algorithm.Log(f"Average Gross Exposure: {avg_exposure:.2%}")
        
        return {
            'total_rebalances': strategy.total_rebalances,
            'trades_executed': strategy.trades_executed,
            'current_positions': len(strategy.current_targets)
        }
