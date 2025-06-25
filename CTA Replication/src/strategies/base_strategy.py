# base_strategy.py - COMPREHENSIVE BASE STRATEGY CLASS FOR CTA STRATEGIES

from AlgorithmImports import *
import numpy as np
from abc import ABC, abstractmethod
from collections import deque

# Import data processing utilities to reduce file size
try:
    from utils.strategy_data_processor import StrategyDataProcessor
    DATA_PROCESSOR_AVAILABLE = True
except ImportError:
    DATA_PROCESSOR_AVAILABLE = False

class BaseStrategy(ABC):
    """
    Base strategy class consolidating common CTA functionality.
    
    PRINCIPLES: Configuration security, continuous contracts, LEAN-first, fail-fast.
    PATTERNS: Config loading, symbol data, validation, targeting, exposure, performance.
    ABSTRACT: generate_signals(), should_rebalance(), _calculate_portfolio_volatility(), _create_symbol_data()
    """
    
    def __init__(self, algorithm, config_manager, strategy_name):
        """Initialize base strategy with centralized configuration."""
        self.algorithm = algorithm
        self.config_manager = config_manager
        self.name = strategy_name
        
        # Initialize core tracking variables (common across all strategies)
        self.symbol_data = {}
        self.current_targets = {}
        self.last_rebalance_date = None
        self.last_update_time = None
        self.trades_executed = 0
        self.total_rebalances = 0
        self.strategy_returns = []
        
        # Availability tracking for enhanced diagnostics
        self._last_unavailable_reason = ""
        self._availability_diagnostics = {}
        
        try:
            # CRITICAL: Get strategy configuration through centralized manager ONLY
            self.config = self.config_manager.get_strategy_config(strategy_name)
            self.algorithm.Log(f"{self.name}: Configuration loaded successfully")
            
            # Get required risk management parameters from centralized config
            risk_config = self.config_manager.get_risk_config()
            self.max_leverage_multiplier = risk_config.get('max_leverage_multiplier', 10.0)
            
            # Validate strategy is enabled
            if not self.config.get('enabled', False):
                error_msg = f"Strategy {self.name} is not enabled in configuration"
                self.algorithm.Error(f"STRATEGY ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            # Load common configuration parameters (found in all strategies)
            self._load_common_config()
            
            # Initialize strategy-specific components (implemented by subclasses)
            self._initialize_strategy_components()
            
            # Log successful initialization
            self._log_initialization_summary()
            
        except Exception as e:
            error_msg = f"CRITICAL ERROR initializing strategy {self.name}: {str(e)}"
            self.algorithm.Error(error_msg)
            # FAIL FAST - no fallback configuration allowed
            raise ValueError(error_msg)
    
    def _load_common_config(self):
        """Load common configuration parameters found across all CTA strategies."""
        try:
            # Common parameters found in MTUM, Kestner, and HMM
            self.target_volatility = self.config.get('target_volatility', 0.15)
            self.max_position_weight = self.config.get('max_position_weight', 0.30)
            self.warmup_days = self.config.get('warmup_days', 100)
            
            self.algorithm.Log(f"{self.name}: Common config loaded - "
                             f"Target Vol: {self.target_volatility:.1%}, "
                             f"Max Position: {self.max_position_weight:.1%}, "
                             f"Warmup: {self.warmup_days} days")
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error loading common config: {str(e)}")
            raise
    
    @abstractmethod
    def _initialize_strategy_components(self):
        """Initialize strategy-specific components. Must validate params and call _initialize_symbol_data()."""
        pass
    
    def _log_initialization_summary(self):
        """Log common initialization summary (found in all strategies)."""
        self.algorithm.Log(f"{self.name}: Base strategy initialized - "
                          f"Target Vol: {self.target_volatility:.1%}, "
                          f"Max Position: {self.max_position_weight:.1%}")
    
    # ============================================================================
    # CONTINUOUS CONTRACT MANAGEMENT (Common across all strategies)
    # ============================================================================
    
    def _initialize_symbol_data(self):
        """Initialize SymbolData for continuous contracts only. Called by subclasses."""
        try:
            # Get all futures symbols from QC's Securities collection
            continuous_contracts = []
            for symbol in self.algorithm.Securities.Keys:
                security = self.algorithm.Securities[symbol]
                if security.Type == SecurityType.Future:
                    symbol_str = str(symbol)
                    # CRITICAL: Only use continuous contracts for signal generation
                    if symbol_str.startswith('/'):
                        continuous_contracts.append(symbol)
                        self.algorithm.Log(f"{self.name}: Using continuous contract {symbol_str}")
                    else:
                        self.algorithm.Log(f"{self.name}: Ignoring underlying contract {symbol_str}")
            
            # Fallback to default liquid futures if none found
            if not continuous_contracts:
                default_symbols = ['ES', 'NQ', 'ZN', 'CL', 'GC']
                self.algorithm.Log(f"{self.name}: No continuous contracts found, using defaults")
                for symbol_str in default_symbols:
                    # Try to find or add the continuous contract
                    symbol = self._find_or_add_continuous_contract(symbol_str)
                    if symbol:
                        continuous_contracts.append(symbol)
            
            # Create SymbolData for each continuous contract
            for symbol in continuous_contracts:
                try:
                    self.symbol_data[symbol] = self._create_symbol_data(symbol)
                    self.algorithm.Log(f"{self.name}: Created symbol data for {symbol}")
                except Exception as e:
                    self.algorithm.Error(f"{self.name}: Failed creating symbol data for {symbol}: {str(e)}")
            
            self.algorithm.Log(f"{self.name}: Initialized {len(self.symbol_data)} continuous contract symbols")
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in _initialize_symbol_data: {str(e)}")
    
    def _find_or_add_continuous_contract(self, symbol_str):
        """Helper to find or add continuous contract for a given ticker."""
        try:
            # Look for existing continuous contract
            for symbol in self.algorithm.Securities.Keys:
                if str(symbol).startswith('/') and symbol_str in str(symbol):
                    return symbol
            
            # If not found, this means the universe setup didn't add it
            # Log this but don't try to add it here (universe management responsibility)
            self.algorithm.Log(f"{self.name}: Continuous contract for {symbol_str} not found in Securities")
            return None
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error finding continuous contract for {symbol_str}: {str(e)}")
            return None
    
    def OnSecuritiesChanged(self, changes):
        """
        Handle securities changes - continuous contracts only.
        
        ARCHITECTURAL PRINCIPLE:
        - Only add continuous contracts to symbol_data
        - Ignore underlying contracts (they're for execution, not signals)
        - Log rollover events for transparency
        """
        try:
            # Add new securities - but only continuous contracts
            for security in changes.AddedSecurities:
                symbol = security.Symbol
                symbol_str = str(symbol)
                
                # Only add continuous contracts for signal generation
                if symbol_str.startswith('/') and symbol not in self.symbol_data:
                    try:
                        self.symbol_data[symbol] = self._create_symbol_data(symbol)
                        self.algorithm.Log(f"{self.name}: Added continuous contract {symbol_str}")
                    except Exception as e:
                        self.algorithm.Error(f"{self.name}: Error adding symbol data for {symbol_str}: {str(e)}")
                elif not symbol_str.startswith('/'):
                    # This is an underlying contract - used for execution, not signals
                    self.algorithm.Log(f"{self.name}: Ignoring underlying contract {symbol_str} (execution only)")
            
            # Remove securities
            for security in changes.RemovedSecurities:
                symbol = security.Symbol
                if symbol in self.symbol_data:
                    try:
                        self.symbol_data[symbol].Dispose()
                        del self.symbol_data[symbol]
                        self.algorithm.Log(f"{self.name}: Removed symbol data for {symbol}")
                    except Exception as e:
                        self.algorithm.Error(f"{self.name}: Error removing symbol data for {symbol}: {str(e)}")
                        
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in OnSecuritiesChanged: {str(e)}")
    
    # ============================================================================
    # COMMON DATA VALIDATION (Found in all strategies)
    # ============================================================================
    
    def update(self, slice_data):
        """
        Common update method found in all strategies.
        
        Updates symbol data with new market data using the common pattern:
        1. Validate slice data using centralized checker
        2. Update each SymbolData object with new bars
        3. Track last update time
        """
        try:
            if not self._validate_slice_data_centralized(slice_data):
                return
            
            self.last_update_time = slice_data.Time if hasattr(slice_data, 'Time') else self.algorithm.Time
            
            # Update symbol data - common pattern across all strategies
            for symbol, bar in slice_data.Bars.items():
                if symbol in self.symbol_data:
                    symbol_data = self.symbol_data[symbol]
                    # Handle different consolidation method names
                    if hasattr(symbol_data, 'OnDataConsolidated'):
                        symbol_data.OnDataConsolidated(None, bar)
                    elif hasattr(symbol_data, 'OnConsolidated'):
                        symbol_data.OnConsolidated(None, bar)
                
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in update: {str(e)}")
    
    def _validate_slice_data_centralized(self, slice_data):
        """
        Common validation using DataIntegrityChecker - found in all strategies.
        
        Uses centralized data validation when available, with fallback to
        QC native validation that properly handles continuous contracts.
        """
        if not slice_data or not slice_data.Bars:
            return False
        
        # Use centralized validation if available
        if hasattr(self.algorithm, 'data_integrity_checker'):
            return self.algorithm.data_integrity_checker.validate_slice(slice_data) is not None
        
        # Fallback validation using QC built-ins with proper continuous contract handling
        for symbol, bar in slice_data.Bars.items():
            if symbol in self.algorithm.Securities:
                security = self.algorithm.Securities[symbol]
                has_data = security.HasData
                price_valid = security.Price > 0
                
                # For continuous contracts, check if mapped contract is tradeable
                is_tradeable = security.IsTradable
                if not is_tradeable and hasattr(security, 'Mapped') and security.Mapped:
                    mapped_contract = security.Mapped
                    if mapped_contract in self.algorithm.Securities:
                        is_tradeable = self.algorithm.Securities[mapped_contract].IsTradable
                
                # During warmup, be more lenient - just check data availability
                if not (has_data and price_valid and (is_tradeable or self.algorithm.IsWarmingUp)):
                    return False
        return True
    
    # ============================================================================
    # COMMON TARGET GENERATION PIPELINE (Found in all strategies)
    # ============================================================================
    
    def generate_targets(self, slice=None):
        """
        Common target generation pipeline found in all strategies.
        
        Pipeline:
        1. Check if rebalancing is needed
        2. Generate raw signals (strategy-specific)
        3. Apply position limits
        4. Apply volatility targeting
        5. Validate trade sizes
        6. Update tracking variables
        7. Log results
        
        Returns:
            dict: Symbol -> target weight mapping
        """
        try:
            self.algorithm.Log(f"{self.name}: generate_targets() called")
            
            should_rebal = self.should_rebalance(self.algorithm.Time)
            self.algorithm.Log(f"{self.name}: should_rebalance() = {should_rebal}")
            
            if not should_rebal:
                self.algorithm.Log(f"{self.name}: No rebalance needed, returning current targets: {len(self.current_targets)}")
                return self.current_targets
            
            self.algorithm.Log(f"{self.name}: Calling generate_signals()")
            signals = self.generate_signals(slice)
            signal_symbols = [str(symbol) for symbol in signals.keys()]
            self.algorithm.Log(f"{self.name}: generate_signals() returned {len(signals)} signals: {signal_symbols}")
            
            if not signals:
                self.algorithm.Log(f"{self.name}: No signals generated, returning empty targets")
                return {}
            
            # Common processing pipeline found in all strategies
            self.algorithm.Log(f"{self.name}: Applying position limits...")
            targets = self._apply_position_limits(signals)
            self.algorithm.Log(f"{self.name}: After position limits: {len(targets)} targets")
            
            self.algorithm.Log(f"{self.name}: Applying volatility targeting...")
            targets = self._apply_volatility_targeting(targets)
            self.algorithm.Log(f"{self.name}: After volatility targeting: {len(targets)} targets")
            
            self.algorithm.Log(f"{self.name}: Validating trade sizes...")
            targets = self._validate_trade_sizes(targets)
            self.algorithm.Log(f"{self.name}: After trade size validation: {len(targets)} targets")
            
            # Update tracking variables (common across all strategies)
            self.current_targets = targets
            self.last_rebalance_date = self.algorithm.Time.date()
            self.total_rebalances += 1
            
            # Log final targets with readable symbol names
            formatted_targets = {str(symbol): f"{weight:.1%}" for symbol, weight in targets.items()}
            self.algorithm.Log(f"{self.name}: Final targets: {formatted_targets}")
            
            # Log current portfolio positions using LEAN's methods
            self._log_portfolio_positions()
            
            return targets
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error generating targets: {str(e)}")
            import traceback
            self.algorithm.Error(f"{self.name}: Traceback: {traceback.format_exc()}")
            return {}
    
    def _apply_position_limits(self, signals):
        """
        Apply position limits - common across all strategies.
        
        Uses max_position_weight from configuration to cap individual positions.
        """
        try:
            limited_signals = {}
            for symbol, signal in signals.items():
                # Apply maximum position limit
                limited_signal = max(-self.max_position_weight, min(self.max_position_weight, signal))
                limited_signals[symbol] = limited_signal
                
                if abs(limited_signal) != abs(signal):
                    self.algorithm.Log(f"{self.name}: Position limit applied to {symbol}: {signal:.1%} → {limited_signal:.1%}")
            
            return limited_signals
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error applying position limits: {str(e)}")
            return signals
    
    def _apply_volatility_targeting(self, signals):
        """
        Apply volatility targeting - common pattern across all strategies.
        
        Scales positions to achieve target portfolio volatility using
        strategy-specific portfolio volatility calculation.
        """
        try:
            if not signals:
                return signals
            
            # Calculate current portfolio volatility (strategy-specific implementation)
            current_vol = self._calculate_portfolio_volatility(signals)
            
            if current_vol <= 0:
                self.algorithm.Log(f"{self.name}: Invalid portfolio volatility {current_vol:.3f}, using unscaled signals")
                return signals
            
            # Scale to target volatility
            vol_scale = self.target_volatility / current_vol
            scaled_signals = {symbol: signal * vol_scale for symbol, signal in signals.items()}
            
            self.algorithm.Log(f"{self.name}: Volatility targeting - Current: {current_vol:.1%}, "
                             f"Target: {self.target_volatility:.1%}, Scale: {vol_scale:.2f}")
            
            return scaled_signals
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error applying volatility targeting: {str(e)}")
            return signals
    
    def _validate_trade_sizes(self, targets):
        """
        Validate trade sizes - common across all strategies.
        
        Removes positions that are too small to be meaningful.
        """
        try:
            min_trade_size = 0.01  # 1% minimum position size
            validated_targets = {}
            
            for symbol, target in targets.items():
                if abs(target) >= min_trade_size:
                    validated_targets[symbol] = target
                else:
                    self.algorithm.Log(f"{self.name}: Removing small position {symbol}: {target:.3f}")
            
            return validated_targets
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error validating trade sizes: {str(e)}")
            return targets
    
    # ============================================================================
    # COMMON PORTFOLIO ANALYSIS (Found in all strategies)
    # ============================================================================
    
    def get_exposure(self):
        """
        Common exposure calculation found in all strategies.
        
        Returns:
            dict: Portfolio exposure metrics
        """
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
            self.algorithm.Error(f"{self.name}: Error calculating exposure: {str(e)}")
            return {'gross_exposure': 0.0, 'net_exposure': 0.0, 'long_exposure': 0.0,
                   'short_exposure': 0.0, 'num_positions': 0}
    
    def _log_portfolio_positions(self):
        """
        Log current portfolio positions using LEAN's native methods.
        
        Shows total contracts and average prices for transparency.
        """
        try:
            positions = []
            for symbol in self.algorithm.Portfolio.Keys:
                holding = self.algorithm.Portfolio[symbol]
                if holding.Quantity != 0:
                    positions.append(f"{symbol}: {holding.Quantity:.0f} @ ${holding.AveragePrice:.2f}")
            
            if positions:
                self.algorithm.Log(f"{self.name}: Current positions: {', '.join(positions)}")
            else:
                self.algorithm.Log(f"{self.name}: No current positions")
                
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error logging portfolio positions: {str(e)}")
    
    # ============================================================================
    # COMMON AVAILABILITY AND DIAGNOSTICS (Enhanced for all strategies)
    # ============================================================================
    
    @property
    def IsAvailable(self):
        """
        Common availability check with enhanced diagnostics.
        
        Strategy is available if:
        1. Enabled in configuration
        2. Has symbol data
        3. Sufficient symbols are ready (≥50% threshold)
        
        Tracks detailed diagnostics for troubleshooting.
        """
        try:
            # Check if strategy is enabled
            if not hasattr(self, 'config') or not self.config.get('enabled', True):
                self._last_unavailable_reason = "Strategy disabled in configuration"
                return False
            
            # Check if we have symbol data
            if not self.symbol_data:
                self._last_unavailable_reason = "No symbol data initialized"
                return False
            
            # Count ready symbols
            ready_symbols = []
            not_ready_symbols = []
            
            for symbol, symbol_data in self.symbol_data.items():
                if hasattr(symbol_data, 'IsReady') and symbol_data.IsReady:
                    ready_symbols.append(str(symbol))
                else:
                    not_ready_symbols.append(str(symbol))
            
            total_symbols = len(self.symbol_data)
            ready_count = len(ready_symbols)
            required_count = max(1, int(total_symbols * 0.5))  # 50% threshold
            
            # Store diagnostics for logging
            self._availability_diagnostics = {
                'ready_count': ready_count,
                'total_count': total_symbols,
                'required_count': required_count,
                'ready_symbols': ready_symbols,
                'not_ready_symbols': not_ready_symbols
            }
            
            # Check if we have enough ready symbols
            is_available = ready_count >= required_count
            
            if not is_available:
                # Create compact diagnostic message
                not_ready_str = ', '.join(not_ready_symbols[:3])  # Show first 3
                if len(not_ready_symbols) > 3:
                    not_ready_str += f" (+{len(not_ready_symbols)-3} more)"
                
                self._last_unavailable_reason = f"Ready: {ready_count}/{total_symbols} (need {required_count}). Not ready: {not_ready_str}"
            
            return is_available
            
        except Exception as e:
            self._last_unavailable_reason = f"Error checking availability: {str(e)}"
            return False
    
    def log_availability_status(self, force=False):
        """
        Log availability status with enhanced diagnostics.
        
        Args:
            force: If True, log even when available (for debugging)
        """
        if force or not self.IsAvailable:
            if hasattr(self, '_last_unavailable_reason') and self._last_unavailable_reason:
                self.algorithm.Log(f"{self.name}: NOT_AVAILABLE - {self._last_unavailable_reason}")
    
    def get_availability_status(self):
        """
        Get detailed availability status for debugging.
        
        Returns:
            dict: Detailed availability information
        """
        is_available = self.IsAvailable  # This updates diagnostics
        return {
            'is_available': is_available,
            'reason': self._last_unavailable_reason,
            'diagnostics': self._availability_diagnostics.copy() if hasattr(self, '_availability_diagnostics') else {}
        }
    
    # ============================================================================
    # COMMON PERFORMANCE TRACKING (Found in all strategies)
    # ============================================================================
    
    def get_performance_metrics(self):
        """Common performance metrics found in all strategies."""
        try:
            exposure = self.get_exposure()
            return {
                'total_rebalances': self.total_rebalances,
                'trades_executed': self.trades_executed,
                'last_rebalance': str(self.last_rebalance_date) if self.last_rebalance_date else 'Never',
                'current_positions': len(self.current_targets),
                'gross_exposure': exposure['gross_exposure'],
                'net_exposure': exposure['net_exposure']
            }
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error getting performance metrics: {str(e)}")
            return {}
    
    def log_status(self):
        """Common status logging found in all strategies."""
        try:
            metrics = self.get_performance_metrics()
            self.algorithm.Log(f"{self.name} Status: {metrics['current_positions']} positions, "
                             f"{metrics['gross_exposure']:.1%} gross exposure, "
                             f"{metrics['total_rebalances']} rebalances")
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error logging status: {str(e)}")
    
    # ============================================================================
    # COMMON HISTORICAL DATA ACCESS (Found in all strategies)
    # ============================================================================
    
    def get_qc_history(self, symbol, periods, resolution=Resolution.Daily):
        """
        Common historical data access using QC native methods.
        
        Args:
            symbol: Symbol to get history for
            periods: Number of periods
            resolution: Data resolution
            
        Returns:
            List: Historical data or None if failed
        """
        try:
            history = self.algorithm.History(symbol, periods, resolution)
            return self._validate_qc_history(history)
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error getting history for {symbol}: {str(e)}")
            return None
    
    def _validate_qc_history(self, history):
        """
        Validate QuantConnect history data.
        
        Args:
            history: QC history data (MemoizingEnumerable[TradeBar])
            
        Returns:
            List: History data as list or None if empty
        """
        if history is None:
            return None
        
        try:
            # Convert to list to check if empty (QC data doesn't support len() directly)
            history_list = list(history)
            return history_list if len(history_list) > 0 else None
        except Exception:
            return None
    
    # ============================================================================
    # ABSTRACT METHODS (Must be implemented by subclasses)
    # ============================================================================
    
    @abstractmethod
    def generate_signals(self, slice=None):
        """
        Generate raw trading signals.
        
        This is the core strategy logic that must be implemented by each subclass.
        
        Args:
            slice: Optional data slice for futures chain analysis
            
        Returns:
            dict: Symbol -> signal strength mapping
            
        Example:
            return {'/ES': 0.25, '/CL': -0.15, '/GC': 0.10}
        """
        pass
    
    @abstractmethod
    def should_rebalance(self, current_time):
        """
        Determine if strategy should rebalance.
        
        Args:
            current_time: Current algorithm time
            
        Returns:
            bool: True if rebalancing is needed
            
        Example:
            # Monthly rebalancing
            if self.last_rebalance_date is None:
                return True
            return current_time.month != self.last_rebalance_date.month
        """
        pass
    
    @abstractmethod
    def _calculate_portfolio_volatility(self, weights):
        """
        Calculate portfolio volatility for given weights.
        
        This is strategy-specific because different strategies use different:
        - Lookback periods
        - Sampling frequencies (daily vs weekly vs monthly)
        - Correlation calculations
        - Volatility estimation methods
        
        Args:
            weights: dict of symbol -> weight mappings
            
        Returns:
            float: Annualized portfolio volatility
            
        Example:
            # Use strategy-specific lookback and correlation matrix
            return self._calculate_vol_with_correlations(weights, self.vol_lookback_days)
            
        QC Native Helper Available:
            # Use QC native methods with fallbacks
            return self._qc_native_portfolio_volatility(weights)
        """
        pass

    # ============================================================================
    # QC NATIVE VOLATILITY HELPER METHODS (Available to all strategies)
    # ============================================================================
    
    def _qc_native_portfolio_volatility(self, weights):
        """QC Native portfolio volatility with multi-fallback: VolatilityModel → defaults → simple."""
        try:
            if not weights or len(weights) == 0:
                return 0.0
            
            symbols = list(weights.keys())
            weight_array = np.array(list(weights.values()))
            
            # Single asset case
            if len(symbols) == 1:
                symbol = symbols[0]
                vol = self._get_qc_native_volatility_base(symbol)
                return abs(weight_array[0]) * vol
            
            # Multi-asset case: Simple diversified approach
            volatilities = []
            valid_weights = []
            
            for i, symbol in enumerate(symbols):
                vol = self._get_qc_native_volatility_base(symbol)
                if vol and 0.001 <= vol <= 2.0:  # Sanity check
                    volatilities.append(vol)
                    valid_weights.append(abs(weight_array[i]))
            
            if not volatilities:
                return self._ultra_simple_portfolio_vol_base(weights)
            
            # Weighted average with diversification
            weighted_vol = sum(w * v for w, v in zip(valid_weights, volatilities))
            
            # Apply diversification factor based on number of assets
            num_assets = len(volatilities)
            if num_assets == 1:
                diversification_factor = 1.0
            elif num_assets <= 3:
                diversification_factor = 0.8
            elif num_assets <= 5:
                diversification_factor = 0.7
            else:
                diversification_factor = 0.6
            
            portfolio_vol = weighted_vol * diversification_factor
            
            return max(0.05, min(0.60, portfolio_vol))
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in QC native portfolio volatility: {str(e)}")
            return self._ultra_simple_portfolio_vol_base(weights)

    def _get_qc_native_volatility_base(self, symbol):
        """Get volatility using QC's native methods with fallbacks."""
        try:
            # Method 1: QC's native VolatilityModel (best if available)
            if symbol in self.algorithm.Securities:
                security = self.algorithm.Securities[symbol]
                
                if hasattr(security, 'VolatilityModel') and security.VolatilityModel:
                    try:
                        daily_vol = security.VolatilityModel.Volatility
                        if daily_vol and daily_vol > 0:
                            annual_vol = daily_vol * (252 ** 0.5)
                            if 0.001 <= annual_vol <= 2.0:  # Sanity check
                                return annual_vol
                    except:
                        pass
            
            # Method 2: Asset class defaults from centralized config
            return self._get_default_volatility_base(symbol)
            
        except Exception as e:
            self.algorithm.Log(f"{self.name}: Error getting QC native volatility for {symbol}: {str(e)}")
            return self._get_default_volatility_base(symbol)

    def _get_default_volatility_base(self, symbol):
        """Get default volatility based on asset class - available to all strategies."""
        try:
            symbol_str = str(symbol)
            
            # Try to get from centralized config first
            if hasattr(self, 'config_manager'):
                try:
                    full_config = self.config_manager.get_full_config()
                    asset_defaults = full_config.get('asset_defaults', {})
                    volatilities = asset_defaults.get('volatilities', {})
                    
                    # Asset class mapping
                    if any(eq in symbol_str for eq in ['ES', 'NQ', 'YM']):
                        return volatilities.get('equity', 0.20)
                    elif any(bond in symbol_str for bond in ['ZN', 'ZB']):
                        return volatilities.get('fixed_income', 0.08)
                    elif any(fx in symbol_str for fx in ['6E', '6J', '6B']):
                        return volatilities.get('forex', 0.12)
                    elif any(comm in symbol_str for comm in ['CL', 'GC', 'SI']):
                        return volatilities.get('commodity', 0.25)
                    else:
                        return volatilities.get('default', 0.20)
                except:
                    pass
            
            # Fallback to hardcoded asset class defaults
            if any(eq in symbol_str for eq in ['ES', 'NQ', 'YM']):
                return 0.20  # 20% equity futures
            elif any(bond in symbol_str for bond in ['ZN', 'ZB']):
                return 0.08  # 8% bond futures
            elif any(fx in symbol_str for fx in ['6E', '6J', '6B']):
                return 0.12  # 12% FX futures
            elif any(comm in symbol_str for comm in ['CL', 'GC', 'SI']):
                return 0.25  # 25% commodity futures
            else:
                return 0.20  # 20% general default
                
        except Exception as e:
            self.algorithm.Log(f"{self.name}: Error getting default volatility for {symbol}: {str(e)}")
            return 0.20

    def _ultra_simple_portfolio_vol_base(self, weights):
        """Ultra simple portfolio volatility - final fallback for all strategies."""
        try:
            if not weights:
                return 0.20
            
            # Gross exposure approach
            gross_exposure = sum(abs(weight) for weight in weights.values())
            avg_futures_vol = 0.18  # 18% reasonable futures average
            
            # Diversification based on position count
            num_positions = len(weights)
            if num_positions == 1:
                diversification_factor = 1.0
            elif num_positions <= 3:
                diversification_factor = 0.8
            elif num_positions <= 5:
                diversification_factor = 0.7
            else:
                diversification_factor = 0.6
            
            portfolio_vol = gross_exposure * avg_futures_vol * diversification_factor
            return max(0.05, min(0.60, portfolio_vol))
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in ultra simple portfolio vol: {str(e)}")
            return 0.20

    # ============================================================================
    # QC NATIVE INDICATOR HELPERS (Available to all strategies)
    # ============================================================================
    
    def setup_qc_native_std_indicators(self, symbols, period=252):
        """Setup QC native STD indicators for volatility calculation."""
        try:
            std_indicators = {}
            
            for symbol in symbols:
                # Use QC's native STD indicator
                std_indicator = self.algorithm.STD(symbol, period)
                std_indicators[symbol] = std_indicator
                self.algorithm.Log(f"{self.name}: Created QC STD({period}) indicator for {symbol}")
            
            return std_indicators
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error setting up QC STD indicators: {str(e)}")
            return {}

    def setup_qc_native_correlation_indicators(self, symbols, period=60):
        """Setup QC native Correlation indicators for correlation calculation."""
        try:
            corr_indicators = {}
            
            # Create correlation indicators for each unique pair
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    if i < j:  # Only upper triangle to avoid duplicates
                        # Use QC's native C() method for Correlation indicator
                        corr_indicator = self.algorithm.C(sym1, sym2, period)
                        corr_indicators[(sym1, sym2)] = corr_indicator
                        self.algorithm.Log(f"{self.name}: Created QC C({period}) indicator for {sym1} vs {sym2}")
            
            self.algorithm.Log(f"{self.name}: Setup {len(corr_indicators)} QC correlation indicators")
            return corr_indicators
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error setting up QC correlation indicators: {str(e)}")
            return {}

    def get_qc_native_correlation(self, sym1, sym2, corr_indicators, fallback_method='default'):
        """Get correlation using QC's native Correlation indicator with fallbacks."""
        try:
            # Same symbol = perfect correlation
            if sym1 == sym2:
                return 1.0
            
            # Try QC's native Correlation indicator
            if (sym1, sym2) in corr_indicators:
                indicator = corr_indicators[(sym1, sym2)]
                if indicator.IsReady:
                    corr = indicator.Current.Value
                    if -1.0 <= corr <= 1.0:  # Valid correlation range
                        return corr
            elif (sym2, sym1) in corr_indicators:
                indicator = corr_indicators[(sym2, sym1)]
                if indicator.IsReady:
                    corr = indicator.Current.Value
                    if -1.0 <= corr <= 1.0:  # Valid correlation range
                        return corr
            
            # Fallback methods
            if fallback_method == 'default':
                return self._get_asset_class_correlation(sym1, sym2)
            else:
                return 0.3  # Simple default
                
        except Exception as e:
            self.algorithm.Log(f"{self.name}: Error getting QC correlation for {sym1} vs {sym2}: {str(e)}")
            return 0.3 if fallback_method == 'simple' else self._get_asset_class_correlation(sym1, sym2)

    def _get_asset_class_correlation(self, sym1, sym2):
        """Get correlation based on asset class relationships."""
        try:
            s1, s2 = str(sym1), str(sym2)
            
            # Try centralized config first
            if hasattr(self, 'config_manager'):
                try:
                    full_config = self.config_manager.get_full_config()
                    asset_defaults = full_config.get('asset_defaults', {})
                    correlations = asset_defaults.get('correlations', {})
                    
                    # Within asset class correlations
                    if any(eq in s1 for eq in ['ES', 'NQ', 'YM']) and any(eq in s2 for eq in ['ES', 'NQ', 'YM']):
                        return correlations.get('within_equity', 0.85)
                    elif any(bond in s1 for bond in ['ZN', 'ZB']) and any(bond in s2 for bond in ['ZN', 'ZB']):
                        return correlations.get('within_fixed_income', 0.70)
                    elif any(fx in s1 for fx in ['6E', '6J', '6B']) and any(fx in s2 for fx in ['6E', '6J', '6B']):
                        return correlations.get('fx_fx', 0.50)
                    elif any(comm in s1 for comm in ['CL', 'GC', 'SI']) and any(comm in s2 for comm in ['CL', 'GC', 'SI']):
                        return correlations.get('within_commodity', 0.30)
                    
                    # Cross-asset correlations
                    elif (any(eq in s1 for eq in ['ES', 'NQ', 'YM']) and any(bond in s2 for bond in ['ZN', 'ZB'])) or \
                         (any(bond in s1 for bond in ['ZN', 'ZB']) and any(eq in s2 for eq in ['ES', 'NQ', 'YM'])):
                        return correlations.get('equity_bond', -0.10)
                    else:
                        return correlations.get('cross_asset_default', 0.15)
                except:
                    pass
            
            # Hardcoded fallbacks
            if any(eq in s1 for eq in ['ES', 'NQ', 'YM']) and any(eq in s2 for eq in ['ES', 'NQ', 'YM']):
                return 0.85  # Equity-equity
            elif any(bond in s1 for bond in ['ZN', 'ZB']) and any(bond in s2 for bond in ['ZN', 'ZB']):
                return 0.70  # Bond-bond
            elif any(comm in s1 for comm in ['CL', 'GC', 'SI']) and any(comm in s2 for comm in ['CL', 'GC', 'SI']):
                return 0.30  # Commodity-commodity
            elif (any(eq in s1 for eq in ['ES', 'NQ', 'YM']) and any(bond in s2 for bond in ['ZN', 'ZB'])) or \
                 (any(bond in s1 for bond in ['ZN', 'ZB']) and any(eq in s2 for eq in ['ES', 'NQ', 'YM'])):
                return -0.10  # Equity-bond negative correlation
            else:
                return 0.15  # Cross-asset default
                
        except Exception as e:
            self.algorithm.Log(f"{self.name}: Error getting asset class correlation: {str(e)}")
            return 0.15
    
    @abstractmethod
    def _create_symbol_data(self, symbol):
        """
        Create strategy-specific SymbolData object.
        
        Args:
            symbol: Symbol to create data for
            
        Returns:
            SymbolData: Strategy-specific symbol data object
            
        Example:
            return self.SymbolData(self.algorithm, symbol, self.momentum_lookbacks, self.vol_lookback_days)
        """
        pass
    
    # ============================================================================
    # BASE SYMBOL DATA CLASS
    # ============================================================================
    
    class BaseSymbolData:
        """
        Base class for strategy-specific SymbolData implementations.
        
        Provides common functionality for historical data access and QC integration.
        """
        
        def __init__(self, algorithm, symbol):
            """Initialize base symbol data."""
            self.algorithm = algorithm
            self.symbol = symbol
            self.history_initialized = False
        
        def get_qc_history(self, periods, resolution=Resolution.Daily):
            """Get historical data using QC native methods."""
            try:
                history = self.algorithm.History(self.symbol, periods, resolution)
                # Convert to list to check if empty (QC data doesn't support len() directly)
                history_list = list(history)
                return history_list if len(history_list) > 0 else None
            except Exception as e:
                self.algorithm.Error(f"Error getting history for {self.symbol}: {str(e)}")
                return None
        
        @property
        @abstractmethod
        def IsReady(self):
            """Check if symbol data is ready for signal generation."""
            pass
        
        def Dispose(self):
            """Clean up resources."""
            pass

    def update_with_data(self, slice):
        """Update strategy with new market data - LEGACY METHOD."""
        try:
            self.current_slice = slice
            self._update_internal_state(slice)
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in update_with_data: {str(e)}")

    def _update_internal_state(self, slice):
        """Update internal state - can be overridden by derived strategies."""
        pass 