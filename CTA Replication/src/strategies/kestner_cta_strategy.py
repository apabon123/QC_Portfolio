# kestner_cta_strategy.py - INHERITS FROM BASE STRATEGY

from AlgorithmImports import *
import numpy as np
from strategies.base_strategy import BaseStrategy

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
        # Initialize base strategy with centralized config
        super().__init__(algorithm, config_manager, strategy_name)
        
        try:
            # All configuration comes from centralized manager
            self._initialize_strategy_components()
            self.algorithm.Log(f"KestnerCTA: Strategy initialized successfully")
            
        except Exception as e:
            error_msg = f"CRITICAL ERROR initializing KestnerCTA: {str(e)}"
            self.algorithm.Error(error_msg)
            raise ValueError(error_msg)
    
    def _initialize_strategy_components(self):
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
            
            self.algorithm.Log(f"KestnerCTA: Initialized with momentum lookbacks {self.momentum_lookbacks}, "
                             f"target volatility {self.target_volatility:.1%}")
            
        except Exception as e:
            error_msg = f"Failed to initialize KestnerCTA components: {str(e)}"
            self.algorithm.Error(f"CRITICAL ERROR: {error_msg}")
            raise ValueError(error_msg)
    
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
                self.algorithm.Log(f"{self.name}: No liquid symbols available for signal generation")
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
                        
                except Exception as e:
                    self.algorithm.Error(f"{self.name}: Error processing {symbol}: {str(e)}")
                    continue
            
            return signals
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error generating signals: {str(e)}")
            return {}
    
    def _get_liquid_symbols(self, slice=None):
        """Get liquid symbols from futures manager with proper slice passing."""
        if self.futures_manager and hasattr(self.futures_manager, 'get_liquid_symbols'):
            liquid_symbols = self.futures_manager.get_liquid_symbols(slice)
            self.algorithm.Log(f"{self.name}: Futures manager returned {len(liquid_symbols)} liquid symbols")
            for symbol in liquid_symbols:
                self.algorithm.Log(f"{self.name}: Liquid symbol: {symbol}")
            return liquid_symbols
        else:
            fallback_symbols = list(self.symbol_data.keys())
            self.algorithm.Log(f"{self.name}: No futures manager, using {len(fallback_symbols)} symbols from symbol_data")
            return fallback_symbols
    
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
                
                if history is None or history.empty:
                    self.algorithm.Error(f"CRITICAL: SymbolData {self.symbol} - No historical data available")
                    self.has_sufficient_data = False
                    self.data_availability_error = "No historical data available"
                    return
                
                # Update rolling windows with historical data
                for index, row in history.iterrows():
                    close_price = row['close']
                    open_price = row['open']
                    
                    self.price_window.Add(close_price)
                    
                    # Calculate daily return
                    if open_price != 0:
                        daily_return = (close_price / open_price) - 1
                        self.ret_window.Add(daily_return)
                    
                    self.data_points_received += 1

                self.algorithm.Log(f"SymbolData {self.symbol}: Initialized with {len(history)} historical bars")

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
