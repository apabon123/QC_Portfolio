# simple_ma_cross_strategy.py - Simple MA Cross Strategy for Testing

from AlgorithmImports import *
import numpy as np
from strategies.base_strategy import BaseStrategy

class SimpleMACrossStrategy(BaseStrategy):
    """
    Simple 5/10 Day Moving Average Crossover Strategy
    - Long when 5DMA > 10DMA
    - Short when 5DMA < 10DMA
    - Minimal warmup (15 days) for quick testing
    """
    
    def __init__(self, algorithm, config_manager, strategy_name):
        """Initialize Simple MA Cross strategy with centralized configuration."""
        # Initialize base strategy with centralized config
        super().__init__(algorithm, config_manager, strategy_name)
        
        try:
            # All configuration comes from centralized manager
            self._initialize_strategy_components()
            self.algorithm.Log(f"SimpleMACross: Strategy initialized successfully")
            
        except Exception as e:
            error_msg = f"CRITICAL ERROR initializing SimpleMACross: {str(e)}"
            self.algorithm.Error(error_msg)
            raise ValueError(error_msg)
    
    def _initialize_strategy_components(self):
        """Initialize Simple MA Cross components using centralized configuration."""
        try:
            # Validate required configuration parameters
            required_params = ['fast_ma_period', 'slow_ma_period', 'target_volatility', 'max_position_weight', 'enabled']
            
            for param in required_params:
                if param not in self.config:
                    error_msg = f"Missing required parameter '{param}' in SimpleMACross configuration"
                    self.algorithm.Error(f"CONFIG ERROR: {error_msg}")
                    raise ValueError(error_msg)
            
            # Initialize strategy parameters from validated config
            self.fast_ma_period = self.config['fast_ma_period']
            self.slow_ma_period = self.config['slow_ma_period']
            self.target_volatility = self.config['target_volatility']
            self.max_position_weight = self.config['max_position_weight']
            
            # Initialize tracking variables
            self.symbol_data = {}
            self.current_targets = {}
            self.last_rebalance_date = None
            self.last_update_time = None
            
            # Performance tracking
            self.trades_executed = 0
            self.total_rebalances = 0
            self.strategy_returns = []
            
            self.algorithm.Log(f"SimpleMACross: Initialized with {self.fast_ma_period}/{self.slow_ma_period} MA periods, "
                             f"target volatility {self.target_volatility:.1%}")
            
        except Exception as e:
            error_msg = f"Failed to initialize SimpleMACross components: {str(e)}"
            self.algorithm.Error(f"CRITICAL ERROR: {error_msg}")
            raise ValueError(error_msg)
    
    def should_rebalance(self, current_time):
        """Determine if strategy should rebalance (daily for testing)."""
        if self.last_rebalance_date is None:
            return True
        
        days_since_rebalance = (current_time.date() - self.last_rebalance_date).days
        return days_since_rebalance >= 1  # Daily rebalancing for testing
    
    def generate_signals(self, slice=None):
        """Generate simple MA crossover signals."""
        try:
            self.algorithm.Log(f"{self.name}: Starting signal generation...")
            signals = {}
            
            # Get liquid symbols (use all available symbols for testing)
            liquid_symbols = self._get_liquid_symbols(slice)
            self.algorithm.Log(f"{self.name}: Found {len(liquid_symbols)} liquid symbols: {[str(s) for s in liquid_symbols]}")
            
            if not liquid_symbols:
                self.algorithm.Log(f"{self.name}: No liquid symbols available for signal generation")
                return signals
            
            for symbol in liquid_symbols:
                try:
                    self.algorithm.Log(f"{self.name}: Processing symbol {symbol}")
                    
                    if symbol not in self.symbol_data:
                        self.algorithm.Log(f"{self.name}: No symbol data for {symbol}")
                        continue
                    
                    symbol_data = self.symbol_data[symbol]
                    if not symbol_data.IsReady:
                        self.algorithm.Log(f"{self.name}: Symbol data not ready for {symbol}")
                        continue
                    
                    # Get moving averages
                    fast_ma = symbol_data.GetFastMA()
                    slow_ma = symbol_data.GetSlowMA()
                    
                    self.algorithm.Log(f"{self.name}: {symbol} - Fast MA: {fast_ma}, Slow MA: {slow_ma}")
                    
                    if fast_ma is None or slow_ma is None:
                        self.algorithm.Log(f"{self.name}: Missing MA values for {symbol}")
                        continue
                    
                    # Generate crossover signal
                    if fast_ma > slow_ma:
                        signal = 1.0  # Long signal
                        self.algorithm.Log(f"{self.name}: {symbol} - LONG signal (Fast > Slow)")
                    elif fast_ma < slow_ma:
                        signal = -1.0  # Short signal
                        self.algorithm.Log(f"{self.name}: {symbol} - SHORT signal (Fast < Slow)")
                    else:
                        signal = 0.0  # No signal
                        self.algorithm.Log(f"{self.name}: {symbol} - NO signal (Fast = Slow)")
                    
                    if abs(signal) > 0.0:
                        signals[symbol] = signal
                        self.algorithm.Log(f"{self.name}: Added signal for {symbol}: {signal}")
                        
                except Exception as e:
                    self.algorithm.Error(f"{self.name}: Error processing {symbol}: {str(e)}")
                    continue
            
            self.algorithm.Log(f"{self.name}: Generated {len(signals)} total signals: {signals}")
            return signals
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error generating signals: {str(e)}")
            import traceback
            self.algorithm.Error(f"{self.name}: Traceback: {traceback.format_exc()}")
            return {}
    
    def _calculate_portfolio_volatility(self, allocation_weights, symbols):
        """
        Calculate portfolio volatility for Simple MA strategy.
        Uses short-term correlations (63 days) appropriate for daily rebalancing.
        """
        try:
            if not allocation_weights or not symbols:
                return 0.0
            
            # Get return data for correlation calculation
            returns_data = {}
            correlation_lookback = self.config.get('volatility_lookback_days', 63)  # Use configured volatility lookback
            
            for symbol in symbols:
                if symbol in allocation_weights and allocation_weights[symbol] != 0:
                    # Get historical returns for correlation calculation
                    history = self.algorithm.History(symbol, correlation_lookback + 1, Resolution.Daily)
                    
                    # Convert to list to check if empty (QC data doesn't support len() directly)
                    if history is not None:
                        history_list = list(history)
                        if len(history_list) > 1:
                            # Calculate daily returns from QC data
                            prices = [bar.Close if hasattr(bar, 'Close') else bar.close for bar in history_list]
                            returns = []
                            for i in range(1, len(prices)):
                                if prices[i-1] > 0:
                                    returns.append((prices[i] / prices[i-1]) - 1)
                            
                            if len(returns) >= 30:  # Need sufficient data (lower threshold for simple strategy)
                                returns_data[symbol] = np.array(returns)
                
            # Handle case with insufficient data (simple fallback)
            if len(returns_data) < 2:
                # Simple weighted average for testing
                total_vol = 0.0
                total_weight = 0.0
                
                for symbol in symbols:
                    if symbol in allocation_weights and allocation_weights[symbol] != 0:
                        # Estimate volatility from recent price movements
                        if symbol in self.algorithm.Securities:
                            # Use a simple default volatility estimate
                            weight = abs(allocation_weights[symbol])
                            vol = 0.15  # Default 15% volatility
                            total_vol += weight * vol
                            total_weight += weight
                
                if total_weight > 0:
                    return total_vol / total_weight
                else:
                    return 0.12  # Default 12% volatility for simple strategy
            
            # Calculate correlation matrix using pandas
            import pandas as pd
            
            # Align returns data to same length
            min_length = min(len(returns) for returns in returns_data.values())
            aligned_returns = {symbol: returns[-min_length:] for symbol, returns in returns_data.items()}
            
            # Create DataFrame
            returns_df = pd.DataFrame(aligned_returns)
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            
            # Replace NaN values with simple default correlations
            correlation_matrix = correlation_matrix.fillna(0.0)
            
            # Apply simple default correlations (less sophisticated than other strategies)
            default_correlation = 0.3  # Simple default correlation
            
            for i, symbol1 in enumerate(correlation_matrix.index):
                for j, symbol2 in enumerate(correlation_matrix.columns):
                    if pd.isna(correlation_matrix.iloc[i, j]) or correlation_matrix.iloc[i, j] == 0:
                        if symbol1 == symbol2:
                            correlation_matrix.iloc[i, j] = 1.0
                        else:
                            correlation_matrix.iloc[i, j] = default_correlation
            
            # Calculate individual volatilities
            volatilities = {}
            for symbol in correlation_matrix.index:
                returns = aligned_returns[symbol]
                vol = np.std(returns) * np.sqrt(252)  # Annualized volatility
                volatilities[symbol] = vol
            
            # Calculate portfolio volatility using covariance matrix
            portfolio_variance = 0.0
            
            for symbol1 in correlation_matrix.index:
                for symbol2 in correlation_matrix.columns:
                    weight1 = allocation_weights.get(symbol1, 0.0)
                    weight2 = allocation_weights.get(symbol2, 0.0)
                    
                    if weight1 != 0 and weight2 != 0:
                        vol1 = volatilities[symbol1]
                        vol2 = volatilities[symbol2]
                        correlation = correlation_matrix.loc[symbol1, symbol2]
                        
                        # Handle extreme correlations
                        correlation = max(-0.9, min(0.9, correlation))
                        
                        covariance = vol1 * vol2 * correlation
                        portfolio_variance += weight1 * weight2 * covariance
            
            portfolio_volatility = np.sqrt(abs(portfolio_variance))
            
            # Sanity check
            if portfolio_volatility <= 0 or np.isnan(portfolio_volatility):
                return 0.12  # Default for simple strategy
            
            return portfolio_volatility
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Portfolio volatility calculation error: {str(e)}")
            return 0.12  # Default for simple strategy

    def _get_liquid_symbols(self, slice=None):
        """Get liquid symbols - use all available symbols for testing."""
        try:
            # For testing, use all symbols we have data for
            available_symbols = list(self.symbol_data.keys())
            self.algorithm.Log(f"{self.name}: Available symbols from symbol_data: {[str(s) for s in available_symbols]}")
            
            # Filter for symbols with data
            liquid_symbols = []
            for symbol in available_symbols:
                if symbol in self.algorithm.Securities:
                    security = self.algorithm.Securities[symbol]
                    if security.HasData and security.Price > 0:
                        liquid_symbols.append(symbol)
                        self.algorithm.Log(f"{self.name}: {symbol} is liquid (HasData: {security.HasData}, Price: {security.Price})")
                    else:
                        self.algorithm.Log(f"{self.name}: {symbol} not liquid (HasData: {security.HasData}, Price: {security.Price})")
            
            return liquid_symbols
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error getting liquid symbols: {str(e)}")
            return []
    
    def _create_symbol_data(self, symbol):
        """Create Simple MA Cross symbol data object."""
        return self.SymbolData(
            self.algorithm,
            symbol,
            self.fast_ma_period,
            self.slow_ma_period
        )
    
    class SymbolData:
        """Simple MA Cross SymbolData for moving average calculations."""

        def __init__(self, algorithm, symbol, fast_period, slow_period):
            self.algorithm = algorithm
            self.symbol = symbol
            self.fast_period = fast_period
            self.slow_period = slow_period
            self.consolidator = None

            # Rolling windows for MA calculations
            max_period = max(fast_period, slow_period)
            self.price_window = RollingWindow[float](max_period + 5)  # Extra buffer
            
            # Track data quality
            self.data_points_received = 0
            self.last_update_time = None

            # Setup consolidator
            try:
                self.consolidator = TradeBarConsolidator(timedelta(days=1))
                self.consolidator.DataConsolidated += self.OnDataConsolidated
                algorithm.SubscriptionManager.AddConsolidator(symbol, self.consolidator)
                algorithm.Log(f"SimpleMACrossSymbolData {symbol}: Consolidator setup successful")
            except Exception as e:
                algorithm.Error(f"SimpleMACrossSymbolData {symbol}: Consolidator setup error: {str(e)}")

            # Initialize with history
            self._initialize_with_history()

        def _initialize_with_history(self):
            """Initialize with historical data."""
            try:
                periods_needed = max(self.fast_period, self.slow_period) + 5
                
                self.algorithm.Log(f"SimpleMACrossSymbolData {self.symbol}: Requesting {periods_needed} periods of history")
                history = self.algorithm.History(self.symbol, periods_needed, Resolution.Daily)
                
                # Convert to list to check if empty (QC data doesn't support len() directly)
                if history is None:
                    history_list = []
                else:
                    history_list = list(history)
                
                if len(history_list) == 0:
                    self.algorithm.Log(f"SimpleMACrossSymbolData {self.symbol}: No historical data available")
                    return
                
                # Add historical prices to window
                for bar in history_list:
                    close_price = float(bar.Close if hasattr(bar, 'Close') else bar.close)
                    self.price_window.Add(close_price)
                    self.data_points_received += 1
                
                self.algorithm.Log(f"SimpleMACrossSymbolData {self.symbol}: Initialized with {self.data_points_received} historical bars")
                
            except Exception as e:
                self.algorithm.Error(f"SimpleMACrossSymbolData {self.symbol}: History initialization error: {str(e)}")

        @property
        def IsReady(self):
            """Check if enough data is available for calculations."""
            required_data = max(self.fast_period, self.slow_period)
            is_ready = self.price_window.Count >= required_data
            if not is_ready:
                self.algorithm.Log(f"SimpleMACrossSymbolData {self.symbol}: Not ready - have {self.price_window.Count}, need {required_data}")
            return is_ready

        def OnDataConsolidated(self, sender, bar: TradeBar):
            """Handle new consolidated bar data."""
            try:
                close_price = float(bar.Close)
                self.price_window.Add(close_price)
                self.data_points_received += 1
                self.last_update_time = bar.Time
                
                if self.data_points_received % 10 == 0:  # Log every 10 bars
                    self.algorithm.Log(f"SimpleMACrossSymbolData {self.symbol}: Updated with bar {self.data_points_received}, price: {close_price}")
                
            except Exception as e:
                self.algorithm.Error(f"SimpleMACrossSymbolData {self.symbol}: Data consolidation error: {str(e)}")

        def GetFastMA(self):
            """Calculate fast moving average."""
            try:
                if self.price_window.Count < self.fast_period:
                    return None
                
                prices = [self.price_window[i] for i in range(self.fast_period)]
                fast_ma = np.mean(prices)
                return fast_ma
                
            except Exception as e:
                self.algorithm.Error(f"SimpleMACrossSymbolData {self.symbol}: Fast MA calculation error: {str(e)}")
                return None

        def GetSlowMA(self):
            """Calculate slow moving average."""
            try:
                if self.price_window.Count < self.slow_period:
                    return None
                
                prices = [self.price_window[i] for i in range(self.slow_period)]
                slow_ma = np.mean(prices)
                return slow_ma
                
            except Exception as e:
                self.algorithm.Error(f"SimpleMACrossSymbolData {self.symbol}: Slow MA calculation error: {str(e)}")
                return None

        def Dispose(self):
            """Clean up resources."""
            try:
                if self.consolidator:
                    self.algorithm.SubscriptionManager.RemoveConsolidator(self.symbol, self.consolidator)
                    self.consolidator = None
            except Exception as e:
                self.algorithm.Error(f"SimpleMACrossSymbolData {self.symbol}: Dispose error: {str(e)}") 