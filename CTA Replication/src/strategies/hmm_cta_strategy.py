# hmm_cta_strategy.py - INHERITS FROM BASE STRATEGY

from AlgorithmImports import *
import numpy as np
from collections import deque
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')
from strategies.base_strategy import BaseStrategy

class HMMCTAStrategy(BaseStrategy):
    """
    Hidden Markov Model CTA Strategy Implementation
    CRITICAL: All configuration comes through centralized config manager only
    """
    
    def __init__(self, algorithm, config_manager, strategy_name):
        """
        Initialize HMM CTA strategy with centralized configuration.
        CRITICAL: NO fallback logic - fail fast if config is invalid.
        """
        # Store algorithm reference first
        self.algorithm = algorithm
        self.config_manager = config_manager
        self.name = strategy_name
        
        # Initialize basic attributes that are expected
        self.current_targets = {}
        self.symbol_data = {}
        self.last_rebalance_date = None
        self.last_update_time = None
        self.trades_executed = 0
        self.total_rebalances = 0
        self.strategy_returns = []
        
        try:
            algorithm.Log(f"HMMCTA: Starting initialization for {strategy_name}")
            
            # Get configuration directly to avoid base class issues
            try:
                self.config = config_manager.get_strategy_config(strategy_name)
                algorithm.Log("HMMCTA: Configuration loaded successfully")
            except Exception as e:
                algorithm.Error(f"HMMCTA: Failed to load config: {str(e)}")
                raise
            
            # Validate strategy is enabled
            if not self.config.get('enabled', False):
                error_msg = f"Strategy {strategy_name} is not enabled in configuration"
                algorithm.Error(f"STRATEGY ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            # Initialize HMM-specific components FIRST
            self._initialize_hmm_components()
            algorithm.Log("HMMCTA: HMM components initialized successfully")
            
            # Now call base class initialization (if needed)
            try:
                # Call parent __init__ but handle failures gracefully
                super().__init__(algorithm, config_manager, strategy_name)
                algorithm.Log("HMMCTA: Base strategy initialization completed")
            except Exception as e:
                # Log the error but don't fail - we have what we need
                algorithm.Log(f"HMMCTA: Base strategy initialization failed: {str(e)}, continuing anyway")
            
            algorithm.Log("HMMCTA: Strategy initialized successfully")
            
        except Exception as e:
            error_msg = f"CRITICAL ERROR initializing HMMCTA: {str(e)}"
            algorithm.Error(error_msg)
            raise ValueError(error_msg)
    
    def _initialize_strategy_components(self):
        """Override base class method to prevent double initialization."""
        # This method is called by the base class, but we handle initialization ourselves
        # in _initialize_hmm_components, so this is just a safe no-op
        pass
    
    def _initialize_hmm_components(self):
        """Initialize HMM-specific components using centralized configuration."""
        try:
            # Validate required configuration parameters
            required_params = [
                'n_components', 'returns_window', 'target_volatility',
                'max_position_weight', 'warmup_days', 'enabled'
            ]
            
            for param in required_params:
                if param not in self.config:
                    error_msg = f"Missing required parameter '{param}' in HMMCTA configuration"
                    self.algorithm.Error(f"CONFIG ERROR: {error_msg}")
                    raise ValueError(error_msg)
            
            # Initialize strategy parameters from validated config
            self.n_components = self.config['n_components']
            self.returns_window = self.config['returns_window']
            self.target_volatility = self.config['target_volatility']
            self.max_position_weight = self.config['max_position_weight']
            self.warmup_days = self.config['warmup_days']
            self.regime_threshold = self.config.get('regime_threshold', 0.50)
            self.regime_persistence_days = self.config.get('regime_persistence_days', 3)
            self.regime_smoothing_alpha = self.config.get('regime_smoothing_alpha', 0.3)
            
            # Initialize strategy-specific tracking
            self.last_retrain_month = -1
            self.regime_buffers = {}
            self.smoothed_regime_probs = {}
            self.total_retrains = 0
            self.regime_changes = 0
            self.regime_persistence_violations = 0
            self.regime_history = {}
            
            # Initialize symbol data for continuous contracts only (basic attributes already initialized in __init__)
            self._initialize_symbol_data()
            
            self.algorithm.Log(f"HMMCTA: Initialized with {self.n_components} components, "
                             f"target volatility {self.target_volatility:.1%}")
            
        except Exception as e:
            error_msg = f"Failed to initialize HMM components: {str(e)}"
            self.algorithm.Error(f"CRITICAL ERROR: {error_msg}")
            raise ValueError(error_msg)
    
    def _initialize_symbol_data(self):
        """Initialize symbol data for continuous contracts only."""
        try:
            # Get all futures symbols from the algorithm
            futures_symbols = []
            for symbol in self.algorithm.Securities.Keys:
                security = self.algorithm.Securities[symbol]
                if security.Type == SecurityType.Future:
                    symbol_str = str(symbol)
                    # CRITICAL FIX: Only use continuous contracts for HMM signal generation
                    # Continuous contracts (like /ES, /CL, /GC) have full historical data
                    # Underlying contracts (like ES WLF0Z3JIJTA9) have limited history and cause availability issues
                    if symbol_str.startswith('/'):
                        futures_symbols.append(symbol)
                        self.algorithm.Log(f"{self.name}: Using continuous contract {symbol_str} for signal generation")
                    else:
                        self.algorithm.Log(f"{self.name}: Ignoring underlying contract {symbol_str} (insufficient history)")
            
            self.algorithm.Log(f"{self.name}: Initializing symbol data for {len(futures_symbols)} continuous contract futures symbols")
            
            # Create symbol data for each continuous contract
            for symbol in futures_symbols:
                try:
                    symbol_data = self._create_symbol_data(symbol)
                    self.symbol_data[symbol] = symbol_data
                    self.algorithm.Log(f"{self.name}: Created symbol data for {symbol}")
                except Exception as e:
                    self.algorithm.Error(f"{self.name}: Failed to create symbol data for {symbol}: {str(e)}")
            
            self.algorithm.Log(f"{self.name}: Symbol data initialized for {len(self.symbol_data)} continuous contract symbols")
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error initializing symbol data: {str(e)}")
            # Don't fail completely - symbol data can be created on-demand
    
    def should_rebalance(self, current_time):
        """Determine if strategy should rebalance (weekly)."""
        if self.last_rebalance_date is None:
            return True
        
        days_since_rebalance = (current_time.date() - self.last_rebalance_date).days
        return days_since_rebalance >= 7  # Weekly rebalancing
    
    def should_retrain_models(self, current_time):
        """Check if models should be retrained (monthly)."""
        return current_time.month != self.last_retrain_month
    
    def retrain_models(self):
        """Retrain HMM models for all symbols."""
        try:
            retrained_count = 0
            for symbol, symbol_data in self.symbol_data.items():
                if symbol_data.IsReady:
                    try:
                        symbol_data.retrain_model()
                        retrained_count += 1
                    except Exception as e:
                        self.algorithm.Error(f"{self.name}: Error retraining {symbol}: {str(e)}")
            
            self.last_retrain_month = self.algorithm.Time.month
            self.total_retrains += 1
            self.algorithm.Log(f"{self.name}: Retrained {retrained_count} models")
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in retrain_models: {str(e)}")
    
    def generate_signals(self, slice=None):
        """
        Generate HMM regime-based signals for all liquid symbols.
        
        Args:
            slice: Optional data slice for futures chain analysis
            
        Returns:
            dict: Symbol -> signal strength mapping
        """
        try:
            signals = {}
            liquid_symbols = self._get_liquid_symbols(slice)
            
            if not liquid_symbols:
                self.algorithm.Log(f"{self.name}: No liquid symbols for signal generation")
                return signals
            
            # Check if monthly retraining is needed
            if self.should_retrain_models(self.algorithm.Time):
                self.retrain_models()
            
            for symbol in liquid_symbols:
                if symbol not in self.symbol_data:
                    continue
                
                symbol_data = self.symbol_data[symbol]
                if not symbol_data.IsReady:
                    continue
                
                try:
                    # Get regime probabilities
                    regime_probs = symbol_data.GetRegimeProbabilities()
                    if regime_probs is None or len(regime_probs) == 0:
                        continue
                    
                    # Apply regime smoothing
                    smoothed_probs = self._smooth_regime_probabilities(symbol, regime_probs)
                    
                    # Determine dominant regime
                    dominant_regime = np.argmax(smoothed_probs)
                    regime_confidence = smoothed_probs[dominant_regime]
                    
                    # Check regime persistence
                    if self._check_regime_persistence(symbol, dominant_regime):
                        # Generate signal based on regime
                        if regime_confidence > self.config['regime_threshold']:
                            # Regime-based position sizing
                            if dominant_regime == 0:  # Trending up regime
                                signal = regime_confidence * 0.5
                            elif dominant_regime == 1:  # Ranging regime
                                signal = 0.0  # Flat position
                            else:  # Trending down regime
                                signal = -regime_confidence * 0.5
                            
                            if abs(signal) > 0.01:
                                signals[symbol] = signal
                                
                                # Track regime history
                                if symbol not in self.regime_history:
                                    self.regime_history[symbol] = deque(maxlen=30)
                                self.regime_history[symbol].append(dominant_regime)
                        
                except Exception as e:
                    self.algorithm.Error(f"{self.name}: Error processing {symbol}: {str(e)}")
                    continue
            
            return signals
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error generating signals: {str(e)}")
            return {}
    
    def _smooth_regime_probabilities(self, symbol, new_probs):
        """Apply exponential smoothing to regime probabilities."""
        alpha = self.config['regime_smoothing_alpha']
        
        if symbol not in self.smoothed_regime_probs:
            self.smoothed_regime_probs[symbol] = new_probs
        else:
            # Exponential smoothing
            smoothed = []
            for i, new_prob in enumerate(new_probs):
                old_prob = self.smoothed_regime_probs[symbol][i] if i < len(self.smoothed_regime_probs[symbol]) else new_prob
                smoothed_prob = alpha * new_prob + (1 - alpha) * old_prob
                smoothed.append(smoothed_prob)
            self.smoothed_regime_probs[symbol] = smoothed
        
        return self.smoothed_regime_probs[symbol]
    
    def _check_regime_persistence(self, symbol, current_regime):
        """Check if regime change meets persistence requirements."""
        persistence_days = self.config['regime_persistence_days']
        
        if symbol not in self.regime_buffers:
            self.regime_buffers[symbol] = deque(maxlen=persistence_days)
        
        buffer = self.regime_buffers[symbol]
        buffer.append(current_regime)
        
        if len(buffer) < persistence_days:
            return False
        
        # Check if all recent regimes are the same
        if all(regime == current_regime for regime in buffer):
            return True
        else:
            self.regime_persistence_violations += 1
            return False
    
    def _calculate_portfolio_volatility(self, allocation_weights, symbols):
        """
        Calculate portfolio volatility using covariance matrix approach.
        HMM uses longer-term correlations (252 days) for regime-based decisions.
        """
        try:
            if not allocation_weights or not symbols:
                return 0.0
            
            # Get return data for correlation calculation
            returns_data = {}
            correlation_lookback = self.config.get('volatility_lookback_days', 252)  # Use configured volatility lookback
            
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
                            
                            if len(returns) >= 100:  # Need sufficient data
                                returns_data[symbol] = np.array(returns)
                
            # Handle case with insufficient data
            if len(returns_data) < 2:
                # Fallback to simple weighted average (conservative estimate)
                total_vol = 0.0
                total_weight = 0.0
                
                for symbol in symbols:
                    if symbol in allocation_weights and allocation_weights[symbol] != 0:
                        # Use HMM's symbol data for volatility if available
                        if symbol in self.symbol_data:
                            vol = self.symbol_data[symbol].GetRecentVolatility()
                            if vol is not None and vol > 0:
                                weight = abs(allocation_weights[symbol])
                                total_vol += weight * vol
                                total_weight += weight
                
                if total_weight > 0:
                    return total_vol / total_weight
                else:
                    return 0.15  # Default 15% volatility for HMM
            
            # Calculate correlation matrix using pandas
            import pandas as pd
            
            # Align returns data to same length
            min_length = min(len(returns) for returns in returns_data.values())
            aligned_returns = {symbol: returns[-min_length:] for symbol, returns in returns_data.items()}
            
            # Create DataFrame
            returns_df = pd.DataFrame(aligned_returns)
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            
            # Replace NaN values with default correlations
            correlation_matrix = correlation_matrix.fillna(0.0)
            
            # Apply default correlations for missing values
            default_correlations = {
                ('equity', 'equity'): 0.7,
                ('commodity', 'commodity'): 0.4,
                ('fx', 'fx'): 0.3,
                ('rates', 'rates'): 0.8,
                ('equity', 'commodity'): 0.1,
                ('equity', 'fx'): -0.1,
                ('equity', 'rates'): -0.3,
                ('commodity', 'fx'): 0.2,
                ('commodity', 'rates'): -0.1,
                ('fx', 'rates'): -0.2,
            }
            
            # Apply default correlations
            for i, symbol1 in enumerate(correlation_matrix.index):
                for j, symbol2 in enumerate(correlation_matrix.columns):
                    if pd.isna(correlation_matrix.iloc[i, j]) or correlation_matrix.iloc[i, j] == 0:
                        category1 = self._get_symbol_category(symbol1)
                        category2 = self._get_symbol_category(symbol2)
                        
                        if symbol1 == symbol2:
                            correlation_matrix.iloc[i, j] = 1.0
                        else:
                            key = tuple(sorted([category1, category2]))
                            correlation_matrix.iloc[i, j] = default_correlations.get(key, 0.1)
            
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
                        correlation = max(-0.95, min(0.95, correlation))
                        
                        covariance = vol1 * vol2 * correlation
                        portfolio_variance += weight1 * weight2 * covariance
            
            portfolio_volatility = np.sqrt(abs(portfolio_variance))
            
            # Sanity check
            if portfolio_volatility <= 0 or np.isnan(portfolio_volatility):
                return 0.15  # Default for HMM
            
            return portfolio_volatility
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Portfolio volatility calculation error: {str(e)}")
            return 0.15  # Default for HMM

    def OnSecuritiesChanged(self, changes):
        """Handle securities changes - continuous contracts only for HMM."""
        try:
            # Add new securities - but only continuous contracts for HMM
            for security in changes.AddedSecurities:
                symbol = security.Symbol
                symbol_str = str(symbol)
                
                # Only add continuous contracts for signal generation
                if symbol_str.startswith('/') and symbol not in self.symbol_data:
                    try:
                        self.symbol_data[symbol] = self._create_symbol_data(symbol)
                        self.algorithm.Log(f"{self.name}: Added continuous contract symbol data for {symbol}")
                    except Exception as e:
                        self.algorithm.Error(f"{self.name}: Failed to create symbol data for {symbol}: {str(e)}")
                elif not symbol_str.startswith('/'):
                    self.algorithm.Log(f"{self.name}: Ignoring underlying contract {symbol_str} (not used for signals)")
            
            # Remove securities - only remove if we were tracking them
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
                                # Post-warmup: require both data and tradeable
                                if security.HasData and is_tradeable:
                                    liquid_symbols.append(symbol)
            
            self.algorithm.Log(f"{self.name}: Found {len(liquid_symbols)} liquid continuous contract symbols")
            
            # Fallback to symbol_data keys if no symbols found (should be continuous contracts only)
            if not liquid_symbols and hasattr(self, 'symbol_data'):
                liquid_symbols = list(self.symbol_data.keys())
                self.algorithm.Log(f"{self.name}: No liquid symbols from Securities, using {len(liquid_symbols)} from symbol_data")
            
            return liquid_symbols
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error getting liquid symbols: {str(e)}")
            # Ultimate fallback
            if hasattr(self, 'symbol_data'):
                return list(self.symbol_data.keys())
            else:
                return []
    
    def _create_symbol_data(self, symbol):
        """Create HMM-specific symbol data object."""
        return self.SymbolData(
            self.algorithm,
            symbol,
            self.config['n_components'],
            self.config['n_iter'],
            self.config['random_state'],
            self.config['returns_window']
        )
    
    class SymbolData:
        """HMM-specific SymbolData for regime analysis."""

        def __init__(self, algorithm, symbol, n_components, n_iter, random_state, returns_window):
            self.algorithm = algorithm
            self.symbol = symbol
            self.n_components = n_components
            self.n_iter = n_iter
            self.random_state = random_state
            self.returns_window = returns_window

            # Rolling windows
            self.price_window = RollingWindow[float](returns_window + 50)
            self.returns_window_data = RollingWindow[float](returns_window)
            
            # HMM model
            self.hmm_model = None
            self.last_retrain_time = None
            
            # Track data quality
            self.data_points_received = 0
            self.last_update_time = None
            self.has_sufficient_data = True
            self.data_availability_error = None

            # Setup consolidator
            try:
                self.consolidator = TradeBarConsolidator(timedelta(days=1))
                self.consolidator.DataConsolidated += self.OnConsolidated
                algorithm.SubscriptionManager.AddConsolidator(symbol, self.consolidator)
            except Exception as e:
                algorithm.Log(f"HMM SymbolData {symbol}: Consolidator setup error: {str(e)}")

            # Initialize with history
            self._initialize_with_history()

        def _initialize_with_history(self):
            """Initialize with historical data using CENTRALIZED data provider."""
            try:
                periods_needed = self.returns_window + 50
                
                # Use centralized data provider if available
                if hasattr(self.algorithm, 'data_integrity_checker') and self.algorithm.data_integrity_checker:
                    history = self.algorithm.data_integrity_checker.get_history(self.symbol, periods_needed, Resolution.Daily)
                else:
                    # Fallback to direct API call (not recommended)
                    self.algorithm.Log(f"HMM SymbolData {self.symbol}: WARNING - No centralized cache, using direct History API")
                    history = self.algorithm.History(self.symbol, periods_needed, Resolution.Daily)
                
                # Convert to list to check if empty (QC data doesn't support len() directly)
                if history is None:
                    history_list = []
                else:
                    history_list = list(history)
                
                if len(history_list) == 0:
                    self.algorithm.Log(f"No history available for {self.symbol}")
                    self.has_sufficient_data = False
                    self.data_availability_error = "No historical data available"
                    return
                
                # Process historical data
                prev_close = None
                for bar in history_list:
                    close_price = bar.Close if hasattr(bar, 'Close') else bar.close
                    self.price_window.Add(close_price)
                    
                    # Calculate returns
                    if prev_close is not None and prev_close > 0:
                        daily_return = (close_price / prev_close) - 1
                        self.returns_window_data.Add(daily_return)
                    
                    prev_close = close_price
                    self.data_points_received += 1
                
                self.algorithm.Log(f"HMM SymbolData {self.symbol}: Initialized with {len(history_list)} bars")

            except Exception as e:
                self.algorithm.Error(f"HMM SymbolData {self.symbol}: History initialization error: {str(e)}")
                self.has_sufficient_data = False
                self.data_availability_error = f"History error: {str(e)}"

        @property
        def IsReady(self):
            """Check if symbol data is ready for HMM analysis."""
            if not self.has_sufficient_data:
                return False
            
            return (self.returns_window_data.Count >= self.returns_window and 
                   self.price_window.Count >= self.returns_window + 10)

        def OnConsolidated(self, sender, bar: TradeBar):
            """Process new daily bar."""
            if bar is None or bar.Close <= 0:
                return
            
            try:
                # Update price window
                prev_price = self.price_window[0] if self.price_window.Count > 0 else bar.Close
                self.price_window.Add(float(bar.Close))
                
                # Calculate daily return
                if prev_price > 0:
                    daily_return = (bar.Close / prev_price) - 1
                    self.returns_window_data.Add(daily_return)
                
                self.data_points_received += 1
                self.last_update_time = bar.Time
                
            except Exception as e:
                self.algorithm.Error(f"HMM SymbolData {self.symbol}: OnConsolidated error: {str(e)}")

        def retrain_model(self):
            """Retrain the HMM model with current return data."""
            try:
                if not self.IsReady:
                    return False
                
                # Get return data for training
                returns_data = [self.returns_window_data[i] for i in range(self.returns_window_data.Count)]
                returns_array = np.array(returns_data).reshape(-1, 1)
                
                # Train Gaussian Mixture Model (as HMM proxy)
                self.hmm_model = GaussianMixture(
                    n_components=self.n_components,
                    max_iter=self.n_iter,
                    random_state=self.random_state,
                    covariance_type='full'
                )
                
                self.hmm_model.fit(returns_array)
                self.last_retrain_time = self.algorithm.Time
                
                return True
                
            except Exception as e:
                self.algorithm.Error(f"HMM {self.symbol}: Model training error: {str(e)}")
                return False

        def GetRegimeProbabilities(self):
            """Get current regime probabilities."""
            try:
                if not self.IsReady or self.hmm_model is None:
                    return None
                
                # Get recent returns for prediction
                recent_returns = [self.returns_window_data[i] for i in range(min(5, self.returns_window_data.Count))]
                if len(recent_returns) == 0:
                    return None
                
                recent_array = np.array(recent_returns).reshape(-1, 1)
                
                # Predict probabilities
                probs = self.hmm_model.predict_proba(recent_array)
                # Return average probabilities over recent period
                avg_probs = np.mean(probs, axis=0)
                
                return avg_probs.tolist()
                
            except Exception as e:
                return None

        def GetRecentVolatility(self):
            """Calculate recent volatility for risk management."""
            try:
                if self.returns_window_data.Count < 20:
                    return None
                
                recent_returns = [self.returns_window_data[i] for i in range(min(20, self.returns_window_data.Count))]
                vol = np.std(recent_returns) * np.sqrt(252)  # Annualized
                return vol if vol > 0 else None
                
            except Exception as e:
                return None

        def GetDataQuality(self):
            """Get data quality metrics."""
            return {
                'symbol': str(self.symbol),
                'data_points_received': self.data_points_received,
                'price_window_count': self.price_window.Count,
                'returns_window_count': self.returns_window_data.Count,
                'returns_window_size': self.returns_window,
                'n_components': self.n_components,
                'has_model': self.hmm_model is not None,
                'is_ready': self.IsReady,
                'last_update': self.last_update_time,
                'last_retrain': self.last_retrain_time
            }

        def Dispose(self):
            """Clean disposal of resources."""
            try:
                if hasattr(self, 'consolidator') and self.consolidator:
                    self.algorithm.SubscriptionManager.RemoveConsolidator(self.symbol, self.consolidator)
                self.price_window.Reset()
                self.returns_window_data.Reset()
                self.hmm_model = None
            except:
                pass


