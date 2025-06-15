# hmm_cta_strategy.py - Optimized for size

from AlgorithmImports import *
import numpy as np
from collections import deque
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

class HMMCTAStrategy:
    """Hidden Markov Model CTA Strategy - Optimized Version"""
    
    def __init__(self, algorithm, futures_manager, name="HMM_CTA", config_manager=None):
        self.algorithm = algorithm
        self.futures_manager = futures_manager
        self.name = name
        self.config_manager = config_manager
        
        self._load_configuration()
        
        # Initialize symbol data for HMM analysis
        self.symbol_data = {}
        
        # Strategy state
        self.current_targets = {}
        self.last_rebalance_date = None
        self.last_retrain_month = -1
        self.strategy_returns = []
        self.portfolio_values = []
        self.last_update_time = None
        
        # Regime tracking for persistence
        self.regime_buffers = {}
        self.smoothed_regime_probs = {}
        
        # Performance tracking
        self.trades_executed = 0
        self.total_rebalances = 0
        self.total_retrains = 0
        self.regime_changes = 0
        self.gross_exposure_history = []
        self.regime_persistence_violations = 0
        self.regime_history = {}
        
        self._log_initialization_summary()
    
    def _load_configuration(self):
        """Load configuration from config_manager with fallback handling."""
        try:
            if self.config_manager:
                strategy_config = self.config_manager.get_strategy_config(self.name)
                if strategy_config:
                    self._build_config_dict(strategy_config)
                    self.algorithm.Log(f"{self.name}: Config loaded from config_manager")
                    return
            
            self.algorithm.Log(f"{self.name}: Using fallback configuration")
            self._load_fallback_config()
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Config loading error: {str(e)}")
            self._load_fallback_config()
    
    def _build_config_dict(self, config):
        """Build complete config dictionary from provided config."""
        self.config_dict = {
            'n_components': config.get('n_components', 3),
            'n_iter': config.get('n_iter', 100),
            'random_state': config.get('random_state', 42),
            'returns_window': config.get('returns_window', 60),
            'retrain_frequency': config.get('retrain_frequency', 'monthly'),
            'target_volatility': config.get('target_volatility', 0.20),
            'rebalance_frequency': config.get('rebalance_frequency', 'weekly'),
            'max_position_weight': config.get('max_position_weight', 0.6),
            'regime_threshold': config.get('regime_threshold', 0.50),
            'regime_persistence_days': config.get('regime_persistence_days', 3),
            'regime_smoothing_alpha': config.get('regime_smoothing_alpha', 0.3),
            'warmup_days': config.get('warmup_days', 80),
            'min_weight_threshold': config.get('min_weight_threshold', 0.01),
            'min_trade_value': config.get('min_trade_value', 1000),
            'max_single_order_value': config.get('max_single_order_value', 50000000),
            'max_leverage_multiplier': config.get('max_leverage_multiplier', 100),
            'max_single_position': config.get('max_single_position', 10.0),
            'daily_stop_loss': config.get('daily_stop_loss', 0.2),
            'enabled': config.get('enabled', True),
            'description': config.get('description', 'HMM CTA Strategy'),
            'expected_sharpe': config.get('expected_sharpe', 0.5),
            'correlation_with_regime': config.get('correlation_with_regime', 0.6)
        }
    
    def _load_fallback_config(self):
        """Load fallback configuration if config_manager is unavailable."""
        self.config_dict = {
            'n_components': 3, 'n_iter': 100, 'random_state': 42, 'returns_window': 60,
            'retrain_frequency': 'monthly', 'target_volatility': 0.20, 'rebalance_frequency': 'weekly',
            'max_position_weight': 0.6, 'regime_threshold': 0.50, 'regime_persistence_days': 3,
            'regime_smoothing_alpha': 0.3, 'warmup_days': 80, 'min_weight_threshold': 0.01,
            'min_trade_value': 1000, 'max_single_order_value': 50000000, 'max_leverage_multiplier': 100,
            'max_single_position': 10.0, 'daily_stop_loss': 0.2, 'enabled': True,
            'description': 'HMM CTA Strategy (Fallback Configuration)', 'expected_sharpe': 0.5,
            'correlation_with_regime': 0.6
        }
    
    def _log_initialization_summary(self):
        """Log initialization summary."""
        self.algorithm.Log(f"{self.name}: Initialized with {self.config_dict['n_components']} regimes, "
                          f"{self.config_dict['target_volatility']:.1%} vol target")
    
    def initialize_symbol_data(self):
        """Symbol data is now initialized in OnSecuritiesChanged."""
        self.algorithm.Log(f"{self.name}: Symbol data initialization deferred to OnSecuritiesChanged")

    def update(self, slice_data):
        """Update strategy with new market data."""
        try:
            if not self._validate_slice_data(slice_data):
                return
            
            self.last_update_time = self.algorithm.Time
            
            # Update symbol data with new bars
            for symbol, bar in slice_data.Bars.items():
                if symbol in self.symbol_data:
                    self.symbol_data[symbol].OnConsolidated(None, bar)
            
            # Check for monthly retraining
            if self.should_retrain_models(self.algorithm.Time):
                self.retrain_models()
                
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in update: {str(e)}")
    
    def _validate_slice_data(self, slice_data):
        """OPTIMIZED validation using centralized DataIntegrityChecker"""
        if not slice_data or not slice_data.Bars:
            return False
        
        # Use centralized validation if available
        if hasattr(self.algorithm, 'data_integrity_checker'):
            return self.algorithm.data_integrity_checker.validate_slice(slice_data) is not None
        
        # Fallback validation using QC built-ins
        for symbol, bar in slice_data.Bars.items():
            if symbol in self.algorithm.Securities:
                security = self.algorithm.Securities[symbol]
                if not (security.HasData and security.IsTradable and security.Price > 0):
                    return False
        return True
    
    def generate_targets(self):
        """Generate target positions based on regime analysis."""
        try:
            if not self.should_rebalance(self.algorithm.Time):
                return self.current_targets
            
            signals = self.generate_signals()
            if not signals:
                return {}
            
            # Apply position limits and volatility targeting
            targets = self._apply_position_limits(signals)
            targets = self._apply_volatility_targeting(targets)
            
            # Validate trade sizes
            targets = self._validate_trade_sizes(targets)
            
            self.current_targets = targets
            self.last_rebalance_date = self.algorithm.Time.date()
            self.total_rebalances += 1
            
            self._log_signal_summary(targets)
            return targets
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error generating targets: {str(e)}")
            return {}
    
    def _log_signal_summary(self, targets):
        """Log signal summary."""
        if targets:
            active_positions = len([w for w in targets.values() if abs(w) > 0.01])
            gross_exposure = sum(abs(w) for w in targets.values())
            self.algorithm.Log(f"{self.name}: Generated {active_positions} positions, "
                              f"{gross_exposure:.1%} gross exposure")
    
    def get_exposure(self):
        """Get current strategy exposure."""
        try:
            if not self.current_targets:
                return {
                    'gross_exposure': 0.0,
                    'net_exposure': 0.0,
                    'long_exposure': 0.0,
                    'short_exposure': 0.0,
                    'num_positions': 0
                }
            
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

    def should_rebalance(self, current_time):
        """Check if strategy should rebalance."""
        if not self.last_rebalance_date:
            return True
        
        days_since = (current_time.date() - self.last_rebalance_date).days
        return days_since >= 7 if self.config_dict['rebalance_frequency'] == 'weekly' else days_since >= 30
    
    def should_retrain_models(self, current_time):
        """Check if models should be retrained."""
        return current_time.month != self.last_retrain_month
    
    def retrain_models(self):
        """Retrain HMM models for all symbols."""
        try:
            retrained_count = 0
            for symbol, symbol_data in self.symbol_data.items():
                if symbol_data.IsReady:
                    symbol_data.retrain_model()
                    retrained_count += 1
            
            self.last_retrain_month = self.algorithm.Time.month
            self.total_retrains += 1
            self.algorithm.Log(f"{self.name}: Retrained {retrained_count} models")
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error retraining models: {str(e)}")
    
    def _smooth_regime_probabilities(self, symbol, new_probs):
        """Apply exponential smoothing to regime probabilities."""
        if symbol not in self.smoothed_regime_probs:
            self.smoothed_regime_probs[symbol] = new_probs
            return new_probs
        
        alpha = self.config_dict['regime_smoothing_alpha']
        old_probs = self.smoothed_regime_probs[symbol]
        smoothed = alpha * new_probs + (1 - alpha) * old_probs
        self.smoothed_regime_probs[symbol] = smoothed
        return smoothed
    
    def _check_regime_persistence(self, symbol, current_regime):
        """Check regime persistence to avoid whipsaws."""
        if symbol not in self.regime_buffers:
            self.regime_buffers[symbol] = deque(maxlen=self.config_dict['regime_persistence_days'])
        
        buffer = self.regime_buffers[symbol]
        buffer.append(current_regime)
        
        if len(buffer) < self.config_dict['regime_persistence_days']:
            return current_regime
        
        # Check if regime has been consistent
        if len(set(buffer)) == 1:
            return current_regime
        else:
            self.regime_persistence_violations += 1
            return buffer[0] if buffer else current_regime

    def generate_signals(self):
        """Generate trading signals based on regime analysis."""
        try:
            signals = {}
            liquid_symbols = self._get_liquid_symbols()
            
            if not liquid_symbols:
                return signals
            
            for symbol in liquid_symbols:
                if symbol not in self.symbol_data or not self.symbol_data[symbol].IsReady:
                    continue
                
                try:
                    symbol_data = self.symbol_data[symbol]
                    regime_probs = symbol_data.GetRegimeProbabilities()
                    
                    if regime_probs is None or len(regime_probs) == 0:
                        continue
                    
                    # Smooth probabilities
                    smoothed_probs = self._smooth_regime_probabilities(symbol, regime_probs)
                    
                    # Determine dominant regime
                    dominant_regime = np.argmax(smoothed_probs)
                    max_prob = smoothed_probs[dominant_regime]
                    
                    # Check regime persistence
                    persistent_regime = self._check_regime_persistence(symbol, dominant_regime)
                    
                    # Generate signal based on regime
                    if max_prob > self.config_dict['regime_threshold']:
                        if persistent_regime == 0:  # Bull regime
                            signals[symbol] = 1.0
                        elif persistent_regime == 1:  # Bear regime
                            signals[symbol] = -1.0
                        else:  # Neutral regime
                            signals[symbol] = 0.0
                    else:
                        signals[symbol] = 0.0
                    
                    # Track regime history
                    if symbol not in self.regime_history:
                        self.regime_history[symbol] = deque(maxlen=252)
                    self.regime_history[symbol].append(persistent_regime)
                    
                except Exception as e:
                    self.algorithm.Error(f"{self.name}: Error processing {symbol}: {str(e)}")
                    continue
            
            return signals
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error generating signals: {str(e)}")
            return {}
    
    def _get_liquid_symbols(self):
        """Get liquid symbols from futures manager."""
        if self.futures_manager and hasattr(self.futures_manager, 'get_liquid_symbols'):
            return self.futures_manager.get_liquid_symbols()
        return list(self.symbol_data.keys())
    
    def _apply_position_limits(self, signals):
        """Apply position size limits."""
        limited_signals = {}
        max_weight = self.config_dict['max_position_weight']
        
        for symbol, signal in signals.items():
            limited_signals[symbol] = max(-max_weight, min(max_weight, signal))
        
        return limited_signals
    
    def _apply_volatility_targeting(self, signals):
        """Apply volatility targeting to signals."""
        try:
            if not signals:
                return signals
            
            # Calculate portfolio volatility
            portfolio_vol = self._calculate_portfolio_volatility(signals)
            target_vol = self.config_dict['target_volatility']
            
            if portfolio_vol > 0:
                vol_scalar = target_vol / portfolio_vol
                vol_scalar = min(vol_scalar, self.config_dict['max_leverage_multiplier'])
                
                scaled_signals = {}
                for symbol, weight in signals.items():
                    scaled_signals[symbol] = weight * vol_scalar
                
                return scaled_signals
            
            return signals
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in volatility targeting: {str(e)}")
            return signals
    
    def _validate_trade_sizes(self, targets):
        """Validate and filter trade sizes."""
        validated_targets = {}
        min_threshold = self.config_dict['min_weight_threshold']
        
        for symbol, weight in targets.items():
            if abs(weight) >= min_threshold:
                validated_targets[symbol] = weight
        
        return validated_targets
    
    def _calculate_portfolio_volatility(self, weights):
        """Calculate portfolio volatility."""
        try:
            if not weights:
                return 0.0
            
            # Simple approach: average individual volatilities
            total_vol = 0.0
            count = 0
            
            for symbol, weight in weights.items():
                if symbol in self.symbol_data and self.symbol_data[symbol].IsReady:
                    vol = self.symbol_data[symbol].GetRecentVolatility()
                    if vol > 0:
                        total_vol += abs(weight) * vol
                        count += 1
            
            return total_vol / count if count > 0 else 0.0
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error calculating portfolio volatility: {str(e)}")
            return 0.0

    def execute_trades(self, new_targets, rollover_tags=None):
        """Execute trades with comprehensive error handling."""
        try:
            if not new_targets:
                return {'success': True, 'trades_executed': 0, 'errors': []}
            
            execution_results = {'success': True, 'trades_executed': 0, 'errors': []}
            
            for symbol, target_weight in new_targets.items():
                try:
                    if abs(target_weight) < self.config_dict['min_weight_threshold']:
                        continue
                    
                    # Calculate position size
                    portfolio_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
                    target_value = target_weight * portfolio_value
                    
                    if abs(target_value) < self.config_dict['min_trade_value']:
                        continue
                    
                    # Get current position
                    current_quantity = 0
                    if symbol in self.algorithm.Portfolio:
                        current_quantity = self.algorithm.Portfolio[symbol].Quantity
                    
                    # Calculate required trade
                    if symbol in self.algorithm.Securities:
                        security = self.algorithm.Securities[symbol]
                        if security.Price > 0:
                            contract_value = security.Price * security.SymbolProperties.ContractMultiplier
                            target_quantity = int(target_value / contract_value)
                            trade_quantity = target_quantity - current_quantity
                            
                            if abs(trade_quantity) > 0:
                                # Determine order tag
                                tag = f"{self.name}_Rebalance_{self.algorithm.Time.strftime('%Y%m%d')}"
                                if rollover_tags and symbol in rollover_tags:
                                    tag = rollover_tags[symbol]
                                
                                # Execute trade
                                order_ticket = self.algorithm.MarketOrder(symbol, trade_quantity, tag=tag)
                                
                                if order_ticket:
                                    execution_results['trades_executed'] += 1
                                    self.trades_executed += 1
                                else:
                                    execution_results['errors'].append(f"Failed to place order for {symbol}")
                
                except Exception as e:
                    error_msg = f"Error trading {symbol}: {str(e)}"
                    execution_results['errors'].append(error_msg)
                    self.algorithm.Error(f"{self.name}: {error_msg}")
            
            if execution_results['errors']:
                execution_results['success'] = False
            
            return execution_results
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Critical error in execute_trades: {str(e)}")
            return {'success': False, 'trades_executed': 0, 'errors': [str(e)]}
    
    def get_performance_metrics(self):
        """Get strategy performance metrics."""
        try:
            return {
                'total_trades': self.trades_executed,
                'total_rebalances': self.total_rebalances,
                'total_retrains': self.total_retrains,
                'regime_changes': self.regime_changes,
                'regime_persistence_violations': self.regime_persistence_violations,
                'avg_gross_exposure': np.mean(self.gross_exposure_history) if self.gross_exposure_history else 0.0,
                'strategy_name': self.name,
                'last_update': self.last_update_time,
                'config_description': self.config_dict.get('description', 'HMM CTA Strategy')
            }
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error getting performance metrics: {str(e)}")
            return {'strategy_name': self.name, 'error': str(e)}
    
    def log_status(self):
        """Log current strategy status."""
        try:
            exposure = self.get_exposure()
            self.algorithm.Log(f"{self.name} Status: {exposure['num_positions']} positions, "
                              f"{exposure['gross_exposure']:.1%} gross exposure")
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error logging status: {str(e)}")
    
    def get_config_status(self):
        """Get configuration status."""
        try:
            return {
                'strategy_name': self.name,
                'enabled': self.config_dict.get('enabled', True),
                'rebalance_frequency': self.config_dict.get('rebalance_frequency', 'weekly'),
                'target_volatility': self.config_dict.get('target_volatility', 0.20),
                'max_position_weight': self.config_dict.get('max_position_weight', 0.6),
                'n_components': self.config_dict.get('n_components', 3),
                'regime_threshold': self.config_dict.get('regime_threshold', 0.50),
                'description': self.config_dict.get('description', 'HMM CTA Strategy'),
                'config_source': 'config_manager' if self.config_manager else 'fallback'
            }
        except Exception as e:
            return {'strategy_name': self.name, 'error': str(e)}

    class SymbolData:
        """Optimized SymbolData class for HMM analysis."""
        
        def __init__(self, algorithm, symbol, n_components, n_iter, random_state, returns_window):
            self.algorithm = algorithm
            self.symbol = symbol
            self.n_components = n_components
            self.n_iter = n_iter
            self.random_state = random_state
            self.returns_window = returns_window
            
            # Data storage
            self.daily_returns = deque(maxlen=returns_window * 2)
            self.prices = deque(maxlen=returns_window * 2)
            self.monthly_returns = deque(maxlen=36)
            
            # HMM model
            self.hmm_model = None
            self.last_regime_probs = None
            self.last_retrain_date = None
            
            # Data quality tracking
            self.data_quality_score = 0.0
            self.consecutive_valid_days = 0
            self.total_data_points = 0
            
            self._initialize_with_history()
        
        def _initialize_with_history(self):
            """Initialize with historical data."""
            try:
                history = self.algorithm.History(self.symbol, self.returns_window + 50, Resolution.Daily)
                
                if history.empty:
                    self.algorithm.Log(f"No history available for {self.symbol}")
                    return
                
                # Process historical data
                for time, row in history.iterrows():
                    if hasattr(row, 'close') and row.close > 0:
                        self.prices.append(float(row.close))
                        self.total_data_points += 1
                
                # Calculate returns
                if len(self.prices) > 1:
                    for i in range(1, len(self.prices)):
                        ret = (self.prices[i] / self.prices[i-1]) - 1
                        self.daily_returns.append(ret)
                
                self._update_monthly_returns()
                self._update_data_quality()
                
                # Initial model training
                if self.IsReady:
                    self.retrain_model()
                
                self.algorithm.Log(f"{self.symbol}: Initialized with {len(self.daily_returns)} returns")
                
            except Exception as e:
                self.algorithm.Error(f"Error initializing {self.symbol}: {str(e)}")
        
        @property
        def IsReady(self):
            """Check if symbol data is ready for analysis."""
            return (len(self.daily_returns) >= self.returns_window and 
                   len(self.monthly_returns) >= 12 and
                   self.data_quality_score > 0.7)
        
        def OnConsolidated(self, sender, bar: TradeBar):
            """Process new bar data."""
            try:
                if bar.Close <= 0:
                    return
                
                # Add new price
                self.prices.append(float(bar.Close))
                self.total_data_points += 1
                
                # Calculate return
                if len(self.prices) > 1:
                    ret = (self.prices[-1] / self.prices[-2]) - 1
                    self.daily_returns.append(ret)
                    self.consecutive_valid_days += 1
                
                # Update monthly returns
                self._update_monthly_returns()
                self._update_data_quality()
                
            except Exception as e:
                self.algorithm.Error(f"Error processing bar for {self.symbol}: {str(e)}")
        
        def _update_monthly_returns(self):
            """Update monthly returns calculation."""
            if len(self.daily_returns) < 22:
                return
            
            # Simple monthly return calculation
            monthly_ret = sum(list(self.daily_returns)[-22:])
            self.monthly_returns.append(monthly_ret)
        
        def retrain_model(self):
            """Retrain HMM model."""
            try:
                if not self.IsReady:
                    return False
                
                # Prepare data for HMM
                returns_array = np.array(list(self.daily_returns)).reshape(-1, 1)
                
                # Train Gaussian Mixture Model
                self.hmm_model = GaussianMixture(
                    n_components=self.n_components,
                    max_iter=self.n_iter,
                    random_state=self.random_state
                )
                
                self.hmm_model.fit(returns_array)
                self.last_retrain_date = self.algorithm.Time
                
                return True
                
            except Exception as e:
                self.algorithm.Error(f"Error retraining model for {self.symbol}: {str(e)}")
                return False
        
        def GetRegimeProbabilities(self):
            """Get current regime probabilities."""
            try:
                if not self.hmm_model or len(self.daily_returns) == 0:
                    return None
                
                # Get recent returns for prediction
                recent_returns = np.array(list(self.daily_returns)[-10:]).reshape(-1, 1)
                probs = self.hmm_model.predict_proba(recent_returns)
                
                # Return average probabilities
                self.last_regime_probs = np.mean(probs, axis=0)
                return self.last_regime_probs
                
            except Exception as e:
                self.algorithm.Error(f"Error getting regime probabilities for {self.symbol}: {str(e)}")
                return None
        
        def GetRecentVolatility(self):
            """Get recent volatility estimate."""
            try:
                if len(self.daily_returns) < 30:
                    return 0.0
                
                recent_returns = list(self.daily_returns)[-30:]
                return np.std(recent_returns) * np.sqrt(252)
                
            except Exception as e:
                return 0.0
        
        def _update_data_quality(self):
            """Update data quality score."""
            if self.total_data_points == 0:
                self.data_quality_score = 0.0
                return
            
            # Simple quality score based on consecutive valid days
            self.data_quality_score = min(1.0, self.consecutive_valid_days / 30.0)
        
        def GetDataQuality(self):
            """Get data quality metrics."""
            return {
                'quality_score': self.data_quality_score,
                'total_points': self.total_data_points,
                'consecutive_valid': self.consecutive_valid_days,
                'returns_count': len(self.daily_returns)
            }
        
        def Dispose(self):
            """Clean up resources."""
            self.daily_returns.clear()
            self.prices.clear()
            self.monthly_returns.clear()
            self.hmm_model = None

    def OnSecuritiesChanged(self, changes):
        """Handle securities changes."""
        try:
            # Add new securities
            for security in changes.AddedSecurities:
                symbol = security.Symbol
                if symbol not in self.symbol_data:
                    self.symbol_data[symbol] = self.SymbolData(
                        algorithm=self.algorithm,
                        symbol=symbol,
                        n_components=self.config_dict['n_components'],
                        n_iter=self.config_dict['n_iter'],
                        random_state=self.config_dict['random_state'],
                        returns_window=self.config_dict['returns_window']
                    )
                    self.algorithm.Log(f"{self.name}: Added symbol data for {symbol}")
            
            # Remove securities
            for security in changes.RemovedSecurities:
                symbol = security.Symbol
                if symbol in self.symbol_data:
                    self.symbol_data[symbol].Dispose()
                    del self.symbol_data[symbol]
                    self.algorithm.Log(f"{self.name}: Removed symbol data for {symbol}")
                    
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in OnSecuritiesChanged: {str(e)}")


