# mtum_cta_strategy.py - MTUM Futures Adaptation (Optimized)

from AlgorithmImports import *
import numpy as np
from collections import deque
from strategies.base_strategy import BaseStrategy

class EfficientPortfolioVolatility:
    """Efficient portfolio volatility using variance-covariance matrix."""
    
    def __init__(self, algorithm, symbols, lookback_days=252):
        self.algorithm = algorithm
        self.symbols = list(symbols)
        self.lookback_days = lookback_days
        self.return_data = {}
        for symbol in self.symbols:
            self.return_data[symbol] = deque(maxlen=lookback_days)
        self.covariance_matrix = None
        self.last_update_time = None
        self.update_frequency_days = 5
        algorithm.Log(f"EfficientPortfolioVol: Initialized for {len(symbols)} assets")
    
    def update_returns(self, slice_data):
        """Update return data with new market data."""
        try:
            for symbol in self.symbols:
                if symbol in slice_data.Bars:
                    bar = slice_data.Bars[symbol]
                    daily_return = 0.0
                    if len(self.return_data[symbol]) > 0:
                        prev_price = self.return_data[symbol][-1]['price']
                        if prev_price > 0:
                            daily_return = (bar.Close - prev_price) / prev_price
                    
                    self.return_data[symbol].append({
                        'price': bar.Close,
                        'return': daily_return,
                        'time': slice_data.Time
                    })
        except Exception as e:
            self.algorithm.Error(f"EfficientPortfolioVol: Error updating returns: {str(e)}")
    
    def should_update_covariance(self):
        """Check if covariance matrix needs updating."""
        if self.covariance_matrix is None or self.last_update_time is None:
            return True
        days_since_update = (self.algorithm.Time - self.last_update_time).days
        return days_since_update >= self.update_frequency_days
    
    def calculate_covariance_matrix(self):
        """Calculate variance-covariance matrix from return data."""
        try:
            # Only consider symbols that have at least one data point
            lengths = [len(self.return_data[s]) for s in self.symbols if len(self.return_data[s]) > 0]
            if not lengths:
                # No data yet – cannot build covariance matrix
                return None

            min_data_points = min(lengths)
            if min_data_points < 30:
                # Insufficient history – wait until we have at least a month of data
                return None
            
            return_matrix = []
            for symbol in self.symbols:
                returns = [data['return'] for data in self.return_data[symbol] if data['return'] != 0]
                if len(returns) < 30:
                    return None
                return_matrix.append(returns[-min_data_points:])
            
            return_matrix = np.array(return_matrix)
            covariance_matrix = np.cov(return_matrix) * 252
            
            self.covariance_matrix = covariance_matrix
            self.last_update_time = self.algorithm.Time
            
            self.algorithm.Debug(f"EfficientPortfolioVol: Updated covariance matrix - "
                               f"shape: {covariance_matrix.shape}, data points: {min_data_points}")
            return covariance_matrix
            
        except Exception as e:
            self.algorithm.Error(f"EfficientPortfolioVol: Error calculating covariance: {str(e)}")
            return None
    
    def calculate_portfolio_volatility(self, weights_dict):
        """Calculate portfolio volatility: σ_p = √(w^T × Σ × w)"""
        try:
            if not weights_dict:
                return 0.20
            
            if self.should_update_covariance():
                self.calculate_covariance_matrix()
            
            if self.covariance_matrix is None:
                return self._fallback_volatility_estimation(weights_dict)
            
            weight_vector = np.array([weights_dict.get(symbol, 0.0) for symbol in self.symbols])
            portfolio_variance = np.dot(weight_vector, np.dot(self.covariance_matrix, weight_vector))
            portfolio_volatility = np.sqrt(max(0, portfolio_variance))
            
            return float(portfolio_volatility)
            
        except Exception as e:
            self.algorithm.Error(f"EfficientPortfolioVol: Error in portfolio volatility: {str(e)}")
            return self._fallback_volatility_estimation(weights_dict)
    
    def _fallback_volatility_estimation(self, weights_dict):
        """Simple fallback when covariance matrix unavailable."""
        gross_exposure = sum(abs(w) for w in weights_dict.values())
        return gross_exposure * 0.18 * 0.7  # avg_vol * diversification_factor

class MTUMCTAStrategy(BaseStrategy):
    """MTUM CTA Strategy - Futures Adaptation"""
    
    def __init__(self, algorithm, config_manager, strategy_name):
        """Initialize MTUM CTA strategy."""
        self.algorithm = algorithm
        self.config_manager = config_manager
        self.name = strategy_name
        self.current_targets = {}
        self.symbol_data = {}
        self.last_rebalance_date = None
        self.last_update_time = None
        self.trades_executed = 0
        self.total_rebalances = 0
        self.strategy_returns = []
        
        try:
            algorithm.Log(f"MTUMCTA: Starting initialization for {strategy_name}")
            
            self.config = config_manager.get_strategy_config(strategy_name)
            algorithm.Log("MTUMCTA: Configuration loaded successfully")
            
            if not self.config.get('enabled', False):
                error_msg = f"Strategy {strategy_name} is not enabled in configuration"
                algorithm.Error(f"STRATEGY ERROR: {error_msg}")
                raise ValueError(error_msg)
            
            self._initialize_mtum_components()
            algorithm.Log("MTUMCTA: MTUM components initialized successfully")
            
            try:
                super().__init__(algorithm, config_manager, strategy_name)
                algorithm.Log("MTUMCTA: Base strategy initialization completed")
            except Exception as e:
                algorithm.Log(f"MTUMCTA: Base strategy initialization failed: {str(e)}, continuing anyway")
            
            algorithm.Log("MTUMCTA: Strategy initialized successfully")
            
        except Exception as e:
            error_msg = f"CRITICAL ERROR initializing MTUMCTA: {str(e)}"
            algorithm.Error(error_msg)
            raise ValueError(error_msg)
    
    def _initialize_strategy_components(self):
        """Override base class method to prevent double initialization."""
        pass
    
    def _initialize_mtum_components(self):
        """Initialize MTUM-specific components."""
        try:
            required_params = [
                'momentum_lookbacks_months', 'volatility_lookback_days', 'target_volatility',
                'max_position_weight', 'warmup_days', 'enabled'
            ]
            
            for param in required_params:
                if param not in self.config:
                    error_msg = f"Missing required parameter '{param}' in MTUM configuration"
                    self.algorithm.Error(f"CONFIG ERROR: {error_msg}")
                    raise ValueError(error_msg)
            
            # Core MTUM parameters
            self.momentum_lookbacks = self.config['momentum_lookbacks_months']
            self.volatility_lookback_days = self.config['volatility_lookback_days']
            self.target_volatility = self.config['target_volatility']
            self.max_position_weight = self.config['max_position_weight']
            
            # Futures adaptation parameters
            self.momentum_threshold = self.config.get('momentum_threshold', 0.0)
            self.enable_long_short = self.config.get('long_short_enabled', True)
            self.signal_strength_weighting = self.config.get('signal_strength_weighting', True)
            
            # Risk management
            self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
            self.signal_clip_threshold = self.config.get('signal_standardization_clip', 3.0)
            
            # Initialize tracking containers
            self.continuous_contracts = []
            self.momentum_indicators = {}
            self.volatility_indicators = {}
            
            # Initialize symbol data and QC indicators
            self.initialize_symbol_data()
            
            # Initialize efficient portfolio volatility calculator
            self.portfolio_vol_calculator = EfficientPortfolioVolatility(
                algorithm=self.algorithm,
                symbols=self.continuous_contracts,
                lookback_days=252
            )
            
            # Log initialization summary
            self._log_initialization_summary()
            
        except Exception as e:
            error_msg = f"Error initializing MTUM components: {str(e)}"
            self.algorithm.Error(error_msg)
            raise ValueError(error_msg)
    
    def _log_initialization_summary(self):
        """Log initialization summary."""
        mode = "LONG-SHORT" if self.enable_long_short else "LONG-ONLY"
        weighting = "SIGNAL-STRENGTH" if self.signal_strength_weighting else "EQUAL-WEIGHT"
        
        self.algorithm.Log(f"{self.name}: MTUM Futures Adaptation - {self.momentum_lookbacks} month lookbacks, "
                          f"{self.target_volatility:.1%} vol target, {mode} mode")
        
        self.algorithm.Log(f"{self.name}: Futures innovations - Threshold: {self.momentum_threshold:.3f}, "
                          f"Weighting: {weighting}, 3Y volatility: {self.volatility_lookback_days}d")
        
        self.algorithm.Log(f"{self.name}: Preserves MTUM DNA - Risk-adjusted momentum, dual-period analysis, "
                          f"±{self.signal_clip_threshold} std dev standardization")
    
    def initialize_symbol_data(self):
        """Initialize QC indicators for all futures."""
        try:
            self.continuous_contracts = []
            for symbol in self.algorithm.Securities.Keys:
                security = self.algorithm.Securities[symbol]
                if security.Type == SecurityType.Future:
                    symbol_str = str(symbol)
                    if symbol_str.startswith('/'):
                        self.continuous_contracts.append(symbol)
                        self.algorithm.Log(f"{self.name}: Using continuous contract {symbol_str}")
                    else:
                        self.algorithm.Log(f"{self.name}: Ignoring underlying contract {symbol_str}")
            
            if not self.continuous_contracts:
                self.algorithm.Log(f"{self.name}: No continuous contracts found in SecuritiesManager. "
                                   f"Strategy will wait for OnSecuritiesChanged events.")

            self._setup_qc_indicators()
            self._warmup_indicators()
            
            self.algorithm.Log(f"{self.name}: Initialized {len(self.continuous_contracts)} symbols with QC indicators")
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error initializing QC indicators: {str(e)}")

    def _setup_qc_indicators(self):
        """Setup QuantConnect's native indicators."""
        self.volatility_indicators = {}
        
        for symbol in self.continuous_contracts:
            self.momentum_indicators[symbol] = {}
            
            for months in self.momentum_lookbacks:
                period = months * 21
                indicator_name = f"roc_{months}m"
                self.momentum_indicators[symbol][indicator_name] = self.algorithm.ROC(symbol, period)
            
            # MTUM OFFICIAL: 3-year standard deviation of WEEKLY returns.
            # To achieve this, we create a weekly consolidator.
            weekly_consolidator = TradeBarConsolidator(timedelta(weeks=1))
            self.algorithm.SubscriptionManager.AddConsolidator(symbol, weekly_consolidator)

            # Create an indicator to calculate weekly returns (ROC with period 1 on weekly bars).
            weekly_return_indicator = RateOfChange(1)
            self.algorithm.RegisterIndicator(symbol, weekly_return_indicator, weekly_consolidator)

            # Create an indicator to calculate the STD of those weekly returns over 3 years (156 weeks).
            volatility_lookback_weeks = 52 * 3  # 156 weeks
            weekly_std_indicator = StandardDeviation(volatility_lookback_weeks)
            
            # Pipe the output of the weekly return indicator into the STD indicator.
            self.volatility_indicators[symbol] = IndicatorExtensions.Of(weekly_std_indicator, weekly_return_indicator)
            
            self.algorithm.Log(f"{self.name}: Setup QC native indicators for {symbol}")
        
        self.algorithm.Log(f"{self.name}: Setup complete - using QC native indicators + efficient var-cov matrix")

    def _warmup_indicators(self):
        """Simplified warmup - let QC handle it naturally."""
        try:
            warmup_days = max(365, self.volatility_lookback_days)
            self.algorithm.Log(f"{self.name}: Using QC's automatic warmup system ({warmup_days} days)")
            
            total_indicators = 0
            for symbol in self.continuous_contracts:
                if symbol in self.momentum_indicators:
                    total_indicators += len(self.momentum_indicators[symbol])
                if symbol in self.volatility_indicators:
                    total_indicators += 1
            
            self.algorithm.Log(f"{self.name}: Setup {total_indicators} QC indicators + var-cov matrix")
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Indicator setup failed: {str(e)}")

    def update(self, slice_data):
        """Update strategy with new market data."""
        try:
            self.last_update_time = slice_data.Time
            
            if not self._validate_slice_data(slice_data):
                return False
            
            if hasattr(self, 'portfolio_vol_calculator'):
                self.portfolio_vol_calculator.update_returns(slice_data)
            
            return True
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error updating strategy: {str(e)}")
            return False
    
    def _validate_slice_data(self, slice_data):
        """Validate slice data."""
        if not slice_data or not slice_data.Bars:
            return False
        
        if hasattr(self.algorithm, 'data_integrity_checker'):
            return self.algorithm.data_integrity_checker.validate_slice(slice_data) is not None
        
        for symbol, bar in slice_data.Bars.items():
            if symbol in self.algorithm.Securities:
                security = self.algorithm.Securities[symbol]
                has_data = security.HasData
                price_valid = security.Price > 0
                
                is_tradeable = security.IsTradable
                if not is_tradeable and hasattr(security, 'Mapped') and security.Mapped:
                    mapped_contract = security.Mapped
                    if mapped_contract in self.algorithm.Securities:
                        is_tradeable = self.algorithm.Securities[mapped_contract].IsTradable
                
                if not (has_data and price_valid and (is_tradeable or self.algorithm.IsWarmingUp)):
                    return False
        return True
    
    def generate_targets(self, slice=None):
        """Generate target positions based on momentum analysis."""
        try:
            if not self.should_rebalance(self.algorithm.Time):
                return self.current_targets
            
            signals = self.generate_signals(slice)
            if not signals:
                return {}
            
            targets = self._apply_position_limits(signals)
            targets = self._apply_volatility_targeting(targets)
            targets = self._validate_trade_sizes(targets)
            
            self.current_targets = targets
            self.last_rebalance_date = self.algorithm.Time.date()
            self.total_rebalances += 1
            
            if targets:
                formatted_targets = {str(symbol): weight for symbol, weight in targets.items()}
                self.algorithm.Log(f"{self.name}: Final targets: {formatted_targets}")
            else:
                self.algorithm.Log(f"{self.name}: Final targets: No positions")
            
            return targets
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error generating targets: {str(e)}")
            return {}
    
    def get_exposure(self):
        """Get current strategy exposure."""
        try:
            if not self.current_targets:
                return {
                    'gross_exposure': 0.0, 'net_exposure': 0.0, 'long_exposure': 0.0,
                    'short_exposure': 0.0, 'num_positions': 0
                }
            
            long_exposure = sum(max(0, weight) for weight in self.current_targets.values())
            short_exposure = sum(min(0, weight) for weight in self.current_targets.values())
            
            return {
                'gross_exposure': long_exposure + abs(short_exposure),
                'net_exposure': long_exposure + short_exposure,
                'long_exposure': long_exposure,
                'short_exposure': short_exposure,
                'num_positions': len(self.current_targets)
            }
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error calculating exposure: {str(e)}")
            return {'gross_exposure': 0.0, 'net_exposure': 0.0, 'long_exposure': 0.0,
                   'short_exposure': 0.0, 'num_positions': 0}

    @property
    def IsAvailable(self):
        """Check if strategy is available for trading."""
        if self.algorithm.IsWarmingUp:
            return False
        
        try:
            ready_symbols = 0
            total_symbols = len(self.continuous_contracts)
            
            if total_symbols == 0:
                return False
            
            for symbol in self.continuous_contracts:
                if symbol in self.momentum_indicators and symbol in self.volatility_indicators:
                    momentum_ready = all(indicator.IsReady for indicator in self.momentum_indicators[symbol].values())
                    volatility_ready = self.volatility_indicators[symbol].IsReady
                    
                    if momentum_ready and volatility_ready:
                        ready_symbols += 1
            
            required_symbols = max(1, int(total_symbols * 0.5))
            is_available = ready_symbols >= required_symbols
            
            return is_available
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error checking availability: {str(e)}")
            return False

    def get_availability_status(self):
        """Get detailed availability status."""
        try:
            if self.algorithm.IsWarmingUp:
                return "WARMING_UP"
            
            ready_symbols = 0
            total_symbols = len(self.continuous_contracts)
            
            for symbol in self.continuous_contracts:
                if symbol in self.momentum_indicators and symbol in self.volatility_indicators:
                    momentum_ready = all(indicator.IsReady for indicator in self.momentum_indicators[symbol].values())
                    volatility_ready = self.volatility_indicators[symbol].IsReady
                    
                    if momentum_ready and volatility_ready:
                        ready_symbols += 1
            
            required_symbols = max(1, int(total_symbols * 0.5))
            
            if ready_symbols >= required_symbols:
                return "AVAILABLE"
            else:
                return f"NOT_AVAILABLE ({ready_symbols}/{total_symbols} ready, need {required_symbols})"
                
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error getting availability status: {str(e)}")
            return "ERROR"

    def should_rebalance(self, current_time):
        """Check if strategy should rebalance."""
        if self.last_rebalance_date is None:
            return True
        
        rebalance_frequency = self.config.get('rebalance_frequency', 'monthly')
        
        if rebalance_frequency == 'monthly':
            return current_time.month != self.last_rebalance_date.month
        elif rebalance_frequency == 'weekly':
            days_diff = (current_time.date() - self.last_rebalance_date).days
            return days_diff >= 7
        else:
            return False

    def generate_signals(self, slice=None):
        """MTUM Futures Adaptation: Generate signals using absolute momentum thresholds."""
        signals = {}
        
        if self.algorithm.Time.day == 1:
            self.debug_volatility_calculation()
        
        ready_symbols = []
        for symbol in self.continuous_contracts:
            if symbol in self.momentum_indicators and symbol in self.volatility_indicators:
                momentum_ready = all(indicator.IsReady for indicator in self.momentum_indicators[symbol].values())
                volatility_ready = self.volatility_indicators[symbol].IsReady
                
                if momentum_ready and volatility_ready:
                    ready_symbols.append(symbol)
        
        if not ready_symbols:
            self.algorithm.Log(f"{self.name}: No QC indicators ready for signal generation")
            return signals
        
        # Calculate risk-adjusted momentum scores
        momentum_scores = {}
        for symbol in ready_symbols:
            try:
                volatility_indicator = self.volatility_indicators[symbol]
                if not volatility_indicator.IsReady:
                    continue
                    
                # This indicator now calculates the standard deviation of WEEKLY returns.
                weekly_return_std = volatility_indicator.Current.Value
                
                # Annualize the weekly volatility.
                volatility = weekly_return_std * (52 ** 0.5)
                
                if volatility <= 0:
                    self.algorithm.Log(f"{self.name}: Invalid volatility for {symbol}: {volatility}")
                    continue
                
                individual_scores = []
                for indicator_name, roc_indicator in self.momentum_indicators[symbol].items():
                    if roc_indicator.IsReady:
                        raw_momentum = roc_indicator.Current.Value
                        months = int(indicator_name.replace('roc_', '').replace('m', ''))
                        
                        if raw_momentum > -1:
                            annualized_momentum = ((1 + raw_momentum) ** (12.0 / months)) - 1
                        else:
                            annualized_momentum = raw_momentum
                        
                        risk_adjusted_score = annualized_momentum / volatility
                        individual_scores.append(risk_adjusted_score)
                        
                        self.algorithm.Debug(f"{self.name}: {symbol} {months}M - "
                                           f"Raw: {raw_momentum:.3f}, Ann: {annualized_momentum:.3f}, "
                                           f"Vol: {volatility:.3f}, Risk-Adj: {risk_adjusted_score:.3f}")
                
                if len(individual_scores) > 0:
                    momentum_scores[symbol] = individual_scores
                    
            except Exception as e:
                self.algorithm.Error(f"{self.name}: Error processing QC indicators for {symbol}: {str(e)}")
                continue
        
        if momentum_scores:
            standardized_scores = self._mtum_standardize_and_average(momentum_scores)
            qualified_signals = self._apply_absolute_momentum_thresholds(standardized_scores)
            signals = self._convert_to_signal_strength_weights(qualified_signals)
            
            self.algorithm.Log(f"{self.name}: Generated {len(signals)} signals from {len(qualified_signals)} qualified "
                             f"(threshold: {self.momentum_threshold:.3f})")
        
        return signals

    def _get_liquid_symbols(self, slice=None):
        """Get liquid symbols using QC native approach."""
        try:
            liquid_symbols = []
            
            for symbol in self.algorithm.Securities.Keys:
                security = self.algorithm.Securities[symbol]
                
                if security.Type == SecurityType.Future:
                    symbol_str = str(symbol)
                    
                    if symbol_str.startswith('/'):
                        if security.HasData:
                            is_tradeable = security.IsTradable
                            if not is_tradeable and hasattr(security, 'Mapped') and security.Mapped:
                                mapped_contract = security.Mapped
                                if mapped_contract in self.algorithm.Securities:
                                    is_tradeable = self.algorithm.Securities[mapped_contract].IsTradable
                            
                            if self.algorithm.IsWarmingUp:
                                if security.HasData:
                                    liquid_symbols.append(symbol)
                            else:
                                if security.HasData and is_tradeable:
                                    liquid_symbols.append(symbol)
            
            return liquid_symbols
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error getting liquid symbols: {str(e)}")
            return list(self.symbol_data.keys()) if self.symbol_data else []

    def _mtum_standardize_and_average(self, momentum_scores):
        """Average the risk-adjusted momentum scores across timeframes per asset.
        Each individual period score is optionally clipped to ±`signal_clip_threshold` before averaging.
        This replaces the previous cross-sectional standardization logic.
        """
        if not momentum_scores:
            return {}
        
        try:
            final_scores = {}
            clip = self.signal_clip_threshold if hasattr(self, 'signal_clip_threshold') else 3.0
            
            for symbol, individual_scores in momentum_scores.items():
                if not individual_scores:
                    continue
                # Clip each period's score to avoid extreme outliers
                clipped_scores = [max(-clip, min(clip, s)) for s in individual_scores]
                final_scores[symbol] = float(np.mean(clipped_scores))
            
            return final_scores
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error averaging momentum scores: {str(e)}")
            return {}

    def _apply_absolute_momentum_thresholds(self, standardized_scores):
        """Apply absolute momentum thresholds - Futures Innovation."""
        if not standardized_scores:
            return {}
        
        try:
            qualified_signals = {}
            
            for symbol, score in standardized_scores.items():
                if abs(score) > self.momentum_threshold:
                    qualified_signals[symbol] = score
                    
                    self.algorithm.Debug(f"{self.name}: {symbol} qualified - Score: {score:.3f} "
                                       f"(threshold: {self.momentum_threshold:.3f})")
                else:
                    self.algorithm.Debug(f"{self.name}: {symbol} below threshold - Score: {score:.3f} "
                                       f"< {self.momentum_threshold:.3f}")
            
            total_assets = len(standardized_scores)
            qualified_assets = len(qualified_signals)
            self.algorithm.Log(f"{self.name}: Threshold filtering - {qualified_assets}/{total_assets} assets qualified "
                             f"(threshold: {self.momentum_threshold:.3f})")
            
            return qualified_signals
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error applying momentum thresholds: {str(e)}")
            return {}
    
    def _convert_to_signal_strength_weights(self, qualified_signals):
        """Convert qualified signals to portfolio weights using signal-strength weighting."""
        if not qualified_signals:
            return {}
        
        try:
            weights = {}
            
            if self.enable_long_short:
                total_abs_signals = sum(abs(signal) for signal in qualified_signals.values())
                
                if total_abs_signals == 0:
                    return {}
                
                for symbol, signal in qualified_signals.items():
                    weight = signal / total_abs_signals
                    weights[symbol] = weight
                    
                    direction = "LONG" if signal > 0 else "SHORT"
                    self.algorithm.Debug(f"{self.name}: {symbol} {direction} weight: {weight:.3f} "
                                       f"(signal: {signal:.3f})")
                
                net_exposure = sum(weights.values())
                gross_exposure = sum(abs(w) for w in weights.values())
                
                self.algorithm.Log(f"{self.name}: Signal-strength weighting - Net: {net_exposure:.1%}, "
                                 f"Gross: {gross_exposure:.1%}")
                
            else:
                positive_signals = {s: max(0, signal) for s, signal in qualified_signals.items()}
                total_positive = sum(positive_signals.values())
                
                if total_positive > 0:
                    for symbol, signal in positive_signals.items():
                        if signal > 0:
                            weights[symbol] = signal / total_positive
                            self.algorithm.Debug(f"{self.name}: {symbol} LONG weight: {weights[symbol]:.3f}")
            
            if weights:
                weights = self._apply_position_limits(weights)
            
            return weights
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error converting to signal-strength weights: {str(e)}")
            return {}

    def _apply_volatility_targeting(self, weights):
        """Apply portfolio-level volatility targeting."""
        if not weights:
            return weights
        
        try:
            current_vol = self._calculate_portfolio_volatility(weights)
            
            if current_vol <= 0:
                self.algorithm.Log(f"{self.name}: Invalid portfolio volatility: {current_vol}, returning original weights")
                return weights
            
            vol_scaling_factor = self.target_volatility / current_vol
            
            scaled_weights = {}
            for symbol, weight in weights.items():
                scaled_weights[symbol] = weight * vol_scaling_factor
            
            if abs(vol_scaling_factor - 1.0) > 0.05:
                self.algorithm.Log(f"{self.name}: Volatility targeting - Current: {current_vol:.1%}, "
                                 f"Target: {self.target_volatility:.1%}, Scale: {vol_scaling_factor:.2f}")
            
            return scaled_weights
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in volatility targeting: {str(e)}")
            return weights
    
    def _apply_position_limits(self, weights):
        """Apply individual position size limits."""
        limited_weights = {}
        max_weight = self.max_position_weight
        
        for symbol, weight in weights.items():
            limited_weights[symbol] = max(-max_weight, min(max_weight, weight))
        
        return limited_weights

    def _calculate_portfolio_volatility(self, weights):
        """Calculate portfolio volatility using efficient var-cov matrix."""
        try:
            if not weights:
                return self.target_volatility
            
            valid_weights = {symbol: weight for symbol, weight in weights.items() if abs(weight) > 0.001}
            if not valid_weights:
                return self.target_volatility
            
            if hasattr(self, 'portfolio_vol_calculator'):
                portfolio_vol = self.portfolio_vol_calculator.calculate_portfolio_volatility(valid_weights)
                
                if 0.05 <= portfolio_vol <= 0.50:
                    if self.algorithm.Time.day == 1:
                        self.algorithm.Log(f"{self.name}: Var-cov portfolio vol: {portfolio_vol:.1%} "
                                         f"(gross exposure: {sum(abs(w) for w in valid_weights.values()):.1%})")
                    return portfolio_vol
                else:
                    self.algorithm.Log(f"{self.name}: Var-cov volatility out of range: {portfolio_vol:.1%}, using fallback")
            
            return self._ultra_simple_portfolio_volatility(valid_weights)
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in var-cov portfolio volatility: {str(e)}")
            return self._ultra_simple_portfolio_volatility(weights)

    def _ultra_simple_portfolio_volatility(self, weights):
        """Ultra-simple portfolio volatility calculation."""
        try:
            if not weights:
                return self.target_volatility
            
            # Get weighted average volatility
            weighted_vol = 0.0
            total_abs_weight = 0.0
            
            for symbol, weight in weights.items():
                if abs(weight) > 0.001:
                    symbol_vol = self._get_default_volatility(symbol)
                    abs_weight = abs(weight)
                    weighted_vol += symbol_vol * abs_weight
                    total_abs_weight += abs_weight
            
            if total_abs_weight > 0:
                avg_vol = weighted_vol / total_abs_weight
                gross_leverage = total_abs_weight
                
                # Apply simple diversification benefit
                diversification_factor = 0.75 if len(weights) > 1 else 1.0
                portfolio_vol = avg_vol * gross_leverage * diversification_factor
                
                return max(0.05, min(0.50, portfolio_vol))
            
            return self.target_volatility
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in ultra-simple portfolio volatility: {str(e)}")
            return self.target_volatility

    def _get_default_volatility(self, symbol):
        """Get default volatility for asset class."""
        try:
            symbol_str = str(symbol).upper()
            
            # Commodity futures
            if any(commodity in symbol_str for commodity in ['CL', 'GC', 'SI', 'HG', 'NG']):
                return 0.25
            # Equity futures
            elif any(equity in symbol_str for equity in ['ES', 'NQ', 'YM', 'RTY']):
                return 0.20
            # Bond futures
            elif any(bond in symbol_str for bond in ['ZN', 'ZB', 'ZF', 'ZT']):
                return 0.08
            # FX futures
            elif any(fx in symbol_str for fx in ['6E', '6J', '6B', '6A', '6C']):
                return 0.12
            else:
                return 0.18
                
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error getting default volatility for {symbol}: {str(e)}")
            return 0.18

    def _validate_trade_sizes(self, targets):
        """Validate trade sizes are reasonable."""
        if not targets:
            return targets
        
        validated_targets = {}
        for symbol, target in targets.items():
            if abs(target) >= 0.001:  # 0.1% minimum
                validated_targets[symbol] = target
        
        return validated_targets

    def execute_trades(self, new_targets, rollover_tags=None):
        """Execute trades based on target positions."""
        try:
            if not new_targets:
                return
            
            for symbol, target_weight in new_targets.items():
                if abs(target_weight) > 0.001:
                    # Use base class or algorithm's execution methods
                    if hasattr(self.algorithm, 'SetHoldings'):
                        self.algorithm.SetHoldings(symbol, target_weight)
                        self.trades_executed += 1
                        
                        direction = "LONG" if target_weight > 0 else "SHORT"
                        self.algorithm.Log(f"{self.name}: {direction} {symbol} {abs(target_weight):.1%}")
                        
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error executing trades: {str(e)}")

    def get_performance_metrics(self):
        """Get strategy performance metrics."""
        try:
            exposure = self.get_exposure()
            
            return {
                'total_rebalances': self.total_rebalances,
                'trades_executed': self.trades_executed,
                'gross_exposure': exposure['gross_exposure'],
                'net_exposure': exposure['net_exposure'],
                'num_positions': exposure['num_positions'],
                'is_available': self.IsAvailable
            }
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error calculating performance metrics: {str(e)}")
            return {}

    def log_status(self):
        """Log current strategy status."""
        try:
            exposure = self.get_exposure()
            availability = self.get_availability_status()
            
            self.algorithm.Log(f"{self.name}: Status - {availability}, "
                             f"Positions: {exposure['num_positions']}, "
                             f"Net: {exposure['net_exposure']:.1%}, "
                             f"Gross: {exposure['gross_exposure']:.1%}")
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error logging status: {str(e)}")

    def _create_symbol_data(self, symbol):
        """Create symbol data container."""
        return {'symbol': symbol, 'initialized': True}

    def OnSecuritiesChanged(self, changes):
        """Handle securities changes - only track continuous contracts."""
        try:
            for security in changes.AddedSecurities:
                symbol_str = str(security.Symbol)
                if symbol_str.startswith('/'):
                    if security.Symbol not in self.symbol_data:
                        # Track symbol
                        self.symbol_data[security.Symbol] = self._create_symbol_data(security.Symbol)
                        self.continuous_contracts.append(security.Symbol)
                        
                        # Setup indicators for this single symbol
                        try:
                            # Momentum indicators dict
                            self.momentum_indicators[security.Symbol] = {}
                            for months in self.momentum_lookbacks:
                                period = months * 21
                                indicator_name = f"roc_{months}m"
                                self.momentum_indicators[security.Symbol][indicator_name] = self.algorithm.ROC(security.Symbol, period)
                            # Weekly consolidator and std
                            weekly_consolidator = TradeBarConsolidator(timedelta(weeks=1))
                            self.algorithm.SubscriptionManager.AddConsolidator(security.Symbol, weekly_consolidator)
                            weekly_return_indicator = RateOfChange(1)
                            self.algorithm.RegisterIndicator(security.Symbol, weekly_return_indicator, weekly_consolidator)
                            volatility_lookback_weeks = 52 * 3
                            weekly_std_indicator = StandardDeviation(volatility_lookback_weeks)
                            self.volatility_indicators[security.Symbol] = IndicatorExtensions.Of(weekly_std_indicator, weekly_return_indicator)
                            self.algorithm.Log(f"{self.name}: Indicators set up for new symbol {symbol_str}")
                        except Exception as ind_e:
                            self.algorithm.Error(f"{self.name}: Error setting up indicators for {symbol_str}: {str(ind_e)}")
                        
                        self.algorithm.Log(f"{self.name}: Added continuous contract {symbol_str}")
                else:
                    self.algorithm.Log(f"{self.name}: Ignoring underlying contract {symbol_str}")
                    
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error handling securities changes: {str(e)}")

    def debug_volatility_calculation(self):
        """Debug volatility calculation on first day of month."""
        try:
            self.algorithm.Log(f"{self.name}: Monthly volatility debug - "
                             f"Target: {self.target_volatility:.1%}, "
                             f"Threshold: {self.momentum_threshold:.3f}")
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in volatility debug: {str(e)}")
