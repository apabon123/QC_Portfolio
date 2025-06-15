# mtum_cta_strategy.py - Optimized for size

from AlgorithmImports import *
import numpy as np
from collections import deque

class MTUMCTAStrategy:
    """MTUM CTA Strategy - Futures Adaptation of MSCI USA Momentum - Optimized Version"""
    
    def __init__(self, algorithm, futures_manager, name="MTUM_CTA", config_manager=None):
        self.algorithm = algorithm
        self.futures_manager = futures_manager
        self.name = name
        self.config_manager = config_manager
        
        self._load_configuration()
        
        # Initialize symbol data for momentum/volatility calculations
        self.symbol_data = {}
        
        # Strategy state
        self.current_targets = {}
        self.last_rebalance_date = None
        self.strategy_returns = []
        self.portfolio_values = []
        self.last_update_time = None
        
        # Performance tracking
        self.trades_executed = 0
        self.total_rebalances = 0
        self.gross_exposure_history = []
        self.momentum_score_history = {}
        
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
            'momentum_lookbacks_months': config.get('momentum_lookbacks_months', [6, 12]),
            'volatility_lookback_days': config.get('volatility_lookback_days', 252 * 3),
            'recent_exclusion_days': config.get('recent_exclusion_days', 22),
            'signal_standardization_clip': config.get('signal_standardization_clip', 3.0),
            'target_volatility': config.get('target_volatility', 0.2),
            'rebalance_frequency': config.get('rebalance_frequency', 'monthly'),
            'max_position_weight': config.get('max_position_weight', 0.5),
            'risk_free_rate': config.get('risk_free_rate', 0.02),
            'warmup_days': config.get('warmup_days', 400),
            'long_short_enabled': config.get('long_short_enabled', True),
            'min_weight_threshold': config.get('min_weight_threshold', 0.01),
            'min_trade_value': config.get('min_trade_value', 1000),
            'max_single_order_value': config.get('max_single_order_value', 50000000),
            'max_leverage_multiplier': config.get('max_leverage_multiplier', 100),
            'max_single_position': config.get('max_single_position', 10.0),
            'daily_stop_loss': config.get('daily_stop_loss', 0.2),
            'enabled': config.get('enabled', True),
            'description': config.get('description', 'MTUM CTA Strategy'),
            'expected_sharpe': config.get('expected_sharpe', 0.6),
            'correlation_with_momentum': config.get('correlation_with_momentum', 0.7)
        }
    
    def _load_fallback_config(self):
        """Load fallback configuration if config_manager is unavailable."""
        self.config_dict = {
            'momentum_lookbacks_months': [6, 12], 'volatility_lookback_days': 252 * 3,
            'recent_exclusion_days': 22, 'signal_standardization_clip': 3.0, 'target_volatility': 0.2,
            'rebalance_frequency': 'monthly', 'max_position_weight': 0.5, 'risk_free_rate': 0.02,
            'warmup_days': 400, 'long_short_enabled': True, 'min_weight_threshold': 0.01,
            'min_trade_value': 1000, 'max_single_order_value': 50000000, 'max_leverage_multiplier': 100,
            'max_single_position': 10.0, 'daily_stop_loss': 0.2, 'enabled': True,
            'description': 'MTUM CTA Strategy (Fallback Config)', 'expected_sharpe': 0.6,
            'correlation_with_momentum': 0.7
        }
    
    def _log_initialization_summary(self):
        """Log initialization summary."""
        self.algorithm.Log(f"{self.name}: Initialized with {self.config_dict['momentum_lookbacks_months']} month lookbacks, "
                          f"{self.config_dict['target_volatility']:.1%} vol target")
    
    def initialize_symbol_data(self):
        """Initialize SymbolData objects for all futures in the manager"""
        try:
            if self.futures_manager and hasattr(self.futures_manager, 'futures_data'):
                symbols = list(self.futures_manager.futures_data.keys())
            else:
                symbols = ['ES', 'NQ', 'ZN']
            
            for symbol in symbols:
                self.symbol_data[symbol] = self.SymbolData(
                    algorithm=self.algorithm,
                    symbol=symbol,
                    lookbackMonthsList=self.config_dict['momentum_lookbacks_months'],
                    volLookbackDays=self.config_dict['volatility_lookback_days'],
                    recent_exclusion_days=self.config_dict['recent_exclusion_days']
                )
                
                self.momentum_score_history[symbol] = deque(maxlen=252)
        
            self.algorithm.Log(f"{self.name}: Initialized {len(self.symbol_data)} symbol data objects")
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error initializing symbol data: {str(e)}")
    
    def update(self, slice_data):
        """Update strategy with new market data."""
        try:
            if not self._validate_slice_data(slice_data):
                return
            
            self.last_update_time = self.algorithm.Time
            
            # Update symbol data with new bars
            for symbol, bar in slice_data.Bars.items():
                if symbol in self.symbol_data:
                    self.symbol_data[symbol].OnDataConsolidated(None, bar)
                    
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
        """Generate target positions based on momentum analysis."""
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
    
    @property
    def IsAvailable(self):
        """Check if strategy is available for trading."""
        try:
            if not self.symbol_data:
                return False
            
            ready_count = sum(1 for sd in self.symbol_data.values() if sd.IsReady)
            total_count = len(self.symbol_data)
            
            # Need at least 50% of symbols ready
            return ready_count >= max(1, total_count * 0.5)
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error checking availability: {str(e)}")
            return False
    
    def should_rebalance(self, current_time):
        """Check if strategy should rebalance."""
        if not self.last_rebalance_date:
            return True
        
        if self.config_dict['rebalance_frequency'] == 'monthly':
            return current_time.month != self.last_rebalance_date.month
        elif self.config_dict['rebalance_frequency'] == 'weekly':
            days_since = (current_time.date() - self.last_rebalance_date).days
            return days_since >= 7
        
        return False
    
    def generate_signals(self):
        """Generate momentum-based trading signals."""
        try:
            signals = {}
            liquid_symbols = self._get_liquid_symbols()
            
            if not liquid_symbols:
                return signals
            
            # Calculate momentum scores for all symbols
            momentum_scores = {}
            for symbol in liquid_symbols:
                if symbol not in self.symbol_data or not self.symbol_data[symbol].IsReady:
                    continue
                
                try:
                    symbol_data = self.symbol_data[symbol]
                    
                    # Calculate risk-adjusted momentum for each lookback period
                    total_score = 0.0
                    valid_scores = 0
                    
                    for lookback_months in self.config_dict['momentum_lookbacks_months']:
                        total_return = symbol_data.GetTotalReturn(lookback_months)
                        volatility = symbol_data.GetVolatility(lookback_months)
                        
                        if total_return is not None and volatility > 0:
                            # Risk-adjusted momentum = (total_return - risk_free) / volatility
                            risk_free_return = self.config_dict['risk_free_rate'] * (lookback_months / 12.0)
                            excess_return = total_return - risk_free_return
                            risk_adjusted_score = excess_return / volatility
                            
                            total_score += risk_adjusted_score
                            valid_scores += 1
                    
                    if valid_scores > 0:
                        momentum_scores[symbol] = total_score / valid_scores
                        
                        # Track momentum score history
                        if symbol in self.momentum_score_history:
                            self.momentum_score_history[symbol].append(momentum_scores[symbol])
                
                except Exception as e:
                    self.algorithm.Error(f"{self.name}: Error calculating momentum for {symbol}: {str(e)}")
                    continue
            
            if not momentum_scores:
                return signals
            
            # Standardize scores
            standardized_scores = self._standardize_scores(momentum_scores, 
                                                         self.config_dict['signal_standardization_clip'])
            
            # Convert to position weights
            position_weights = self._convert_scores_to_weights(standardized_scores)
            
            return position_weights
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error generating signals: {str(e)}")
            return {}
    
    def _get_liquid_symbols(self):
        """Get liquid symbols from futures manager."""
        if self.futures_manager and hasattr(self.futures_manager, 'get_liquid_symbols'):
            return self.futures_manager.get_liquid_symbols()
        return list(self.symbol_data.keys())
    
    def _standardize_scores(self, scores, clip_at=3.0):
        """Standardize momentum scores using z-score normalization."""
        try:
            if not scores:
                return scores
            
            score_values = list(scores.values())
            mean_score = np.mean(score_values)
            std_score = np.std(score_values)
            
            if std_score == 0:
                return {symbol: 0.0 for symbol in scores.keys()}
            
            standardized = {}
            for symbol, score in scores.items():
                z_score = (score - mean_score) / std_score
                clipped_score = np.clip(z_score, -clip_at, clip_at)
                standardized[symbol] = clipped_score
            
            return standardized
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error standardizing scores: {str(e)}")
            return scores
    
    def _convert_scores_to_weights(self, scores):
        """Convert standardized scores to position weights."""
        try:
            if not scores:
                return {}
            
            weights = {}
            
            if self.config_dict['long_short_enabled']:
                # Long/short: use scores directly as weights
                for symbol, score in scores.items():
                    weights[symbol] = score / len(scores)  # Normalize by number of assets
            else:
                # Long-only: only positive scores
                positive_scores = {s: max(0, score) for s, score in scores.items()}
                total_positive = sum(positive_scores.values())
                
                if total_positive > 0:
                    for symbol, score in positive_scores.items():
                        weights[symbol] = score / total_positive
                else:
                    weights = {symbol: 0.0 for symbol in scores.keys()}
            
            return weights
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error converting scores to weights: {str(e)}")
            return {}
    
    def _apply_position_limits(self, weights):
        """Apply individual position size limits."""
        limited_weights = {}
        max_weight = self.config_dict['max_position_weight']
        
        for symbol, weight in weights.items():
            limited_weights[symbol] = max(-max_weight, min(max_weight, weight))
        
        return limited_weights
    
    def _apply_volatility_targeting(self, weights):
        """Apply portfolio-level volatility targeting."""
        try:
            if not weights:
                return weights
            
            # Calculate portfolio volatility
            portfolio_vol = self._calculate_portfolio_volatility(weights)
            target_vol = self.config_dict['target_volatility']
            
            if portfolio_vol > 0:
                vol_scalar = target_vol / portfolio_vol
                vol_scalar = min(vol_scalar, self.config_dict['max_leverage_multiplier'])
                
                scaled_weights = {}
                for symbol, weight in weights.items():
                    scaled_weights[symbol] = weight * vol_scalar
                
                return scaled_weights
            
            return weights
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in volatility targeting: {str(e)}")
            return weights
    
    def _validate_trade_sizes(self, targets):
        """Validate and filter trade sizes."""
        validated_targets = {}
        min_threshold = self.config_dict['min_weight_threshold']
        
        for symbol, weight in targets.items():
            if abs(weight) >= min_threshold:
                validated_targets[symbol] = weight
        
        return validated_targets
    
    def _calculate_portfolio_volatility(self, weights):
        """Calculate expected portfolio volatility."""
        try:
            if not weights:
                return 0.0
            
            # Simple approach: weighted average of individual volatilities
            total_vol = 0.0
            total_weight = 0.0
            
            for symbol, weight in weights.items():
                if symbol in self.symbol_data and self.symbol_data[symbol].IsReady:
                    vol = self.symbol_data[symbol].GetVolatility()
                    if vol > 0:
                        total_vol += abs(weight) * vol
                        total_weight += abs(weight)
            
            return total_vol / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error calculating portfolio volatility: {str(e)}")
            return 0.0
    
    def _log_signal_summary(self, final_targets, momentum_scores):
        """Log summary of generated signals."""
        if final_targets:
            active_positions = len([w for w in final_targets.values() if abs(w) > 0.01])
            gross_exposure = sum(abs(w) for w in final_targets.values())
            self.algorithm.Log(f"{self.name}: Generated {active_positions} positions, "
                              f"{gross_exposure:.1%} gross exposure")
    
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
                'avg_gross_exposure': np.mean(self.gross_exposure_history) if self.gross_exposure_history else 0.0,
                'strategy_name': self.name,
                'last_update': self.last_update_time,
                'config_description': self.config_dict.get('description', 'MTUM CTA Strategy')
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
                'rebalance_frequency': self.config_dict.get('rebalance_frequency', 'monthly'),
                'target_volatility': self.config_dict.get('target_volatility', 0.2),
                'max_position_weight': self.config_dict.get('max_position_weight', 0.5),
                'momentum_lookbacks': self.config_dict.get('momentum_lookbacks_months', [6, 12]),
                'long_short_enabled': self.config_dict.get('long_short_enabled', True),
                'description': self.config_dict.get('description', 'MTUM CTA Strategy'),
                'config_source': 'config_manager' if self.config_manager else 'fallback'
            }
        except Exception as e:
            return {'strategy_name': self.name, 'error': str(e)}

    class SymbolData:
        """Optimized SymbolData class for momentum analysis."""
        
        def __init__(self, algorithm, symbol, lookbackMonthsList, volLookbackDays, recent_exclusion_days=22):
            self.algorithm = algorithm
            self.symbol = symbol
            self.lookbackMonthsList = lookbackMonthsList
            self.volLookbackDays = volLookbackDays
            self.recent_exclusion_days = recent_exclusion_days
            
            # Data storage
            self.daily_prices = deque(maxlen=volLookbackDays + 100)
            self.monthly_prices = deque(maxlen=max(lookbackMonthsList) + 5)
            self.daily_returns = deque(maxlen=volLookbackDays)
            
            # Data quality tracking
            self.data_quality_score = 0.0
            self.consecutive_valid_days = 0
            self.total_data_points = 0
            
            self._initialize_with_history()
        
        def _initialize_with_history(self):
            """Initialize with historical data."""
            try:
                history_days = self.volLookbackDays + 100
                history = self.algorithm.History(self.symbol, history_days, Resolution.Daily)
                
                if history.empty:
                    self.algorithm.Log(f"No history available for {self.symbol}")
                    return
                
                # Process historical data
                for time, row in history.iterrows():
                    if hasattr(row, 'close') and row.close > 0:
                        self.daily_prices.append(float(row.close))
                        self.total_data_points += 1
                
                # Calculate returns
                if len(self.daily_prices) > 1:
                    for i in range(1, len(self.daily_prices)):
                        ret = (self.daily_prices[i] / self.daily_prices[i-1]) - 1
                        self.daily_returns.append(ret)
                
                self._update_monthly_prices()
                self._update_data_quality()
                
                self.algorithm.Log(f"{self.symbol}: Initialized with {len(self.daily_prices)} prices")
                
            except Exception as e:
                self.algorithm.Error(f"Error initializing {self.symbol}: {str(e)}")
        
        @property
        def IsReady(self):
            """Check if symbol data is ready for analysis."""
            min_days_needed = max(self.lookbackMonthsList) * 22 + self.recent_exclusion_days + 50
            return (len(self.daily_prices) >= min_days_needed and 
                   len(self.monthly_prices) >= max(self.lookbackMonthsList) and
                   self.data_quality_score > 0.7)
        
        def OnDataConsolidated(self, sender, bar: TradeBar):
            """Process new bar data."""
            try:
                if bar.Close <= 0:
                    return
                
                # Add new price
                self.daily_prices.append(float(bar.Close))
                self.total_data_points += 1
                
                # Calculate return
                if len(self.daily_prices) > 1:
                    ret = (self.daily_prices[-1] / self.daily_prices[-2]) - 1
                    self.daily_returns.append(ret)
                    self.consecutive_valid_days += 1
                
                # Update monthly prices
                self._update_monthly_prices()
                self._update_data_quality()
                
            except Exception as e:
                self.algorithm.Error(f"Error processing bar for {self.symbol}: {str(e)}")
        
        def _update_monthly_prices(self):
            """Update monthly price sampling."""
            if len(self.daily_prices) < 22:
                return
            
            # Sample prices every ~22 trading days for monthly calculation
            if len(self.daily_prices) % 22 == 0:
                self.monthly_prices.append(self.daily_prices[-1])
        
        def GetTotalReturn(self, lookbackMonths):
            """Calculate total return over specified months."""
            try:
                if len(self.monthly_prices) < lookbackMonths + 1:
                    return None
                
                # Get prices from lookbackMonths ago and current
                start_price = self.monthly_prices[-(lookbackMonths + 1)]
                end_price = self.monthly_prices[-1]
                
                if start_price > 0:
                    # Exclude recent period
                    if len(self.daily_prices) >= self.recent_exclusion_days:
                        excluded_price = self.daily_prices[-self.recent_exclusion_days]
                        if excluded_price > 0:
                            end_price = excluded_price
                    
                    return (end_price / start_price) - 1.0
                
                return None
                
            except Exception as e:
                self.algorithm.Error(f"Error calculating total return for {self.symbol}: {str(e)}")
                return None
        
        def GetVolatility(self, lookbackMonths=None):
            """Calculate volatility over specified period."""
            try:
                if lookbackMonths:
                    # Use monthly returns for longer periods
                    if len(self.monthly_prices) < lookbackMonths + 1:
                        return None
                    
                    monthly_returns = []
                    for i in range(1, min(lookbackMonths + 1, len(self.monthly_prices))):
                        if self.monthly_prices[-i-1] > 0:
                            ret = (self.monthly_prices[-i] / self.monthly_prices[-i-1]) - 1
                            monthly_returns.append(ret)
                    
                    if len(monthly_returns) > 1:
                        return np.std(monthly_returns) * np.sqrt(12)  # Annualize
                else:
                    # Use daily returns for shorter periods
                    if len(self.daily_returns) < 30:
                        return None
                    
                    recent_returns = list(self.daily_returns)[-252:]  # Last year
                    if len(recent_returns) > 1:
                        return np.std(recent_returns) * np.sqrt(252)  # Annualize
                
                return None
                
            except Exception as e:
                self.algorithm.Error(f"Error calculating volatility for {self.symbol}: {str(e)}")
                return None
        
        def _update_data_quality(self):
            """Update data quality score."""
            if self.total_data_points == 0:
                self.data_quality_score = 0.0
                return
            
            # Simple quality score based on consecutive valid days
            self.data_quality_score = min(1.0, self.consecutive_valid_days / 50.0)
        
        def GetDataQuality(self):
            """Get data quality metrics."""
            return {
                'quality_score': self.data_quality_score,
                'total_points': self.total_data_points,
                'consecutive_valid': self.consecutive_valid_days,
                'daily_prices_count': len(self.daily_prices),
                'monthly_prices_count': len(self.monthly_prices)
            }
        
        def Dispose(self):
            """Clean up resources."""
            self.daily_prices.clear()
            self.monthly_prices.clear()
            self.daily_returns.clear()

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
                        lookbackMonthsList=self.config_dict['momentum_lookbacks_months'],
                        volLookbackDays=self.config_dict['volatility_lookback_days'],
                        recent_exclusion_days=self.config_dict['recent_exclusion_days']
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
