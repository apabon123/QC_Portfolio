# mtum_cta_strategy.py - INHERITS FROM BASE STRATEGY

from AlgorithmImports import *
import numpy as np
from collections import deque
from strategies.base_strategy import BaseStrategy

class MTUMCTAStrategy(BaseStrategy):
    """
    MTUM CTA Strategy Implementation
    CRITICAL: All configuration comes through centralized config manager only
    """
    
    def __init__(self, algorithm, config_manager, strategy_name):
        """
        Initialize MTUM CTA strategy with centralized configuration.
        CRITICAL: NO fallback logic - fail fast if config is invalid.
        """
        # Initialize base strategy with centralized config
        super().__init__(algorithm, config_manager, strategy_name)
        
        try:
            # All configuration comes from centralized manager
            self._initialize_strategy_components()
            self.algorithm.Log(f"MTUMCTA: Strategy initialized successfully")
            
        except Exception as e:
            error_msg = f"CRITICAL ERROR initializing MTUMCTA: {str(e)}"
            self.algorithm.Error(error_msg)
            raise ValueError(error_msg)
    
    def _initialize_strategy_components(self):
        """Initialize MTUM-specific components using centralized configuration."""
        try:
            # Validate required configuration parameters
            required_params = [
                'momentum_lookbacks_months', 'volatility_lookback_days', 'target_volatility',
                'max_position_weight', 'warmup_days', 'enabled'
            ]
            
            for param in required_params:
                if param not in self.config:
                    error_msg = f"Missing required parameter '{param}' in MTUMCTA configuration"
                    self.algorithm.Error(f"CONFIG ERROR: {error_msg}")
                    raise ValueError(error_msg)
            
            # Initialize strategy parameters from validated config
            self.momentum_lookbacks_months = self.config['momentum_lookbacks_months']
            self.volatility_lookback_days = self.config['volatility_lookback_days']
            self.target_volatility = self.config['target_volatility']
            self.max_position_weight = self.config['max_position_weight']
            self.warmup_days = self.config['warmup_days']
            self.recent_exclusion_days = self.config.get('recent_exclusion_days', 22)
            self.signal_standardization_clip = self.config.get('signal_standardization_clip', 3.0)
            
            # Initialize tracking variables
            self.symbol_data = {}
            self.current_targets = {}
            self.last_rebalance_date = None
            self.last_update_time = None
            
            # Performance tracking
            self.trades_executed = 0
            self.total_rebalances = 0
            self.strategy_returns = []
            
            self.algorithm.Log(f"MTUMCTA: Initialized with momentum lookbacks {self.momentum_lookbacks_months}, "
                             f"target volatility {self.target_volatility:.1%}")
            
        except Exception as e:
            error_msg = f"Failed to initialize MTUMCTA components: {str(e)}"
            self.algorithm.Error(f"CRITICAL ERROR: {error_msg}")
            raise ValueError(error_msg)
    
    # REMOVED: All fallback configuration logic
    # All configuration MUST come through the centralized config manager
    
    def _log_initialization_summary(self):
        """Log initialization summary."""
        self.algorithm.Log(f"{self.name}: Initialized with {self.config['momentum_lookbacks_months']} month lookbacks, "
                          f"{self.config['target_volatility']:.1%} vol target")
    
    def initialize_symbol_data(self):
        """Initialize SymbolData objects for all futures in the manager"""
        try:
            if self.futures_manager and hasattr(self.futures_manager, 'futures_data'):
                symbols = list(self.futures_manager.futures_data.keys())
            else:
                symbols = ['ES', 'NQ', 'ZN']
            
            for symbol in symbols:
                self.symbol_data[symbol] = self._create_symbol_data(symbol)
                
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
        """Determine if strategy should rebalance (monthly)."""
        if self.last_rebalance_date is None:
            return True
        
        # Rebalance monthly on the first trading day
        return (current_time.month != self.last_rebalance_date.month and
                current_time.day <= 5)  # First 5 days of month
    
    def generate_signals(self, slice=None):
        """
        Generate MTUM-style signals for all liquid symbols.
        
        Args:
            slice: Optional data slice for futures chain analysis
            
        Returns:
            dict: Symbol -> signal strength mapping
        """
        signals = {}
        liquid_symbols = self._get_liquid_symbols(slice)
        
        if not liquid_symbols:
            self.algorithm.Log(f"{self.name}: No liquid symbols for signal generation")
            return signals
        
        # Calculate momentum scores for all symbols
        momentum_scores = {}
        for symbol in liquid_symbols:
            if symbol not in self.symbol_data:
                continue
            
            symbol_data = self.symbol_data[symbol]
            if not symbol_data.IsReady:
                continue
            
            try:
                # Calculate total return momentum (excluding recent period)
                total_momentum = 0
                valid_lookbacks = 0
                
                for lookback_months in self.config['momentum_lookbacks_months']:
                    momentum = symbol_data.GetTotalReturn(lookback_months)
                    if momentum is not None:
                        total_momentum += momentum
                        valid_lookbacks += 1
                
                if valid_lookbacks > 0:
                    avg_momentum = total_momentum / valid_lookbacks
                    momentum_scores[symbol] = avg_momentum
                    
            except Exception as e:
                self.algorithm.Error(f"{self.name}: Error processing {symbol}: {str(e)}")
                continue
        
        if momentum_scores:
            # Standardize scores
            standardized_scores = self._standardize_scores(momentum_scores)
            
            # Convert to position weights
            signals = self._convert_scores_to_weights(standardized_scores)
        
        return signals
    
    def _get_liquid_symbols(self, slice=None):
        """Get liquid symbols from futures manager."""
        if self.futures_manager and hasattr(self.futures_manager, 'get_liquid_symbols'):
            return self.futures_manager.get_liquid_symbols(slice)
        return list(self.symbol_data.keys())
    
    def _standardize_scores(self, scores, clip_at=None):
        """Standardize momentum scores using z-score normalization."""
        if not scores:
            return {}
        
        if clip_at is None:
            clip_at = self.config['signal_standardization_clip']
        
        try:
            values = list(scores.values())
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val <= 0:
                return {symbol: 0.0 for symbol in scores}
            
            standardized = {}
            for symbol, score in scores.items():
                z_score = (score - mean_val) / std_val
                # Clip extreme values
                clipped_score = max(-clip_at, min(clip_at, z_score))
                standardized[symbol] = clipped_score
            
            return standardized
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Score standardization error: {str(e)}")
            return {}
    
    def _convert_scores_to_weights(self, scores):
        """Convert standardized scores to position weights."""
        if not scores:
            return {}
        
        try:
            weights = {}
            
            if self.config['long_short_enabled']:
                # Long-short strategy
                for symbol, score in scores.items():
                    # Convert score to weight (normalized by number of positions)
                    weight = score / len(scores)
                    weights[symbol] = weight
            else:
                # Long-only strategy
                positive_scores = {s: max(0, score) for s, score in scores.items()}
                total_positive = sum(positive_scores.values())
                
                if total_positive > 0:
                    for symbol, score in positive_scores.items():
                        weights[symbol] = score / total_positive
            
            return weights
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Weight conversion error: {str(e)}")
            return {}
    
    def _apply_position_limits(self, weights):
        """Apply individual position size limits."""
        limited_weights = {}
        max_weight = self.config['max_position_weight']
        
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
            target_vol = self.config['target_volatility']
            
            if portfolio_vol > 0:
                vol_scalar = target_vol / portfolio_vol
                vol_scalar = min(vol_scalar, self.config['max_leverage_multiplier'])
                
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
        min_threshold = self.config['min_weight_threshold']
        
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
                    if abs(target_weight) < self.config['min_weight_threshold']:
                        continue
                    
                    # Calculate position size
                    portfolio_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
                    target_value = target_weight * portfolio_value
                    
                    if abs(target_value) < self.config['min_trade_value']:
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
                'config_description': self.config.get('description', 'MTUM CTA Strategy')
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
                'enabled': self.config.get('enabled', True),
                'rebalance_frequency': self.config.get('rebalance_frequency', 'monthly'),
                'target_volatility': self.config.get('target_volatility', 0.2),
                'max_position_weight': self.config.get('max_position_weight', 0.5),
                'momentum_lookbacks': self.config.get('momentum_lookbacks_months', [6, 12]),
                'long_short_enabled': self.config.get('long_short_enabled', True),
                'description': self.config.get('description', 'MTUM CTA Strategy'),
                'config_source': 'config_manager' if self.config_manager else 'fallback'
            }
        except Exception as e:
            return {'strategy_name': self.name, 'error': str(e)}

    def _create_symbol_data(self, symbol):
        """Create MTUM-specific symbol data object."""
        return self.SymbolData(
            algorithm=self.algorithm,
            symbol=symbol,
            lookbackMonthsList=self.config['momentum_lookbacks_months'],
            volLookbackDays=self.config['volatility_lookback_days'],
            recent_exclusion_days=self.config['recent_exclusion_days']
        )

    class SymbolData:
        """MTUM-specific SymbolData for momentum calculations."""

        def __init__(self, algorithm, symbol, lookbackMonthsList, volLookbackDays, recent_exclusion_days=22):
            self.algorithm = algorithm
            self.symbol = symbol
            self.lookbackMonthsList = lookbackMonthsList
            self.volLookbackDays = volLookbackDays
            self.recent_exclusion_days = recent_exclusion_days

            # Rolling windows
            max_lookback_days = max(lookbackMonthsList) * 22 + volLookbackDays + 50  # 22 trading days/month
            self.price_window = RollingWindow[float](max_lookback_days)
            self.monthly_prices = deque(maxlen=max(lookbackMonthsList) + 5)
            
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
                algorithm.Log(f"MTUM SymbolData {symbol}: Consolidator setup error: {str(e)}")

            # Initialize with history
            self._initialize_with_history()

        def _initialize_with_history(self):
            """Initialize with historical data using CENTRALIZED data provider."""
            try:
                history_days = self.volLookbackDays + 100
                
                # Use centralized data provider if available
                if hasattr(self.algorithm, 'data_integrity_checker') and self.algorithm.data_integrity_checker:
                    history = self.algorithm.data_integrity_checker.get_history(self.symbol, history_days, Resolution.Daily)
                else:
                    # Fallback to direct API call (not recommended)
                    self.algorithm.Log(f"MTUM SymbolData {self.symbol}: WARNING - No centralized cache, using direct History API")
                    history = self.algorithm.History(self.symbol, history_days, Resolution.Daily)
                
                if history is None or history.empty:
                    self.algorithm.Log(f"No history available for {self.symbol}")
                    self.has_sufficient_data = False
                    self.data_availability_error = "No historical data available"
                    return
                
                # Process historical data
                for index, row in history.iterrows():
                    close_price = row['close']
                    self.price_window.Add(close_price)
                    self.data_points_received += 1
                
                # Update monthly prices
                self._update_monthly_prices()
                
                self.algorithm.Log(f"MTUM SymbolData {self.symbol}: Initialized with {len(history)} bars")

            except Exception as e:
                self.algorithm.Error(f"MTUM SymbolData {self.symbol}: History initialization error: {str(e)}")
                self.has_sufficient_data = False
                self.data_availability_error = f"History error: {str(e)}"

        @property
        def IsReady(self):
            """Check if symbol data is ready for calculations."""
            if not self.has_sufficient_data:
                return False
            
            min_data_points = max(self.lookbackMonthsList) * 22 + self.recent_exclusion_days + 10
            return (self.price_window.Count >= min_data_points and 
                   len(self.monthly_prices) >= max(self.lookbackMonthsList))

        def OnDataConsolidated(self, sender, bar: TradeBar):
            """Process new daily bar."""
            if bar is None or bar.Close <= 0:
                return
            
            try:
                self.price_window.Add(float(bar.Close))
                self.data_points_received += 1
                self.last_update_time = bar.Time
                
                # Update monthly prices on month-end
                if bar.Time.day >= 25:  # Approximate month-end
                    self._update_monthly_prices()
                
            except Exception as e:
                self.algorithm.Error(f"MTUM SymbolData {self.symbol}: OnDataConsolidated error: {str(e)}")

        def _update_monthly_prices(self):
            """Update monthly price snapshots."""
            if self.price_window.Count > 0:
                current_price = self.price_window[0]
                if not self.monthly_prices or current_price != self.monthly_prices[-1]:
                    self.monthly_prices.append(current_price)

        def GetTotalReturn(self, lookbackMonths):
            """Calculate total return over specified lookback period (excluding recent period)."""
            try:
                if not self.IsReady or len(self.monthly_prices) < lookbackMonths + 1:
                    return None
                
                # Get current price (excluding recent period)
                recent_days_back = min(self.recent_exclusion_days, self.price_window.Count - 1)
                current_price = self.price_window[recent_days_back] if recent_days_back > 0 else self.price_window[0]
                
                # Get lookback price
                lookback_price = self.monthly_prices[-(lookbackMonths + 1)]
                
                if lookback_price <= 0:
                    return None
                
                # Calculate total return
                total_return = (current_price / lookback_price) - 1
                return total_return
                
            except Exception as e:
                return None

        def GetVolatility(self, lookbackMonths=None):
            """Calculate volatility over specified period."""
            try:
                if not self.IsReady:
                    return None
                
                # Use default volatility lookback if not specified
                lookback_days = self.volLookbackDays if lookbackMonths is None else lookbackMonths * 22
                
                if self.price_window.Count < lookback_days + 1:
                    return None
                
                # Calculate daily returns
                returns = []
                for i in range(1, min(lookback_days + 1, self.price_window.Count)):
                    if self.price_window[i] > 0:
                        daily_return = (self.price_window[i-1] / self.price_window[i]) - 1
                        returns.append(daily_return)
                
                if len(returns) < 10:
                    return None
                
                vol = np.std(returns) * np.sqrt(252)  # Annualized
                return vol if vol > 0 else None
                
            except Exception as e:
                return None

        def GetDataQuality(self):
            """Get data quality metrics."""
            return {
                'symbol': str(self.symbol),
                'data_points_received': self.data_points_received,
                'price_window_count': self.price_window.Count,
                'monthly_prices_count': len(self.monthly_prices),
                'lookback_months': self.lookbackMonthsList,
                'vol_lookback_days': self.volLookbackDays,
                'recent_exclusion_days': self.recent_exclusion_days,
                'is_ready': self.IsReady,
                'last_update': self.last_update_time
            }

        def Dispose(self):
            """Clean disposal of resources."""
            try:
                if hasattr(self, 'consolidator') and self.consolidator:
                    self.algorithm.SubscriptionManager.RemoveConsolidator(self.symbol, self.consolidator)
                self.price_window.Reset()
                self.monthly_prices.clear()
            except:
                pass

    def OnSecuritiesChanged(self, changes):
        """Handle securities changes."""
        try:
            # Add new securities
            for security in changes.AddedSecurities:
                symbol = security.Symbol
                if symbol not in self.symbol_data:
                    self.symbol_data[symbol] = self._create_symbol_data(symbol)
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
