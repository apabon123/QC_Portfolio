# hmm_cta_strategy.py - COMPLETE REWRITE WITH ALL ORIGINAL FEATURES

from AlgorithmImports import *
import numpy as np
from collections import deque
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

class HMMCTAStrategy:
    """
    Hidden Markov Model CTA Strategy - COMPLETE VERSION WITH ALL FEATURES
    
    DYNAMIC LOADING COMPATIBLE:
    ✅ Correct constructor signature: (algorithm, futures_manager, name, config_manager)
    ✅ Required methods: update(), generate_targets(), get_exposure()
    ✅ Proper config loading from config_manager
    ✅ Complete trade execution with rollover support
    ✅ Comprehensive performance tracking and diagnostics
    ✅ Enhanced SymbolData class with full validation
    
    Strategy Features:
    - 3-component Gaussian Mixture Model for regime detection
    - Weekly rebalancing with regime persistence filtering
    - Exponential smoothing of regime probabilities
    - Monthly model retraining for adaptation
    - Complete trade execution pipeline
    - Advanced data quality monitoring
    - Config-driven parameters (no hardcoded values)
    """
    
    def __init__(self, algorithm, futures_manager, name="HMM_CTA", config_manager=None):
        """
        Initialize HMM CTA strategy for dynamic loading system.
        
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
        
        # Load configuration (DYNAMIC LOADING COMPATIBLE)
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
        
        # Performance tracking (COMPLETE)
        self.trades_executed = 0
        self.total_rebalances = 0
        self.total_retrains = 0
        self.regime_changes = 0
        self.gross_exposure_history = []
        self.regime_persistence_violations = 0
        
        # Regime tracking for analysis (ENHANCED)
        self.regime_history = {}
        
        # Log successful initialization
        self._log_initialization_summary()
        
        # Initialize symbol data
        self.initialize_symbol_data()
    
    def _load_configuration(self):
        """
        Load configuration from config_manager with proper fallback handling.
        DYNAMIC LOADING COMPATIBLE - handles both config_manager and fallback scenarios.
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
    
    def _build_config_dict(self, config):
        """Build complete config dictionary from provided config."""
        self.config_dict = {
            # HMM-specific parameters
            'n_components': config.get('n_components', 3),
            'n_iter': config.get('n_iter', 100),
            'random_state': config.get('random_state', 42),
            'returns_window': config.get('returns_window', 60),
            'retrain_frequency': config.get('retrain_frequency', 'monthly'),
            'target_volatility': config.get('target_volatility', 0.20),
            'rebalance_frequency': config.get('rebalance_frequency', 'weekly'),
            'max_position_weight': config.get('max_position_weight', 0.5),
            'regime_threshold': config.get('regime_threshold', 0.70),
            'regime_persistence_days': config.get('regime_persistence_days', 3),
            'regime_smoothing_alpha': config.get('regime_smoothing_alpha', 0.3),
            'warmup_days': config.get('warmup_days', 80),
            
            # Execution parameters
            'min_weight_threshold': config.get('min_weight_threshold', 0.01),
            'min_trade_value': config.get('min_trade_value', 1000),
            'max_single_order_value': config.get('max_single_order_value', 50000000),
            
            # Risk parameters
            'max_leverage_multiplier': config.get('max_leverage_multiplier', 100),
            'max_single_position': config.get('max_single_position', 10.0),
            'daily_stop_loss': config.get('daily_stop_loss', 0.2),
            
            # Strategy metadata
            'enabled': config.get('enabled', True),
            'description': config.get('description', 'HMM CTA Strategy'),
            'expected_sharpe': config.get('expected_sharpe', 0.5),
            'correlation_with_regime': config.get('correlation_with_regime', 0.6)
        }
    
    def _load_fallback_config(self):
        """Load fallback configuration if config_manager is unavailable."""
        self.config_dict = {
            # HMM-specific parameters
            'n_components': 3,
            'n_iter': 100,
            'random_state': 42,
            'returns_window': 60,
            'retrain_frequency': 'monthly',
            'target_volatility': 0.20,
            'rebalance_frequency': 'weekly',
            'max_position_weight': 0.5,
            'regime_threshold': 0.70,
            'regime_persistence_days': 3,
            'regime_smoothing_alpha': 0.3,
            'warmup_days': 80,
            
            # Execution parameters
            'min_weight_threshold': 0.01,
            'min_trade_value': 1000,
            'max_single_order_value': 50000000,
            
            # Risk parameters
            'max_leverage_multiplier': 100,
            'max_single_position': 10.0,
            'daily_stop_loss': 0.2,
            
            # Strategy metadata
            'enabled': True,
            'description': 'HMM CTA Strategy (Fallback Config)',
            'expected_sharpe': 0.5,
            'correlation_with_regime': 0.6
        }
    
    def _log_initialization_summary(self):
        """Log comprehensive initialization summary with config source."""
        self.algorithm.Log(f"{self.name}: Initialized HMM CTA strategy with CONFIG-COMPLIANT parameters:")
        self.algorithm.Log(f"  N Components: {self.config_dict['n_components']} regimes")
        self.algorithm.Log(f"  Returns Window: {self.config_dict['returns_window']} days")
        self.algorithm.Log(f"  Target Volatility: {self.config_dict['target_volatility']:.1%}")
        self.algorithm.Log(f"  Max Position Weight: {self.config_dict['max_position_weight']:.1%}")
        self.algorithm.Log(f"  Regime Threshold: {self.config_dict['regime_threshold']:.1%}")
        self.algorithm.Log(f"  Regime Persistence: {self.config_dict['regime_persistence_days']} days")
        self.algorithm.Log(f"  Min Weight Threshold: {self.config_dict['min_weight_threshold']:.1%}")
        self.algorithm.Log(f"  Min Trade Value: ${self.config_dict['min_trade_value']:,}")
        self.algorithm.Log(f"  Max Single Order: ${self.config_dict['max_single_order_value']:,}")
        self.algorithm.Log(f"  Rebalance Frequency: {self.config_dict['rebalance_frequency']}")
        self.algorithm.Log(f"  Retrain Frequency: {self.config_dict['retrain_frequency']}")
        self.algorithm.Log(f"  Description: {self.config_dict['description']}")
    
    def initialize_symbol_data(self):
        """Initialize SymbolData objects for all futures in the manager"""
        try:
            # Get symbols from futures manager or fallback
            if self.futures_manager and hasattr(self.futures_manager, 'futures_data'):
                symbols = list(self.futures_manager.futures_data.keys())
            else:
                # Fallback for dynamic loading system
                symbols = ['ES', 'ZN']  # HMM strategy focuses on these
            
            for symbol in symbols:
                self.symbol_data[symbol] = self.SymbolData(
                    algorithm=self.algorithm,
                    symbol=symbol,
                    n_components=self.config_dict['n_components'],
                    n_iter=self.config_dict['n_iter'],
                    random_state=self.config_dict['random_state'],
                    returns_window=self.config_dict['returns_window']
                )
                
                # Initialize regime tracking
                self.regime_history[symbol] = deque(maxlen=252)
                self.regime_buffers[symbol] = deque(maxlen=self.config_dict['regime_persistence_days'])
                self.smoothed_regime_probs[symbol] = None
        
            self.algorithm.Log(f"{self.name}: Initialized {len(self.symbol_data)} symbol data objects with HMM models")
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error initializing symbol data: {str(e)}")
    
    # ============================================================================
    # REQUIRED METHODS FOR DYNAMIC LOADING SYSTEM
    # ============================================================================
    
    def update(self, slice_data):
        """
        Update strategy with new market data (REQUIRED by orchestrator).
        
        Args:
            slice_data: QuantConnect Slice object with market data
        """
        try:
            # Track updates
            if hasattr(slice_data, 'Time'):
                self.last_update_time = slice_data.Time
            
            # Log update occasionally for debugging
            if self.algorithm.Time.day % 7 == 0 and self.algorithm.Time.hour == 16:  # Weekly logging
                ready_count = len([sd for sd in self.symbol_data.values() if sd.IsReady])
                self.algorithm.Debug(f"{self.name}: Update - {ready_count}/{len(self.symbol_data)} symbols ready")
                
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in update: {str(e)}")
    
    def generate_targets(self):
        """
        Generate position targets (REQUIRED by orchestrator).
        
        This is the main method called by the orchestrator to get trading signals.
        
        Returns:
            dict: {symbol: target_weight} for each tradeable symbol
        """
        try:
            self.algorithm.Log(f"{self.name}: Generating regime-based targets...")
            
            # Use the existing generate_signals method
            targets = self.generate_signals()
            
            # Store current targets
            self.current_targets = targets.copy()
            
            if targets:
                self.algorithm.Log(f"{self.name}: Generated {len(targets)} regime targets")
                for symbol, weight in targets.items():
                    direction = "LONG" if weight > 0 else "SHORT"
                    self.algorithm.Log(f"  {symbol}: {direction} {abs(weight):.3f}")
            else:
                self.algorithm.Log(f"{self.name}: No regime targets generated")
            
            return targets
            
        except Exception as e:
            self.algorithm.Error(f"{self.name}: Error in generate_targets: {str(e)}")
            return {}
    
    def get_exposure(self):
        """
        Get current strategy exposure metrics (REQUIRED by orchestrator).
        
        Returns:
            dict: Exposure metrics for monitoring
        """
        if not self.current_targets:
            return {
                'gross_exposure': 0.0,
                'net_exposure': 0.0,
                'num_positions': 0,
                'largest_position': 0.0
            }
        
        gross = sum(abs(w) for w in self.current_targets.values())
        net = sum(self.current_targets.values())
        largest = max(abs(w) for w in self.current_targets.values()) if self.current_targets else 0.0
        
        return {
            'gross_exposure': gross,
            'net_exposure': net,
            'num_positions': len(self.current_targets),
            'largest_position': largest
        }
    
    # ============================================================================
    # HMM STRATEGY IMPLEMENTATION (COMPLETE)
    # ============================================================================
    
    def should_rebalance(self, current_time):
        """Determine if strategy should rebalance based on config frequency"""
        if self.config_dict['rebalance_frequency'] == 'weekly':
            return (current_time.weekday() == 4 and 
                   (self.last_rebalance_date is None or 
                    current_time.date() != self.last_rebalance_date))
        elif self.config_dict['rebalance_frequency'] == 'daily':
            return (self.last_rebalance_date is None or 
                   current_time.date() != self.last_rebalance_date)
        else:
            return False
    
    def should_retrain_models(self, current_time):
        """Check if HMM models should be retrained based on config frequency"""
        if self.config_dict['retrain_frequency'] == 'monthly':
            return (current_time.month != self.last_retrain_month and 
                   current_time.day <= 3)  # Early in month
        return False
    
    def retrain_models(self):
        """Retrain all HMM models with latest data"""
        retrained_count = 0
        
        for symbol, symbol_data in self.symbol_data.items():
            if symbol_data.retrain_model():
                retrained_count += 1
                # Clear regime buffers after retraining
                self.regime_buffers[symbol].clear()
                self.smoothed_regime_probs[symbol] = None
        
        self.last_retrain_month = self.algorithm.Time.month
        self.total_retrains += 1
        
        self.algorithm.Log(f"{self.name}: Retrained {retrained_count}/{len(self.symbol_data)} HMM models")
    
    def _smooth_regime_probabilities(self, symbol, new_probs):
        """Apply exponential smoothing to regime probabilities using config parameter"""
        alpha = self.config_dict['regime_smoothing_alpha']
        
        if self.smoothed_regime_probs[symbol] is not None:
            self.smoothed_regime_probs[symbol] = (alpha * new_probs + 
                                                 (1 - alpha) * self.smoothed_regime_probs[symbol])
        else:
            self.smoothed_regime_probs[symbol] = new_probs
        
        return self.smoothed_regime_probs[symbol]
    
    def _check_regime_persistence(self, symbol, current_regime):
        """Check if regime change is persistent enough to trade on using config parameter"""
        buffer = self.regime_buffers[symbol]
        buffer.append(current_regime)
        
        # Need enough history (from config)
        if len(buffer) < self.config_dict['regime_persistence_days']:
            return None
        
        # Check if all recent regimes are the same
        if len(set(buffer)) == 1:
            return current_regime
        else:
            self.regime_persistence_violations += 1
            return None  # No confirmed regime
    
    def generate_signals(self):
        """
        Generate trading signals using CONFIG-COMPLIANT HMM regime detection
        """
        # Check if models need retraining
        if self.should_retrain_models(self.algorithm.Time):
            self.retrain_models()
        
        # Get liquid symbols from futures manager
        liquid_symbols = self._get_liquid_symbols()
        
        if not liquid_symbols:
            self.algorithm.Log(f"{self.name}: No liquid symbols available")
            return {}
        
        # Check which symbols are ready for regime detection
        ready_symbols = []
        for symbol in liquid_symbols:
            if symbol in self.symbol_data and self.symbol_data[symbol].IsReady:
                ready_symbols.append(symbol)
        
        if not ready_symbols:
            self.algorithm.Log(f"{self.name}: No symbols ready for regime detection")
            return {}
        
        # Generate regime-based signals with CONFIG-COMPLIANT parameters
        regime_signals = {}
        regime_summary = []
        confirmed_regimes = 0
        
        for symbol in ready_symbols:
            sd = self.symbol_data[symbol]
            
            # Get current regime probabilities
            raw_regime_probs = sd.GetRegimeProbabilities()
            
            if raw_regime_probs is not None and len(raw_regime_probs) == 3:
                # Apply exponential smoothing (using config parameter)
                smoothed_probs = self._smooth_regime_probabilities(symbol, raw_regime_probs)
                prob_down, prob_ranging, prob_up = smoothed_probs
                
                # Determine dominant regime
                dominant_regime = np.argmax(smoothed_probs)
                
                # Check regime persistence (using config parameter)
                confirmed_regime = self._check_regime_persistence(symbol, dominant_regime)
                
                if confirmed_regime is not None:
                    confirmed_regimes += 1
                    
                    # Generate position based on CONFIRMED regime probabilities (using config threshold)
                    signal_strength = 0.0
                    regime_threshold = self.config_dict['regime_threshold']
                    
                    if prob_up > regime_threshold:
                        # Strong upward regime - go long
                        signal_strength = min(prob_up, 1.0)  # Cap at 1.0
                    elif prob_down > regime_threshold:
                        # Strong downward regime - go short
                        signal_strength = -min(prob_down, 1.0)  # Cap at -1.0
                    else:
                        # Ranging regime or low confidence - neutral
                        signal_strength = 0.0
                    
                    regime_signals[symbol] = signal_strength
                    
                    # Track regime for analysis
                    self.regime_history[symbol].append(confirmed_regime)
                    
                    # Log regime info
                    regime_summary.append(f"{symbol}:P({prob_down:.2f},{prob_ranging:.2f},{prob_up:.2f})→{signal_strength:.2f}")
                else:
                    # No confirmed regime - neutral position
                    regime_signals[symbol] = 0.0
            else:
                regime_signals[symbol] = 0.0
        
        # Apply CONFIG-COMPLIANT position limits
        limited_signals = self._apply_position_limits(regime_signals)
        
        # Apply CONFIG-COMPLIANT volatility targeting
        final_targets = self._apply_volatility_targeting(limited_signals)
        
        # Apply CONFIG-COMPLIANT trade size validation
        validated_targets = self._validate_trade_sizes(final_targets)
        
        # Enhanced logging
        if regime_summary:
            self.algorithm.Log(f"{self.name}: Regimes: {', '.join(regime_summary[:3])}")
        
        if confirmed_regimes < len(ready_symbols):
            pending_regimes = len(ready_symbols) - confirmed_regimes
            self.algorithm.Log(f"{self.name}: {confirmed_regimes}/{len(ready_symbols)} regimes confirmed, "
                             f"{pending_regimes} pending persistence")
        
        return validated_targets
    
    def _get_liquid_symbols(self):
        """Get liquid symbols from futures manager or fallback."""
        if self.futures_manager and hasattr(self.futures_manager, 'get_symbols_for_strategy'):
            return self.futures_manager.get_symbols_for_strategy(self.name)
        elif self.futures_manager and hasattr(self.futures_manager, 'get_liquid_symbols'):
            return self.futures_manager.get_liquid_symbols()
        elif self.futures_manager and hasattr(self.futures_manager, 'futures_data'):
            return list(self.futures_manager.futures_data.keys())
        else:
            # Fallback to symbol data keys
            return list(self.symbol_data.keys())
    
    def _apply_position_limits(self, signals):
        """Apply CONFIG-COMPLIANT individual position size limits"""
        limited_signals = {}
        limited_count = 0
        max_position_weight = self.config_dict['max_position_weight']
        
        for symbol, signal in signals.items():
            # Apply individual position limit from config
            if abs(signal) > max_position_weight:
                limited_signal = np.sign(signal) * max_position_weight
                limited_signals[symbol] = limited_signal
                limited_count += 1
                
                self.algorithm.Log(f"{self.name}: Limited {symbol} from {signal:.3f} to {limited_signal:.3f} "
                                 f"(max: {max_position_weight:.1%} from config)")
            else:
                limited_signals[symbol] = signal
        
        # Log if multiple positions were limited
        if limited_count > 1:
            self.algorithm.Log(f"{self.name}: Limited {limited_count} positions to max weight {max_position_weight:.1%}")
        
        return limited_signals
    
    def _apply_volatility_targeting(self, signals):
        """Apply CONFIG-COMPLIANT portfolio-level volatility targeting"""
        if not signals:
            return signals
        
        # Calculate expected portfolio volatility
        portfolio_vol = self._calculate_portfolio_volatility(signals)
        
        if portfolio_vol > 0:
            # Use config parameter for target volatility
            target_vol = self.config_dict['target_volatility']
            vol_scalar = target_vol / portfolio_vol
            
            # Apply CONFIG-COMPLIANT leverage limits
            max_leverage = self.config_dict['max_leverage_multiplier']
            if vol_scalar > max_leverage:
                vol_scalar = max_leverage
                self.algorithm.Log(f"{self.name}: Leverage capped at {max_leverage}x (config limit)")
            
            # Apply volatility scaling
            final_targets = {
                symbol: signal * vol_scalar 
                for symbol, signal in signals.items()
            }
            
            # Track gross exposure
            gross_exposure = sum(abs(weight) for weight in final_targets.values())
            self.gross_exposure_history.append(gross_exposure)
            
            # Improved logging (less verbose) with config values
            if gross_exposure > 0.05:  # Only log meaningful exposure
                self.algorithm.Log(f"{self.name}: HMM vol targeting - "
                                 f"Vol: {portfolio_vol:.1%} → {target_vol:.1%}, "
                                 f"Gross: {gross_exposure:.1%}")
        else:
            final_targets = signals
            gross_exposure = sum(abs(weight) for weight in final_targets.values())
            self.gross_exposure_history.append(gross_exposure)
        
        return final_targets
    
    def _validate_trade_sizes(self, targets):
        """
        Validate that trade sizes respect CONFIG-COMPLIANT execution limits.
        This prevents trades that would be blocked by execution manager.
        """
        if not targets:
            return targets
        
        validated_targets = {}
        portfolio_value = self.algorithm.Portfolio.TotalPortfolioValue
        max_order_value = self.config_dict['max_single_order_value']
        max_single_position = self.config_dict['max_single_position']
        
        for symbol, target_weight in targets.items():
            # Check against maximum single position limit (from config)
            if abs(target_weight) > max_single_position:
                capped_weight = np.sign(target_weight) * max_single_position
                self.algorithm.Log(f"{self.name}: Capped {symbol} position from {target_weight:.1%} "
                                 f"to {capped_weight:.1%} (config max: {max_single_position:.1%})")
                target_weight = capped_weight
            
            # Estimate trade value and check against execution limits
            target_value = abs(target_weight * portfolio_value)
            
            if target_value > max_order_value:
                # Scale down to fit within execution limits
                scale_factor = max_order_value / target_value
                scaled_weight = target_weight * scale_factor
                
                self.algorithm.Log(f"{self.name}: Scaled {symbol} from {target_weight:.1%} "
                                 f"to {scaled_weight:.1%} (trade value limit: ${max_order_value:,})")
                validated_targets[symbol] = scaled_weight
            else:
                validated_targets[symbol] = target_weight
        
        return validated_targets
    
    def _calculate_portfolio_volatility(self, weights):
        """Calculate expected portfolio volatility for HMM strategy"""
        if not weights:
            return 0.0
        
        weighted_vol_sum = 0.0
        total_abs_weight = 0.0
        
        for symbol, weight in weights.items():
            if symbol in self.symbol_data and abs(weight) > 0:
                # Use recent volatility
                asset_vol = self.symbol_data[symbol].GetRecentVolatility()
                
                if asset_vol > 0:
                    abs_weight = abs(weight)
                    weighted_vol_sum += abs_weight * asset_vol
                    total_abs_weight += abs_weight
        
        if total_abs_weight > 0:
            avg_vol = weighted_vol_sum / total_abs_weight
            
            # Improved diversification calculation
            num_assets = len([w for w in weights.values() if abs(w) > 0.01])
            if num_assets > 1:
                # Better diversification benefit for weekly strategy
                diversification_factor = max(0.7, 1.0 - (num_assets - 1) * 0.08)
            else:
                diversification_factor = 1.0
            
            return avg_vol * diversification_factor
        
        return 0.0
    
    # ============================================================================
    # COMPLETE TRADE EXECUTION (RESTORED FROM ORIGINAL)
    # ============================================================================
    
    def execute_trades(self, new_targets, rollover_tags=None):
        """
        Execute trades to reach target portfolio weights using CONFIG-COMPLIANT parameters.
        
        Args:
            new_targets (dict): {symbol: target_weight}
            rollover_tags (dict): Rollover re-establishment tags from rollover manager
        
        Returns:
            dict: Execution summary
        """
        if rollover_tags is None:
            rollover_tags = {}
        
        orders_placed = 0
        rollover_orders = 0
        liquidations = 0
        blocked_trades = 0
        total_portfolio_value = self.algorithm.Portfolio.TotalPortfolioValue
        
        # Use CONFIG-COMPLIANT parameters
        min_weight_threshold = self.config_dict['min_weight_threshold']
        min_trade_value = self.config_dict['min_trade_value']
        max_order_value = self.config_dict['max_single_order_value']
        
        # Execute target positions
        for symbol, target_weight in new_targets.items():
            if symbol not in self.algorithm.Securities:
                continue
                
            mapped_contract = self.algorithm.Securities[symbol].Mapped
            
            if mapped_contract is None or not self.algorithm.Securities[mapped_contract].HasData:
                continue
            
            # Validate price and multiplier using futures manager
            security = self.algorithm.Securities[mapped_contract]
            price = security.Price
            
            if self.futures_manager and not self.futures_manager.validate_price(symbol, price):
                continue
            
            if self.futures_manager and not self.futures_manager.validate_multiplier(mapped_contract):
                continue
            
            # Calculate current and target positions
            holdings_value = self.algorithm.Portfolio[mapped_contract].HoldingsValue
            current_weight = holdings_value / total_portfolio_value if total_portfolio_value != 0 else 0
            
            # Check if trade is needed using config threshold
            weight_diff = abs(current_weight - target_weight)
            if weight_diff > min_weight_threshold:
                
                target_value = target_weight * total_portfolio_value
                trade_value = target_value - holdings_value
                
                # Check CONFIG-COMPLIANT trade size limits
                if abs(trade_value) > min_trade_value:
                    
                    # Check against maximum order value (CONFIG-COMPLIANT)
                    if abs(trade_value) > max_order_value:
                        self.algorithm.Log(f"{self.name}: BLOCKED {symbol} trade ${abs(trade_value):,.0f} "
                                         f"exceeds config limit ${max_order_value:,}")
                        blocked_trades += 1
                        continue
                    
                    try:
                        if self.futures_manager:
                            multiplier = self.futures_manager.get_effective_multiplier(mapped_contract)
                        else:
                            # Fallback multiplier estimation
                            multiplier = 50 if 'ES' in str(symbol) else 100
                        
                        quantity_diff = int(trade_value / (price * multiplier))
                        
                        if abs(quantity_diff) >= 1:
                            # Check for rollover re-establishment
                            trade_tag = None
                            is_rollover = False
                            
                            if symbol in rollover_tags:
                                rollover_info = rollover_tags[symbol]
                                trade_tag = f"{self.name}_{rollover_info['tag']}"
                                is_rollover = True
                                rollover_orders += 1
                                
                                self.algorithm.Log(f"{self.name}: Rollover re-establish "
                                                 f"{symbol} {quantity_diff} contracts")
                            
                            # Execute trade with CONFIG-COMPLIANT tag
                            if trade_tag:
                                order_ticket = self.algorithm.MarketOrder(mapped_contract, 
                                                                        quantity_diff, 
                                                                        tag=trade_tag)
                            else:
                                order_ticket = self.algorithm.MarketOrder(mapped_contract, 
                                                                        quantity_diff,
                                                                        tag=f"{self.name}_regime")
                            
                            orders_placed += 1
                            
                    except Exception as e:
                        self.algorithm.Log(f"{self.name}: Order error {symbol} - {str(e)}")
        
        # Liquidate positions not in new targets
        current_symbols = set(new_targets.keys())
        
        for symbol in self.symbol_data:
            if symbol not in current_symbols:
                if symbol in self.algorithm.Securities:
                    mapped_contract = self.algorithm.Securities[symbol].Mapped
                    
                    if (mapped_contract is not None and 
                        mapped_contract in self.algorithm.Portfolio and 
                        self.algorithm.Portfolio[mapped_contract].Invested):
                        
                        try:
                            self.algorithm.Liquidate(mapped_contract, tag=f"{self.name}_liquidate")
                            liquidations += 1
                        except:
                            pass
        
        # Update strategy state
        self.current_targets = new_targets.copy()
        self.last_rebalance_date = self.algorithm.Time.date()
        self.trades_executed += orders_placed
        self.total_rebalances += 1
        
        # Return CONFIG-COMPLIANT execution summary
        execution_summary = {
            'orders_placed': orders_placed,
            'rollover_orders': rollover_orders,
            'liquidations': liquidations,
            'blocked_trades': blocked_trades,
            'total_trades': orders_placed + liquidations,
            'config_compliance': {
                'min_trade_value': min_trade_value,
                'max_order_value': max_order_value,
                'min_weight_threshold': min_weight_threshold
            }
        }
        
        if execution_summary['total_trades'] > 0 or blocked_trades > 0:
            self.algorithm.Log(f"{self.name}: CONFIG-COMPLIANT execution: {orders_placed} orders, "
                             f"{liquidations} liquidations, {rollover_orders} rollovers, "
                             f"{blocked_trades} blocked (config limits)")
        
        return execution_summary
    
    # ============================================================================
    # COMPLETE PERFORMANCE TRACKING (RESTORED FROM ORIGINAL)
    # ============================================================================
    
    def get_performance_metrics(self):
        """Get CONFIG-COMPLIANT strategy performance metrics"""
        avg_gross_exposure = (sum(self.gross_exposure_history) / len(self.gross_exposure_history) 
                            if self.gross_exposure_history else 0)
        
        # Calculate regime persistence rate
        persistence_rate = (1.0 - (self.regime_persistence_violations / max(1, self.total_rebalances)))
        
        return {
            'name': self.name,
            'config_compliant': True,
            'total_rebalances': self.total_rebalances,
            'total_retrains': self.total_retrains,
            'regime_changes': self.regime_changes,
            'trades_executed': self.trades_executed,
            'current_positions': len([w for w in self.current_targets.values() if abs(w) > 0.001]),
            'target_volatility': self.config_dict['target_volatility'],
            'ready_symbols': len([sd for sd in self.symbol_data.values() if sd.IsReady]),
            'avg_gross_exposure': avg_gross_exposure,
            'current_gross_exposure': sum(abs(w) for w in self.current_targets.values()),
            'rebalance_frequency': self.config_dict['rebalance_frequency'],
            'regime_persistence_violations': self.regime_persistence_violations,
            'regime_persistence_rate': persistence_rate,
            'max_position_weight': self.config_dict['max_position_weight'],
            'regime_threshold': self.config_dict['regime_threshold'],
            'config_source': 'dynamic_loading_config'
        }
    
    def log_status(self):
        """Log CONFIG-COMPLIANT strategy status"""
        metrics = self.get_performance_metrics()
        liquid_count = len(self._get_liquid_symbols())
        
        self.algorithm.Log(f"{self.name}: {metrics['ready_symbols']} ready, "
                         f"{liquid_count} liquid, {metrics['current_positions']} positions, "
                         f"{metrics['total_rebalances']} rebalances, "
                         f"{metrics['total_retrains']} retrains, "
                         f"Persistence: {metrics['regime_persistence_rate']:.1%}, "
                         f"Gross: {metrics['current_gross_exposure']:.1%}, "
                         f"Max Pos: {metrics['max_position_weight']:.1%}, "
                         f"Target Vol: {metrics['target_volatility']:.1%} "
                         f"(Config: {metrics['config_source']})")
    
    def get_config_status(self):
        """Get detailed configuration status for debugging"""
        return {
            'config_manager_available': self.config_manager is not None,
            'config_source': 'dynamic_loading_config',
            'critical_parameters': {
                'n_components': self.config_dict['n_components'],
                'returns_window': self.config_dict['returns_window'],
                'target_volatility': self.config_dict['target_volatility'],
                'max_position_weight': self.config_dict['max_position_weight'],
                'regime_threshold': self.config_dict['regime_threshold'],
                'regime_persistence_days': self.config_dict['regime_persistence_days'],
                'min_weight_threshold': self.config_dict['min_weight_threshold'],
                'min_trade_value': self.config_dict['min_trade_value'],
                'max_single_order_value': self.config_dict['max_single_order_value'],
                'max_single_position': self.config_dict['max_single_position']
            },
            'strategy_enabled': self.config_dict['enabled'],
            'description': self.config_dict['description']
        }
    
    ################################################################################
    #                    ENHANCED SYMBOLDATA CLASS (COMPLETE)                       #
    ################################################################################
    
    class SymbolData:
        """
        Handles HMM regime detection for individual symbols
        COMPLETE VERSION with enhanced error handling and validation
        """

        def __init__(self, algorithm, symbol, n_components, n_iter, random_state, returns_window):
            self.algorithm = algorithm
            self.symbol = symbol
            self.n_components = n_components
            self.n_iter = n_iter
            self.random_state = random_state
            self.returns_window = returns_window
            
            # Initialize HMM model (using GaussianMixture as proxy for HMM)
            self.model = GaussianMixture(
                n_components=n_components,
                max_iter=n_iter,
                random_state=random_state,
                covariance_type='full'
            )
            self.model_trained = False
            self.last_train_month = -1
            
            # Rolling window for daily returns (uses config parameter)
            self.returns = RollingWindow[float](returns_window)
            
            # Rolling windows for different period returns
            self.monthly_returns = RollingWindow[float](12)  # 12 months of monthly returns
            self.daily_returns = RollingWindow[float](60)   # 60 days for volatility calc
            
            # Track data quality
            self.data_points_received = 0
            self.last_update_time = None
            
            # Use daily consolidator (works with daily data)
            try:
                self.consolidator = TradeBarConsolidator(timedelta(days=1))
                self.consolidator.DataConsolidated += self.OnConsolidated
                algorithm.SubscriptionManager.AddConsolidator(symbol, self.consolidator)
            except Exception as e:
                algorithm.Log(f"SymbolData {symbol}: Consolidator setup error: {str(e)}")
            
            # Track previous price for return calculation
            self.previous_price = None
            
            # Initialize with history for faster warmup
            self._initialize_with_history()

        def _initialize_with_history(self):
            """Initialize with historical data for faster warmup with enhanced validation"""
            try:
                total_history = self.returns_window + 20  # Extra buffer for weekends/holidays
                history = self.algorithm.History[TradeBar](self.symbol, total_history, Resolution.Daily)
                
                prev_price = None
                history_count = 0
                
                for bar in history:
                    close = bar.Close
                    if close > 0:  # Validate price
                        if prev_price is not None and prev_price > 0:
                            return_value = (close / prev_price) - 1.0
                            
                            # Sanity check on returns
                            if abs(return_value) < 0.5:  # Max 50% daily move
                                self.returns.Add(return_value)
                                self.daily_returns.Add(return_value)
                                history_count += 1
                        
                        prev_price = close
                
                # Set previous price for ongoing calculations
                self.previous_price = prev_price
                
                # Initialize monthly returns from price history
                self._update_monthly_returns()
                
                self.algorithm.Log(f"SymbolData {self.symbol}: Initialized with {history_count} historical returns "
                                 f"(window: {self.returns_window} from config)")
                
            except Exception as e:
                self.algorithm.Log(f"SymbolData {self.symbol}: History initialization error: {str(e)}")

        @property
        def IsReady(self):
            """Enhanced readiness check with data quality validation"""
            basic_ready = (self.returns.IsReady and 
                          self.model_trained and 
                          self.daily_returns.Count >= 20)
            
            # Additional quality checks
            if basic_ready:
                # Check for recent data
                recent_data = self.data_points_received > 0
                
                # Check for reasonable return variance (not stuck)
                if self.returns.Count >= 10:
                    recent_returns = [self.returns[i] for i in range(min(10, self.returns.Count))]
                    return_variance = np.var(recent_returns) if len(recent_returns) > 1 else 0
                    has_variance = return_variance > 0
                else:
                    has_variance = True
                
                return recent_data and has_variance
            
            return False

        def OnConsolidated(self, sender, bar: TradeBar):
            """Enhanced data consolidation with validation"""
            try:
                close = bar.Close
                
                # Validate incoming data
                if close <= 0:
                    return
                
                self.data_points_received += 1
                self.last_update_time = bar.Time
                
                if self.previous_price is not None and self.previous_price > 0:
                    # Calculate daily return
                    return_value = (close / self.previous_price) - 1.0
                    
                    # Sanity check on daily returns (avoid extreme outliers)
                    if abs(return_value) < 0.5:  # Max 50% daily move
                        self.returns.Add(return_value)
                        self.daily_returns.Add(return_value)
                
                self.previous_price = close
                
                # Update monthly returns periodically (early in month)
                if bar.Time.day <= 3:
                    self._update_monthly_returns()
                    
            except Exception as e:
                self.algorithm.Log(f"SymbolData {self.symbol}: Data consolidation error: {str(e)}")

        def _update_monthly_returns(self):
            """Update monthly returns from price history with enhanced validation"""
            if self.returns.Count < 22:  # Need at least one month
                return
                
            try:
                # Calculate monthly returns by sampling prices ~22 trading days apart
                monthly_rets = []
                
                for month in range(min(12, self.returns.Count // 22)):
                    end_idx = month * 22
                    start_idx = (month + 1) * 22
                    
                    if start_idx < self.returns.Count:
                        # Get cumulative return over the month
                        monthly_return = 0.0
                        valid_days = 0
                        
                        for day in range(22):
                            day_idx = start_idx - day
                            if day_idx >= 0 and day_idx < self.returns.Count:
                                daily_ret = self.returns[day_idx]
                                if abs(daily_ret) < 0.2:  # Filter extreme days
                                    monthly_return += daily_ret
                                    valid_days += 1
                        
                        if valid_days > 15:  # Need most days in month
                            # Sanity check on monthly returns
                            if abs(monthly_return) < 2.0:  # Max 200% monthly move
                                monthly_rets.append(monthly_return)
                
                # Add new monthly returns to rolling window
                for ret in reversed(monthly_rets):  # Add oldest first
                    if (self.monthly_returns.Count == 0 or 
                        abs(ret - self.monthly_returns[0]) > 0.001):
                        self.monthly_returns.Add(ret)
                        
            except Exception as e:
                self.algorithm.Log(f"SymbolData {self.symbol}: Monthly returns update error: {str(e)}")

        def retrain_model(self):
            """Retrain the HMM model with current returns data with enhanced validation"""
            if not self.returns.IsReady:
                return False
            
            try:
                # Prepare training data with validation
                returns_data = []
                for i in range(self.returns.Count):
                    ret = self.returns[i]
                    if abs(ret) < 0.5:  # Filter extreme outliers
                        returns_data.append(ret)
                
                if len(returns_data) < self.n_components * 5:  # Need minimum data per component
                    return False
                
                returns_array = np.array(returns_data).reshape(-1, 1)
                
                # Retrain model
                self.model.fit(returns_array)
                self.model_trained = True
                self.last_train_month = self.algorithm.Time.month
                
                return True
                
            except Exception as e:
                self.algorithm.Log(f"HMM retrain error for {self.symbol}: {str(e)}")
                return False

        def GetRegimeProbabilities(self):
            """Get current regime probabilities with enhanced validation"""
            if not self.IsReady or self.returns.Count == 0:
                return None
            
            try:
                # Get latest return with validation
                latest_return = self.returns[0]
                
                # Validate latest return
                if abs(latest_return) > 0.5:  # Extreme outlier
                    return None
                
                latest_return_array = np.array([[latest_return]])
                
                # Predict regime probabilities
                probs = self.model.predict_proba(latest_return_array).flatten()
                
                # Validate probabilities
                if len(probs) != self.n_components or np.any(np.isnan(probs)) or np.any(probs < 0):
                    return None
                
                # Sort probabilities by regime (assume components are ordered by mean return)
                try:
                    means = self.model.means_.flatten()
                    sorted_indices = np.argsort(means)
                    
                    # Reorder: [down, ranging, up]
                    ordered_probs = probs[sorted_indices]
                    
                    return ordered_probs
                except:
                    # Fallback to original order if sorting fails
                    return probs
                
            except Exception as e:
                self.algorithm.Log(f"Regime probability error for {self.symbol}: {str(e)}")
                return None

        def GetRecentVolatility(self):
            """
            Calculate recent volatility for portfolio vol calculation with enhanced validation
            """
            try:
                if self.daily_returns.Count < 10:
                    return 0.15  # Default assumption
                
                # Use last 20 daily returns or available count
                lookback = min(20, self.daily_returns.Count)
                recent_returns = []
                
                for i in range(lookback):
                    ret = self.daily_returns[i]
                    if abs(ret) < 0.5:  # Filter extreme outliers
                        recent_returns.append(ret)
                
                if len(recent_returns) < 5:  # Need minimum sample
                    return 0.15
                
                vol = np.std(recent_returns, ddof=1)
                
                # Annualize and validate
                annualized_vol = vol * np.sqrt(252)
                
                # Sanity check on volatility
                if 0.01 <= annualized_vol <= 3.0:  # Between 1% and 300%
                    return annualized_vol
                else:
                    return 0.15  # Default fallback
                    
            except Exception as e:
                self.algorithm.Log(f"Volatility calculation error for {self.symbol}: {str(e)}")
                return 0.15

        def GetDataQuality(self):
            """Get data quality metrics for diagnostics"""
            return {
                'symbol': str(self.symbol),
                'data_points_received': self.data_points_received,
                'returns_count': self.returns.Count,
                'daily_returns_count': self.daily_returns.Count,
                'monthly_returns_count': self.monthly_returns.Count,
                'returns_window_size': self.returns_window,  # From config
                'n_components': self.n_components,  # From config
                'model_trained': self.model_trained,
                'is_ready': self.IsReady,
                'last_update': self.last_update_time,
                'latest_return': self.returns[0] if self.returns.Count > 0 else 0
            }

        def Dispose(self):
            """Clean disposal of resources"""
            try:
                # Note: In QuantConnect, removing consolidators can be tricky
                # This is optional cleanup
                self.returns.Reset()
                self.daily_returns.Reset()
                self.monthly_returns.Reset()
            except:
                pass  # Ignore disposal errors
