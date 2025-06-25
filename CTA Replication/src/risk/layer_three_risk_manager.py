# layer_three_risk_manager.py - COMPLETE FINAL VERSION

from AlgorithmImports import *
from collections import deque
import numpy as np

class LayerThreeRiskManager:
    """
    Layer 3: Enhanced Portfolio Risk Management System
    Complete implementation with daily return tracking and real covariance matrix.
    """
    
    def __init__(self, algorithm, config_manager):
        """
        Initialize Layer Three Risk Manager with centralized configuration.
        CRITICAL: All configuration comes through centralized config manager only.
        """
        self.algorithm = algorithm
        self.config_manager = config_manager
        
        try:
            # Get risk configuration through centralized manager ONLY
            self.risk_config = self.config_manager.get_risk_config()
            
            # Initialize risk parameters from validated config
            self.target_portfolio_vol = self.risk_config['target_portfolio_vol']
            self.min_notional_exposure = self.risk_config['min_notional_exposure']
            self.max_leverage_multiplier = self.risk_config['max_leverage_multiplier']
            self.daily_stop_loss = self.risk_config['daily_stop_loss']
            self.max_drawdown_stop = self.risk_config['max_drawdown_stop']
            self.max_single_position = self.risk_config['max_single_position']
            
            # Initialize correlation and volatility parameters from allocation config
            allocation_config = self.config_manager.get_allocation_config()
            self.use_correlation = allocation_config.get('use_correlation', True)
            self.correlation_lookback_days = allocation_config.get('correlation_lookback_days', 126)
            
            # Initialize tracking variables
            self.portfolio_returns = deque(maxlen=self.risk_config.get('vol_estimation_days', 252))
            self.symbol_returns = {}
            self.volatility_estimates = {}
            self.correlation_matrix = {}
            self.last_portfolio_value = None
            self.peak_portfolio_value = None
            self.current_drawdown = 0.0
            
            # Initialize risk management counters and flags
            self.volatility_scalings = 0
            self.position_limits_applied = 0
            self.emergency_stop_triggered = False
            self.risk_metrics_history = []
            
            self.algorithm.Log(f"LayerThreeRiskManager: Initialized with target vol {self.target_portfolio_vol:.1%}, "
                             f"max leverage {self.max_leverage_multiplier}x")
            
        except Exception as e:
            error_msg = f"CRITICAL ERROR initializing LayerThreeRiskManager: {str(e)}"
            self.algorithm.Error(error_msg)
            raise ValueError(error_msg)
    
    def apply_portfolio_risk_management(self, combined_targets):
        """Main entry point for Layer 3 risk management."""
        try:
            self.algorithm.Log(f"LAYER 3: Applying risk management to {len(combined_targets)} positions")
            
            if self._check_emergency_stops():
                return {}
            
            portfolio_metrics = self._calculate_portfolio_metrics(combined_targets)
            vol_adjusted = self._apply_volatility_targeting(combined_targets, portfolio_metrics)
            limited_targets = self._apply_position_limits(vol_adjusted)
            final_targets = self._apply_minimum_exposure(limited_targets)
            
            self._record_risk_metrics(final_targets, portfolio_metrics)
            self._log_risk_summary(combined_targets, final_targets, portfolio_metrics)
            
            return final_targets
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 3: Risk management error: {str(e)}")
            return combined_targets
    
    def update_daily_returns(self):
        """Update daily return data for all symbols."""
        try:
            current_time = self.algorithm.Time
            if hasattr(self, '_last_return_update_date') and self._last_return_update_date == current_time.date():
                return
            
            self._last_return_update_date = current_time.date()
            symbols_updated = 0
            
            for symbol in self.algorithm.Securities.Keys:
                try:
                    security = self.algorithm.Securities[symbol]
                    if security.Price <= 0:
                        continue
                    
                    # Use centralized data provider if available
                    if hasattr(self.algorithm, 'data_integrity_checker') and self.algorithm.data_integrity_checker:
                        history = self.algorithm.data_integrity_checker.get_history(symbol, 2, Resolution.Daily)
                    else:
                        # Fallback to direct API call (not recommended)
                        self.algorithm.Log(f"RiskManager: WARNING - No centralized cache for {symbol}, using direct History API")
                        history = self.algorithm.History([symbol], 2, Resolution.Daily)
                    
                    # Convert to list to check if empty (QC data doesn't support len() directly)
                    if history is not None:
                        history_list = list(history)
                        if len(history_list) >= 2:
                            # Extract prices from QC data
                            prices = [bar.Close if hasattr(bar, 'Close') else bar.close for bar in history_list]
                            if len(prices) >= 2 and prices[-2] > 0:
                                daily_return = (prices[-1] - prices[-2]) / prices[-2]
                                self._update_return_data(symbol, daily_return)
                                symbols_updated += 1
                except:
                    continue
            
            if symbols_updated > 0 and current_time.day == 1:
                quality = self._get_data_quality_status()
                self.algorithm.Log(f"LAYER 3: Data quality: {quality}")
                
        except Exception as e:
            self.algorithm.Error(f"LAYER 3: Daily return update error: {str(e)}")
    
    def _update_return_data(self, symbol, return_value):
        """Update return data for a symbol."""
        if symbol not in self.symbol_returns:
            self.symbol_returns[symbol] = deque(maxlen=self.correlation_lookback_days)
        
        self.symbol_returns[symbol].append(return_value)
        
        # Clear cached estimates periodically
        if len(self.symbol_returns[symbol]) % 30 == 0:
            if symbol in self.volatility_estimates:
                del self.volatility_estimates[symbol]
            keys_to_remove = [k for k in self.correlation_matrix.keys() if str(symbol) in k]
            for key in keys_to_remove:
                del self.correlation_matrix[key]
    
    def _get_data_quality_status(self):
        """Get current data quality status."""
        sufficient_count = sum(1 for returns in self.symbol_returns.values() if len(returns) >= 30)
        total_count = len(self.symbol_returns)
        
        if sufficient_count >= 3:
            return 'excellent'
        elif sufficient_count >= 2:
            return 'good'
        elif sufficient_count >= 1:
            return 'limited'
        else:
            return 'insufficient'
    
    def _calculate_portfolio_metrics(self, targets):
        """Calculate portfolio metrics including enhanced volatility."""
        try:
            portfolio_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
            gross_exposure = sum(abs(weight) for weight in targets.values())
            estimated_vol = self._estimate_portfolio_volatility(targets)
            
            return {
                'portfolio_value': portfolio_value,
                'gross_exposure': gross_exposure,
                'estimated_volatility': estimated_vol,
                'position_count': len([w for w in targets.values() if abs(w) > 0.001]),
                'max_position': max(abs(w) for w in targets.values()) if targets else 0,
                'net_exposure': sum(targets.values()),
                'data_quality': self._get_data_quality_status()
            }
        except Exception as e:
            self.algorithm.Error(f"LAYER 3: Portfolio metrics error: {str(e)}")
            return {'portfolio_value': 0, 'gross_exposure': 0, 'estimated_volatility': 0.15, 
                   'position_count': 0, 'max_position': 0, 'net_exposure': 0, 'data_quality': 'error'}
    
    def _estimate_portfolio_volatility(self, targets):
        """Estimate portfolio volatility using enhanced covariance when available."""
        try:
            if not targets:
                return 0.1
            
            symbols = list(targets.keys())
            weights = np.array(list(targets.values()))
            
            if len(symbols) == 1:
                asset_vol = self._get_asset_volatility(symbols[0])
                return asset_vol * abs(weights[0])
            
            if self.use_correlation:
                data_quality = self._get_data_quality_status()
                
                if data_quality in ['good', 'excellent']:
                    try:
                        portfolio_vol = self._calculate_enhanced_volatility(symbols, weights)
                        self.algorithm.Log(f"LAYER 3: Enhanced volatility ({data_quality}): {portfolio_vol:.1%}")
                        return portfolio_vol
                    except:
                        pass
                
                # Fallback to basic covariance
                portfolio_vol = self._calculate_basic_volatility(symbols, weights)
                self.algorithm.Log(f"LAYER 3: Basic volatility ({data_quality}): {portfolio_vol:.1%}")
                return portfolio_vol
            
            # Simple fallback
            return self._calculate_simple_volatility(symbols, weights)
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 3: Volatility estimation error: {str(e)}")
            return 0.15
    
    def _calculate_enhanced_volatility(self, symbols, weights):
        """Calculate volatility using real return data."""
        volatilities = np.array([self._get_real_volatility(symbol) for symbol in symbols])
        n = len(symbols)
        correlation_matrix = np.eye(n)
        
        for i, symbol_i in enumerate(symbols):
            for j, symbol_j in enumerate(symbols):
                if i != j:
                    correlation_matrix[i, j] = self._get_real_correlation(symbol_i, symbol_j)
        
        vol_matrix = np.diag(volatilities)
        covariance_matrix = vol_matrix @ correlation_matrix @ vol_matrix
        portfolio_variance = weights.T @ covariance_matrix @ weights
        return float(np.sqrt(max(0, portfolio_variance)))
    
    def _calculate_basic_volatility(self, symbols, weights):
        """Calculate volatility using default correlations."""
        volatilities = np.array([self._get_asset_volatility(symbol) for symbol in symbols])
        n = len(symbols)
        correlation_matrix = np.eye(n)
        
        for i, symbol_i in enumerate(symbols):
            for j, symbol_j in enumerate(symbols):
                if i != j:
                    correlation_matrix[i, j] = self._get_default_correlation(symbol_i, symbol_j)
        
        vol_matrix = np.diag(volatilities)
        covariance_matrix = vol_matrix @ correlation_matrix @ vol_matrix
        portfolio_variance = weights.T @ covariance_matrix @ weights
        return float(np.sqrt(max(0, portfolio_variance)))
    
    def _calculate_simple_volatility(self, symbols, weights):
        """Simple volatility calculation as ultimate fallback."""
        volatilities = np.array([self._get_asset_volatility(symbol) for symbol in symbols])
        avg_correlation = 0.3
        
        sum_of_variances = np.sum((weights * volatilities) ** 2)
        sum_of_covariances = 0
        
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                covariance = avg_correlation * volatilities[i] * volatilities[j] * weights[i] * weights[j]
                sum_of_covariances += 2 * covariance
        
        portfolio_variance = sum_of_variances + sum_of_covariances
        return float(np.sqrt(max(0, portfolio_variance)))
    
    def _get_real_volatility(self, symbol):
        """Get volatility from real return data."""
        if symbol in self.symbol_returns and len(self.symbol_returns[symbol]) >= 30:
            returns = np.array(list(self.symbol_returns[symbol]))
            vol = np.std(returns) * np.sqrt(252)
            return max(0.05, min(2.0, vol))
        return self._get_asset_volatility(symbol)
    
    def _get_real_correlation(self, symbol1, symbol2):
        """Get correlation from real return data."""
        corr_key = tuple(sorted([str(symbol1), str(symbol2)]))
        if corr_key in self.correlation_matrix:
            return self.correlation_matrix[corr_key]
        
        if (symbol1 in self.symbol_returns and symbol2 in self.symbol_returns and
            len(self.symbol_returns[symbol1]) >= 30 and len(self.symbol_returns[symbol2]) >= 30):
            
            returns1 = np.array(list(self.symbol_returns[symbol1]))
            returns2 = np.array(list(self.symbol_returns[symbol2]))
            min_length = min(len(returns1), len(returns2))
            returns1 = returns1[-min_length:]
            returns2 = returns2[-min_length:]
            
            correlation = np.corrcoef(returns1, returns2)[0, 1]
            if not np.isnan(correlation):
                correlation = max(-0.95, min(0.95, correlation))
                self.correlation_matrix[corr_key] = correlation
                return correlation
        
        return self._get_default_correlation(symbol1, symbol2)
    
    def _get_asset_volatility(self, symbol):
        """Get default volatility by asset type using centralized config."""
        if symbol in self.volatility_estimates:
            return self.volatility_estimates[symbol]
        
        # Use centralized ASSET_DEFAULTS from config
        full_config = self.config_manager.get_full_config()
        asset_defaults = full_config.get('asset_defaults', {})
        volatilities = asset_defaults.get('volatilities', {})
        
        symbol_str = str(symbol)
        if 'ES' in symbol_str or 'NQ' in symbol_str or 'YM' in symbol_str:
            return volatilities.get('equities', 0.20)
        elif 'ZN' in symbol_str or 'ZB' in symbol_str:
            return volatilities.get('bonds', 0.08)
        elif any(fx in symbol_str for fx in ['6E', '6J', '6B']):
            return volatilities.get('fx', 0.12)
        elif any(commodity in symbol_str for commodity in ['CL', 'GC']):
            return volatilities.get('commodities', 0.25)  # This drives the 14.9% calculation
        else:
            return volatilities.get('default', 0.20)
    
    def _get_default_correlation(self, symbol1, symbol2):
        """Get default correlation between asset types using centralized config."""
        # Use centralized ASSET_DEFAULTS from config
        full_config = self.config_manager.get_full_config()
        asset_defaults = full_config.get('asset_defaults', {})
        correlations = asset_defaults.get('correlations', {})
        
        s1, s2 = str(symbol1), str(symbol2)
        
        # Within asset class correlations
        if all(eq in s for eq in ['ES', 'NQ', 'YM'] for s in [s1, s2]):
            return correlations.get('within_equity', 0.85)
        elif all(bond in s for bond in ['ZN', 'ZB'] for s in [s1, s2]):
            return correlations.get('within_fixed_income', 0.70)
        elif all(commodity in s for commodity in ['CL', 'GC'] for s in [s1, s2]):
            return correlations.get('within_commodity', 0.30)
        
        # Cross-asset class correlations
        elif any(eq in s1 for eq in ['ES', 'NQ', 'YM']) and any(bond in s2 for bond in ['ZN', 'ZB']):
            return correlations.get('equity_bond', -0.10)
        elif any(bond in s1 for bond in ['ZN', 'ZB']) and any(eq in s2 for eq in ['ES', 'NQ', 'YM']):
            return correlations.get('equity_bond', -0.10)
        elif any(eq in s1 for eq in ['ES', 'NQ', 'YM']) and any(commodity in s2 for commodity in ['CL', 'GC']):
            return correlations.get('equity_commodity', 0.20)
        elif any(commodity in s1 for commodity in ['CL', 'GC']) and any(eq in s2 for eq in ['ES', 'NQ', 'YM']):
            return correlations.get('equity_commodity', 0.20)
        elif any(bond in s1 for bond in ['ZN', 'ZB']) and any(commodity in s2 for commodity in ['CL', 'GC']):
            return correlations.get('bond_commodity', -0.05)
        elif any(commodity in s1 for commodity in ['CL', 'GC']) and any(bond in s2 for bond in ['ZN', 'ZB']):
            return correlations.get('bond_commodity', -0.05)
        
        # Default cross-asset correlation
        else:
            return correlations.get('cross_asset_default', 0.15)
    
    def _check_emergency_stops(self):
        """Check for emergency stop conditions."""
        try:
            current_portfolio_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
            
            # Initialize peak portfolio value if not set
            if self.peak_portfolio_value is None:
                self.peak_portfolio_value = current_portfolio_value
            elif current_portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = current_portfolio_value
            
            if self.peak_portfolio_value > 0:
                self.current_drawdown = (self.peak_portfolio_value - current_portfolio_value) / self.peak_portfolio_value
            
            if self.last_portfolio_value is not None and self.last_portfolio_value > 0:
                daily_return = (current_portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
                if daily_return < -self.daily_stop_loss:
                    self.algorithm.Log(f"LAYER 3: DAILY STOP LOSS TRIGGERED: {daily_return:.2%}")
                    # Defensive check for attribute existence (QuantConnect caching issue)
                    if not hasattr(self, 'emergency_stop_triggered'):
                        self.emergency_stop_triggered = False
                    self.emergency_stop_triggered = True
                    return True
            
            if self.current_drawdown > self.max_drawdown_stop:
                self.algorithm.Log(f"LAYER 3: MAX DRAWDOWN STOP TRIGGERED: {self.current_drawdown:.2%}")
                # Defensive check for attribute existence (QuantConnect caching issue)
                if not hasattr(self, 'emergency_stop_triggered'):
                    self.emergency_stop_triggered = False
                self.emergency_stop_triggered = True
                return True
            
            self.last_portfolio_value = current_portfolio_value
            return False
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 3: Emergency stop error: {str(e)}")
            return False
    
    def _apply_volatility_targeting(self, targets, portfolio_metrics):
        """Apply volatility targeting to scale positions."""
        try:
            estimated_vol = portfolio_metrics['estimated_volatility']
            if estimated_vol <= 0:
                return targets
            
            vol_scaling_factor = self.target_portfolio_vol / estimated_vol
            vol_scaling_factor = min(vol_scaling_factor, self.max_leverage_multiplier)
            
            scaled_targets = {symbol: weight * vol_scaling_factor for symbol, weight in targets.items()}
            
            if abs(vol_scaling_factor - 1.0) > 0.1:
                # Defensive check for attribute existence (QuantConnect caching issue)
                if not hasattr(self, 'volatility_scalings'):
                    self.volatility_scalings = 0
                self.volatility_scalings += 1
                data_quality = portfolio_metrics.get('data_quality', 'unknown')
                self.algorithm.Log(f"LAYER 3: Vol targeting ({data_quality}): {estimated_vol:.1%} -> {self.target_portfolio_vol:.1%} "
                                 f"(scaling: {vol_scaling_factor:.2f}x)")
            
            return scaled_targets
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 3: Volatility targeting error: {str(e)}")
            return targets
    
    def _apply_position_limits(self, targets):
        """Apply position size limits."""
        try:
            limited_targets = {}
            limits_applied = False
            
            for symbol, weight in targets.items():
                if abs(weight) > self.max_single_position:
                    original_weight = weight
                    weight = self.max_single_position if weight > 0 else -self.max_single_position
                    
                    if not limits_applied:
                        # Defensive check for attribute existence (QuantConnect caching issue)
                        if not hasattr(self, 'position_limits_applied'):
                            self.position_limits_applied = 0
                        self.position_limits_applied += 1
                        limits_applied = True
                    
                    self.algorithm.Log(f"LAYER 3: Position limit applied to {symbol}: "
                                     f"{original_weight:.1%} -> {weight:.1%}")
                
                limited_targets[symbol] = weight
            
            return limited_targets
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 3: Position limits error: {str(e)}")
            return targets
    
    def _apply_minimum_exposure(self, targets):
        """Apply minimum exposure requirements."""
        try:
            gross_exposure = sum(abs(weight) for weight in targets.values())
            
            if gross_exposure < self.min_notional_exposure:
                scaling_factor = self.min_notional_exposure / gross_exposure if gross_exposure > 0 else 1.0
                scaled_targets = {symbol: weight * scaling_factor for symbol, weight in targets.items()}
                self.algorithm.Log(f"LAYER 3: Scaled up to meet minimum exposure {self.min_notional_exposure:.1f}x")
                return scaled_targets
            
            return targets
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 3: Minimum exposure error: {str(e)}")
            return targets
    
    def _record_risk_metrics(self, final_targets, portfolio_metrics):
        """Record risk metrics for monitoring."""
        try:
            risk_metrics = {
                'timestamp': self.algorithm.Time,
                'portfolio_value': portfolio_metrics['portfolio_value'],
                'gross_exposure': sum(abs(w) for w in final_targets.values()),
                'net_exposure': sum(final_targets.values()),
                'estimated_volatility': portfolio_metrics['estimated_volatility'],
                'target_volatility': self.target_portfolio_vol,
                'data_quality': portfolio_metrics.get('data_quality', 'unknown'),
                'current_drawdown': self.current_drawdown
            }
            # Defensive check for attribute existence (QuantConnect caching issue)
            if not hasattr(self, 'risk_metrics_history'):
                self.risk_metrics_history = []
            self.risk_metrics_history.append(risk_metrics)
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 3: Risk metrics recording error: {str(e)}")
    
    def _log_risk_summary(self, original_targets, final_targets, portfolio_metrics):
        """Log risk management summary."""
        try:
            original_gross = sum(abs(w) for w in original_targets.values())
            final_gross = sum(abs(w) for w in final_targets.values())
            scaling_applied = final_gross / original_gross if original_gross > 0 else 1.0
            
            self.algorithm.Log("LAYER 3 ENHANCED RISK SUMMARY:")
            self.algorithm.Log(f"  Input gross exposure: {original_gross:.1%}")
            self.algorithm.Log(f"  Final gross exposure: {final_gross:.1%}")
            self.algorithm.Log(f"  Risk scaling applied: {scaling_applied:.2f}x")
            self.algorithm.Log(f"  Estimated portfolio vol: {portfolio_metrics['estimated_volatility']:.1%}")
            self.algorithm.Log(f"  Target portfolio vol: {self.target_portfolio_vol:.1%}")
            self.algorithm.Log(f"  Data quality: {portfolio_metrics.get('data_quality', 'unknown')}")
            self.algorithm.Log(f"  Current drawdown: {self.current_drawdown:.1%}")
            
        except Exception as e:
            self.algorithm.Error(f"LAYER 3: Risk summary error: {str(e)}")
    
    def get_risk_status(self):
        """Get current risk status."""
        # Defensive checks for attributes (QuantConnect caching issue)
        if not hasattr(self, 'emergency_stop_triggered'):
            self.emergency_stop_triggered = False
        if not hasattr(self, 'volatility_scalings'):
            self.volatility_scalings = 0
        if not hasattr(self, 'position_limits_applied'):
            self.position_limits_applied = 0
            
        return {
            'target_volatility': self.target_portfolio_vol,
            'current_drawdown': self.current_drawdown,
            'emergency_stop_active': self.emergency_stop_triggered,
            'volatility_scalings': self.volatility_scalings,
            'position_limits_applied': self.position_limits_applied,
            'data_quality': self._get_data_quality_status(),
            'symbols_tracked': len(self.symbol_returns)
        }
