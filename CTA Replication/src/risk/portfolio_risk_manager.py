# ========================================
# LAYER 3: portfolio_risk_manager.py
# ========================================

from AlgorithmImports import *
import numpy as np
from collections import deque

class PortfolioRiskManager:
    """
    Layer 3: Portfolio-Level Risk Management
    
    Handles:
    - Portfolio volatility targeting
    - Minimum notional exposure
    - Portfolio-level stops
    - Overall leverage management
    
    UPDATED: Now uses QC's built-in performance tracking when configured
    """
    
    def __init__(self, algorithm, config=None):
        self.algorithm = algorithm
        
        # Default configuration
        self.config = {
            'target_portfolio_vol': 0.20,        # 20% target volatility
            'min_notional_exposure': 3,       # Minimum 60% deployed
            'max_leverage_multiplier': 10.0,      # Maximum 10x leverage
            'daily_stop_loss': 0.20,             # 2% daily portfolio stop
            'max_drawdown_stop': 0.75,           # 15% max drawdown stop
            'vol_estimation_days': 63,           # 3 months for vol estimation
            
            # NEW: QC Integration Settings
            'use_qc_performance': True,          # Use QC's built-in performance tracking
            'use_qc_portfolio': True,            # Use QC's Portfolio object
            'log_performance': True,             # Log performance metrics
        }
        
        if config:
            self.config.update(config)
        
        # Legacy tracking (kept as fallback or when QC features disabled)
        self.portfolio_returns = deque(maxlen=self.config['vol_estimation_days'])
        self.portfolio_peak = 0.0
        self.stop_triggered = False
        self.last_portfolio_value = 0.0
        
        # QC Performance tracking
        self.use_qc_features = self.config.get('use_qc_performance', True)
        self.initial_portfolio_value = None
        
        self.algorithm.Log(f"Layer3 RiskManager: Initialized with QC features {'ENABLED' if self.use_qc_features else 'DISABLED'}")
        self.algorithm.Log(f"Layer3 RiskManager: Config: {self.config}")
    
    def update_portfolio_performance(self, current_portfolio_value=None):
        """
        Update portfolio performance tracking using QC's built-in features when available
        
        Args:
            current_portfolio_value (float): Optional override, otherwise uses QC's Portfolio.TotalPortfolioValue
        """
        # Use QC's portfolio value if not provided
        if current_portfolio_value is None:
            current_portfolio_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
        
        # Initialize tracking on first call
        if self.initial_portfolio_value is None:
            self.initial_portfolio_value = current_portfolio_value
            self.portfolio_peak = current_portfolio_value
            self.last_portfolio_value = current_portfolio_value
            return
        
        # Update performance tracking
        if self.use_qc_features:
            # Use QC's built-in statistics when available
            self._update_qc_performance_tracking(current_portfolio_value)
        else:
            # Fall back to our custom tracking
            self._update_custom_performance_tracking(current_portfolio_value)
        
        # Always update peak for drawdown calculation
        if current_portfolio_value > self.portfolio_peak:
            self.portfolio_peak = current_portfolio_value
        
        # Log performance if configured
        if self.config.get('log_performance', False):
            self._log_performance_summary(current_portfolio_value)
    
    def _update_qc_performance_tracking(self, current_portfolio_value):
        """Update performance using QC's built-in features"""
        # Calculate return for our vol estimation (QC doesn't expose this directly)
        if self.last_portfolio_value > 0:
            portfolio_return = (current_portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
            self.portfolio_returns.append(portfolio_return)
        
        self.last_portfolio_value = current_portfolio_value
        
        # Note: QC automatically tracks performance statistics in self.algorithm.Portfolio
        # We can access: TotalPortfolioValue, TotalHoldingsValue, Cash, etc.
    
    def _update_custom_performance_tracking(self, current_portfolio_value):
        """Fallback to custom performance tracking"""
        if self.last_portfolio_value > 0:
            portfolio_return = (current_portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
            self.portfolio_returns.append(portfolio_return)
        
        self.last_portfolio_value = current_portfolio_value
    
    def _log_performance_summary(self, current_portfolio_value):
        """Log performance summary using QC or custom data"""
        if self.initial_portfolio_value and self.initial_portfolio_value > 0:
            total_return = (current_portfolio_value - self.initial_portfolio_value) / self.initial_portfolio_value
            drawdown = (self.portfolio_peak - current_portfolio_value) / self.portfolio_peak if self.portfolio_peak > 0 else 0
            
            if self.use_qc_features:
                # Access QC's portfolio statistics
                cash = float(self.algorithm.Portfolio.Cash)
                margin_used = float(self.algorithm.Portfolio.TotalMarginUsed)
                holdings_value = float(self.algorithm.Portfolio.TotalHoldingsValue)
                
                if len(self.portfolio_returns) >= 10:  # Need some data for vol calc
                    realized_vol = self.calculate_portfolio_volatility(None)
                    self.algorithm.Log(f"Layer3 Performance: Total Return: {total_return:.2%}, "
                                     f"Drawdown: {drawdown:.2%}, Vol: {realized_vol:.1%}, "
                                     f"Cash: ${cash:,.0f}, Margin: ${margin_used:,.0f}")
            else:
                # Use custom tracking
                self.algorithm.Log(f"Layer3 Performance: Total Return: {total_return:.2%}, "
                                 f"Drawdown: {drawdown:.2%}")
    
    def calculate_portfolio_volatility(self, combined_targets):
        """
        Calculate expected portfolio volatility
        
        Args:
            combined_targets (dict): {symbol: weight} from Layer 2 (can be None)
            
        Returns:
            float: Expected portfolio volatility
        """
        if len(self.portfolio_returns) < 20:
            return 0.15  # Default assumption
        
        # Use realized portfolio volatility
        returns_array = np.array(list(self.portfolio_returns))
        portfolio_vol = np.std(returns_array, ddof=1) * np.sqrt(252)
        
        return max(portfolio_vol, 0.05)  # Minimum 5% vol assumption
    
    def get_qc_portfolio_summary(self):
        """Get portfolio summary using QC's built-in features"""
        if not self.use_qc_features:
            return None
        
        return {
            'total_portfolio_value': float(self.algorithm.Portfolio.TotalPortfolioValue),
            'cash': float(self.algorithm.Portfolio.Cash),
            'total_holdings_value': float(self.algorithm.Portfolio.TotalHoldingsValue),
            'total_margin_used': float(self.algorithm.Portfolio.TotalMarginUsed),
            'total_fees': float(self.algorithm.Portfolio.TotalFees),
            'total_profit': float(self.algorithm.Portfolio.TotalUnrealizedProfit),
            'invested': self.algorithm.Portfolio.Invested,
            'positions_count': len([h for h in self.algorithm.Portfolio.Values if h.Invested])
        }
    
    def apply_portfolio_risk_management(self, combined_targets):
        """
        Apply portfolio-level risk management
        
        Args:
            combined_targets (dict): Combined strategy targets from Layer 2
            
        Returns:
            dict: Risk-adjusted final targets
        """
        if not combined_targets:
            return combined_targets
        
        # Step 1: Check portfolio-level stops
        if self.check_portfolio_stops():
            self.algorithm.Log("Layer3: Portfolio stop triggered - reducing exposure")
            return {symbol: weight * 0.5 for symbol, weight in combined_targets.items()}
        
        # Step 2: Calculate current portfolio characteristics
        current_gross_exposure = sum(abs(weight) for weight in combined_targets.values())
        
        # CRITICAL FIX: Handle zero gross exposure case
        if current_gross_exposure <= 0.001:  # Essentially zero
            self.algorithm.Log("Layer3: Combined targets have zero gross exposure - returning empty targets")
            return {}
        
        portfolio_vol = self.calculate_portfolio_volatility(combined_targets)
        
        # Step 3: Apply minimum notional exposure constraint
        if current_gross_exposure < self.config['min_notional_exposure']:
            boost_factor = self.config['min_notional_exposure'] / current_gross_exposure  # Now safe from division by zero
            combined_targets = {symbol: weight * boost_factor for symbol, weight in combined_targets.items()}
            
            self.algorithm.Log(f"Layer3: Boosted exposure from {current_gross_exposure:.1%} "
                             f"to {self.config['min_notional_exposure']:.1%}")
            
            # Recalculate after boost
            current_gross_exposure = sum(abs(weight) for weight in combined_targets.values())
            portfolio_vol = self.calculate_portfolio_volatility(combined_targets) * boost_factor
        
        # Step 4: Apply volatility targeting
        if portfolio_vol > 0:
            vol_scalar = self.config['target_portfolio_vol'] / portfolio_vol
            
            # Apply leverage limits
            vol_scalar = min(vol_scalar, self.config['max_leverage_multiplier'])
            
            final_targets = {symbol: weight * vol_scalar for symbol, weight in combined_targets.items()}
            
            final_gross_exposure = sum(abs(weight) for weight in final_targets.values())
            
            self.algorithm.Log(f"Layer3: Vol targeting - Portfolio vol {portfolio_vol:.1%} → "
                             f"Target {self.config['target_portfolio_vol']:.1%}, "
                             f"Scalar: {vol_scalar:.2f}, Final exposure: {final_gross_exposure:.1%}")
        else:
            final_targets = combined_targets
        
        return final_targets
    
    def check_portfolio_stops(self):
        """
        Check if portfolio-level stops should be triggered
        
        Returns:
            bool: True if stops should be triggered
        """
        current_portfolio_value = self.algorithm.Portfolio.TotalPortfolioValue
        
        # Daily stop loss
        if len(self.portfolio_returns) > 0:
            daily_return = self.portfolio_returns[-1]
            if daily_return < -self.config['daily_stop_loss']:
                self.algorithm.Log(f"Layer3: Daily stop triggered - {daily_return:.2%} loss")
                return True
        
        # Maximum drawdown stop
        if self.portfolio_peak > 0:
            current_drawdown = (self.portfolio_peak - current_portfolio_value) / self.portfolio_peak
            if current_drawdown > self.config['max_drawdown_stop']:
                self.algorithm.Log(f"Layer3: Max drawdown stop triggered - {current_drawdown:.2%} drawdown")
                return True
        
        return False
    
    def get_risk_metrics(self):
        """Get current risk metrics for monitoring"""
        current_portfolio_value = self.algorithm.Portfolio.TotalPortfolioValue
        
        metrics = {
            'portfolio_value': current_portfolio_value,
            'portfolio_peak': self.portfolio_peak,
            'target_vol': self.config['target_portfolio_vol'],
            'min_exposure': self.config['min_notional_exposure']
        }
        
        if self.portfolio_peak > 0:
            metrics['current_drawdown'] = (self.portfolio_peak - current_portfolio_value) / self.portfolio_peak
        
        if len(self.portfolio_returns) > 20:
            returns_array = np.array(list(self.portfolio_returns))
            metrics['realized_vol'] = np.std(returns_array, ddof=1) * np.sqrt(252)
        
        return metrics
