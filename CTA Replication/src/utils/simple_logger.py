"""
Simple Logger for QuantConnect CTA Framework
Fallback logger that works without complex imports or dependencies.
"""

class SimpleLogger:
    """
    Simple logging utility with basic component-level control.
    No external dependencies, works reliably in QuantConnect environment.
    """
    
    def __init__(self, algorithm, component_name: str):
        """
        Initialize simple logger for a specific component.
        
        Args:
            algorithm: QC Algorithm instance
            component_name: Name of the component
        """
        self.algorithm = algorithm
        self.component_name = component_name
        self.daily_log_count = 0
        self.max_daily_logs = 1000
        self.last_reset_date = None
        
        # Simple component-level configuration
        self.component_levels = {
            'kestner_cta': 'DEBUG',
            'execution_manager': 'DEBUG', 
            'system_reporter': 'INFO',
            'warmup': 'ERROR',
            'data_validator': 'WARNING',
            'orchestrator': 'INFO',
            'risk_manager': 'INFO',
        }
        
        self.level_values = {
            'DEBUG': 10,
            'INFO': 20,
            'WARNING': 30,
            'ERROR': 40,
            'CRITICAL': 50
        }
    
    def _should_log(self, level: str) -> bool:
        """Check if message should be logged."""
        # Check daily log limit
        if hasattr(self.algorithm, 'Time'):
            current_date = self.algorithm.Time.date()
            if self.last_reset_date != current_date:
                self.daily_log_count = 0
                self.last_reset_date = current_date
        
        if self.daily_log_count >= self.max_daily_logs:
            return False
        
        # Check log level
        component_level = self.component_levels.get(self.component_name, 'INFO')
        return self.level_values.get(level, 20) >= self.level_values.get(component_level, 20)
    
    def _format_message(self, level: str, message: str) -> str:
        """Format log message."""
        formatted_message = f"{level}: [{self.component_name}] {message}"
        
        # Truncate if too long
        if len(formatted_message) > 200:
            formatted_message = formatted_message[:197] + "..."
        
        return formatted_message
    
    def _log(self, level: str, message: str):
        """Internal logging method."""
        if not self._should_log(level):
            return
        
        formatted_message = self._format_message(level, message)
        
        # Use appropriate QC logging method
        if level == 'DEBUG':
            self.algorithm.Debug(formatted_message)
        elif level == 'ERROR':
            self.algorithm.Error(formatted_message)
        else:
            self.algorithm.Log(formatted_message)
        
        self.daily_log_count += 1
    
    # Public logging methods
    def debug(self, message: str):
        """Log debug message."""
        self._log('DEBUG', message)
    
    def info(self, message: str):
        """Log info message."""
        self._log('INFO', message)
    
    def warning(self, message: str):
        """Log warning message."""
        self._log('WARNING', message)
    
    def error(self, message: str):
        """Log error message."""
        self._log('ERROR', message)
    
    def critical(self, message: str):
        """Log critical message."""
        self._log('CRITICAL', message)
    
    # Special methods for specific use cases
    def log_trade_execution(self, symbol, action, quantity, price=None):
        """Log trade execution."""
        price_str = f" @ ${price:.2f}" if price else ""
        self.info(f"TRADE: {action} {quantity} {symbol}{price_str}")
    
    def log_equity_mismatch(self, portfolio_value: float, calculated_return: float, qc_return: float):
        """Log equity/return mismatch."""
        mismatch = abs(calculated_return - qc_return)
        self.critical(f"EQUITY_MISMATCH: Portfolio=${portfolio_value:,.0f}, Our={calculated_return:.2%}, QC={qc_return:.2%}, Diff={mismatch:.2%}")
    
    def log_large_move(self, previous_value: float, current_value: float, daily_return: float):
        """Log large portfolio moves."""
        self.critical(f"LARGE_MOVE: ${previous_value:,.0f} -> ${current_value:,.0f} ({daily_return:.2%})")
    
    def is_debug_enabled(self) -> bool:
        """Check if debug logging is enabled."""
        component_level = self.component_levels.get(self.component_name, 'INFO')
        return component_level == 'DEBUG' 