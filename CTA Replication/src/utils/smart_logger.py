"""
Smart Logging Utility for QuantConnect CTA Framework

Provides component-level logging control with configurable verbosity levels.
Designed to work within QC's 100KB log limits while maintaining debugging capability.
"""

from typing import Optional, Dict, Any
from datetime import datetime

class SmartLogger:
    """
    Smart logging utility with component-level control and condensed modes.
    
    Usage:
        logger = SmartLogger(algorithm, 'kestner_cta')
        logger.debug("Detailed debug information")
        logger.info("General information")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical issue")
    """
    
    # Log level hierarchy (lower number = more verbose)
    LOG_LEVELS = {
        'DEBUG': 10,
        'INFO': 20,
        'WARNING': 30,
        'ERROR': 40,
        'CRITICAL': 50
    }
    
    def __init__(self, algorithm, component_name: str, config_manager=None):
        """
        Initialize smart logger for a specific component.
        
        Args:
            algorithm: QC Algorithm instance
            component_name: Name of the component (matches config keys)
            config_manager: Optional config manager for dynamic settings
        """
        self.algorithm = algorithm
        self.component_name = component_name
        self.config_manager = config_manager
        
        # Load logging configuration
        self._load_logging_config()
        
        # Initialize log counters for rate limiting
        self._daily_log_count = 0
        self._last_reset_date = algorithm.Time.date() if hasattr(algorithm, 'Time') else None
        
    def _load_logging_config(self):
        """Load logging configuration from config manager or defaults."""
        try:
            if self.config_manager:
                # Try to get from config manager
                algo_config = self.config_manager.get_algorithm_config()
                self.logging_config = algo_config.get('logging', {})
            else:
                # Try to import config directly
                try:
                    from config.config_market_strategy import LOGGING_CONFIG
                    self.logging_config = LOGGING_CONFIG
                except ImportError:
                    # Try alternative import path
                    try:
                        from src.config.config_market_strategy import LOGGING_CONFIG
                        self.logging_config = LOGGING_CONFIG
                    except ImportError:
                        # Use fallback config
                        raise ImportError("Cannot import logging config")
                
        except (ImportError, AttributeError):
            # Fallback to minimal config
            self.logging_config = {
                'component_levels': {self.component_name: 'INFO'},
                'global_log_level': 'INFO',
                'condensed_mode': True,
                'max_daily_logs': 1000,
                'debug_modes': {
                    'equity_mismatch_detection': True,
                    'large_move_detection': True,
                    'trade_execution_details': True,
                    'performance_tracking': True,
                },
                'condensed_settings': {'suppress_routine_validation': True},
                'formatting': {'include_component': True, 'use_prefixes': True}
            }
    
    def _should_log(self, level: str) -> bool:
        """Check if message should be logged based on component level and daily limits."""
        # Check daily log limit
        if hasattr(self.algorithm, 'Time'):
            current_date = self.algorithm.Time.date()
            if self._last_reset_date != current_date:
                self._daily_log_count = 0
                self._last_reset_date = current_date
        
        max_logs = self.logging_config.get('max_daily_logs', 1000)
        if self._daily_log_count >= max_logs:
            return False
        
        # Check log level
        component_level = self.logging_config.get('component_levels', {}).get(
            self.component_name, 
            self.logging_config.get('global_log_level', 'INFO')
        )
        
        return self.LOG_LEVELS.get(level, 20) >= self.LOG_LEVELS.get(component_level, 20)
    
    def _format_message(self, level: str, message: str) -> str:
        """Format log message according to configuration."""
        formatting = self.logging_config.get('formatting', {})
        
        # Build prefix
        prefix_parts = []
        if formatting.get('use_prefixes', True):
            prefix_parts.append(f"{level}:")
        if formatting.get('include_component', True):
            prefix_parts.append(f"[{self.component_name}]")
        
        prefix = " ".join(prefix_parts)
        formatted_message = f"{prefix} {message}" if prefix else message
        
        # Truncate if needed
        max_length = formatting.get('max_message_length', 200)
        if len(formatted_message) > max_length:
            formatted_message = formatted_message[:max_length-3] + "..."
        
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
        
        self._daily_log_count += 1
    
    # Public logging methods
    def debug(self, message: str):
        """Log debug message (most verbose)."""
        self._log('DEBUG', message)
    
    def info(self, message: str):
        """Log info message (general information)."""
        self._log('INFO', message)
    
    def warning(self, message: str):
        """Log warning message."""
        self._log('WARNING', message)
    
    def error(self, message: str):
        """Log error message."""
        self._log('ERROR', message)
    
    def critical(self, message: str):
        """Log critical message (highest priority)."""
        self._log('CRITICAL', message)
    
    # Special debug mode methods
    def debug_if_enabled(self, mode_name: str, message: str):
        """Log debug message only if specific debug mode is enabled."""
        debug_modes = self.logging_config.get('debug_modes', {})
        if debug_modes.get(mode_name, False):
            self.debug(message)
    
    def log_trade_execution(self, symbol, action, quantity, price=None):
        """Log trade execution with special formatting."""
        if self.logging_config.get('debug_modes', {}).get('trade_execution_details', False):
            price_str = f" @ ${price:.2f}" if price else ""
            self.info(f"TRADE: {action} {quantity} {symbol}{price_str}")
    
    def log_performance_metric(self, metric_name: str, value: Any):
        """Log performance metrics with special formatting."""
        if self.logging_config.get('debug_modes', {}).get('performance_tracking', False):
            self.info(f"PERF: {metric_name} = {value}")
    
    def log_equity_mismatch(self, portfolio_value: float, calculated_return: float, qc_return: float):
        """Log equity/return mismatch with special formatting."""
        if self.logging_config.get('debug_modes', {}).get('equity_mismatch_detection', False):
            mismatch = abs(calculated_return - qc_return)
            self.critical(f"EQUITY_MISMATCH: Portfolio=${portfolio_value:,.0f}, Our={calculated_return:.2%}, QC={qc_return:.2%}, Diff={mismatch:.2%}")
    
    def log_large_move(self, previous_value: float, current_value: float, daily_return: float):
        """Log large portfolio moves with special formatting."""
        if self.logging_config.get('debug_modes', {}).get('large_move_detection', False):
            self.critical(f"LARGE_MOVE: ${previous_value:,.0f} -> ${current_value:,.0f} ({daily_return:.2%})")
    
    # Condensed logging methods
    def log_condensed_daily_summary(self, summary_data: Dict[str, Any]):
        """Log condensed daily summary."""
        if self.logging_config.get('condensed_mode', True):
            portfolio_value = summary_data.get('portfolio_value', 0)
            daily_return = summary_data.get('daily_return', 0)
            active_positions = summary_data.get('active_positions', 0)
            
            self.info(f"DAILY: ${portfolio_value:,.0f} ({daily_return:+.2%}) | {active_positions} positions")
    
    def log_condensed_trade_summary(self, trades_executed: int, total_value: float):
        """Log condensed trade summary."""
        if trades_executed > 0:
            self.info(f"TRADES: {trades_executed} executed, ${total_value:,.0f} total value")
    
    # Utility methods
    def is_debug_enabled(self) -> bool:
        """Check if debug logging is enabled for this component."""
        component_level = self.logging_config.get('component_levels', {}).get(
            self.component_name, 
            self.logging_config.get('global_log_level', 'INFO')
        )
        return component_level == 'DEBUG'
    
    def get_daily_log_count(self) -> int:
        """Get current daily log count."""
        return self._daily_log_count
    
    def get_remaining_logs(self) -> int:
        """Get remaining logs for today."""
        max_logs = self.logging_config.get('max_daily_logs', 1000)
        return max(0, max_logs - self._daily_log_count) 