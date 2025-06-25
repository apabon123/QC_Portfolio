# warmup_manager.py - Warmup Management Utilities
"""
Warmup management utilities extracted from main.py to stay under 64KB limit.
Handles QC's native warm-up system setup and validation.
"""

from AlgorithmImports import *

class WarmupManager:
    """Manages algorithm warmup process using QC's native warmup system."""
    
    def __init__(self, algorithm, config_manager):
        self.algorithm = algorithm
        self.config_manager = config_manager
        
    def setup_enhanced_warmup(self):
        """
        Setup QC's native warm-up system based on strategy requirements and QC primer.
        Implements proper warm-up period calculation and configuration.
        """
        try:
            warmup_config = self.config_manager.get_warmup_config()
            
            if not warmup_config.get('enabled', False):
                self.algorithm.Log("WARMUP: Disabled in configuration")
                return
            
            # Calculate required warm-up period based on enabled strategies
            warmup_days = self.config_manager.calculate_max_warmup_needed()
            
            if warmup_days <= 0:
                self.algorithm.Log("WARMUP: No warm-up required")
                return
            
            # Use QC's native warm-up system
            warmup_method = warmup_config.get('method', 'time_based')
            warmup_resolution = getattr(Resolution, warmup_config.get('resolution', 'Daily'))
            
            if warmup_method == 'time_based':
                # Time-based warm-up (recommended for CTA strategies)
                warmup_period = timedelta(days=warmup_days)
                self.algorithm.SetWarmUp(warmup_period, warmup_resolution)
                self.algorithm.Log(f"WARMUP: Set time-based warm-up for {warmup_days} days at {warmup_config.get('resolution', 'Daily')} resolution")
            else:
                # Bar-count based warm-up
                self.algorithm.SetWarmUp(warmup_days, warmup_resolution)
                self.algorithm.Log(f"WARMUP: Set bar-count warm-up for {warmup_days} bars at {warmup_config.get('resolution', 'Daily')} resolution")
            
            # Enable automatic indicator warm-up
            self.algorithm.Settings.AutomaticIndicatorWarmUp = True
            self.algorithm.Log("WARMUP: Enabled automatic indicator warm-up")
            
            # Store warm-up info for progress tracking
            self.algorithm._warmup_config = warmup_config
            self.algorithm._warmup_start_time = None
            self.algorithm._warmup_total_days = warmup_days
            
            # Only log detailed warmup info if warmup logging is enabled
            log_warmup_details = self._should_log_component('warmup', 'INFO')
            
            if log_warmup_details:
                self.algorithm.Log("=" * 60)
                self.algorithm.Log("ENHANCED WARM-UP SYSTEM CONFIGURED")
                self.algorithm.Log(f"Method: {warmup_method}")
                self.algorithm.Log(f"Period: {warmup_days} days")
                self.algorithm.Log(f"Resolution: {warmup_config.get('resolution', 'Daily')}")
                self.algorithm.Log(f"Automatic Indicators: Enabled")
                
                # Log strategy-specific requirements
                progress_info = self.config_manager.get_warmup_progress_info()
                enabled_strategies = progress_info.get('enabled_strategies', [])
                self.algorithm.Log(f"Enabled Strategies: {enabled_strategies}")
                
                for strategy_name in enabled_strategies:
                    strategy_config = self.config_manager.get_strategy_config(strategy_name)
                    strategy_warmup = strategy_config.get('warmup_config', {})
                    required_days = strategy_warmup.get('required_days', 0)
                    if required_days > 0:
                        self.algorithm.Log(f"  - {strategy_name}: {required_days} days required")
                
                self.algorithm.Log("=" * 60)
            else:
                # Always log summary even if details are suppressed
                self.algorithm.Log(f"WARMUP: {warmup_days} days configured for {len(self.config_manager.get_enabled_strategies())} strategies")
            
        except Exception as e:
            self.algorithm.Error(f"Failed to setup enhanced warm-up: {str(e)}")
            # Continue without warm-up rather than failing completely
            self.algorithm.Log("WARNING: Continuing without warm-up due to setup error")

    def on_warmup_finished(self):
        """
        QC's native warm-up completion callback.
        Validates that all indicators and strategies are ready for trading.
        """
        try:
            # Check if we should log detailed warmup completion info
            log_warmup_details = self._should_log_component('warmup', 'INFO')
            
            if log_warmup_details:
                self.algorithm.Log("=" * 80)
                self.algorithm.Log("WARM-UP PERIOD COMPLETED - VALIDATING SYSTEM READINESS")
                self.algorithm.Log("=" * 80)
            else:
                # Always log the completion, just less verbose
                self.algorithm.Log("WARM-UP COMPLETED - System ready for trading")
            
            # Mark warm-up as completed
            self.algorithm._warmup_completed = True
            
            # Get warm-up configuration
            warmup_config = self.config_manager.get_warmup_config()
            
            # Validate indicators are ready (if enabled)
            if warmup_config.get('validate_indicators_ready', True):
                self.validate_indicators_ready()
            
            # Validate strategies are ready
            self.validate_strategies_ready()
            
            # Validate universe is ready
            self.validate_universe_ready()
            
            # Log system status
            if log_warmup_details:
                self.log_warmup_completion_status()
            
            # Optional: Trigger immediate test rebalance to verify system
            if warmup_config.get('test_rebalance_on_completion', False):
                self.algorithm.Log("TESTING: Triggering immediate rebalance to test system...")
                try:
                    self.algorithm.WeeklyRebalance()
                except Exception as test_e:
                    self.algorithm.Error(f"WARMUP TEST FAILED: {str(test_e)}")
            
            if log_warmup_details:
                self.algorithm.Log("=" * 80)
                self.algorithm.Log("SYSTEM IS READY FOR LIVE TRADING")
                self.algorithm.Log("=" * 80)
            
        except Exception as e:
            self.algorithm.Error(f"Error in OnWarmupFinished: {str(e)}")
            # Don't raise - allow trading to continue even if validation has issues

    def validate_indicators_ready(self):
        """Validate that all required indicators are ready after warm-up."""
        try:
            enabled_strategies = self.config_manager.get_enabled_strategies()
            
            for strategy_name in enabled_strategies:
                indicator_ready = self.config_manager.validate_warmup_indicators(strategy_name)
                if indicator_ready:
                    self.algorithm.Log(f"WARMUP VALIDATION: {strategy_name} indicators ready")
                else:
                    self.algorithm.Log(f"WARMUP WARNING: {strategy_name} indicators may not be ready")
                    
        except Exception as e:
            self.algorithm.Error(f"Failed to validate indicators: {str(e)}")

    def validate_strategies_ready(self):
        """Validate that all strategies are ready for trading using available methods."""
        try:
            if hasattr(self.algorithm, 'orchestrator') and self.algorithm.orchestrator:
                # Use available orchestrator methods to validate strategies
                if hasattr(self.algorithm.orchestrator, 'strategy_loader') and self.algorithm.orchestrator.strategy_loader:
                    loaded_strategies = self.algorithm.orchestrator.strategy_loader.get_loaded_strategies()
                    self.algorithm.Log(f"WARMUP VALIDATION: {len(loaded_strategies)} strategies loaded")
                    self.algorithm.Log(f"WARMUP VALIDATION: Strategy names: {list(loaded_strategies)}")
                    
                    if len(loaded_strategies) > 0:
                        self.algorithm.Log("WARMUP VALIDATION: Strategies ready for trading")
                    else:
                        self.algorithm.Log("WARMUP WARNING: No strategies loaded")
                else:
                    self.algorithm.Log("WARMUP WARNING: Strategy loader not available for validation")
            else:
                self.algorithm.Log("WARMUP WARNING: Orchestrator not available for strategy validation")
                
        except Exception as e:
            self.algorithm.Error(f"Failed to validate strategies: {str(e)}")

    def validate_universe_ready(self):
        """Validate that the universe is properly populated and ready for trading."""
        try:
            if hasattr(self.algorithm, 'futures_symbols') and self.algorithm.futures_symbols:
                symbol_count = len(self.algorithm.futures_symbols)
                self.algorithm.Log(f"WARMUP VALIDATION: {symbol_count} futures symbols in universe")
                
                # Check data availability for symbols
                valid_symbols = 0
                for symbol in self.algorithm.futures_symbols:
                    if symbol in self.algorithm.Securities:
                        security = self.algorithm.Securities[symbol]
                        if security.HasData:
                            valid_symbols += 1
                
                self.algorithm.Log(f"WARMUP VALIDATION: {valid_symbols}/{symbol_count} symbols have data")
                
                if valid_symbols == 0:
                    self.algorithm.Log("WARMUP WARNING: No symbols have data available")
                elif valid_symbols < symbol_count:
                    self.algorithm.Log(f"WARMUP WARNING: Only {valid_symbols}/{symbol_count} symbols have data")
                else:
                    self.algorithm.Log("WARMUP VALIDATION: All symbols have data - universe ready")
            else:
                self.algorithm.Log("WARMUP WARNING: No futures symbols found in universe")
                
        except Exception as e:
            self.algorithm.Error(f"Failed to validate universe: {str(e)}")

    def log_warmup_completion_status(self):
        """Log comprehensive warmup completion status."""
        try:
            self.algorithm.Log("WARMUP COMPLETION STATUS:")
            self.algorithm.Log(f"  - Algorithm Time: {self.algorithm.Time}")
            self.algorithm.Log(f"  - Is Warming Up: {self.algorithm.IsWarmingUp}")
            self.algorithm.Log(f"  - Portfolio Value: ${self.algorithm.Portfolio.TotalPortfolioValue:,.2f}")
            self.algorithm.Log(f"  - Available Cash: ${self.algorithm.Portfolio.Cash:,.2f}")
            
            if hasattr(self.algorithm, 'futures_symbols'):
                self.algorithm.Log(f"  - Universe Size: {len(self.algorithm.futures_symbols)} futures")
                
        except Exception as e:
            self.algorithm.Error(f"Failed to log warmup completion status: {str(e)}")

    def log_warmup_progress(self):
        """Log warmup progress periodically."""
        try:
            if hasattr(self.algorithm, '_warmup_total_days') and self.algorithm._warmup_total_days > 0:
                # Calculate progress if possible
                if hasattr(self.algorithm, '_warmup_start_time') and self.algorithm._warmup_start_time:
                    elapsed = (self.algorithm.Time - self.algorithm._warmup_start_time).days
                    progress_pct = min(100, (elapsed / self.algorithm._warmup_total_days) * 100)
                    self.algorithm.Log(f"WARMUP PROGRESS: {progress_pct:.1f}% ({elapsed}/{self.algorithm._warmup_total_days} days)")
                else:
                    self.algorithm.Log(f"WARMUP: In progress (target: {self.algorithm._warmup_total_days} days)")
            else:
                self.algorithm.Log("WARMUP: In progress")
                
        except Exception as e:
            self.algorithm.Error(f"Failed to log warmup progress: {str(e)}")

    def _should_log_component(self, component_name, level):
        """Check if we should log for a component at a given level."""
        try:
            if hasattr(self.algorithm, 'config_manager') and self.algorithm.config_manager:
                from config.config_market_strategy import get_log_level_for_component
                component_level = get_log_level_for_component(component_name)
                
                # Simple level comparison (DEBUG < INFO < WARNING < ERROR < CRITICAL)
                levels = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'CRITICAL': 4}
                return levels.get(level, 1) >= levels.get(component_level, 1)
            return True  # Default to logging if config not available
        except:
            return True  # Default to logging if there's any error 