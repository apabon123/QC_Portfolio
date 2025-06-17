# main_core.py - Streamlined Main Algorithm for QuantConnect Cloud
"""
Three-Layer CTA Portfolio System - Streamlined Version
Optimized for QuantConnect Cloud deployment (under 64KB limit)

This is the main algorithm entry point that coordinates all components
without containing the bulk implementation details.
"""

from AlgorithmImports import *
import sys
import os

# Add src directory to Python path for QuantConnect compatibility
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import core components from organized folders
# Use QuantConnect cloud-compatible import paths
try:
    # Try importing with relative paths (QC cloud compatible)
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'config'))
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'components'))
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'utils'))
    
    # Import configuration first (most critical)
    from algorithm_config_manager import AlgorithmConfigManager
    
    # Import other components
    from three_layer_orchestrator import ThreeLayerOrchestrator  
    from portfolio_execution_manager import PortfolioExecutionManager
    from system_reporter import SystemReporter
    from universe import FuturesManager
    from optimized_symbol_manager import OptimizedSymbolManager
    
    # Defensive import for AssetFilterManager
    try:
        from asset_filter_manager import AssetFilterManager
    except ImportError:
        # AssetFilterManager will be available from universe.py fallback
        AssetFilterManager = None
    from qc_native_data_accessor import QCNativeDataAccessor
    from simplified_data_integrity_checker import SimplifiedDataIntegrityChecker
    from unified_data_interface import UnifiedDataInterface
    
    # Import utilities
    from universe_helpers import UniverseHelpers
    from futures_helpers import FuturesHelpers
    
    IMPORTS_SUCCESSFUL = True
    IMPORT_ERROR = None
except ImportError as e:
    # Fallback for QuantConnect cloud environment
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)

class ThreeLayerCTAPortfolio(QCAlgorithm):
    """
    Professional three-layer CTA portfolio system.
    Streamlined for QuantConnect Cloud deployment.
    
    Architecture:
    - Component 1: AlgorithmConfigManager (Configuration Management)
    - Component 2: ThreeLayerOrchestrator (Strategy Coordination)  
    - Component 3: PortfolioExecutionManager (Trade Execution)
    - Component 4: SystemReporter (Analytics & Reporting)
    - Component 5: FuturesManager (Universe & Rollover Management)
    """
    
    def Initialize(self):
        """
        Algorithm initialization with ENHANCED SECURITY validation.
        Uses centralized configuration management with complete validation.
        """
        try:
            # STEP 1: Initialize configuration management FIRST
            self.config_manager = AlgorithmConfigManager(self)
            self.config = self.config_manager.load_and_validate_config(variant="full")
            
            # STEP 2: CRITICAL - Validate complete configuration
            self.config_manager.validate_complete_configuration()
            
            # STEP 3: Generate and log configuration audit report
            audit_report = self.config_manager.get_config_audit_report()
            for line in audit_report.split('\n'):
                self.Log(line)
            
            # STEP 4: QC NATIVE WARM-UP SETUP (BASED ON PRIMER)
            self._setup_enhanced_warmup()
            
            # STEP 5: Initialize QC native features from validated config
            algo_config = self.config_manager.get_algorithm_config()
            self.SetStartDate(algo_config['start_date']['year'], 
                             algo_config['start_date']['month'], 
                             algo_config['start_date']['day'])
            
            if 'end_date' in algo_config:
                self.SetEndDate(algo_config['end_date']['year'],
                               algo_config['end_date']['month'],
                               algo_config['end_date']['day'])
            
            self.SetCash(algo_config['initial_cash'])
            self.SetBenchmark(algo_config.get('benchmark', 'SPY'))
            
            # STEP 6: Initialize OPTIMIZED symbol management (Phase 1)
            self.symbol_manager = OptimizedSymbolManager(self, self.config_manager)
            self.shared_symbols = self.symbol_manager.setup_shared_subscriptions()
            
            # Initialize QC native data accessor (Phase 2)
            self.data_accessor = QCNativeDataAccessor(self)
            
            # Use simplified data integrity checker for validation only (Phase 2)
            self.data_integrity_checker = SimplifiedDataIntegrityChecker(self, self.config_manager)
            
            # Initialize unified data interface (Phase 3)
            self.unified_data = UnifiedDataInterface(
                self, 
                self.config_manager, 
                self.data_accessor, 
                self.data_integrity_checker
            )
            
            # STEP 7: Initialize components with centralized config and shared symbols
            self.orchestrator = ThreeLayerOrchestrator(self, self.config_manager, self.shared_symbols)
            self.execution_manager = PortfolioExecutionManager(self, self.config_manager)
            
            # Initialize universe manager using shared symbols
            self.universe_manager = FuturesManager(self, self.config_manager, self.shared_symbols)
            self.universe_manager.initialize_universe()
            
            # Initialize performance reporting
            self.reporter = SystemReporter(self, self.config_manager)
            
            # Schedule weekly rebalancing (market close on Fridays)
            self.Schedule.On(
                self.DateRules.WeekEnd(),
                self.TimeRules.BeforeMarketClose("SPY", 30),
                self.WeeklyRebalance
            )
            
            # Schedule monthly reporting
            self.Schedule.On(
                self.DateRules.MonthStart(),
                self.TimeRules.AfterMarketOpen("SPY", 30),
                self.MonthlyReport
            )

            # Initialize tracking variables for defensive programming
            self._warmup_completed = False
            self._first_rebalance_attempted = False
            self._rollover_events_count = 0
            self._algorithm_start_time = self.Time

            self.Log("=" * 80)
            self.Log("THREE-LAYER CTA PORTFOLIO ALGORITHM INITIALIZED SUCCESSFULLY")
            self.Log("=" * 80)
            
        except Exception as e:
            self.Error(f"CRITICAL ERROR during initialization: {str(e)}")
            raise

    def _setup_enhanced_warmup(self):
        """
        Setup QC's native warm-up system based on strategy requirements and QC primer.
        Implements proper warm-up period calculation and configuration.
        """
        try:
            warmup_config = self.config_manager.get_warmup_config()
            
            if not warmup_config.get('enabled', False):
                self.Log("WARMUP: Disabled in configuration")
                return
            
            # Calculate required warm-up period based on enabled strategies
            warmup_days = self.config_manager.calculate_max_warmup_needed()
            
            if warmup_days <= 0:
                self.Log("WARMUP: No warm-up required")
                return
            
            # Use QC's native warm-up system
            warmup_method = warmup_config.get('method', 'time_based')
            warmup_resolution = getattr(Resolution, warmup_config.get('resolution', 'Daily'))
            
            if warmup_method == 'time_based':
                # Time-based warm-up (recommended for CTA strategies)
                warmup_period = timedelta(days=warmup_days)
                self.SetWarmUp(warmup_period, warmup_resolution)
                self.Log(f"WARMUP: Set time-based warm-up for {warmup_days} days at {warmup_config.get('resolution', 'Daily')} resolution")
            else:
                # Bar-count based warm-up
                self.SetWarmUp(warmup_days, warmup_resolution)
                self.Log(f"WARMUP: Set bar-count warm-up for {warmup_days} bars at {warmup_config.get('resolution', 'Daily')} resolution")
            
            # Enable automatic indicator warm-up
            self.Settings.AutomaticIndicatorWarmUp = True
            self.Log("WARMUP: Enabled automatic indicator warm-up")
            
            # Store warm-up info for progress tracking
            self._warmup_config = warmup_config
            self._warmup_start_time = None
            self._warmup_total_days = warmup_days
            
            self.Log("=" * 60)
            self.Log("ENHANCED WARM-UP SYSTEM CONFIGURED")
            self.Log(f"Method: {warmup_method}")
            self.Log(f"Period: {warmup_days} days")
            self.Log(f"Resolution: {warmup_config.get('resolution', 'Daily')}")
            self.Log(f"Automatic Indicators: Enabled")
            
            # Log strategy-specific requirements
            progress_info = self.config_manager.get_warmup_progress_info()
            enabled_strategies = progress_info.get('enabled_strategies', [])
            self.Log(f"Enabled Strategies: {enabled_strategies}")
            
            for strategy_name in enabled_strategies:
                strategy_config = self.config_manager.get_strategy_config(strategy_name)
                strategy_warmup = strategy_config.get('warmup_config', {})
                required_days = strategy_warmup.get('required_days', 0)
                if required_days > 0:
                    self.Log(f"  - {strategy_name}: {required_days} days required")
            
            self.Log("=" * 60)
            
        except Exception as e:
            self.Error(f"Failed to setup enhanced warm-up: {str(e)}")
            # Continue without warm-up rather than failing completely
            self.Log("WARNING: Continuing without warm-up due to setup error")

    def OnWarmupFinished(self):
        """
        QC's native warm-up completion callback.
        Validates that all indicators and strategies are ready for trading.
        """
        try:
            self.Log("=" * 80)
            self.Log("WARM-UP PERIOD COMPLETED - VALIDATING SYSTEM READINESS")
            self.Log("=" * 80)
            
            # Mark warm-up as completed
            self._warmup_completed = True
            
            # Get warm-up configuration
            warmup_config = self.config_manager.get_warmup_config()
            
            # Validate indicators are ready (if enabled)
            if warmup_config.get('validate_indicators_ready', True):
                self._validate_indicators_ready()
            
            # Validate strategies are ready
            self._validate_strategies_ready()
            
            # Validate universe is ready
            self._validate_universe_ready()
            
            # Log system status
            self._log_warmup_completion_status()
            
            # Optional: Trigger immediate test rebalance to verify system
            if warmup_config.get('test_rebalance_on_completion', False):
                self.Log("TESTING: Triggering immediate rebalance to test system...")
                try:
                    self.WeeklyRebalance()
                except Exception as test_e:
                    self.Error(f"WARMUP TEST FAILED: {str(test_e)}")
            
            self.Log("=" * 80)
            self.Log("SYSTEM IS READY FOR LIVE TRADING")
            self.Log("=" * 80)
            
        except Exception as e:
            self.Error(f"Error in OnWarmupFinished: {str(e)}")
            # Don't raise - allow trading to continue even if validation has issues

    def _validate_indicators_ready(self):
        """Validate that all required indicators are ready after warm-up."""
        try:
            enabled_strategies = self.config_manager.get_enabled_strategies()
            
            for strategy_name in enabled_strategies:
                indicator_ready = self.config_manager.validate_warmup_indicators(strategy_name)
                if indicator_ready:
                    self.Log(f"WARMUP VALIDATION: {strategy_name} indicators ready")
                else:
                    self.Log(f"WARMUP WARNING: {strategy_name} indicators may not be ready")
                    
        except Exception as e:
            self.Error(f"Failed to validate indicators: {str(e)}")

    def _validate_strategies_ready(self):
        """Validate that all strategies are ready for trading."""
        try:
            if hasattr(self, 'orchestrator') and self.orchestrator:
                strategies_ready = self.orchestrator.validate_strategies_ready()
                self.Log(f"WARMUP VALIDATION: Strategies ready: {strategies_ready}")
            else:
                self.Log("WARMUP WARNING: Orchestrator not available for strategy validation")
                
        except Exception as e:
            self.Error(f"Failed to validate strategies: {str(e)}")

    def _validate_universe_ready(self):
        """Validate that the universe is ready with liquid symbols."""
        try:
            if hasattr(self, 'universe_manager') and self.universe_manager:
                # Get liquid symbols after warm-up (no slice needed for validation)
                liquid_symbols = self.universe_manager.get_liquid_symbols()
                self.Log(f"WARMUP VALIDATION: {len(liquid_symbols)} liquid symbols available")
                
                if len(liquid_symbols) == 0:
                    self.Log("WARMUP WARNING: No liquid symbols found - trading may be limited")
                else:
                    # Log sample of liquid symbols
                    sample_symbols = list(liquid_symbols)[:5]
                    self.Log(f"WARMUP VALIDATION: Sample liquid symbols: {sample_symbols}")
            else:
                self.Log("WARMUP WARNING: Universe manager not available for validation")
                
        except Exception as e:
            self.Error(f"Failed to validate universe: {str(e)}")

    def _log_warmup_completion_status(self):
        """Log comprehensive warm-up completion status."""
        try:
            # Algorithm status
            self.Log(f"Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
            self.Log(f"Trading Start Time: {self.Time}")
            
            # Warm-up timing
            if hasattr(self, '_warmup_start_time') and self._warmup_start_time:
                warmup_duration = self.Time - self._warmup_start_time
                self.Log(f"Warm-up Duration: {warmup_duration}")
            
            # Data status
            total_securities = len(self.Securities)
            self.Log(f"Total Securities: {total_securities}")
            
            # Strategy status
            enabled_strategies = self.config_manager.get_enabled_strategies()
            self.Log(f"Enabled Strategies: {list(enabled_strategies.keys())}")
            
        except Exception as e:
            self.Error(f"Failed to log warmup completion status: {str(e)}")

    def _log_warmup_progress(self):
        """Log warm-up progress periodically."""
        try:
            if hasattr(self, '_warmup_start_time') and self._warmup_start_time:
                elapsed = self.Time - self._warmup_start_time
                total_days = getattr(self, '_warmup_total_days', 0)
                
                if total_days > 0:
                    progress = min(100, (elapsed.days / total_days) * 100)
                    self.Log(f"WARMUP PROGRESS: {progress:.1f}% complete ({elapsed.days}/{total_days} days)")
                else:
                    self.Log(f"WARMUP PROGRESS: {elapsed.days} days elapsed")
            else:
                self.Log("WARMUP PROGRESS: Tracking not initialized")
                
        except Exception as e:
            self.Error(f"Failed to log warmup progress: {str(e)}")

    def OnData(self, slice):
        """
        Handle market data updates with ENHANCED warm-up awareness and futures chain analysis.
        Implements proper QC warm-up patterns from the primer.
        """
        try:
            # Defensive check: Initialize tracking variables if not already set
            if not hasattr(self, '_warmup_completed'):
                self._warmup_completed = False
                self._first_rebalance_attempted = False
                self._rollover_events_count = 0
                self._algorithm_start_time = self.Time
                self.Log("WARNING: Tracking variables initialized in OnData (should have been in Initialize)")
            
            # QC WARM-UP HANDLING (BASED ON PRIMER)
            if self.IsWarmingUp:
                # During warm-up: Only update components for state building, NO TRADING
                self._handle_warmup_data(slice)
                return
            
            # POST WARM-UP: Normal trading logic
            if not self._warmup_completed:
                self._warmup_completed = True
                self._handle_warmup_completion()
            
            # NORMAL OPERATION: Process data with full validation
            self._handle_normal_trading_data(slice)
                
        except Exception as e:
            self.Error(f"Error in OnData: {str(e)}")

    def _handle_warmup_data(self, slice):
        """
        Handle data during warm-up period with PROPER futures chain passing.
        
        During warm-up: Update indicators and build state, but NO TRADING.
        CRITICAL: Pass slice to all components that need futures chain analysis.
        """
        try:
            # Track warm-up progress
            if not hasattr(self, '_warmup_start_time') or not self._warmup_start_time:
                self._warmup_start_time = self.Time
                self.Log("WARMUP: First data received - starting warm-up tracking")
            
            # CRITICAL: Pass slice to universe manager for futures chain analysis
            if hasattr(self, 'universe_manager') and self.universe_manager:
                self.universe_manager.update_during_warmup(slice)
            
            # Update orchestrator with slice during warm-up
            if hasattr(self, 'orchestrator') and self.orchestrator:
                self.orchestrator.update_during_warmup(slice)
            
            # Handle rollover events during warm-up (positions don't exist yet, but track for analysis)
            if hasattr(slice, 'SymbolChangedEvents') and slice.SymbolChangedEvents:
                for symbolChanged in slice.SymbolChangedEvents.Values:
                    self._rollover_events_count += 1
                    self.Log(f"WARMUP ROLLOVER: {symbolChanged.OldSymbol} -> {symbolChanged.NewSymbol} "
                           f"(Event #{self._rollover_events_count})")
            
            # Periodic warm-up progress logging
            if self.Time.hour == 0 and self.Time.minute == 0:  # Once per day
                self._log_warmup_progress()
                
        except Exception as e:
            self.Error(f"Error handling warmup data: {str(e)}")

    def _handle_warmup_completion(self):
        """Handle the transition from warm-up to normal trading."""
        try:
            # ENHANCED WARMUP COMPLETION LOGGING
            warmup_info = self._get_warmup_status()
            self.Log("="*80)
            self.Log("WARMUP COMPLETED - SYSTEM NOW READY FOR TRADING!")
            self.Log(f"WARMUP INFO: {warmup_info}")
            self.Log(f"TRADING START DATE: {self.Time}")
            self.Log(f"PORTFOLIO VALUE: ${self.Portfolio.TotalPortfolioValue:,.2f}")
            self.Log("="*80)
            
            # IMMEDIATE TEST: Trigger first rebalancing to test the system
            self.Log("="*50)
            self.Log("WARMUP COMPLETE - TESTING IMMEDIATE REBALANCING")
            self.Log("NOTE: Using daily data resolution - rebalancing scheduled for market close")
            self.Log("="*50)
            try:
                # Force immediate rebalancing test (will use current daily bar data)
                self.Log("FORCING IMMEDIATE REBALANCING TEST...")
                self.WeeklyRebalance()
                
                # Note about proper scheduling
                self.Log("SCHEDULING INFO: Future rebalancing will occur at market close (end of week)")
                self.Log("This prevents stale fills that occur with intraday scheduling on daily data")
                
            except Exception as test_e:
                self.Error(f"IMMEDIATE REBALANCE TEST FAILED: {str(test_e)}")
            self.Log("="*50)
            
        except Exception as e:
            self.Error(f"Error handling warmup completion: {str(e)}")

    def _handle_normal_trading_data(self, slice):
        """
        Handle data during normal trading using unified data interface (Phase 3).
        
        STREAMLINED: Uses unified data interface instead of direct slice manipulation.
        """
        try:
            # Handle rollover events first (before trading logic)
            if hasattr(slice, 'SymbolChangedEvents') and slice.SymbolChangedEvents:
                for symbolChanged in slice.SymbolChangedEvents.Values:
                    self._rollover_events_count += 1
                    self.Log(f"ROLLOVER: {symbolChanged.OldSymbol} -> {symbolChanged.NewSymbol} "
                           f"(Event #{self._rollover_events_count})")
            
            # PHASE 3: Use unified data interface for standardized data access
            if hasattr(self, 'unified_data') and self.unified_data:
                # Get standardized slice data through unified interface
                unified_slice_data = self.unified_data.get_slice_data(
                    slice, 
                    symbols=list(self.shared_symbols.keys()) if hasattr(self, 'shared_symbols') else None,
                    data_types=['bars', 'chains']
                )
                
                # Pass unified data to orchestrator
                if hasattr(self, 'orchestrator') and self.orchestrator:
                    self.orchestrator.update_with_unified_data(unified_slice_data, slice)
                
                # Pass unified data to universe manager for liquidity analysis
                if hasattr(self, 'universe_manager') and self.universe_manager:
                    self.universe_manager.update_with_unified_data(unified_slice_data, slice)
                
                # Update system reporter with unified data
                if hasattr(self, 'system_reporter') and self.system_reporter:
                    self.system_reporter.update_with_unified_data(unified_slice_data, slice)
            else:
                # Fallback to direct slice passing (backward compatibility)
                self.Log("WARNING: Unified data interface not available, using direct slice access")
                
                if hasattr(self, 'orchestrator') and self.orchestrator:
                    self.orchestrator.update_with_data(slice)
                
                if hasattr(self, 'universe_manager') and self.universe_manager:
                    self.universe_manager.update_with_slice(slice)
                
                if hasattr(self, 'system_reporter') and self.system_reporter:
                    self.system_reporter.update_with_slice(slice)
                
        except Exception as e:
            self.Error(f"Error handling normal trading data: {str(e)}")

    def _get_warmup_status(self):
        """Get current warmup status information."""
        try:
            warmup_info = self.config_manager.get_warmup_progress_info()
            
            if warmup_info.get('enabled', False):
                return f"Current: {self.Time.strftime('%Y-%m-%d')}, IsWarmingUp: {self.IsWarmingUp}, " \
                       f"MaxDays: {warmup_info.get('max_days_needed', 'N/A')}, " \
                       f"Method: {warmup_info.get('method', 'N/A')}"
            else:
                return f"Current: {self.Time.strftime('%Y-%m-%d')}, IsWarmingUp: {self.IsWarmingUp} (warmup disabled)"
        except Exception as e:
            return f"Current: {self.Time.strftime('%Y-%m-%d')}, IsWarmingUp: {self.IsWarmingUp} (error: {str(e)})"
    
    def _add_core_futures(self):
        """Add core futures contracts with error handling."""
        try:
            futures_helper = FuturesHelpers(self)
            self.futures_symbols = futures_helper.add_core_futures(['ES', 'NQ', 'ZN'])
            self.Log(f"Successfully added {len(self.futures_symbols)} futures contracts")
        except Exception as e:
            self.Error(f"Failed to add futures: {str(e)}")
            self.futures_symbols = []
    
    def _schedule_rebalancing(self):
        """Schedule weekly and monthly rebalancing compatible with daily data resolution."""
        try:
            # CRITICAL: For daily data resolution, use END-OF-DAY scheduling to avoid stale fills
            # Daily bars are only complete at end of day (~6 PM for futures)
            
            # Weekly rebalancing at END OF WEEK - use end of day timing
            # This ensures we have the complete daily bar with actual closing prices
            self.Schedule.On(
                self.DateRules.WeekEnd(), 
                self.TimeRules.At(23, 30),  # 11:30 PM - after daily bar is complete
                self.WeeklyRebalance
            )
            
            # Monthly performance reporting at month end - use end of day timing
            self.Schedule.On(
                self.DateRules.MonthEnd(),
                self.TimeRules.At(23, 45),  # 11:45 PM - after daily bar is complete
                self.MonthlyReporting
            )
            
            # Daily continuous contract validation - use end of day timing
            self.Schedule.On(
                self.DateRules.EveryDay(),
                self.TimeRules.At(23, 0),   # 11:00 PM - after daily bar is complete
                self.ValidateContinuousContracts
            )
            
            self.Log("Rebalancing schedule configured for DAILY DATA RESOLUTION (END-OF-DAY):")
            self.Log("  - Weekly: End of week at 11:30 PM (after daily bar completion)")
            self.Log("  - Monthly: Month end at 11:45 PM (after daily bar completion)")
            self.Log("  - Daily: Contract validation at 11:00 PM (after daily bar completion)")
            self.Log("  - PREVENTS STALE FILLS: Uses actual daily closing prices, not previous day's prices")
            self.Log("  - Daily bars complete around 6 PM for futures, scheduling at 11+ PM ensures fresh data")
            
        except Exception as e:
            self.Error(f"Failed to schedule rebalancing: {str(e)}")
    
    def _emergency_fallback_initialization(self):
        """
        SECURITY-COMPLIANT Emergency fallback - NO TRADING CONFIGURATION FALLBACKS
        Only initializes tracking variables and minimal component stubs to prevent crashes.
        DOES NOT create fallback trading parameters - algorithm will not trade.
        """
        self.Log("EMERGENCY FALLBACK: Initializing crash-prevention system only...")
        self.Error("CRITICAL: Configuration failed - Algorithm will NOT trade")
        self.Error("SECURITY: No fallback trading parameters will be used")
        
        try:
            # Initialize tracking variables ONLY (critical for OnData to not crash)
            self._warmup_completed = False
            self._first_rebalance_attempted = False
            self._rollover_events_count = 0
            self._algorithm_start_time = self.Time
            
            # Initialize empty containers to prevent AttributeError crashes
            self.futures_symbols = []
            self.config = None  # Explicitly set to None - no fallback config
            self.config_manager = None  # Explicitly set to None - no fallback config manager
            
            # Create minimal component stubs ONLY to prevent crashes (no trading functionality)
            class CrashPreventionOrchestrator:
                """Minimal stub to prevent crashes - does not trade"""
                def __init__(self, algorithm):
                    self.algorithm = algorithm
                
                def OnSecuritiesChanged(self, changes):
                    self.algorithm.Log("CrashPreventionOrchestrator: OnSecuritiesChanged called (no action)")
                
                def update_with_data(self, slice):
                    self.algorithm.Log("CrashPreventionOrchestrator: update_with_data called (no action)")
                
                def update_during_warmup(self, slice):
                    self.algorithm.Log("CrashPreventionOrchestrator: update_during_warmup called (no action)")
                
                def generate_portfolio_targets(self):
                    self.algorithm.Log("CrashPreventionOrchestrator: No targets generated (no configuration)")
                    return {'status': 'failed', 'reason': 'No valid configuration available'}
            
            class CrashPreventionExecutionManager:
                """Minimal stub to prevent crashes - does not execute trades"""
                def __init__(self, algorithm):
                    self.algorithm = algorithm
                
                def execute_rebalance_result(self, result):
                    self.algorithm.Log("CrashPreventionExecutionManager: No trades executed (no configuration)")
                    return {'status': 'failed', 'reason': 'No valid configuration available'}
            
            class CrashPreventionSystemReporter:
                """Minimal stub to prevent crashes - basic logging only"""
                def __init__(self, algorithm):
                    self.algorithm = algorithm
                
                def track_rebalance_performance(self, result):
                    self.algorithm.Log("CrashPreventionSystemReporter: No performance tracking (no configuration)")
                
                def generate_monthly_performance_report(self):
                    self.algorithm.Log("CrashPreventionSystemReporter: No monthly report (no configuration)")
                
                def generate_final_algorithm_report(self):
                    self.algorithm.Log("CrashPreventionSystemReporter: Algorithm ended without valid configuration")
                    self.algorithm.Log("SECURITY: No trading occurred due to configuration failure")
            
            # Initialize crash prevention components
            self.orchestrator = CrashPreventionOrchestrator(self)
            self.execution_manager = CrashPreventionExecutionManager(self)
            self.system_reporter = CrashPreventionSystemReporter(self)
            
            # Set minimal QC parameters to prevent QC framework errors
            # These are NOT trading parameters - just framework requirements
            try:
                self.SetStartDate(2020, 1, 1)  # Minimal date range
                self.SetEndDate(2020, 1, 2)    # 1 day only
                self.SetCash(1000000)          # Minimal cash (no trades will be made)
                self.Log("Emergency fallback: Set minimal QC framework parameters")
                self.Log("IMPORTANT: Algorithm will not trade - configuration required")
            except Exception as qc_e:
                self.Error(f"Failed to set minimal QC parameters: {str(qc_e)}")
            
            self.Log("Emergency fallback initialization complete - CRASH PREVENTION ONLY")
            self.Log("SECURITY COMPLIANCE: No trading configuration fallbacks used")
            
        except Exception as e:
            self.Error(f"Emergency fallback failed: {str(e)}")
            self.Error("CRITICAL: Algorithm cannot initialize even crash prevention mode")
            # Don't attempt any further fallbacks - let it fail
    
    def OnSecuritiesChanged(self, changes):
        """Forward security changes to orchestrator for strategy initialization."""
        try:
            # Log securities changes for diagnostics (helps understand /ZN timing issues)
            for security in changes.AddedSecurities:
                symbol = security.Symbol
                symbol_str = str(symbol)
                
                # Log continuous vs underlying contracts
                if symbol_str.startswith('/') or symbol_str.startswith('futures/'):
                    self.Log(f"QC Added: Continuous contract {symbol_str}")
                elif security.Type == SecurityType.Future:
                    self.Log(f"QC Added: Underlying contract {symbol_str}")
            
            for security in changes.RemovedSecurities:
                self.Log(f"QC Removed: {security.Symbol}")
            
            # Forward to orchestrator - QC handles all the mapping automatically
            if hasattr(self, 'orchestrator'):
                self.orchestrator.OnSecuritiesChanged(changes)
                
        except Exception as e:
            self.Error(f"Error in OnSecuritiesChanged: {str(e)}")
    
    def OnSymbolChangedEvents(self, symbolChangedEvents):
        """Handle futures rollover events using config-driven settings."""
        try:
            # SECURITY: No fallback configuration - fail fast if config is invalid
            if not self.config_manager or not self.config:
                self.Error("SECURITY: No valid configuration available for rollover handling")
                self.Error("CRITICAL: Cannot execute rollover without validated configuration")
                return
            
            # Get rollover configuration from validated config manager ONLY
            try:
                execution_config = self.config_manager.get_execution_config()
                rollover_config = execution_config.get('rollover_config', {})
                
                if not rollover_config:
                    self.Error("SECURITY: No rollover_config found in validated configuration")
                    self.Error("CRITICAL: Cannot execute rollover without rollover configuration")
                    return
                    
            except Exception as config_error:
                self.Error(f"SECURITY: Failed to get rollover configuration: {str(config_error)}")
                self.Error("CRITICAL: Cannot execute rollover without validated configuration")
                return
            
            # Check if rollover handling is enabled
            if not rollover_config.get('enabled', True):
                self.Log("Rollover handling is disabled in config")
                return
            
            for symbol, changedEvent in symbolChangedEvents.items():
                oldSymbol = changedEvent.OldSymbol
                newSymbol = changedEvent.NewSymbol
                quantity = self.Portfolio[oldSymbol].Quantity
                
                # Create rollover tags using config
                tag_prefix = rollover_config.get('rollover_tag_prefix', 'ROLLOVER')
                tag = f"{tag_prefix} - {self.Time}: {oldSymbol} -> {newSymbol}"
                
                # Validate new contract if configured
                if rollover_config.get('validate_rollover_contracts', True):
                    if not self._validate_rollover_contract(newSymbol):
                        if rollover_config.get('emergency_liquidation', True):
                            self.Liquidate(oldSymbol, tag=f"{tag_prefix} - EMERGENCY LIQUIDATION")
                            self.Error(f"Failed to validate new contract {newSymbol}, emergency liquidation executed")
                        continue
                
                # Execute rollover with retry logic
                success = self._execute_rollover_with_retry(
                    oldSymbol, newSymbol, quantity, tag, rollover_config
                )
                
                # Log rollover event if configured
                if rollover_config.get('log_rollover_events', True):
                    self.Log(f"ROLLOVER {'SUCCESS' if success else 'FAILED'}: {oldSymbol} -> {newSymbol}, quantity: {quantity}")
                
                # Track rollover costs if configured
                if rollover_config.get('track_rollover_costs', True) and hasattr(self, 'system_reporter'):
                    self.system_reporter.track_rollover_cost(oldSymbol, newSymbol, quantity)
            
            # Defensive check: Initialize tracking variables if not already set
            if not hasattr(self, '_rollover_events_count'):
                self._warmup_completed = False
                self._first_rebalance_attempted = False
                self._rollover_events_count = 0
                self._algorithm_start_time = self.Time
                self.Log("WARNING: Tracking variables initialized in OnSymbolChangedEvents (should have been in Initialize)")
            
            self._rollover_events_count += len(symbolChangedEvents)
            
        except Exception as e:
            self.Error(f"Error handling rollover events: {str(e)}")
    
    def _validate_rollover_contract(self, symbol):
        """Validate that a rollover contract is tradeable."""
        try:
            # Basic validation - check if symbol exists in securities
            if symbol in self.Securities:
                return True
            
            # Additional validation could be added here
            return True
            
        except Exception as e:
            self.Error(f"Error validating rollover contract {symbol}: {str(e)}")
            return False
    
    def _execute_rollover_with_retry(self, oldSymbol, newSymbol, quantity, tag, rollover_config):
        """Execute rollover with retry logic based on config."""
        max_attempts = rollover_config.get('retry_attempts', 3)
        order_type = rollover_config.get('order_type', 'market')
        
        for attempt in range(max_attempts):
            try:
                # Liquidate old position
                self.Liquidate(oldSymbol, tag=f"{tag} - CLOSE (attempt {attempt + 1})")
                
                # Open new position if we had quantity
                if quantity != 0:
                    if order_type.lower() == 'market':
                        self.MarketOrder(newSymbol, quantity, tag=f"{tag} - OPEN (attempt {attempt + 1})")
                    else:
                        # Could add other order types like MarketOnOpen if needed
                        self.MarketOrder(newSymbol, quantity, tag=f"{tag} - OPEN (attempt {attempt + 1})")
                
                return True  # Success
                
            except Exception as e:
                if attempt == max_attempts - 1:
                    # Final attempt failed
                    self.Error(f"Rollover failed after {max_attempts} attempts: {str(e)}")
                    
                    # Emergency liquidation if configured
                    if rollover_config.get('emergency_liquidation', True):
                        try:
                            self.Liquidate(oldSymbol, tag=f"{tag} - EMERGENCY")
                        except:
                            pass
                    
                    return False
                else:
                    self.Log(f"Rollover attempt {attempt + 1} failed, retrying: {str(e)}")
        
        return False
    
    def WeeklyRebalance(self):
        """Execute weekly portfolio rebalancing."""
        try:
            # Defensive check: Initialize tracking variables if not already set
            if not hasattr(self, '_first_rebalance_attempted'):
                self._warmup_completed = False
                self._first_rebalance_attempted = False
                self._rollover_events_count = 0
                self._algorithm_start_time = self.Time
                self.Log("WARNING: Tracking variables initialized in WeeklyRebalance (should have been in Initialize)")
            
            # CRITICAL: Check if we're still warming up
            if self.IsWarmingUp:
                warmup_info = self._get_warmup_status()
                self.Log(f"REBALANCE SKIPPED: Still warming up - {warmup_info}")
                return
                
            self.Log("="*50)
            self.Log("EXECUTING WEEKLY REBALANCE")
            self.Log("="*50)
            
            # Generate portfolio targets through three-layer process
            rebalance_result = self.orchestrator.generate_portfolio_targets()
            
            if rebalance_result and rebalance_result.get('status') == 'success':
                # Execute the rebalance
                execution_result = self.execution_manager.execute_rebalance_result(rebalance_result)
                
                # Track performance
                self.system_reporter.track_rebalance_performance(execution_result)
                
                targets = rebalance_result.get('final_targets', {})
                self.Log(f"Weekly rebalance complete: {len(targets)} targets executed")
            else:
                self.Log("No targets generated for weekly rebalance")
                
            self._first_rebalance_attempted = True
            
        except Exception as e:
            self.Error(f"Error in weekly rebalance: {str(e)}")
    
    def MonthlyReporting(self):
        """Generate monthly performance reports."""
        try:
            self.Log("="*50)
            self.Log("GENERATING MONTHLY REPORT")
            self.Log("="*50)
            
            if hasattr(self, 'system_reporter'):
                self.system_reporter.generate_monthly_performance_report()
            
            # Report data integrity status
            if hasattr(self, 'data_integrity_checker') and self.data_integrity_checker:
                quarantine_status = self.data_integrity_checker.get_quarantine_status()
                self.Log(f"Data Integrity Status:")
                self.Log(f"  Quarantined symbols: {quarantine_status['quarantined_count']}")
                self.Log(f"  Total symbols tracked: {quarantine_status['total_symbols_tracked']}")
                
                if quarantine_status['quarantined_symbols']:
                    self.Log("  Quarantined details:")
                    for symbol_info in quarantine_status['quarantined_symbols']:
                        self.Log(f"    {symbol_info['ticker']}: {symbol_info['reason']} ({symbol_info['days_quarantined']} days)")
            
            # Report bad data position management status
            if hasattr(self, 'bad_data_position_manager') and self.bad_data_position_manager:
                position_status = self.bad_data_position_manager.get_status_report()
                self.Log(f"Bad Data Position Management:")
                for line in position_status.split('\n'):
                    self.Log(f"  {line}")
                
                # Cleanup resolved issues
                self.bad_data_position_manager.cleanup_resolved_issues()
            
        except Exception as e:
            self.Error(f"Error in monthly reporting: {str(e)}")
    
    def ValidateContinuousContracts(self):
        """Daily validation using QC native functionality to diagnose /ZN vs ZN issues."""
        try:
            if hasattr(self, 'contract_resolver') and self.contract_resolver:
                self.Log("Starting daily QC native contract validation...")
                
                # Log QC's native status for all futures
                self.contract_resolver.log_qc_native_status()
                
                # Get diagnostics report
                diagnostics = self.contract_resolver.get_diagnostics_report()
                
                self.Log(f"QC Native Status: "
                        f"Failed history requests: {diagnostics['failed_history_requests']}, "
                        f"Initialized symbols: {diagnostics['initialized_symbols']}")
                
                # Alert if there are persistent failures
                if diagnostics['failed_history_requests'] > 0:
                    self.Log(f"WARNING: {diagnostics['failed_history_requests']} symbols have history failures: "
                           f"{diagnostics['failed_symbols']}")
                    
                    # Test history requests for failed symbols to understand the timing issue
                    for symbol_str in diagnostics['failed_symbols'][:3]:  # Test first 3
                        symbol = next((s for s in self.Securities.Keys if str(s) == symbol_str), None)
                        if symbol:
                            self.Log(f"Testing history for {symbol_str}...")
                            test_history = self.contract_resolver.get_history_with_diagnostics(symbol, 10)
                            if test_history is not None:
                                self.Log(f"  SUCCESS: Got {len(test_history)} bars")
                            else:
                                self.Log(f"  FAILED: No history returned")
        
        except Exception as e:
            self.Error(f"Error in QC native contract validation: {str(e)}")
    
    def OnEndOfAlgorithm(self):
        """Generate final algorithm report."""
        try:
            # Defensive check: Initialize tracking variables if not already set
            if not hasattr(self, '_rollover_events_count'):
                self._warmup_completed = False
                self._first_rebalance_attempted = False
                self._rollover_events_count = 0
                self._algorithm_start_time = self.Time
                self.Log("WARNING: Tracking variables initialized in OnEndOfAlgorithm (should have been in Initialize)")
            
            self.Log("="*80)
            self.Log("ALGORITHM ENDING - GENERATING FINAL REPORT")
            self.Log("="*80)
            
            if hasattr(self, 'system_reporter'):
                self.system_reporter.generate_final_algorithm_report()
            
            # Log final system metrics
            self.Log(f"Final system metrics:")
            self.Log(f"  Rollover events handled: {self._rollover_events_count}")
            self.Log(f"  First rebalance attempted: {self._first_rebalance_attempted}")
            self.Log(f"  Futures symbols tracked: {len(getattr(self, 'futures_symbols', []))}")
            
        except Exception as e:
            self.Error(f"Error in OnEndOfAlgorithm: {str(e)}")
    
    def _initialize_qc_native_features(self):
        """Initialize QuantConnect native features based on configuration."""
        try:
            qc_config = self.config.get('qc_native', {})
            
            # Portfolio Performance Tracking
            portfolio_config = qc_config.get('portfolio_tracking', {})
            if portfolio_config.get('use_qc_statistics', True):
                self.Log("QC Native: Using built-in Statistics for performance tracking")
                # QC automatically provides Statistics - no setup needed
                
            if portfolio_config.get('track_benchmark', True):
                benchmark = self.config.get('algorithm', {}).get('benchmark', 'SPY')
                try:
                    self.SetBenchmark(benchmark)
                    self.Log(f"QC Native: Benchmark set to {benchmark}")
                except Exception as e:
                    self.Log(f"Warning: Could not set benchmark {benchmark}: {str(e)}")
            
            if portfolio_config.get('enable_runtime_statistics', True):
                # QC automatically tracks runtime statistics
                self.Log("QC Native: Runtime statistics enabled")
            
            # Order Management
            order_config = qc_config.get('order_management', {})
            if order_config.get('use_qc_transactions', True):
                self.Log("QC Native: Using built-in Transactions for order tracking")
                # QC automatically provides Transactions - no setup needed
            
            # Data Management
            data_config = qc_config.get('data_management', {})
            if data_config.get('enable_data_normalization', True):
                self.Log("QC Native: Using built-in data normalization")
                # Applied via futures configuration
            
            # Futures Management
            futures_config = qc_config.get('futures_management', {})
            if futures_config.get('use_qc_rollover', True):
                self.Log("QC Native: Using built-in futures rollover system")
                # Applied via DataMappingMode and DataNormalizationMode
            
            self.Log("SUCCESS: QC native features initialized")
            
        except Exception as e:
            self.Log(f"Warning: QC native features initialization failed: {str(e)}")
    
    def OnOrderEvent(self, orderEvent):
        """Handle order events using QC native features when configured."""
        try:
            # Get order management configuration
            qc_config = self.config.get('qc_native', {})
            order_config = qc_config.get('order_management', {})
            
            if order_config.get('use_qc_order_events', True):
                # Use QC's built-in order event handling + our custom logic
                if hasattr(self, 'execution_manager'):
                    self.execution_manager.track_order_execution(orderEvent)
            else:
                # Fall back to purely custom tracking
                if hasattr(self, 'execution_manager'):
                    self.execution_manager.track_order_execution(orderEvent)
                    
        except Exception as e:
            self.Error(f"Error in OnOrderEvent: {str(e)}")
    
    def get_portfolio_performance_summary(self):
        """Get portfolio performance summary using QC native features when available."""
        try:
            qc_config = self.config.get('qc_native', {})
            portfolio_config = qc_config.get('portfolio_tracking', {})
            
            if portfolio_config.get('use_qc_statistics', True):
                # Use QC's built-in statistics
                return self._get_qc_performance_summary()
            else:
                # Use custom tracking
                return self._get_custom_performance_summary()
                
        except Exception as e:
            self.Log(f"Error getting performance summary: {str(e)}")
            return {}
    
    def _get_qc_performance_summary(self):
        """Get performance summary using QC's built-in Statistics."""
        try:
            portfolio = self.Portfolio
            
            return {
                'total_portfolio_value': float(portfolio.TotalPortfolioValue),
                'total_profit': float(portfolio.TotalUnrealizedProfit),
                'total_fees': float(portfolio.TotalFees),
                'cash': float(portfolio.Cash),
                'invested': portfolio.Invested,
                'positions_count': sum(1 for h in portfolio.Values if h.Invested),
                'benchmark_symbol': str(self.Benchmark) if hasattr(self, 'Benchmark') else None,
                'timestamp': self.Time
                # Note: Additional statistics available in QC's Statistics object
                # which is automatically populated during backtesting
            }
        except Exception as e:
            self.Log(f"Error getting QC performance summary: {str(e)}")
            return {}
    
    def _get_custom_performance_summary(self):
        """Fallback to custom performance tracking."""
        return {
            'total_portfolio_value': float(self.Portfolio.TotalPortfolioValue),
            'cash': float(self.Portfolio.Cash),
            'positions_count': sum(1 for h in self.Portfolio.Values if h.Invested),
            'timestamp': self.Time,
            'note': 'Using custom performance tracking'
        } 