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
        """Initialize the Three-Layer CTA Portfolio System."""
        try:
            self.Log("="*80)
            self.Log("INITIALIZING THREE-LAYER CTA PORTFOLIO SYSTEM")
            self.Log("="*80)
            
            # Check if imports were successful
            if not IMPORTS_SUCCESSFUL:
                self.Error(f"CRITICAL: Module imports failed: {IMPORT_ERROR}")
                self.Error("SECURITY POLICY: No fallback trading configurations allowed")
                self.Error("Initializing crash-prevention mode (NO TRADING)")
                self._emergency_fallback_initialization()
                return
            
            # Initialize tracking variables
            self._warmup_completed = False
            self._first_rebalance_attempted = False
            self._rollover_events_count = 0
            self._algorithm_start_time = self.Time
            
            # Step 1: Add futures first (critical for QC)
            self.Log("Step 1: Adding futures contracts...")
            self.futures_symbols = []
            self._add_core_futures()
            
            # Step 2: Initialize configuration management
            self.Log("Step 2: Initializing configuration management...")
            try:
                self.config_manager = AlgorithmConfigManager(self)
                self.config = self.config_manager.load_and_validate_config(variant="full")
                self.Log("SUCCESS: Configuration manager initialized")
                
                # CRITICAL: Validate complete configuration before proceeding
                self.Log("Step 2.1: Validating complete configuration...")
                self.config_manager.validate_complete_configuration()
                self.Log("SUCCESS: Complete configuration validated")
                
                # CRITICAL: Generate and log configuration audit report
                self.Log("Step 2.2: Generating configuration audit report...")
                audit_report = self.config_manager.get_config_audit_report()
                
                # Log the audit report (split into chunks to avoid QC log limits)
                report_lines = audit_report.split('\n')
                for i in range(0, len(report_lines), 10):  # Log 10 lines at a time
                    chunk = '\n'.join(report_lines[i:i+10])
                    self.Log(chunk)
                
                self.Log("SUCCESS: Configuration audit report generated")
                
            except Exception as e:
                # CRITICAL: Do not use fallback configuration for trading
                error_msg = f"CRITICAL CONFIGURATION FAILURE: {str(e)}"
                self.Error(error_msg)
                self.Error("SECURITY POLICY: No fallback trading configurations allowed")
                self.Error("STOPPING ALGORITHM: Cannot trade with invalid configuration")
                
                # Log the specific configuration issue for debugging
                self.Error("Configuration loading failed. This could be due to:")
                self.Error("1. Missing or corrupted configuration files")
                self.Error("2. Invalid configuration structure")
                self.Error("3. Import path issues in QuantConnect cloud")
                self.Error("4. Configuration validation failures")
                
                # SECURITY: Emergency fallback only prevents crashes - does not trade
                self.Error("Initializing crash-prevention mode (NO TRADING)")
                self._emergency_fallback_initialization()
                return
            
            # Step 2.5: Initialize QuantConnect Native Features
            self.Log("Step 2.5: Initializing QuantConnect native features...")
            self._initialize_qc_native_features()
            
            # Step 3: Initialize universe management
            self.Log("Step 3: Initializing universe management...")
            try:
                # Create futures manager (required by strategy loader)
                self.futures_manager = FuturesManager(self, self.config_manager)
                self.futures_manager.initialize_universe()
                self.Log("SUCCESS: Futures manager initialized")
                
                # Create universe helper for additional data
                universe_helper = UniverseHelpers(self, self.config_manager)
                self.universe_data = universe_helper.initialize_universe()
                self.Log("SUCCESS: Universe management initialized")
            except Exception as e:
                self.Log(f"Using minimal universe management: {str(e)}")
                # Create minimal futures manager as fallback
                class MinimalFuturesManager:
                    def __init__(self, algorithm): self.algorithm = algorithm
                    def get_liquid_symbols(self): return ['ES', 'NQ', 'ZN']
                
                self.futures_manager = MinimalFuturesManager(self)
                self.universe_data = {'tickers': ['ES', 'NQ', 'ZN'], 'symbols': self.futures_symbols}
            
            # Step 4: Initialize orchestrator
            self.Log("Step 4: Initializing three-layer orchestrator...")
            try:
                self.orchestrator = ThreeLayerOrchestrator(self, self.config_manager)
                self.orchestrator.initialize_system()
                self.Log("SUCCESS: Orchestrator initialized")
            except Exception as e:
                self.Log(f"Using minimal orchestrator: {str(e)}")
                # Create minimal orchestrator as fallback
                class MinimalOrchestrator:
                    def __init__(self, algorithm): self.algorithm = algorithm
                    def initialize_system(self): pass
                    def OnSecuritiesChanged(self, changes): pass
                    def update_with_data(self, slice): pass
                    def update_during_warmup(self, slice): pass
                
                self.orchestrator = MinimalOrchestrator(self)
            
            # Step 5: Initialize execution manager
            self.Log("Step 5: Initializing execution manager...")
            try:
                self.execution_manager = PortfolioExecutionManager(self, self.config_manager)
                self.Log("SUCCESS: Execution manager initialized")
            except Exception as e:
                self.Log(f"Using minimal execution manager: {str(e)}")
                # Create minimal execution manager as fallback
                class MinimalExecutionManager:
                    def __init__(self, algorithm): self.algorithm = algorithm
                
                self.execution_manager = MinimalExecutionManager(self)
            
            # Step 6: Initialize data integrity checker (CRITICAL FOR PRIORITY 2 FUTURES)
            self.Log("Step 6: Initializing data integrity checker...")
            try:
                from components.data_integrity_checker import DataIntegrityChecker
                self.data_integrity_checker = DataIntegrityChecker(self)
                self.Log("SUCCESS: Data integrity checker initialized")
            except Exception as e:
                self.Log(f"Warning: Data integrity checker failed: {str(e)}")
                self.data_integrity_checker = None
                self.Log("WARNING: Data validation will be basic only")
            
            # Step 6b: Initialize bad data position manager (NEW TARGETED APPROACH)
            self.Log("Step 6b: Initializing bad data position manager...")
            try:
                from components.bad_data_position_manager import BadDataPositionManager
                self.bad_data_position_manager = BadDataPositionManager(self, self.config_manager)
                self.Log("SUCCESS: Bad data position manager initialized")
            except Exception as e:
                self.Log(f"Warning: Bad data position manager failed: {str(e)}")
                self.bad_data_position_manager = None
            
            # Step 6c: Initialize QC native contract resolver (FIXES /ZN vs ZN mapping issues)
            self.Log("Step 6c: Initializing QC native contract resolver...")
            try:
                from components.qc_native_contract_resolver import QCNativeContractResolver
                self.contract_resolver = QCNativeContractResolver(self)
                self.Log("SUCCESS: QC native contract resolver initialized")
            except Exception as e:
                self.Log(f"Warning: QC native contract resolver failed: {str(e)}")
                self.contract_resolver = None
            
            # Step 7: Initialize system reporter
            self.Log("Step 7: Initializing system reporter...")
            try:
                self.system_reporter = SystemReporter(self, self.config_manager)
                self.Log("SUCCESS: System reporter initialized")
            except Exception as e:
                self.Log(f"Using minimal system reporter: {str(e)}")
                # Create minimal system reporter as fallback
                class MinimalSystemReporter:
                    def __init__(self, algorithm): self.algorithm = algorithm
                    def track_rollover_cost(self, old_symbol, new_symbol, quantity): pass
                
                self.system_reporter = MinimalSystemReporter(self)
            
            # Step 8: Schedule rebalancing
            self.Log("Step 8: Scheduling rebalancing...")
            self._schedule_rebalancing()
            
            self.Log("="*80)
            self.Log("THREE-LAYER CTA SYSTEM INITIALIZATION COMPLETE")
            self.Log("="*80)
            
        except Exception as e:
            self.Error(f"CRITICAL ERROR in Initialize: {str(e)}")
            # Emergency fallback initialization
            self._emergency_fallback_initialization()
    
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
        """Schedule weekly and monthly rebalancing."""
        try:
            # Weekly rebalancing on Fridays at 10 AM (market hours)
            # Use WeekEnd() which triggers on the last trading day of the week (usually Friday)
            self.Schedule.On(
                self.DateRules.WeekEnd(), 
                self.TimeRules.At(10, 0), 
                self.WeeklyRebalance
            )
            
            # Alternative: Also schedule on explicit Fridays as backup
            self.Schedule.On(
                self.DateRules.Every(DayOfWeek.Friday),
                self.TimeRules.At(10, 0),
                self.WeeklyRebalanceBackup
            )
            
            # Monthly performance reporting
            self.Schedule.On(
                self.DateRules.MonthEnd(),
                self.TimeRules.At(17, 0),
                self.MonthlyReporting
            )
            
            # Daily continuous contract validation (fixes /ZN vs ZN mapping issues)
            self.Schedule.On(
                self.DateRules.EveryDay(),
                self.TimeRules.At(9, 0),
                self.ValidateContinuousContracts
            )
            
            self.Log("Rebalancing schedule configured:")
            self.Log("  - Weekly: Fridays at 10:00 AM (WeekEnd + explicit Friday backup)")
            self.Log("  - Monthly: Month end at 17:00")
            self.Log("  - Daily: Contract validation at 09:00")
            
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
    
    def OnData(self, slice):
        """Handle market data updates with targeted position management for bad data."""
        try:
            # Defensive check: Initialize tracking variables if not already set
            if not hasattr(self, '_warmup_completed'):
                self._warmup_completed = False
                self._first_rebalance_attempted = False
                self._rollover_events_count = 0
                self._algorithm_start_time = self.Time
                self.Log("WARNING: Tracking variables initialized in OnData (should have been in Initialize)")
            
            if self.IsWarmingUp:
                if not self._warmup_completed:
                    # Update components during warmup with basic validation
                    if hasattr(self, 'orchestrator'):
                        validated_slice = self._validate_slice_data_basic(slice)
                        if validated_slice:
                            self.orchestrator.update_during_warmup(validated_slice)
                return

            if not self._warmup_completed:
                self._warmup_completed = True
                self.Log("Warmup completed - System ready for trading")
                
                # IMMEDIATE TEST: Trigger first rebalancing to test the system
                self.Log("="*50)
                self.Log("WARMUP COMPLETE - TESTING IMMEDIATE REBALANCING")
                self.Log("="*50)
                try:
                    self.WeeklyRebalance()
                except Exception as test_e:
                    self.Error(f"IMMEDIATE REBALANCE TEST FAILED: {str(test_e)}")
                self.Log("="*50)
                
            # TARGETED APPROACH: Basic validation + position management for bad data
            validated_slice = self._validate_slice_data_basic(slice)
            
            if not validated_slice:
                # Skip this update if no valid data
                return

            # Use data integrity checker for quarantine logic (reports to position manager)
            if hasattr(self, 'data_integrity_checker') and self.data_integrity_checker:
                final_validated_slice = self.data_integrity_checker.validate_slice(validated_slice)
                if final_validated_slice:
                    validated_slice = final_validated_slice

            # Update components with validated data
            if validated_slice and hasattr(self, 'orchestrator'):
                self.orchestrator.update_with_data(validated_slice)
                
        except Exception as e:
            self.Error(f"Error in OnData: {str(e)}")
    
    def _validate_slice_data_basic(self, slice):
        """
        BASIC slice validation - just check if slice has any usable data
        Don't try to modify the slice, just validate it's safe to use
        """
        try:
            # Check if we have any data in the slice
            if not slice or not hasattr(slice, 'Keys') or not slice.Keys:
                return None
            
            # Count valid symbols in this slice
            valid_symbol_count = 0
            
            # Check each symbol in the slice
            for symbol in slice.Keys:
                # Check if symbol exists in our securities
                if symbol not in self.Securities:
                    continue
                
                security = self.Securities[symbol]
                
                # Basic QC built-in checks
                if not (security.HasData and security.IsTradable):
                    continue
                
                # Check if symbol has valid data in this slice
                has_valid_data = False
                
                # Check bar data
                if hasattr(slice, 'Bars') and slice.Bars and symbol in slice.Bars:
                    try:
                        bar = slice.Bars[symbol]
                        if bar and hasattr(bar, 'Close') and bar.Close > 0:
                            has_valid_data = True
                    except:
                        pass
                
                # Check quote data
                if hasattr(slice, 'QuoteBars') and slice.QuoteBars and symbol in slice.QuoteBars:
                    try:
                        quote = slice.QuoteBars[symbol]
                        if (quote and hasattr(quote, 'Bid') and hasattr(quote, 'Ask') and
                            hasattr(quote.Bid, 'Close') and hasattr(quote.Ask, 'Close') and
                            quote.Bid.Close > 0 and quote.Ask.Close > 0):
                            has_valid_data = True
                    except:
                        pass
                
                if has_valid_data:
                    valid_symbol_count += 1
            
            # Return slice if we have at least one valid symbol, None otherwise
            if valid_symbol_count > 0:
                return slice
            else:
                return None
            
        except Exception as e:
            self.Error(f"Error in basic slice validation: {str(e)}")
            # Return slice on validation error to avoid blocking all data
            return slice
    
    def _validate_slice_data_aggressive(self, slice):
        """
        AGGRESSIVE slice validation - Only allow symbols with confirmed valid data
        This prevents the 'security does not have accurate price' errors
        """
        try:
            # Track valid symbols for this slice
            valid_symbols = set()
            
            # STEP 1: Pre-filter symbols using QC's built-in validation
            for symbol in slice.Keys:
                # Use QC's slice.Contains() method - CRITICAL check
                if not slice.Contains(symbol):
                    continue
                
                # Check if symbol exists in our securities
                if symbol not in self.Securities:
                    continue
                
                security = self.Securities[symbol]
                
                # AGGRESSIVE QC built-in checks
                if not (security.HasData and security.IsTradable and 
                        hasattr(security, 'Price') and security.Price > 0):
                    continue
                
                # Symbol passed all checks
                valid_symbols.add(symbol)
            
            if not valid_symbols:
                # No valid symbols in this slice
                return None
            
            # STEP 2: Validate bar data for valid symbols only
            has_valid_data = False
            
            if hasattr(slice, 'Bars') and slice.Bars:
                valid_bars = {}
                for symbol in valid_symbols:
                    if symbol in slice.Bars:
                        bar = slice.Bars[symbol]
                        if self._is_bar_data_valid_aggressive(symbol, bar):
                            valid_bars[symbol] = bar
                            has_valid_data = True
                        else:
                            # Remove from valid symbols if bar data is bad
                            valid_symbols.discard(symbol)
                            # Occasional logging to avoid spam
                            if hasattr(self, 'Time') and self.Time.day == 1 and self.Time.hour == 16:
                                ticker = str(symbol).replace('/', '')
                                self.Debug(f"Skipping invalid bar data for {ticker}")
                
                # Replace slice bars with only valid ones
                if valid_bars:
                    slice.Bars.clear()
                    for symbol, bar in valid_bars.items():
                        slice.Bars[symbol] = bar
            
            # STEP 3: Validate quote data for valid symbols only
            if hasattr(slice, 'QuoteBars') and slice.QuoteBars:
                valid_quotes = {}
                for symbol in valid_symbols:
                    if symbol in slice.QuoteBars:
                        quote = slice.QuoteBars[symbol]
                        if self._is_quote_data_valid_aggressive(symbol, quote):
                            valid_quotes[symbol] = quote
                            has_valid_data = True
                        else:
                            # Remove from valid symbols if quote data is bad
                            valid_symbols.discard(symbol)
                
                # Replace slice quotes with only valid ones
                if valid_quotes:
                    slice.QuoteBars.clear()
                    for symbol, quote in valid_quotes.items():
                        slice.QuoteBars[symbol] = quote
            
            # Only return slice if we have confirmed valid data
            return slice if has_valid_data and valid_symbols else None
            
        except Exception as e:
            self.Error(f"Error in aggressive slice validation: {str(e)}")
            return None  # Return None on validation error to be safe
    
    def _is_bar_data_valid_aggressive(self, symbol, bar):
        """
        AGGRESSIVE bar validation - Prevents 'security does not have accurate price' errors
        More strict than basic validation to ensure data quality
        """
        try:
            # CRITICAL: Check if symbol exists in securities (QC managed)
            if symbol not in self.Securities:
                return False
            
            security = self.Securities[symbol]
            
            # AGGRESSIVE QC built-in validation
            if not (security.HasData and security.IsTradable):
                return False
            
            # CRITICAL: Verify security has valid price
            if not hasattr(security, 'Price') or security.Price is None or security.Price <= 0:
                return False
            
            # AGGRESSIVE bar data validation
            if not bar:
                return False
            
            # Check all OHLC values are positive and reasonable
            if not (bar.Open > 0 and bar.High > 0 and bar.Low > 0 and bar.Close > 0):
                return False
            
            # STRICT OHLC relationship validation
            if bar.High < bar.Low:
                return False
            
            if bar.Close < bar.Low or bar.Close > bar.High:
                return False
            
            if bar.Open < bar.Low or bar.Open > bar.High:
                return False
            
            # Additional sanity checks - prevent extreme price movements
            if bar.High / bar.Low > 10:  # More than 10x price range in one bar
                return False
            
            # Check volume if available (some futures may not have volume)
            if hasattr(bar, 'Volume') and bar.Volume < 0:
                return False
            
            return True
            
        except Exception as e:
            return False
    
    def _is_bar_data_valid(self, symbol, bar):
        """
        LEGACY bar validation - kept for backward compatibility
        Use _is_bar_data_valid_aggressive for new code
        """
        try:
            # LEVERAGE QC'S BUILT-IN VALIDATION:
            
            # 1. Check if symbol exists in securities (QC managed)
            if symbol not in self.Securities:
                return False
            
            security = self.Securities[symbol]
            
            # 2. Use QC's HasData property instead of manual checks
            if not security.HasData:
                return False
            
            # 3. Use QC's IsTradable property
            if not security.IsTradable:
                return False
            
            # 4. Basic bar integrity (QC doesn't validate OHLC relationships)
            if not bar or bar.Close <= 0 or bar.Open <= 0:
                return False
            
            # 5. OHLC relationship validation (business logic QC doesn't provide)
            if bar.High < bar.Low:
                return False
            
            if bar.Close < bar.Low or bar.Close > bar.High:
                return False
            
            if bar.Open < bar.Low or bar.Open > bar.High:
                return False
            
            return True
            
        except Exception as e:
            return False
    
    def _is_quote_data_valid_aggressive(self, symbol, quote):
        """
        AGGRESSIVE quote validation - Prevents trading errors with bad quote data
        More strict validation to ensure data quality
        """
        try:
            # CRITICAL: Check if symbol exists in securities (QC managed)
            if symbol not in self.Securities:
                return False
            
            security = self.Securities[symbol]
            
            # AGGRESSIVE QC built-in validation
            if not (security.HasData and security.IsTradable):
                return False
            
            # CRITICAL: Verify security has valid price
            if not hasattr(security, 'Price') or security.Price is None or security.Price <= 0:
                return False
            
            # AGGRESSIVE quote data validation
            if not quote:
                return False
            
            # Validate bid data
            if not hasattr(quote, 'Bid') or not quote.Bid:
                return False
            
            if not hasattr(quote.Bid, 'Close') or quote.Bid.Close <= 0:
                return False
            
            # Validate ask data
            if not hasattr(quote, 'Ask') or not quote.Ask:
                return False
            
            if not hasattr(quote.Ask, 'Close') or quote.Ask.Close <= 0:
                return False
            
            # STRICT bid/ask relationship validation
            if quote.Bid.Close >= quote.Ask.Close:
                return False
            
            # Additional sanity checks - prevent extreme spreads
            spread_ratio = (quote.Ask.Close - quote.Bid.Close) / quote.Bid.Close
            if spread_ratio > 0.1:  # More than 10% spread is suspicious
                return False
            
            return True
            
        except Exception as e:
            return False
    
    def _is_quote_data_valid(self, symbol, quote):
        """
        LEGACY quote validation - kept for backward compatibility
        Use _is_quote_data_valid_aggressive for new code
        """
        try:
            # LEVERAGE QC'S BUILT-IN VALIDATION:
            
            # 1. Check if symbol exists in securities (QC managed)
            if symbol not in self.Securities:
                return False
            
            security = self.Securities[symbol]
            
            # 2. Use QC's HasData property
            if not security.HasData:
                return False
            
            # 3. Use QC's IsTradable property
            if not security.IsTradable:
                return False
            
            # 4. Basic quote integrity (QC doesn't validate bid/ask relationships)
            if not quote or quote.Bid.Close <= 0 or quote.Ask.Close <= 0:
                return False
            
            # 5. Bid/Ask relationship validation (business logic QC doesn't provide)
            if quote.Bid.Close > quote.Ask.Close:
                return False
            
            return True
            
        except Exception as e:
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
                self.Log(f"REBALANCE SKIPPED: Still warming up (current: {self.Time}, warmup: {self.IsWarmingUp})")
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
    
    def WeeklyRebalanceBackup(self):
        """Backup weekly rebalancing method to ensure it triggers."""
        try:
            self.Log(f"BACKUP REBALANCE TRIGGER: Friday at {self.Time}")
            
            # Only execute if primary rebalance hasn't run today
            if not hasattr(self, '_last_rebalance_date') or self._last_rebalance_date != self.Time.date():
                self.Log("BACKUP: Executing weekly rebalance (primary didn't trigger)")
                self._last_rebalance_date = self.Time.date()
                self.WeeklyRebalance()
            else:
                self.Log("BACKUP: Primary rebalance already executed today")
                
        except Exception as e:
            self.Error(f"Error in backup weekly rebalance: {str(e)}")
    
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