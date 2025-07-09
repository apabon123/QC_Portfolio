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
    
    # Import configuration first (most critical) - try multiple import paths
    try:
        from algorithm_config_manager import AlgorithmConfigManager
    except ImportError:
        try:
            from src.config.algorithm_config_manager import AlgorithmConfigManager
        except ImportError:
            # Final fallback - direct path
            import sys
            import os
            config_path = os.path.join(os.path.dirname(__file__), 'src', 'config')
            if config_path not in sys.path:
                sys.path.insert(0, config_path)
            from algorithm_config_manager import AlgorithmConfigManager
    
    # Import other components
    from three_layer_orchestrator import ThreeLayerOrchestrator  
    from portfolio_execution_manager import PortfolioExecutionManager
    from system_reporter import SystemReporter
    from futures_rollover_manager import FuturesRolloverManager
    # Removed FuturesManager and OptimizedSymbolManager - using QC native methods instead
    
    # Defensive import for AssetFilterManager
    try:
        from asset_filter_manager import AssetFilterManager
    except ImportError:
        # AssetFilterManager will be available from universe.py fallback
        AssetFilterManager = None
    # Removed QCNativeDataAccessor, SimplifiedDataIntegrityChecker, UnifiedDataInterface
    # Using direct QuantConnect/LEAN native methods instead of custom wrappers
    
    # Import utilities
    from universe_helpers import UniverseHelpers
    from futures_helpers import FuturesHelpers
    from warmup_manager import WarmupManager
    # Removed rollover_handler import - using QuantConnect's simple official pattern
    
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
            # CHECK: Verify imports were successful
            if not IMPORTS_SUCCESSFUL:
                self.Error(f"CRITICAL: Import failed during startup: {IMPORT_ERROR}")
                raise Exception(f"Import failure: {IMPORT_ERROR}")
            
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
            self.warmup_manager = WarmupManager(self, self.config_manager)
            self.warmup_manager.setup_enhanced_warmup()
            
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
            
            # STEP 6: Symbol management now handled directly by QC native methods
            # No need for custom symbol management - QC handles this internally
            # Data access now uses direct LEAN methods (self.History, self.Securities, etc.)
            
            # STEP 7: Initialize centralized data validator FIRST (used by all components)
            from src.components.centralized_data_validator import CentralizedDataValidator
            self.data_validator = CentralizedDataValidator(self, self.config_manager)
            
            # STEP 8: Initialize components with centralized config
            self.orchestrator = ThreeLayerOrchestrator(self, self.config_manager)
            
            # CRITICAL: Initialize the orchestrator system (loads strategies)
            if not self.orchestrator.initialize_system():
                self.Error("CRITICAL: Orchestrator system initialization failed")
                raise Exception("Orchestrator initialization failed")
            
            # Initialize bad data position manager BEFORE execution manager
            from src.components.bad_data_position_manager import BadDataPositionManager
            self.bad_data_manager = BadDataPositionManager(self, self.config_manager)
            
            # Initialize execution manager
            self.execution_manager = PortfolioExecutionManager(self, self.config_manager)
            
            # INTEGRATION: Connect bad data manager to execution manager
            self.execution_manager.set_bad_data_manager(self.bad_data_manager)
            
            # SIMPLIFIED: Direct QC-native universe setup (no FuturesManager needed)
            self.Log("MAIN: Setting up futures universe using QC native methods...")
            
            try:
                # Initialize universe directly using QC's native AddFuture
                self.futures_symbols = []
                self._setup_futures_universe()
                
                self.Log(f"MAIN: Universe setup completed with {len(self.futures_symbols)} futures")
                
            except Exception as e:
                self.Error(f"MAIN: Universe setup failed: {str(e)}")
                import traceback
                self.Error(f"MAIN: Full traceback: {traceback.format_exc()}")
                raise
            
            # Initialize performance reporting
            self.system_reporter = SystemReporter(self, self.config_manager)
            
            # Initialize portfolio valuation manager to prevent "accurate price" errors
            from src.components.portfolio_valuation_manager import PortfolioValuationManager
            self.portfolio_valuation_manager = PortfolioValuationManager(
                self, self.config_manager, self.bad_data_manager
            )
            
            # Initialize CRITICAL rollover manager (isolated for safety)
            self.rollover_manager = FuturesRolloverManager(self, self.config_manager)
            
            # Setup scheduling using futures-compatible timing (not market-dependent)
            self._schedule_rebalancing()

            # Initialize tracking variables for defensive programming
            self._warmup_completed = False
            self._first_rebalance_attempted = False
            self._rollover_events_count = 0
            self._algorithm_start_time = self.Time

            # Initialize custom charts for strategy allocation tracking
            self._initialize_custom_charts()
            
            self.Log("=" * 80)
            self.Log("THREE-LAYER CTA PORTFOLIO ALGORITHM INITIALIZED SUCCESSFULLY")
            self.Log("=" * 80)
            
        except Exception as e:
            self.Error(f"CRITICAL ERROR during initialization: {str(e)}")
            raise

    def _initialize_custom_charts(self):
        """Initialize custom charts for tracking strategy allocations and performance."""
        try:
            # Get enabled strategies from configuration
            enabled_strategies = list(self.config_manager.get_enabled_strategies().keys())
            
            # Initialize tracking variables for allocations
            self.strategy_allocations = {strategy: 0.0 for strategy in enabled_strategies}
            self.last_chart_update = None
            
            # Log chart initialization
            self.Log(f"CHARTS: Initialized strategy allocation tracking for {len(enabled_strategies)} strategies: {enabled_strategies}")
            
        except Exception as e:
            self.Error(f"Error initializing custom charts: {str(e)}")

    def _update_strategy_allocation_chart(self, strategy_allocations):
        """Update the strategy allocation chart with current allocations."""
        try:
            if not strategy_allocations:
                return
            
            # Update our tracking
            self.strategy_allocations.update(strategy_allocations)
            self.last_chart_update = self.Time
            
            # Plot each strategy's allocation percentage
            for strategy_name, allocation in strategy_allocations.items():
                # Convert to percentage (0.70 -> 70%)
                allocation_percentage = allocation * 100
                
                # Plot on the "Strategy Allocation" chart
                self.Plot("Strategy Allocation", strategy_name, allocation_percentage)
            
            # Also plot total allocation to verify it sums to 100%
            total_allocation = sum(strategy_allocations.values()) * 100
            self.Plot("Strategy Allocation", "Total", total_allocation)
            
            # Also update portfolio performance chart
            portfolio_value = self.Portfolio.TotalPortfolioValue
            self.Plot("Portfolio Performance", "Total Value", portfolio_value)
            
            # Track cash vs invested
            cash_percentage = (self.Portfolio.Cash / portfolio_value) * 100 if portfolio_value > 0 else 0
            invested_percentage = 100 - cash_percentage
            self.Plot("Portfolio Composition", "Cash %", cash_percentage)
            self.Plot("Portfolio Composition", "Invested %", invested_percentage)
            
            # Log the update (but not too frequently)
            if hasattr(self, '_last_allocation_log_time'):
                if (self.Time - self._last_allocation_log_time).days >= 7:  # Log weekly
                    allocation_str = ", ".join([f"{name}: {alloc:.1%}" for name, alloc in strategy_allocations.items()])
                    self.Log(f"CHARTS: Strategy allocations updated - {allocation_str}")
                    self._last_allocation_log_time = self.Time
            else:
                allocation_str = ", ".join([f"{name}: {alloc:.1%}" for name, alloc in strategy_allocations.items()])
                self.Log(f"CHARTS: Strategy allocations updated - {allocation_str}")
                self._last_allocation_log_time = self.Time
            
        except Exception as e:
            self.Error(f"Error updating strategy allocation chart: {str(e)}")

    def _should_log_component(self, component_name, level):
        """Check if we should log for a component at a given level."""
        try:
            if hasattr(self, 'config_manager') and self.config_manager:
                from config.config_market_strategy import get_log_level_for_component
                component_level = get_log_level_for_component(component_name)
                
                # Simple level comparison (DEBUG < INFO < WARNING < ERROR < CRITICAL)
                levels = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'CRITICAL': 4}
                return levels.get(level, 1) >= levels.get(component_level, 1)
            return True  # Default to logging if config not available
        except:
            return True  # Default to logging if there's any error

    def _setup_futures_universe(self):
        """
        Setup futures universe using QC's TWO-CONTRACT ARCHITECTURE for perfect rollover prices.
        
        ARCHITECTURE: Add both front month (offset=0) and second month (offset=1) continuous contracts.
        This ensures we ALWAYS have rollover prices because the second continuous contract is already
        tracking the "about to be front month" contract with real price data.
        
        GENIUS SOLUTION: When front month rolls A→B, Contract B is already the .Mapped property
        of the second continuous contract, so we can get its price during rollover events.
        """
        try:
            # Check if we should log universe details (based on logging config)
            log_universe_details = self._should_log_component('universe', 'INFO')
            
            if log_universe_details:
                self.Log("UNIVERSE: Setting up TWO-CONTRACT ARCHITECTURE for perfect rollover prices")
            
            # Get futures universe from configuration (completely configurable)
            # Get the raw universe config, not the transformed priority groups
            raw_config = self.config_manager.get_full_config()
            universe_config = raw_config.get('universe', {})
            loading_config = universe_config.get('loading', {})
            simple_selection = universe_config.get('simple_selection', {})
            
            # DEBUG: Log configuration loading details
            self.Log(f"DEBUG: universe_config keys: {list(universe_config.keys()) if universe_config else 'None'}")
            self.Log(f"DEBUG: loading_config: {loading_config}")
            self.Log(f"DEBUG: simple_selection: {simple_selection}")
            
            # Check if simple selection is enabled (easy way to specify exact futures list)
            if simple_selection.get('enabled', False):
                # Use simple futures list (easiest configuration method)
                futures_to_add = simple_selection.get('futures_list', [])
                excluded_symbols = loading_config.get('exclude_problematic_symbols', [])
                
                # Apply exclusions to simple list
                futures_to_add = [symbol for symbol in futures_to_add if symbol not in excluded_symbols]
                
                if log_universe_details:
                    self.Log(f"UNIVERSE: Using SIMPLE SELECTION mode")
                    self.Log(f"UNIVERSE: Simple list: {simple_selection.get('futures_list', [])}")
                    self.Log(f"UNIVERSE: After exclusions: {futures_to_add}")
            else:
                # Use priority-based filtering (advanced configuration method)
                futures_config = universe_config.get('futures', {})
                max_priority = loading_config.get('max_priority', 2)
                excluded_symbols = loading_config.get('exclude_problematic_symbols', [])
                
                # DEBUG: Log priority-based filtering details
                self.Log(f"DEBUG: futures_config keys: {list(futures_config.keys()) if futures_config else 'None'}")
                self.Log(f"DEBUG: max_priority: {max_priority}")
                self.Log(f"DEBUG: excluded_symbols: {excluded_symbols}")
                
                if log_universe_details:
                    self.Log(f"UNIVERSE: Using PRIORITY-BASED filtering (max_priority={max_priority})")
                
                # Build futures list from configuration based on priority
                futures_to_add = []
                
                # Iterate through all categories in futures config
                for category_name, category_futures in futures_config.items():
                    if log_universe_details:
                        self.Log(f"UNIVERSE: Processing category '{category_name}' with {len(category_futures)} futures")
                    
                    for symbol, symbol_config in category_futures.items():
                        symbol_priority = symbol_config.get('priority', 999)  # High number = low priority
                        
                        # Include if priority is within max_priority and not excluded
                        if symbol_priority <= max_priority and symbol not in excluded_symbols:
                            futures_to_add.append(symbol)
                            if log_universe_details:
                                self.Log(f"UNIVERSE: Added {symbol} (priority {symbol_priority}) from {category_name}")
                        else:
                            if log_universe_details:
                                reason = "excluded" if symbol in excluded_symbols else f"priority {symbol_priority} > max {max_priority}"
                                self.Log(f"UNIVERSE: Skipped {symbol} ({reason}) from {category_name}")
                
                # Remove duplicates while preserving order
                seen = set()
                unique_futures = []
                for symbol in futures_to_add:
                    if symbol not in seen:
                        seen.add(symbol)
                        unique_futures.append(symbol)
                futures_to_add = unique_futures
            
            # EMERGENCY FALLBACK: If no futures found, use hardcoded list to prevent crash
            if not futures_to_add:
                self.Log("WARNING: No futures found in configuration! Using emergency fallback list.")
                futures_to_add = ['ES', 'CL', 'GC']  # Emergency fallback to prevent crash
                self.Log(f"WARNING: Emergency fallback futures: {futures_to_add}")
            
            if log_universe_details:
                self.Log(f"UNIVERSE: Priority filtering - max_priority={max_priority}, excluded={excluded_symbols}")
                self.Log(f"UNIVERSE: Final futures list: {futures_to_add}")
                self.Log("UNIVERSE: Adding BOTH front month (offset=0) AND second month (offset=1) contracts")
            
            # Initialize storage for configurable contract depth
            self.futures_symbols = []  # Front month contracts (for trading)
            self.additional_contracts = {}  # Additional contracts by depth (for rollover prices)
            self.symbol_mappings = {}  # Map front month symbol to additional contracts
            
            # Get contract depth configuration
            execution_config = self.config_manager.get_execution_config()
            depth_config = execution_config.get('futures_config', {}).get('contract_depth_config', {})
            
            # Add contracts with configurable depth
            total_added = 0
            for symbol_str in futures_to_add:
                try:
                    # Determine contract depth for this symbol
                    contract_depth = self._get_contract_depth_for_symbol(symbol_str, depth_config)
                    
                    if log_universe_details:
                        self.Log(f"UNIVERSE: {symbol_str} configured for {contract_depth} contract depth")
                    
                    # 1. Always add FRONT MONTH continuous contract (contractDepthOffset=0)
                    front_future = self.AddFuture(
                        symbol_str,
                        Resolution.Daily,
                        dataMappingMode=DataMappingMode.OpenInterest,
                        dataNormalizationMode=DataNormalizationMode.BackwardsRatio,
                        contractDepthOffset=0  # Front month
                    )
                    
                    # Store front month contract
                    self.futures_symbols.append(front_future.Symbol)
                    self.additional_contracts[front_future.Symbol] = []
                    
                    if log_universe_details:
                        self.Log(f"UNIVERSE: Added {symbol_str} FRONT -> {front_future.Symbol}")
                    
                    total_added += 1
                    
                    # 2. Add additional contracts based on configured depth
                    additional_symbols = []
                    for depth_offset in range(1, contract_depth):
                        additional_future = self.AddFuture(
                            symbol_str,
                            Resolution.Daily,
                            dataMappingMode=DataMappingMode.OpenInterest,
                            dataNormalizationMode=DataNormalizationMode.BackwardsRatio,
                            contractDepthOffset=depth_offset
                        )
                        
                        additional_symbols.append(additional_future.Symbol)
                        total_added += 1
                        
                        if log_universe_details:
                            self.Log(f"UNIVERSE: Added {symbol_str} DEPTH-{depth_offset} -> {additional_future.Symbol}")
                    
                    # Store additional contracts and create mappings
                    self.additional_contracts[front_future.Symbol] = additional_symbols
                    
                    # Primary mapping: front month -> second month (for backward compatibility)
                    if len(additional_symbols) > 0:
                        self.symbol_mappings[front_future.Symbol] = additional_symbols[0]  # Second month
                    
                except Exception as e:
                    self.Log(f"UNIVERSE: Failed to add {symbol_str}: {str(e)}")
                    # Continue with other symbols
            
            # Always log summary (even at WARNING level)
            self.Log(f"UNIVERSE: CONFIGURABLE-DEPTH ARCHITECTURE - Successfully added {total_added} contracts")
            self.Log(f"UNIVERSE: {len(self.futures_symbols)} front month + {sum(len(contracts) for contracts in self.additional_contracts.values())} additional contracts")
            
            if log_universe_details:
                self.Log(f"UNIVERSE: Front month symbols: {[str(s) for s in self.futures_symbols]}")
                for front_symbol, additional_symbols in self.additional_contracts.items():
                    if additional_symbols:
                        self.Log(f"UNIVERSE: {front_symbol} -> {len(additional_symbols)} additional: {[str(s) for s in additional_symbols]}")
                self.Log(f"UNIVERSE: Symbol mappings: {len(self.symbol_mappings)} front->second mappings created")
            
            if len(self.futures_symbols) == 0:
                self.Error("UNIVERSE ERROR: No front month futures contracts were added!")
                raise Exception("Universe setup failed - no front month contracts added")
            
            # Validate that we have additional contracts for rollover price tracking
            total_additional = sum(len(contracts) for contracts in self.additional_contracts.values())
            if total_additional == 0:
                self.Log("UNIVERSE WARNING: No additional contracts added - rollover price tracking may be limited")
            
        except Exception as e:
            self.Error(f"UNIVERSE: Failed to setup CONFIGURABLE-DEPTH futures universe: {str(e)}")
            raise

    def _get_contract_depth_for_symbol(self, symbol_str, depth_config):
        """
        Determine the appropriate contract depth for a given futures symbol.
        
        Args:
            symbol_str (str): Futures symbol (e.g., 'ES', 'VX', 'CL')
            depth_config (dict): Contract depth configuration
            
        Returns:
            int: Number of contracts to add (including front month)
        """
        try:
            # Search through each category to find the symbol
            for category, config in depth_config.items():
                if category == 'default':
                    continue
                    
                contracts = config.get('contracts', [])
                if symbol_str in contracts:
                    depth = config.get('depth', 2)
                    self.Log(f"UNIVERSE: {symbol_str} found in {category} category, depth: {depth}")
                    return depth
            
            # Use default if not found in any category
            default_depth = depth_config.get('default', {}).get('depth', 2)
            self.Log(f"UNIVERSE: {symbol_str} using default depth: {default_depth}")
            return default_depth
            
        except Exception as e:
            self.Log(f"UNIVERSE: Error determining depth for {symbol_str}: {str(e)}")
            return 2  # Safe fallback

    def OnWarmupFinished(self):
        """QC's native warm-up completion callback - delegates to WarmupManager."""
        if hasattr(self, 'warmup_manager'):
            self.warmup_manager.on_warmup_finished()
        else:
            self.Log("WARM-UP COMPLETED - System ready for trading")

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
            
            # Use LEAN's native futures chain analysis directly instead of custom universe manager
            if hasattr(slice, 'FuturesChains') and slice.FuturesChains:
                self.Log(f"WARMUP: Processing {len(slice.FuturesChains)} futures chains")
            
            # Update orchestrator with slice during warm-up
            if hasattr(self, 'orchestrator') and self.orchestrator:
                self.orchestrator.update_during_warmup(slice)
            
            # Handle rollover events during warm-up (positions don't exist yet, but track for analysis)
            if hasattr(slice, 'SymbolChangedEvents') and slice.SymbolChangedEvents:
                for symbolChanged in slice.SymbolChangedEvents.Values:
                    self._rollover_events_count += 1
                    # Smart rollover logging: only log first few events and periodic summaries
                    if self._rollover_events_count <= 3:
                        self.Log(f"WARMUP ROLLOVER: {symbolChanged.OldSymbol} -> {symbolChanged.NewSymbol} "
                               f"(Event #{self._rollover_events_count})")
                    elif self._rollover_events_count % 10 == 0:  # Every 10th rollover
                        self.Log(f"WARMUP ROLLOVER SUMMARY: {self._rollover_events_count} total events")
            
            # Periodic warm-up progress logging
            if self.Time.hour == 0 and self.Time.minute == 0:  # Once per day
                if hasattr(self, 'warmup_manager'):
                    self.warmup_manager.log_warmup_progress()
                else:
                    self.Log("WARMUP: In progress")
                
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
            # Store current slice for QC's recommended slice.Contains() validation
            self.current_slice = slice
            
            # CRITICAL: Validate existing positions before any portfolio operations
            # This prevents "security does not have an accurate price" errors
            if hasattr(self, 'portfolio_valuation_manager'):
                validation_results = self.portfolio_valuation_manager.validate_portfolio_before_valuation()
                if not validation_results['can_proceed_with_valuation']:
                    self.Log("WARNING: Portfolio valuation validation failed - skipping this data event")
                    return
            
            # Handle rollover events first (before trading logic)
            if hasattr(slice, 'SymbolChangedEvents') and slice.SymbolChangedEvents:
                for symbolChanged in slice.SymbolChangedEvents.Values:
                    self._rollover_events_count += 1
                    # Smart rollover logging: always log during trading (more important than warmup)
                    # But keep it concise
                    old_ticker = str(symbolChanged.OldSymbol).split()[0] if ' ' in str(symbolChanged.OldSymbol) else str(symbolChanged.OldSymbol)
                    new_ticker = str(symbolChanged.NewSymbol).split()[0] if ' ' in str(symbolChanged.NewSymbol) else str(symbolChanged.NewSymbol)
                    self.Log(f"ROLLOVER: {old_ticker} -> {new_ticker} (#{self._rollover_events_count})")
            
            # Use direct LEAN slice access instead of custom unified data interface
            if hasattr(self, 'orchestrator') and self.orchestrator:
                self.orchestrator.update_with_data(slice)
            
            # Track daily performance for accurate monthly return calculations
            if hasattr(self, 'system_reporter') and self.system_reporter:
                self.system_reporter.generate_daily_performance_update()
                
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
            
            # WEEKLY REBALANCE (ROLLING 7-DAY) ----------------------------------
            # Instead of hard-coding a calendar weekday we trigger every day at 23:30
            # and execute the rebalance only if ≥ 7 days have elapsed since the last.
            # This avoids Friday-holiday skips and ensures daily bars are complete.
            self.Schedule.On(
                self.DateRules.EveryDay(),
                self.TimeRules.At(23, 30),  # 11:30 PM NY – after daily bars are final
                self.MaybeWeeklyRebalance
            )

            # Monthly performance reporting at month end - use end-of-day timing
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
            self.Log("  - Weekly: Rolling 7-day check at 11:30 PM (after daily bar)")
            self.Log("  - Monthly: Month end at 11:45 PM (after daily bar completion)")
            self.Log("  - Daily: Contract validation at 11:00 PM (after daily bar completion)")
            self.Log("  - PREVENTS STALE FILLS: Uses actual daily closing prices, not previous day's prices")
            self.Log("  - Daily bars complete around 6 PM for futures, scheduling at 11+ PM ensures fresh data")
            
        except Exception as e:
            self.Error(f"Failed to schedule rebalancing: {str(e)}")
    
    def _emergency_fallback_initialization(self):
        """Emergency fallback - NO TRADING CONFIGURATION FALLBACKS"""
        self.Error("CRITICAL: Configuration failed - Algorithm will NOT trade")
        
        try:
            # Initialize tracking variables
            self._warmup_completed = False
            self._first_rebalance_attempted = False
            self._rollover_events_count = 0
            self._algorithm_start_time = self.Time
            self.futures_symbols = []
            self.config = None
            self.config_manager = None
            
            # Create minimal component stubs
            class CrashPrevention:
                def __init__(self, algorithm): self.algorithm = algorithm
                def OnSecuritiesChanged(self, changes): pass
                def update_with_data(self, slice): pass
                def update_during_warmup(self, slice): pass
                def generate_portfolio_targets(self): return {'status': 'failed'}
                def execute_rebalance_result(self, result): return {'status': 'failed'}
                def track_rebalance_performance(self, result): pass
                def generate_monthly_performance_report(self): pass
                def generate_final_algorithm_report(self): 
                    self.algorithm.Log("SECURITY: No trading occurred due to configuration failure")
            
            # Initialize crash prevention components
            fallback = CrashPrevention(self)
            self.orchestrator = fallback
            self.execution_manager = fallback
            self.system_reporter = fallback
            
            # Set minimal QC parameters
            try:
                self.SetStartDate(2020, 1, 1)
                self.SetEndDate(2020, 1, 2)
                self.SetCash(1000000)
                self.Log("Emergency fallback: Minimal QC parameters set")
            except Exception as qc_e:
                self.Error(f"Failed to set minimal QC parameters: {str(qc_e)}")
            
        except Exception as e:
            self.Error(f"Emergency fallback failed: {str(e)}")
    
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
        """Handle futures rollover using dedicated rollover manager."""
        try:
            # Delegate to the CRITICAL rollover manager component
            self.rollover_manager.handle_symbol_changed_events(symbolChangedEvents)
            
            # Update local tracking for backward compatibility
            self._rollover_events_count = getattr(self, '_rollover_events_count', 0) + len(symbolChangedEvents)
                
        except Exception as e:
            self.Error(f"Error in OnSymbolChangedEvents: {str(e)}")
    
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
                # Get monitoring configuration for warmup logging
                monitoring_config = self.config_manager.get_execution_config().get('monitoring', {})
                progress_frequency = monitoring_config.get('warmup_progress_frequency_days', 90)
                
                # Only log rebalance skip periodically during warmup to reduce noise
                should_log_skip = False
                if progress_frequency > 0:
                    if not hasattr(self, '_last_rebalance_skip_log'):
                        self._last_rebalance_skip_log = self.Time
                        should_log_skip = True
                    elif (self.Time - self._last_rebalance_skip_log).days >= (progress_frequency / 4):  # Log rebalance skips more frequently than monthly
                        self._last_rebalance_skip_log = self.Time
                        should_log_skip = True
                
                if should_log_skip:
                    warmup_info = self._get_warmup_status()
                    self.Log(f"REBALANCE SKIPPED: Still warming up - {warmup_info}")
                return
                
            self.Log("="*50)
            self.Log("EXECUTING WEEKLY REBALANCE")
            self.Log("="*50)
            
            # CRITICAL: Validate existing positions before rebalancing
            # This prevents "security does not have an accurate price" errors during portfolio operations
            if hasattr(self, 'portfolio_valuation_manager'):
                validation_results = self.portfolio_valuation_manager.validate_portfolio_before_valuation()
                if not validation_results['can_proceed_with_valuation']:
                    self.Log("WARNING: Portfolio validation failed - skipping rebalance to prevent errors")
                    return
            
            # Generate portfolio targets through three-layer process
            rebalance_result = self.orchestrator.weekly_rebalance()
            
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
        """Generate monthly performance reports with QC mismatch detection."""
        try:
            # Get monitoring configuration
            monitoring_config = self.config_manager.get_execution_config().get('monitoring', {})
            skip_during_warmup = monitoring_config.get('skip_reports_during_warmup', True)
            progress_frequency = monitoring_config.get('warmup_progress_frequency_days', 90)
            
            # Skip monthly reports during warmup to reduce logging noise
            if self.IsWarmingUp and skip_during_warmup:
                # Only log warmup progress periodically if configured
                if progress_frequency > 0:
                    if hasattr(self, '_last_warmup_monthly_log'):
                        days_since_last_log = (self.Time - self._last_warmup_monthly_log).days
                        if days_since_last_log < progress_frequency:
                            return
                    
                    self._last_warmup_monthly_log = self.Time
                    warmup_config = self.config_manager.get_algorithm_config().get('warmup', {})
                    warmup_days = warmup_config.get('calculated_days', 831)
                    
                    # Estimate warmup progress (rough calculation)
                    start_time = self.StartDate
                    # Ensure we never show negative progress when Time < StartDate (during historical warm-up)
                    elapsed_days = max(0, (self.Time - start_time).days)
                    progress_pct = min(100, (elapsed_days / warmup_days) * 100)
                    
                    self.Log(f"WARMUP PROGRESS: {progress_pct:.1f}% ({elapsed_days}/{warmup_days} days) - Monthly reports disabled during warmup")
                return
            
            if hasattr(self, 'system_reporter'):
                self.system_reporter.generate_monthly_performance_report()
            
            # CRITICAL: Add QC equity vs return mismatch detection
            current_portfolio_value = self.Portfolio.TotalPortfolioValue
            initial_capital = 10000000  # From config
            
            # Calculate our own return for comparison with QC's charts
            our_calculated_return = (current_portfolio_value - initial_capital) / initial_capital
            
            # Get QC's built-in statistics if available
            qc_return = 0.0
            try:
                if hasattr(self, 'Statistics') and 'Total Return' in self.Statistics:
                    qc_return = float(self.Statistics['Total Return'].replace('%', '')) / 100.0
            except:
                pass
            
            # Detect significant mismatches (>1% difference)
            return_mismatch = abs(our_calculated_return - qc_return)
            if return_mismatch > 0.01:  # 1% threshold
                self.Log(f"CRITICAL: QC MISMATCH DETECTED")
                self.Log(f"  Portfolio Value: ${current_portfolio_value:,.0f}")
                self.Log(f"  Our Calculated Return: {our_calculated_return:.2%}")
                self.Log(f"  QC Statistics Return: {qc_return:.2%}")
                self.Log(f"  Mismatch: {return_mismatch:.2%}")
                
                # Log position details during mismatch
                self.Log(f"  Active Positions during mismatch:")
                for holding in self.Portfolio.Values:
                    if holding.Invested:
                        symbol_str = str(holding.Symbol).split()[0]
                        price = holding.Price
                        quantity = holding.Quantity
                        value = holding.HoldingsValue
                        self.Log(f"    {symbol_str}: {quantity} @ ${price:.2f} = ${value:,.0f}")
            
            # Quick validation using centralized validator
            valid_symbols = 0
            current_slice = getattr(self, 'current_slice', None)
            
            for symbol in self.futures_symbols:
                if hasattr(self, 'data_validator'):
                    validation_result = self.data_validator.validate_symbol_for_trading(symbol, current_slice)
                    if validation_result['is_valid']:
                        valid_symbols += 1
                else:
                    # Fallback validation
                    if symbol in self.Securities and self.Securities[symbol].HasData:
                        valid_symbols += 1
            
            # Check for active positions
            total_positions = sum(1 for holding in self.Portfolio.Values if holding.Invested)
            
            # Only log monthly report if there are issues or positions
            issues_detected = valid_symbols < len(self.futures_symbols)
            
            if issues_detected or total_positions > 0 or return_mismatch > 0.01:
                self.Log("================================================================================")
                self.Log("MONTHLY PERFORMANCE REPORT")
                self.Log("================================================================================")
                self.Log(f"Portfolio Value: ${self.Portfolio.TotalPortfolioValue:,.0f}")
                self.Log(f"Active Positions: {total_positions}")
                self.Log(f"Valid Symbols: {valid_symbols}/{len(self.futures_symbols)}")
                
                # Only show position details if there are positions
                if total_positions > 0:
                    for holding in self.Portfolio.Values:
                        if holding.Invested:
                            symbol_str = str(holding.Symbol).split()[0] if ' ' in str(holding.Symbol) else str(holding.Symbol)
                            weight = holding.HoldingsValue / self.Portfolio.TotalPortfolioValue
                            self.Log(f"  {symbol_str}: {weight:.1%} (${holding.HoldingsValue:,.0f})")
                
                if issues_detected:
                    invalid_count = len(self.futures_symbols) - valid_symbols
                    self.Log(f"WARNING: {invalid_count} symbols have data issues")
            
        except Exception as e:
            self.Error(f"MonthlyReporting error: {str(e)}")
    
    def ValidateContinuousContracts(self):
        """Daily validation using centralized data validator."""
        try:
            valid_count = 0
            invalid_count = 0
            critical_issues = []
            current_slice = getattr(self, 'current_slice', None)
            
            for symbol in self.futures_symbols:
                # Use centralized validator with slice data
                validation_result = self.data_validator.validate_symbol_for_trading(symbol, current_slice)
                
                if validation_result['is_valid']:
                    valid_count += 1
                else:
                    invalid_count += 1
                    # Only track critical issues
                    if validation_result['reason'] in ['symbol_not_in_securities', 'no_data', 'invalid_price', 'no_current_slice_data']:
                        critical_issues.append(f"{symbol}: {validation_result['reason']}")
            
            # Only log if there are critical issues or it's post-warmup with problems
            if critical_issues:
                self.Log(f"CRITICAL VALIDATION ISSUES: {len(critical_issues)} symbols")
                for issue in critical_issues[:3]:  # Limit to first 3
                    self.Log(f"  {issue}")
                if len(critical_issues) > 3:
                    self.Log(f"  ... and {len(critical_issues) - 3} more")
            
            # Smart logging: only log if there are issues or significant changes
            if not hasattr(self, '_last_validation_counts'):
                self._last_validation_counts = {'valid': 0, 'invalid': 0, 'last_log_time': None}
            
            # Determine if we should log
            should_log_validation = False
            log_reason = ""
            
            # Always log critical issues
            if critical_issues:
                should_log_validation = True
                log_reason = "critical_issues"
            
            # Log if validation counts changed
            elif (valid_count != self._last_validation_counts['valid'] or 
                  invalid_count != self._last_validation_counts['invalid']):
                should_log_validation = True
                log_reason = "count_change"
            
            # Log if no valid contracts (critical situation)
            elif valid_count == 0 and not self.IsWarmingUp:
                should_log_validation = True
                log_reason = "no_valid_contracts"
            
            # Periodic summary (once per week) for healthy validation
            elif (not self._last_validation_counts['last_log_time'] or 
                  (self.Time - self._last_validation_counts['last_log_time']).days >= 7):
                if valid_count > 0 and invalid_count == 0 and not self.IsWarmingUp:
                    should_log_validation = True
                    log_reason = "weekly_healthy_summary"
                    self._last_validation_counts['last_log_time'] = self.Time
            
            if should_log_validation:
                if log_reason == "no_valid_contracts":
                    self.Log("CRITICAL: No valid contracts - trading will not occur!")
                elif log_reason == "weekly_healthy_summary":
                    self.Log(f"Contract validation healthy: {valid_count} valid, all contracts operational")
                else:
                    self.Log(f"Contract validation: {valid_count} valid, {invalid_count} invalid")
            
            # Update tracking
            self._last_validation_counts.update({
                'valid': valid_count,
                'invalid': invalid_count
            })
            
        except Exception as e:
            self.Error(f"ValidateContinuousContracts error: {str(e)}")
    
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
            
            # Log rollover manager statistics if available
            if hasattr(self, 'rollover_manager'):
                rollover_stats = self.rollover_manager.get_rollover_statistics()
                self.Log(f"  Rollover Manager Statistics:")
                self.Log(f"    Total rollover events: {rollover_stats.get('total_rollover_events', 0)}")
                self.Log(f"    Rollover history entries: {rollover_stats.get('rollover_history_length', 0)}")
                if rollover_stats.get('last_rollover'):
                    last_rollover = rollover_stats['last_rollover']
                    self.Log(f"    Last rollover: {last_rollover['old_symbol']} -> {last_rollover['new_symbol']} on {last_rollover['timestamp']}")
            
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

    # ---------------------------------------------------------------------------
    # Rolling-7-day rebalance helper – executes WeeklyRebalance only when needed
    # ---------------------------------------------------------------------------
    def MaybeWeeklyRebalance(self):
        """Trigger WeeklyRebalance if ≥7 days have passed since the last run."""
        try:
            # Initialise tracking attribute if missing (defensive for warm-start runs)
            if not hasattr(self, '_last_weekly_rebalance'):
                self._last_weekly_rebalance = None

            if (self._last_weekly_rebalance is None or
                (self.Time.date() - self._last_weekly_rebalance).days >= 7):
                self.WeeklyRebalance()
                # Record date regardless of success; skipped weeks will be retried next day
                self._last_weekly_rebalance = self.Time.date()
        except Exception as e:
            self.Error(f"MaybeWeeklyRebalance error: {str(e)}") 