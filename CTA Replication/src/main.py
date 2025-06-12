# main_core.py - Streamlined Main Algorithm for QuantConnect Cloud
"""
Three-Layer CTA Portfolio System - Streamlined Version
Optimized for QuantConnect Cloud deployment (under 64KB limit)

This is the main algorithm entry point that coordinates all components
without containing the bulk implementation details.
"""

from AlgorithmImports import *

# Import core components from organized folders
from components.algorithm_config_manager import AlgorithmConfigManager
from components.three_layer_orchestrator import ThreeLayerOrchestrator  
from components.portfolio_execution_manager import PortfolioExecutionManager
from components.system_reporter import SystemReporter
from universe import FuturesManager

# Import fallback implementations
from utils.fallback_implementations import FallbackComponents
from utils.universe_helpers import UniverseHelpers
from utils.futures_helpers import FuturesHelpers

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
            except Exception as e:
                self.Log(f"Using fallback configuration: {str(e)}")
                fallback = FallbackComponents()
                self.config = fallback.create_fallback_config()
                self.config_manager = fallback.create_fallback_config_manager(self.config)
            
            # Step 3: Initialize universe management
            self.Log("Step 3: Initializing universe management...")
            try:
                # Create universe manager without storing complex objects
                universe_helper = UniverseHelpers(self, self.config_manager)
                self.universe_data = universe_helper.initialize_universe()
                self.Log("SUCCESS: Universe management initialized")
            except Exception as e:
                self.Log(f"Using minimal universe management: {str(e)}")
                self.universe_data = {'tickers': ['ES', 'NQ', 'ZN'], 'symbols': self.futures_symbols}
            
            # Step 4: Initialize orchestrator
            self.Log("Step 4: Initializing three-layer orchestrator...")
            try:
                self.orchestrator = ThreeLayerOrchestrator(self, self.config_manager, self.universe_data)
                self.orchestrator.initialize_system()
                self.Log("SUCCESS: Orchestrator initialized")
            except Exception as e:
                self.Log(f"Using minimal orchestrator: {str(e)}")
                fallback = FallbackComponents()
                self.orchestrator = fallback.create_minimal_orchestrator(self)
            
            # Step 5: Initialize execution manager
            self.Log("Step 5: Initializing execution manager...")
            try:
                self.execution_manager = PortfolioExecutionManager(self, self.config_manager, self.universe_data)
                self.Log("SUCCESS: Execution manager initialized")
            except Exception as e:
                self.Log(f"Using minimal execution manager: {str(e)}")
                fallback = FallbackComponents()
                self.execution_manager = fallback.create_minimal_execution_manager(self)
            
            # Step 6: Initialize system reporter
            self.Log("Step 6: Initializing system reporter...")
            try:
                self.system_reporter = SystemReporter(self)
                self.Log("SUCCESS: System reporter initialized")
            except Exception as e:
                self.Log(f"Using minimal system reporter: {str(e)}")
                fallback = FallbackComponents()
                self.system_reporter = fallback.create_minimal_system_reporter(self)
            
            # Step 7: Schedule rebalancing
            self.Log("Step 7: Scheduling rebalancing...")
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
            # Weekly rebalancing on Fridays at 4 PM
            self.Schedule.On(
                self.DateRules.WeekEnd(), 
                self.TimeRules.At(16, 0), 
                self.WeeklyRebalance
            )
            
            # Monthly performance reporting
            self.Schedule.On(
                self.DateRules.MonthEnd(),
                self.TimeRules.At(17, 0),
                self.MonthlyReporting
            )
            
            self.Log("Rebalancing schedule configured: Weekly + Monthly")
            
        except Exception as e:
            self.Error(f"Failed to schedule rebalancing: {str(e)}")
    
    def _emergency_fallback_initialization(self):
        """Emergency fallback if main initialization fails."""
        self.Log("EMERGENCY FALLBACK: Initializing minimal system...")
        
        try:
            fallback = FallbackComponents()
            self.config = fallback.create_fallback_config()
            self.orchestrator = fallback.create_minimal_orchestrator(self)
            self.execution_manager = fallback.create_minimal_execution_manager(self)
            self.system_reporter = fallback.create_minimal_system_reporter(self)
            
            # Try to add at least one future
            try:
                future = self.AddFuture('ES')
                future.SetFilter(0, 182)
                self.futures_symbols = [future.Symbol]
                self.Log("Emergency fallback: Added ES future")
            except:
                self.futures_symbols = []
            
            self.Log("Emergency fallback initialization complete")
            
        except Exception as e:
            self.Error(f"Emergency fallback failed: {str(e)}")
    
    def OnSymbolChangedEvents(self, symbolChangedEvents):
        """Handle futures rollover events."""
        try:
            if hasattr(self, 'execution_manager'):
                self.execution_manager.handle_rollover_events(symbolChangedEvents)
            self._rollover_events_count += len(symbolChangedEvents)
        except Exception as e:
            self.Error(f"Error handling rollover events: {str(e)}")
    
    def OnData(self, slice):
        """Handle market data updates."""
        try:
            if self.IsWarmingUp:
                if not self._warmup_completed:
                    # Update components during warmup
                    if hasattr(self, 'orchestrator'):
                        self.orchestrator.update_during_warmup(slice)
                return
            
            if not self._warmup_completed:
                self._warmup_completed = True
                self.Log("Warmup completed - System ready for trading")
            
            # Update all components with new data
            if hasattr(self, 'orchestrator'):
                self.orchestrator.update_with_data(slice)
                
        except Exception as e:
            self.Error(f"Error in OnData: {str(e)}")
    
    def WeeklyRebalance(self):
        """Execute weekly portfolio rebalancing."""
        try:
            if self.IsWarmingUp:
                return
                
            self.Log("="*50)
            self.Log("EXECUTING WEEKLY REBALANCE")
            self.Log("="*50)
            
            # Generate portfolio targets through three-layer process
            targets = self.orchestrator.generate_portfolio_targets()
            
            if targets:
                # Execute the rebalance
                result = self.execution_manager.execute_portfolio_rebalance(targets)
                
                # Track performance
                self.system_reporter.track_rebalance_performance(result)
                
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
            
        except Exception as e:
            self.Error(f"Error in monthly reporting: {str(e)}")
    
    def OnEndOfAlgorithm(self):
        """Generate final algorithm report."""
        try:
            self.Log("="*80)
            self.Log("ALGORITHM ENDING - GENERATING FINAL REPORT")
            self.Log("="*80)
            
            if hasattr(self, 'system_reporter'):
                self.system_reporter.generate_final_algorithm_report()
            
            # Log final system metrics
            self.Log(f"Final system metrics:")
            self.Log(f"  Rollover events handled: {self._rollover_events_count}")
            self.Log(f"  First rebalance attempted: {self._first_rebalance_attempted}")
            self.Log(f"  Futures symbols tracked: {len(self.futures_symbols)}")
            
        except Exception as e:
            self.Error(f"Error in OnEndOfAlgorithm: {str(e)}") 