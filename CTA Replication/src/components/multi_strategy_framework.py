# multi_strategy_framework.py
"""
Updated Multi-Strategy Framework for Three-Layer Architecture

This framework now supports:
- Layer 1: Individual "naive" strategies 
- Layer 2: Dynamic strategy allocation (handled by main.py)
- Layer 3: Portfolio risk management (handled by main.py)

The framework focuses on shared infrastructure and basic coordination.

FIXED VERSION: Correct imports to match actual universe.py structure
"""

from AlgorithmImports import *
import numpy as np
# Import both classes from universe.py to match your actual structure
from universe import AssetFilterManager, FuturesManager
# Note: futuresroll.py is not being used in the new architecture


class MultiStrategyFuturesAlgorithm(QCAlgorithm):
    """
    Multi-strategy futures algorithm framework
    
    Provides shared infrastructure for three-layer portfolio system:
    - Futures universe management via AssetFilterManager
    - Basic algorithm setup and configuration
    - Shared scheduling and monitoring infrastructure
    
    Layer 2 and Layer 3 functionality is handled by main.py
    The new architecture uses OnSymbolChangedEvents for rollover handling
    """

    def Initialize(self):
        """
        Basic algorithm initialization - will be extended by main.py
        
        This provides minimal setup. The main.py will override this method
        to implement the full three-layer system with config-driven setup.
        """
        # 1) Basic algorithm setup (will be overridden by config in main.py)
        self.SetStartDate(2015, 1, 1)
        self.SetEndDate(2020, 12, 31)
        self.SetCash(10_000_000)
        
        # Store starting cash for health checks
        self.starting_cash = 10_000_000
        
        # Settings for futures trading
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0
        self.Settings.RebalancePortfolioOnSecurityChanges = False
        self.Settings.RebalancePortfolioOnInsightChanges = False

        # 2) Initialize shared universe manager (minimal setup)
        # NOTE: main.py will create a proper config-driven FuturesManager
        # This is just a fallback for the base class
        try:
            # For the base class, we'll skip universe manager initialization
            # and let main.py handle it completely. This avoids conflicts.
            self.universe_manager = None
            self.Log("Base class: Universe manager initialization deferred to main.py")
            
        except Exception as e:
            self.Log(f"WARNING: Could not initialize base universe manager: {str(e)}")
            self.universe_manager = None

        # 3) Initialize strategies placeholder (Layer 1)
        # This will be overridden by main.py to setup the three-layer system
        self.strategies = {}
        self._initialize_strategies()

        # 4) Basic warmup (will be overridden by config in main.py)
        self.SetWarmUp(timedelta(days=80), Resolution.Daily)

        # 5) Schedule basic rebalancing and monitoring
        self._setup_basic_scheduling()

        self.Log(f"Multi-strategy framework base class initialized")

    def _initialize_strategies(self):
        """
        Initialize strategies - override this in your main.py
        
        This base implementation does nothing. Your main.py should override this
        method to configure the three-layer system (strategies, allocator, risk manager).
        """
        # Base class does nothing - main.py will override this
        pass

    def _setup_basic_scheduling(self):
        """
        Setup basic scheduled events - main.py will override with full scheduling
        
        This provides minimal scheduling for the base class only.
        """
        
        # Basic weekly check (will be overridden by main.py)
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Friday),
            self.TimeRules.At(14, 0),
            self.BasicWeeklyCheck
        )
        
        # Basic daily monitoring
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(16, 30),
            self.BasicDailyCheck
        )

    def OnData(self, slice: Slice):
        """
        Handle incoming data - basic implementation
        
        Main.py will override this with full rollover handling and data processing.
        """
        
        # Only update universe manager if it exists and has the method
        if (self.universe_manager and 
            hasattr(self.universe_manager, 'update_data_quality')):
            try:
                self.universe_manager.update_data_quality(slice)
            except Exception as e:
                # Silently handle errors in base class to avoid spam
                pass
        
        # Handle symbol changed events for rollover (basic handling)
        if slice.SymbolChangedEvents:
            self._handle_basic_symbol_changes(slice.SymbolChangedEvents)
            
            # Also update rollover status if universe manager supports it
            if (self.universe_manager and 
                hasattr(self.universe_manager, 'update_rollover_status')):
                try:
                    self.universe_manager.update_rollover_status(slice)
                except Exception as e:
                    # Silently handle errors in base class
                    pass

    def _handle_basic_symbol_changes(self, symbol_changed_events):
        """
        Basic symbol change handling - main.py will implement full rollover logic
        """
        if not symbol_changed_events:
            return
            
        # Just log the events for the base class (FIX: Handle KeyValuePair correctly)
        for event in symbol_changed_events:
            # Access the KeyValuePair correctly - get the Value which is the SymbolChangedEvent
            event_obj = event.Value
            self.Log(f"SYMBOL CHANGE: {event_obj.OldSymbol} → {event_obj.NewSymbol}")

    def BasicWeeklyCheck(self):
        """
        Basic weekly check - override in main.py for full three-layer rebalancing
        """
        if self.IsWarmingUp:
            return
        
        self.Log(f"=== BASIC WEEKLY CHECK {self.Time.date()} ===")
        
        # Basic portfolio status
        portfolio_value = self.Portfolio.TotalPortfolioValue
        positions = [h for h in self.Portfolio.Values if h.Invested]
        
        self.Log(f"Portfolio: ${portfolio_value:,.0f}, {len(positions)} positions")
        
        # This is where main.py will implement the three-layer rebalancing

    def BasicDailyCheck(self):
        """
        Basic daily monitoring - main.py will enhance with full risk management
        """
        if self.IsWarmingUp:
            return
            
        portfolio_value = self.Portfolio.TotalPortfolioValue
        
        # Basic monitoring around known problem dates
        if self.Time.year == 2019 and self.Time.month >= 10:
            margin_used = self.Portfolio.TotalMarginUsed
            cash = self.Portfolio.Cash
            
            self.Log(f"DAILY CHECK {self.Time.date()}: PV=${portfolio_value:,.0f}, "
                    f"Cash=${cash:,.0f}, Margin=${margin_used:,.0f}")

    def OnEndOfAlgorithm(self):
        """Basic final reporting - main.py will enhance with comprehensive analysis"""
        self.Log(f"\n=== BASIC ALGORITHM COMPLETION ===")
        
        final_value = self.Portfolio.TotalPortfolioValue
        total_return = (final_value - self.starting_cash) / self.starting_cash
        
        self.Log(f"Final Portfolio Value: ${final_value:,.0f}")
        self.Log(f"Total Return: {total_return:.2%}")

    # =============================================================================
    # UTILITY METHODS - For compatibility and base functionality
    # =============================================================================

    def add_futures_contract(self, symbol, data_mapping_mode=None, data_normalization_mode=None):
        """
        Add a futures contract with proper configuration
        
        This is a utility method for manual contract addition.
        The main universe management is handled by AssetFilterManager.
        """
        try:
            # Use provided modes or defaults
            mapping_mode = data_mapping_mode or DataMappingMode.OpenInterest
            normalization_mode = data_normalization_mode or DataNormalizationMode.BackwardsRatio
            
            # Add the futures contract
            future = self.AddFuture(
                symbol,
                dataMappingMode=mapping_mode,
                dataNormalizationMode=normalization_mode,
                contractDepthOffset=0,
                extendedMarketHours=True
            )
            
            # Set contract filter for 6 months of contracts
            future.SetFilter(timedelta(0), timedelta(182))
            
            self.Log(f"Added futures contract: {symbol}")
            return future
            
        except Exception as e:
            self.Log(f"ERROR adding futures contract {symbol}: {str(e)}")
            return None

    def get_portfolio_summary(self):
        """Get basic portfolio summary"""
        portfolio_value = float(self.Portfolio.TotalPortfolioValue)
        cash = float(self.Portfolio.Cash)
        margin_used = float(self.Portfolio.TotalMarginUsed)
        
        positions = [h for h in self.Portfolio.Values if h.Invested]
        total_holdings_value = sum(abs(float(h.HoldingsValue)) for h in positions)
        gross_exposure = total_holdings_value / portfolio_value if portfolio_value > 0 else 0
        
        return {
            'portfolio_value': portfolio_value,
            'cash': cash,
            'margin_used': margin_used,
            'position_count': len(positions),
            'gross_exposure': gross_exposure,
            'timestamp': self.Time
        }

    def get_position_summary(self):
        """Get detailed position summary"""
        positions = []
        for holding in self.Portfolio.Values:
            if holding.Invested:
                positions.append({
                    'symbol': str(holding.Symbol),
                    'quantity': float(holding.Quantity),
                    'holdings_value': float(holding.HoldingsValue),
                    'unrealized_pnl': float(holding.UnrealizedProfit),
                    'average_price': float(holding.AveragePrice),
                    'market_price': float(holding.Price)
                })
        
        return {
            'positions': positions,
            'total_positions': len(positions),
            'timestamp': self.Time
        }

    def emergency_liquidate_all(self, reason="manual_trigger"):
        """Emergency liquidation of all positions"""
        try:
            self.Log(f"=== EMERGENCY LIQUIDATION: {reason} ===")
            
            positions = [h for h in self.Portfolio.Values if h.Invested]
            if not positions:
                self.Log("No positions to liquidate")
                return True
            
            liquidation_orders = []
            for holding in positions:
                try:
                    order = self.Liquidate(holding.Symbol, tag=f"emergency_{reason}")
                    if order:
                        liquidation_orders.append(order)
                        self.Log(f"EMERGENCY LIQUIDATE: {holding.Symbol} - {holding.Quantity} contracts")
                except Exception as e:
                    self.Log(f"ERROR liquidating {holding.Symbol}: {str(e)}")
            
            self.Log(f"Emergency liquidation: {len(liquidation_orders)} orders placed")
            return len(liquidation_orders) > 0
            
        except Exception as e:
            self.Error(f"CRITICAL ERROR in emergency liquidation: {str(e)}")
            return False

    def validate_algorithm_health(self):
        """Basic algorithm health validation"""
        try:
            health_status = {
                'timestamp': self.Time,
                'is_warming_up': self.IsWarmingUp,
                'algorithm_healthy': True,
                'issues': []
            }
            
            # Check portfolio value
            portfolio_value = float(self.Portfolio.TotalPortfolioValue)
            if portfolio_value <= 0:
                health_status['algorithm_healthy'] = False
                health_status['issues'].append('zero_or_negative_portfolio_value')
            
            # Check for excessive margin usage
            margin_used = float(self.Portfolio.TotalMarginUsed)
            if margin_used > portfolio_value * 0.95:  # 95% margin usage
                health_status['issues'].append('excessive_margin_usage')
            
            # Check cash levels
            cash = float(self.Portfolio.Cash)
            if cash < 0:
                health_status['issues'].append('negative_cash')
            
            # Check for stuck positions (positions with zero market value)
            stuck_positions = []
            for holding in self.Portfolio.Values:
                if holding.Invested and holding.Price <= 0:
                    stuck_positions.append(str(holding.Symbol))
            
            if stuck_positions:
                health_status['issues'].append(f'stuck_positions: {stuck_positions}')
            
            # Check universe manager health
            if self.universe_manager:
                try:
                    if hasattr(self.universe_manager, 'get_universe_health'):
                        universe_health = self.universe_manager.get_universe_health()
                        if not universe_health.get('healthy', True):
                            health_status['issues'].append('universe_manager_unhealthy')
                except:
                    health_status['issues'].append('universe_manager_error')
            
            # Set overall health
            if health_status['issues']:
                health_status['algorithm_healthy'] = False
            
            return health_status
            
        except Exception as e:
            self.Error(f"ERROR validating algorithm health: {str(e)}")
            return {
                'timestamp': self.Time,
                'algorithm_healthy': False,
                'issues': [f'health_check_error: {str(e)}']
            }

    def log_system_status(self, context="BASIC"):
        """Log basic system status"""
        try:
            self.Log(f"=== SYSTEM STATUS ({context}) ===")
            
            # Portfolio summary
            portfolio_summary = self.get_portfolio_summary()
            self.Log(f"Portfolio: ${portfolio_summary['portfolio_value']:,.0f}")
            self.Log(f"Cash: ${portfolio_summary['cash']:,.0f}")
            self.Log(f"Positions: {portfolio_summary['position_count']}")
            self.Log(f"Gross Exposure: {portfolio_summary['gross_exposure']:.1%}")
            
            # Health check
            health = self.validate_algorithm_health()
            health_status = "HEALTHY" if health['algorithm_healthy'] else "ISSUES"
            self.Log(f"Health: {health_status}")
            
            if health['issues']:
                self.Log(f"Issues: {', '.join(health['issues'])}")
            
            # Universe status
            if self.universe_manager and hasattr(self.universe_manager, 'get_config_summary'):
                try:
                    universe_summary = self.universe_manager.get_config_summary()
                    self.Log(f"Universe: {universe_summary.get('universe_size', 0)} contracts")
                except:
                    self.Log("Universe: Status unavailable")
            else:
                self.Log("Universe: Not initialized in base class")
            
            self.Log(f"=== END STATUS ({context}) ===")
            
        except Exception as e:
            self.Error(f"ERROR logging system status: {str(e)}")

    # =============================================================================
    # LEGACY COMPATIBILITY METHODS
    # =============================================================================

    def add_strategy(self, strategy_name, strategy_class, config):
        """
        Legacy method for adding strategies - kept for compatibility
        
        Note: The new three-layer system uses dynamic strategy loading
        through the config system, not direct strategy addition.
        """
        self.Log(f"WARNING: add_strategy is deprecated. Use config-driven strategy loading in main.py")
        return False
    
    def remove_strategy(self, strategy_name):
        """
        Legacy method for removing strategies - kept for compatibility
        """
        self.Log(f"WARNING: remove_strategy is deprecated. Use config-driven strategy management in main.py")
        return False
    
    def get_strategy_status(self):
        """
        Legacy method for strategy status - kept for compatibility
        """
        return {
            'note': 'Strategy status now handled by main.py three-layer system',
            'legacy_strategies': len(self.strategies)
        }
