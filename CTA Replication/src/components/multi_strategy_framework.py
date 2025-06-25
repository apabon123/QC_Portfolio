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
from universe import FuturesManager
from asset_filter_manager import AssetFilterManager
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

        # 4) Warmup will be configured by main.py using strategy-specific requirements
        # Do not set warmup here - let main.py handle it with proper strategy requirements

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
        Handle symbol changes using QC's built-in rollover management.
        This is much cleaner than manual symbol matching.
        """
        if not symbol_changed_events:
            return
        
        self.Log(f"=== FUTURES ROLLOVER EVENTS ===")
        
        for event in symbol_changed_events:
            # Access the KeyValuePair correctly - get the Value which is the SymbolChangedEvent
            event_obj = event.Value
            old_symbol = event_obj.OldSymbol
            new_symbol = event_obj.NewSymbol
            
            self.Log(f"ROLLOVER: {old_symbol} → {new_symbol}")
            
            # Get current position in the old contract
            old_holding = self.Portfolio[old_symbol]
            if old_holding.Invested:
                quantity = old_holding.Quantity
                
                # QC's recommended rollover approach: liquidate old, buy new
                self.Log(f"  Rolling {quantity:+.0f} contracts from {old_symbol} to {new_symbol}")
                
                # Liquidate the old contract
                self.Liquidate(old_symbol, tag=f"Rollover_Liquidate_{self.Time.strftime('%Y%m%d')}")
                
                # Immediately establish position in new contract
                if quantity != 0:
                    self.MarketOrder(new_symbol, quantity, tag=f"Rollover_Establish_{self.Time.strftime('%Y%m%d')}")
                
                self.Log(f"  ✓ Rollover completed: {old_symbol} → {new_symbol}")
            else:
                self.Log(f"  No position to roll for {old_symbol}")

    def OnSymbolChangedEvents(self, symbol_changed_events):
        """
        QC's built-in rollover event handler.
        This is called automatically when continuous contracts roll over.
        """
        try:
            self._handle_basic_symbol_changes(symbol_changed_events)
            
            # If we have additional managers that need to know about rollovers, notify them
            if hasattr(self, 'execution_manager') and self.execution_manager:
                # The execution manager can track these events for reporting
                for event in symbol_changed_events:
                    event_obj = event.Value
                    self.execution_manager.track_rollover_event(event_obj.OldSymbol, event_obj.NewSymbol)
                    
        except Exception as e:
            self.Log(f"ERROR in OnSymbolChangedEvents: {str(e)}")
            # Don't let rollover errors crash the algorithm
            import traceback
            self.Log(f"Rollover error traceback: {traceback.format_exc()}")

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

    def get_portfolio_summary(self, use_qc_native=True):
        """
        Get portfolio summary using QC's built-in features when available
        
        Args:
            use_qc_native (bool): Whether to use QC's native portfolio features
        """
        if use_qc_native:
            return self._get_qc_portfolio_summary()
        else:
            return self._get_custom_portfolio_summary()
    
    def _get_qc_portfolio_summary(self):
        """Get portfolio summary using QC's built-in Portfolio object"""
        # QC provides all these values efficiently
        portfolio = self.Portfolio
        
        # Get position count using QC's optimized method
        invested_holdings = [h for h in portfolio.Values if h.Invested]
        total_holdings_value = sum(abs(float(h.HoldingsValue)) for h in invested_holdings)
        
        portfolio_value = float(portfolio.TotalPortfolioValue)
        gross_exposure = total_holdings_value / portfolio_value if portfolio_value > 0 else 0
        
        return {
            'portfolio_value': portfolio_value,
            'cash': float(portfolio.Cash),
            'margin_used': float(portfolio.TotalMarginUsed),
            'total_holdings_value': float(portfolio.TotalHoldingsValue),
            'total_fees': float(portfolio.TotalFees),
            'total_profit': float(portfolio.TotalUnrealizedProfit),
            'position_count': len(invested_holdings),
            'gross_exposure': gross_exposure,
            'net_liquidation_value': portfolio_value,  # Same as portfolio value but more explicit
            'buying_power': float(portfolio.TotalMarginUsed),  # Available buying power
            'invested': portfolio.Invested,
            'timestamp': self.Time
        }
    
    def _get_custom_portfolio_summary(self):
        """Fallback to manual calculation (legacy)"""
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

    def get_position_summary(self, use_qc_native=True):
        """
        Get detailed position summary using QC's built-in features when available
        
        Args:
            use_qc_native (bool): Whether to use QC's native position features
        """
        if use_qc_native:
            return self._get_qc_position_summary()
        else:
            return self._get_custom_position_summary()
    
    def _get_qc_position_summary(self):
        """Get position summary using QC's Portfolio.Values efficiently"""
        positions = []
        
        # Use QC's optimized iteration over invested holdings
        for holding in self.Portfolio.Values:
            if holding.Invested:
                # QC provides all these properties efficiently
                positions.append({
                    'symbol': str(holding.Symbol),
                    'symbol_id': holding.Symbol.ID,  # QC's unique identifier
                    'quantity': float(holding.Quantity),
                    'holdings_value': float(holding.HoldingsValue),
                    'unrealized_pnl': float(holding.UnrealizedProfit),
                    'realized_pnl': float(holding.Profit),
                    'average_price': float(holding.AveragePrice),
                    'market_price': float(holding.Price),
                    'total_sale_volume': float(holding.TotalSaleVolume),
                    'net_profit': float(holding.NetProfit),
                    'is_long': holding.IsLong,
                    'is_short': holding.IsShort,
                    'currency': str(holding.QuoteCurrency.Symbol) if hasattr(holding, 'QuoteCurrency') else 'USD'
                })
        
        return {
            'positions': positions,
            'total_positions': len(positions),
            'total_holdings_value': float(self.Portfolio.TotalHoldingsValue),
            'total_unrealized_profit': float(self.Portfolio.TotalUnrealizedProfit),
            'timestamp': self.Time
        }
    
    def _get_custom_position_summary(self):
        """Fallback to manual position calculation (legacy)"""
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
    
    def get_portfolio_metrics(self):
        """Get comprehensive portfolio metrics using QC's built-in features"""
        portfolio = self.Portfolio
        
        # Core portfolio metrics from QC
        metrics = {
            'portfolio_value': float(portfolio.TotalPortfolioValue),
            'cash': float(portfolio.Cash),
            'holdings_value': float(portfolio.TotalHoldingsValue),
            'margin_used': float(portfolio.TotalMarginUsed),
            'margin_remaining': float(portfolio.MarginRemaining),
            'total_fees': float(portfolio.TotalFees),
            'unrealized_profit': float(portfolio.TotalUnrealizedProfit),
            'invested': portfolio.Invested,
            'positions_count': sum(1 for h in portfolio.Values if h.Invested)
        }
        
        # Calculate derived metrics
        if metrics['portfolio_value'] > 0:
            metrics['cash_percentage'] = metrics['cash'] / metrics['portfolio_value']
            metrics['margin_utilization'] = metrics['margin_used'] / metrics['portfolio_value']
            
            # Calculate gross and net exposure
            long_value = sum(h.HoldingsValue for h in portfolio.Values if h.IsLong)
            short_value = sum(abs(h.HoldingsValue) for h in portfolio.Values if h.IsShort)
            
            metrics['long_exposure'] = long_value / metrics['portfolio_value']
            metrics['short_exposure'] = short_value / metrics['portfolio_value']
            metrics['gross_exposure'] = (long_value + short_value) / metrics['portfolio_value']
            metrics['net_exposure'] = (long_value - short_value) / metrics['portfolio_value']
        
        return metrics

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
