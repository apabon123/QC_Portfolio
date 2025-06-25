# futures_rollover_manager.py - CRITICAL ROLLOVER LOGIC COMPONENT

from AlgorithmImports import *

class FuturesRolloverManager:
    """
    CRITICAL COMPONENT: Futures Rollover Management
    
    This component isolates the critical rollover logic to prevent accidental modifications.
    The rollover logic is based on QuantConnect's official pattern and handles the transition
    from expiring contracts to new contracts seamlessly.
    
    ARCHITECTURE: Uses QuantConnect's TWO-CONTRACT system where both front month and
    second month contracts are tracked, ensuring rollover prices are always available.
    """
    
    def __init__(self, algorithm, config_manager):
        """
        Initialize the Futures Rollover Manager.
        
        Args:
            algorithm: QuantConnect algorithm instance
            config_manager: Centralized configuration manager
        """
        self.algorithm = algorithm
        self.config_manager = config_manager
        
        # Rollover tracking
        self.rollover_events_count = 0
        self.rollover_history = []
        
        # Configuration
        self.execution_config = config_manager.get_execution_config()
        self.rollover_config = self.execution_config.get('rollover', {})
        
        # Logging configuration
        self.log_rollover_details = self.rollover_config.get('log_rollover_details', True)
        self.log_rollover_prices = self.rollover_config.get('log_rollover_prices', True)
        
        self.algorithm.Log("ROLLOVER MANAGER: Initialized with QuantConnect official rollover pattern")
    
    def handle_symbol_changed_events(self, symbolChangedEvents):
        """
        Handle QuantConnect's OnSymbolChangedEvents using the official rollover pattern.
        
        This is the CRITICAL method that handles futures contract rollovers.
        Based on QuantConnect's official documentation and best practices.
        
        Args:
            symbolChangedEvents: Dictionary of symbol change events from QuantConnect
        """
        try:
            for symbol, changedEvent in symbolChangedEvents.items():
                oldSymbol = changedEvent.OldSymbol
                newSymbol = changedEvent.NewSymbol
                
                # SAFETY CHECK: Ensure oldSymbol still exists in Securities before accessing
                if oldSymbol not in self.algorithm.Securities:
                    # This can happen if the security was removed before the event is processed
                    if self.log_rollover_details:
                        self.algorithm.Log(f"ROLLOVER WARNING: {oldSymbol} not found in Securities collection – skipping event.")
                    continue

                quantity = self.algorithm.Portfolio[oldSymbol].Quantity
                
                if quantity != 0:
                    # CRITICAL: Subscribe to new contract explicitly
                    actual_symbol = self._ensure_new_contract_subscription(newSymbol)
                    
                    # REAL-WORLD TIMING FIX: Schedule rollover for market close
                    # QC triggers at midnight (artificial), but real volume rolls execute at COB
                    # This maintains continuous exposure - no gap between old and new contracts
                    if self.log_rollover_details:
                        self.algorithm.Log(f"ROLLOVER DETECTED: {oldSymbol} -> {actual_symbol} (scheduled for market close)")
                    
                    self.algorithm.Schedule.On(
                        self.algorithm.DateRules.Today,
                        self.algorithm.TimeRules.BeforeMarketClose(actual_symbol, 30),
                        lambda os=oldSymbol, ns=actual_symbol, q=quantity: self._execute_end_of_day_rollover(os, ns, q)
                    )
                    
                self.rollover_events_count += 1
                
        except Exception as e:
            self.algorithm.Error(f"ROLLOVER MANAGER: Error in symbol changed events: {str(e)}")
    
    def _ensure_new_contract_subscription(self, newSymbol):
        """
        Ensure the new contract is properly subscribed and available for trading.
        
        Args:
            newSymbol: The new contract symbol to subscribe to
            
        Returns:
            Symbol: The actual symbol that was subscribed (may differ from input)
        """
        try:
            added_contract = self.algorithm.AddFutureContract(newSymbol)
            actual_symbol = added_contract.Symbol
            
            if self.log_rollover_details:
                self.algorithm.Log(f"ROLLOVER: Successfully added new contract {actual_symbol}")
            
            return actual_symbol
            
        except Exception as add_e:
            self.algorithm.Log(f"ROLLOVER: Failed to add contract {newSymbol}: {add_e}")
            return newSymbol  # Try anyway with original symbol
    
    def _execute_end_of_day_rollover(self, oldSymbol, newSymbol, quantity):
        """
        Execute the actual rollover using QuantConnect's official pattern:
        1. Liquidate old position
        2. Open new position with same quantity
        
        This follows the simple 6-8 line pattern recommended by QuantConnect.
        
        Args:
            oldSymbol: Symbol of the expiring contract
            newSymbol: Symbol of the new contract
            quantity: Number of contracts to roll over (captured when event scheduled)
        """
        try:
            # REAL-TIME SAFETY CHECK ─ fetch current position size in case it has changed
            current_qty = 0
            if oldSymbol in self.algorithm.Portfolio:
                current_qty = self.algorithm.Portfolio[oldSymbol].Quantity

            if current_qty == 0:
                # Position has already been rolled by another event; skip duplicate
                if self.log_rollover_details:
                    self.algorithm.Log(
                        f"ROLLOVER SKIPPED: No remaining position in {oldSymbol} (duplicate event)"
                    )
                return

            # Use the up-to-date quantity for the rollover (can differ from captured value)
            quantity = current_qty
            if self.log_rollover_details:
                self.algorithm.Log(f"EXECUTING ROLLOVER: {oldSymbol} -> {newSymbol}")
            
            # CRITICAL FIX: Check for zero quantity before placing rollover order
            if quantity == 0:
                if self.log_rollover_details:
                    self.algorithm.Log(f"ROLLOVER SKIPPED: Zero quantity for {newSymbol}")
                return
            
            # Step 1: Close old position (QuantConnect official pattern)
            close_tickets = self.algorithm.Liquidate(oldSymbol, tag=f"Rollover-Close-{oldSymbol}")
            
            # Step 2: Open new position (QuantConnect official pattern)
            open_ticket = self.algorithm.MarketOrder(newSymbol, quantity, tag=f"Rollover-Open-{newSymbol}")
            
            # Log rollover execution details
            self._log_rollover_execution(oldSymbol, newSymbol, quantity, close_tickets, open_ticket)
            
            # Track rollover for reporting
            self._track_rollover_event(oldSymbol, newSymbol, quantity)
                
        except Exception as e:
            self.algorithm.Error(f"ROLLOVER EXECUTION FAILED: {oldSymbol} -> {newSymbol}: {e}")
    
    def _log_rollover_execution(self, oldSymbol, newSymbol, quantity, close_tickets, open_ticket):
        """
        Log detailed rollover execution information.
        
        Args:
            oldSymbol: Old contract symbol
            newSymbol: New contract symbol  
            quantity: Quantity rolled over
            close_tickets: Result from Liquidate call
            open_ticket: Result from MarketOrder call
        """
        try:
            if not self.log_rollover_details:
                return
            
            self.algorithm.Log(f"  Closed old position: {oldSymbol}")
            
            # Log new contract price if available
            if self.log_rollover_prices and newSymbol in self.algorithm.Securities:
                security = self.algorithm.Securities[newSymbol]
                self.algorithm.Log(f"  New contract price: ${security.Price:.2f}")
            
            if open_ticket:
                self.algorithm.Log(f"  ROLLOVER SUCCESS: {oldSymbol} -> {newSymbol} ({quantity} contracts)")
                
                # Handle close ticket logging (can be single ticket or list)
                close_info = self._format_close_ticket_info(close_tickets)
                self.algorithm.Log(f"    {close_info}")
                self.algorithm.Log(f"    Open Order ID: {open_ticket.OrderId}, Status: {open_ticket.Status}")
            else:
                self.algorithm.Error(f"  ROLLOVER FAILED: MarketOrder returned None for {newSymbol}")
                
            self.algorithm.Log("=" * 60)
            
        except Exception as e:
            self.algorithm.Error(f"ROLLOVER LOGGING ERROR: {str(e)}")
    
    def _format_close_ticket_info(self, close_tickets):
        """
        Format close ticket information for logging.
        
        Args:
            close_tickets: Result from Liquidate call (can be various types)
            
        Returns:
            str: Formatted close ticket information
        """
        try:
            if close_tickets:
                if hasattr(close_tickets, 'OrderId'):
                    return f"Close Order ID: {close_tickets.OrderId}"
                elif isinstance(close_tickets, list) and len(close_tickets) > 0:
                    return f"Close Orders: {len(close_tickets)} tickets"
                else:
                    return "Close Order: Completed"
            else:
                return "Close Order: None returned"
                
        except Exception:
            return "Close Order: Logging error"
    
    def _track_rollover_event(self, oldSymbol, newSymbol, quantity):
        """
        Track rollover event for reporting and analysis.
        
        Args:
            oldSymbol: Old contract symbol
            newSymbol: New contract symbol
            quantity: Quantity rolled over
        """
        try:
            rollover_event = {
                'timestamp': self.algorithm.Time,
                'old_symbol': str(oldSymbol),
                'new_symbol': str(newSymbol),
                'quantity': quantity,
                'event_count': self.rollover_events_count
            }
            
            self.rollover_history.append(rollover_event)
            
            # Keep only recent rollover history to prevent memory issues
            max_history = self.rollover_config.get('max_rollover_history', 100)
            if len(self.rollover_history) > max_history:
                self.rollover_history = self.rollover_history[-max_history:]
                
        except Exception as e:
            self.algorithm.Error(f"ROLLOVER TRACKING ERROR: {str(e)}")
    
    def get_rollover_statistics(self):
        """
        Get rollover statistics for reporting.
        
        Returns:
            dict: Rollover statistics and history
        """
        try:
            recent_rollovers = self.rollover_history[-10:] if self.rollover_history else []
            
            return {
                'total_rollover_events': self.rollover_events_count,
                'rollover_history_length': len(self.rollover_history),
                'recent_rollovers': recent_rollovers,
                'last_rollover': self.rollover_history[-1] if self.rollover_history else None
            }
            
        except Exception as e:
            self.algorithm.Error(f"ROLLOVER STATISTICS ERROR: {str(e)}")
            return {'error': str(e)}
    
    def validate_rollover_readiness(self):
        """
        Validate that the rollover system is ready for operation.
        
        Returns:
            dict: Validation results
        """
        try:
            validation_results = {
                'is_ready': True,
                'issues': [],
                'warnings': []
            }
            
            # Check algorithm reference
            if not self.algorithm:
                validation_results['is_ready'] = False
                validation_results['issues'].append('Algorithm reference is None')
            
            # Check configuration
            if not self.config_manager:
                validation_results['is_ready'] = False
                validation_results['issues'].append('Config manager reference is None')
            
            # Check if we have the necessary QC methods
            required_methods = ['Liquidate', 'MarketOrder', 'AddFutureContract', 'Schedule']
            for method in required_methods:
                if not hasattr(self.algorithm, method):
                    validation_results['is_ready'] = False
                    validation_results['issues'].append(f'Algorithm missing required method: {method}')
            
            return validation_results
            
        except Exception as e:
            return {
                'is_ready': False,
                'issues': [f'Validation error: {str(e)}'],
                'warnings': []
            } 