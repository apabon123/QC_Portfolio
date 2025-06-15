"""
QC Native Contract Resolver

Leverages QuantConnect's native functionality for continuous contract handling.
Provides minimal backup logic only when QC's built-in methods fail.

Key QC Native Features Used:
- AddFuture() for continuous contracts
- .Mapped property for underlying contracts  
- OnSymbolChangedEvents for rollover handling
- HasData, IsTradable properties for validation
- Built-in DataMappingMode and DataNormalizationMode
"""

from AlgorithmImports import *

class QCNativeContractResolver:
    """
    Minimal wrapper around QC's native contract handling.
    Only provides backup when QC's native methods fail.
    """
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        
        # Track history request failures to avoid repeated failed requests
        self.failed_history_requests = {}
        
        # Track symbol initialization to prevent duplicate logs
        self.initialized_symbols = set()
        
        self.algorithm.Log("QCNativeContractResolver: Initialized (leveraging QC native functionality)")
    
    def get_symbol_for_data(self, symbol):
        """
        Get symbol for data requests (strategies use continuous contracts).
        Uses QC's native continuous contract handling.
        
        Args:
            symbol: Continuous contract symbol
            
        Returns:
            Symbol to use for data requests (usually the same continuous contract)
        """
        symbol_str = str(symbol)
        
        # For continuous contracts, QC handles everything natively
        if symbol_str.startswith('/') or symbol_str.startswith('futures/'):
            # Check if QC says this symbol has data
            if symbol in self.algorithm.Securities:
                security = self.algorithm.Securities[symbol]
                
                # Use QC's native HasData property
                if security.HasData:
                    return symbol
                else:
                    # QC says no data - log this for diagnostics
                    if symbol_str not in self.failed_history_requests:
                        self.algorithm.Log(f"QC Native: Continuous contract {symbol_str} has no data (QC HasData=False)")
                        self.failed_history_requests[symbol_str] = self.algorithm.Time
                    
                    # Return symbol anyway - let QC handle it
                    return symbol
            else:
                self.algorithm.Log(f"QC Native: Symbol {symbol_str} not in Securities collection")
                return symbol
        
        # Not a continuous contract - return as-is
        return symbol
    
    def get_symbol_for_trading(self, symbol):
        """
        Get symbol for trading (use QC's .Mapped property).
        This is where QC automatically handles the continuous -> underlying mapping.
        
        Args:
            symbol: Continuous contract symbol
            
        Returns:
            Underlying contract symbol from QC's .Mapped property
        """
        symbol_str = str(symbol)
        
        try:
            if symbol in self.algorithm.Securities:
                security = self.algorithm.Securities[symbol]
                
                # Use QC's native .Mapped property - this is the key QC functionality
                if hasattr(security, 'Mapped') and security.Mapped:
                    mapped_contract = security.Mapped
                    
                    # Only log if this is a new mapping to avoid spam
                    mapped_str = str(mapped_contract)
                    if f"{symbol_str}->{mapped_str}" not in self.initialized_symbols:
                        self.algorithm.Log(f"QC Native: {symbol_str} -> {mapped_str} (via .Mapped)")
                        self.initialized_symbols.add(f"{symbol_str}->{mapped_str}")
                    
                    return mapped_contract
                else:
                    self.algorithm.Log(f"QC Native: No .Mapped contract for {symbol_str}")
                    return symbol
            else:
                return symbol
                
        except Exception as e:
            self.algorithm.Log(f"QC Native: Error getting mapped contract for {symbol_str}: {str(e)}")
            return symbol
    
    def validate_symbol_for_trading(self, symbol):
        """
        Validate using QC's native properties.
        
        Args:
            symbol: Symbol to validate
            
        Returns:
            dict: Validation result with QC native properties
        """
        try:
            # Get the trading symbol (underlying contract)
            trading_symbol = self.get_symbol_for_trading(symbol)
            
            if trading_symbol not in self.algorithm.Securities:
                return {
                    'valid': False,
                    'reason': f"Trading symbol {trading_symbol} not in Securities",
                    'qc_properties': {}
                }
            
            security = self.algorithm.Securities[trading_symbol]
            
            # Collect QC's native properties
            qc_properties = {
                'HasData': getattr(security, 'HasData', False),
                'IsTradable': getattr(security, 'IsTradable', False),
                'Price': getattr(security, 'Price', None),
                'Mapped': str(getattr(security, 'Mapped', None))
            }
            
            # Use QC's validation
            if not qc_properties['HasData']:
                return {
                    'valid': False,
                    'reason': f"QC HasData=False for {trading_symbol}",
                    'qc_properties': qc_properties
                }
            
            if not qc_properties['IsTradable']:
                return {
                    'valid': False,
                    'reason': f"QC IsTradable=False for {trading_symbol}",
                    'qc_properties': qc_properties
                }
            
            if not qc_properties['Price'] or qc_properties['Price'] <= 0:
                return {
                    'valid': False,
                    'reason': f"Invalid price: {qc_properties['Price']}",
                    'qc_properties': qc_properties
                }
            
            return {
                'valid': True,
                'trading_symbol': trading_symbol,
                'qc_properties': qc_properties
            }
            
        except Exception as e:
            return {
                'valid': False,
                'reason': f"Validation error: {str(e)}",
                'qc_properties': {}
            }
    
    def get_history_with_diagnostics(self, symbol, periods, resolution=Resolution.Daily):
        """
        Get history using QC's native History() method with diagnostics.
        This helps understand why some history requests fail while others succeed.
        
        Args:
            symbol: Symbol to get history for
            periods: Number of periods
            resolution: Data resolution
            
        Returns:
            History data or None if failed
        """
        symbol_str = str(symbol)
        
        try:
            # Use QC's native History method
            history = self.algorithm.History(symbol, periods, resolution)
            
            if history is not None and not history.empty:
                # Success - log if this was previously failing
                if symbol_str in self.failed_history_requests:
                    self.algorithm.Log(f"QC Native: History recovered for {symbol_str} ({len(history)} bars)")
                    del self.failed_history_requests[symbol_str]
                
                return history
            else:
                # Failed - track and log with diagnostics
                if symbol_str not in self.failed_history_requests:
                    self.algorithm.Log(f"QC Native: No history for {symbol_str} (periods={periods}, resolution={resolution})")
                    
                    # Add QC diagnostic info
                    if symbol in self.algorithm.Securities:
                        security = self.algorithm.Securities[symbol]
                        self.algorithm.Log(f"  QC Diagnostics: HasData={getattr(security, 'HasData', 'N/A')}, "
                                         f"IsTradable={getattr(security, 'IsTradable', 'N/A')}, "
                                         f"Price={getattr(security, 'Price', 'N/A')}")
                        
                        if hasattr(security, 'Mapped'):
                            mapped = security.Mapped
                            self.algorithm.Log(f"  QC Mapped: {mapped}")
                            if mapped and mapped in self.algorithm.Securities:
                                mapped_security = self.algorithm.Securities[mapped]
                                self.algorithm.Log(f"  Mapped HasData: {getattr(mapped_security, 'HasData', 'N/A')}")
                    
                    self.failed_history_requests[symbol_str] = self.algorithm.Time
                
                return None
                
        except Exception as e:
            self.algorithm.Log(f"QC Native: History error for {symbol_str}: {str(e)}")
            return None
    
    def handle_rollover_qc_native(self, symbol_changed_event):
        """
        Handle rollover using QC's native OnSymbolChangedEvents.
        This is QC's built-in rollover mechanism.
        
        Args:
            symbol_changed_event: QC's SymbolChangedEvent
        """
        old_symbol = symbol_changed_event.OldSymbol
        new_symbol = symbol_changed_event.NewSymbol
        
        self.algorithm.Log(f"QC Native Rollover: {old_symbol} -> {new_symbol}")
        
        # Update our tracking
        old_str = str(old_symbol)
        new_str = str(new_symbol)
        
        # Remove old symbol from failed requests if it was there
        if old_str in self.failed_history_requests:
            del self.failed_history_requests[old_str]
        
        # Clear old initialization tracking
        keys_to_remove = [key for key in self.initialized_symbols if key.startswith(old_str)]
        for key in keys_to_remove:
            self.initialized_symbols.remove(key)
    
    def get_diagnostics_report(self):
        """Get diagnostics about symbol handling."""
        return {
            'failed_history_requests': len(self.failed_history_requests),
            'initialized_symbols': len(self.initialized_symbols),
            'failed_symbols': list(self.failed_history_requests.keys()),
        }
    
    def log_qc_native_status(self):
        """Log QC's native properties for debugging."""
        self.algorithm.Log("=== QC Native Status ===")
        
        futures_count = 0
        has_data_count = 0
        tradable_count = 0
        
        for symbol in self.algorithm.Securities.Keys:
            security = self.algorithm.Securities[symbol]
            
            if security.Type == SecurityType.Future:
                futures_count += 1
                
                if getattr(security, 'HasData', False):
                    has_data_count += 1
                
                if getattr(security, 'IsTradable', False):
                    tradable_count += 1
        
        self.algorithm.Log(f"QC Native: {futures_count} futures, {has_data_count} with data, {tradable_count} tradable")
        
        if self.failed_history_requests:
            self.algorithm.Log(f"QC Native: {len(self.failed_history_requests)} symbols with history failures")
            for symbol_str in list(self.failed_history_requests.keys())[:5]:  # Show first 5
                self.algorithm.Log(f"  {symbol_str}") 