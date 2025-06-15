"""
Continuous Contract Manager

Handles the mapping between continuous contracts and their underlying tradeable contracts
to resolve issues where continuous contracts show "No history available" while underlying 
contracts have valid data.

This addresses the specific issue seen in logs:
- /ZN: Initialized with 856 prices (continuous contract works)
- No history available for /ZN (same continuous contract later fails)
- ZN VU1EHIDJYKH1: Initialized with 293 prices (underlying contract works)
- ZN VWJ060NWX82T: Initialized with 129 prices (another underlying works)
"""

from AlgorithmImports import *
from datetime import datetime, timedelta

class ContinuousContractManager:
    """
    Manages the relationship between continuous contracts and their underlying contracts.
    Provides fallback logic when continuous contracts lose their data mapping.
    """
    
    def __init__(self, algorithm, config_manager=None):
        self.algorithm = algorithm
        self.config_manager = config_manager
        
        # Track mappings between continuous and underlying contracts  
        self.contract_mappings = {}
        
        # Track which continuous contracts have data issues
        self.problematic_continuous = set()
        
        # Track valid underlying contracts as fallbacks
        self.valid_underlying_contracts = {}
        
        # Configuration
        self.config = self._load_config()
        
        self.algorithm.Log("ContinuousContractManager: Initialized")
    
    def _load_config(self):
        """Load configuration for continuous contract management."""
        default_config = {
            'enable_fallback_mapping': True,
            'max_days_without_data': 3,
            'prefer_front_month': True,
            'validate_underlying_data': True,
            'log_mapping_changes': True,
            'retry_failed_continuous': True
        }
        
        if self.config_manager:
            try:
                config = self.config_manager.get_config().get('continuous_contracts', {})
                return {**default_config, **config}
            except:
                pass
        
        return default_config
    
    def register_continuous_contract(self, continuous_symbol, underlying_symbols=None):
        """
        Register a continuous contract and its underlying contracts.
        
        Args:
            continuous_symbol: The continuous contract symbol (e.g., /ZN)
            underlying_symbols: List of underlying contract symbols (e.g., [ZN VU1EHIDJYKH1])
        """
        continuous_str = str(continuous_symbol)
        
        if underlying_symbols is None:
            underlying_symbols = []
        
        # Initialize mapping
        self.contract_mappings[continuous_str] = {
            'continuous_symbol': continuous_symbol,
            'underlying_symbols': underlying_symbols,
            'has_data': True,
            'last_data_check': self.algorithm.Time,
            'fallback_active': False,
            'primary_underlying': None
        }
        
        # Track underlying contracts
        for underlying in underlying_symbols:
            underlying_str = str(underlying)
            if underlying_str not in self.valid_underlying_contracts:
                self.valid_underlying_contracts[underlying_str] = {
                    'symbol': underlying,
                    'continuous_parent': continuous_str,
                    'has_data': True,
                    'last_price': None,
                    'last_update': self.algorithm.Time
                }
        
        if self.config['log_mapping_changes']:
            self.algorithm.Log(f"ContinuousContractManager: Registered {continuous_str} with {len(underlying_symbols)} underlying contracts")
    
    def update_contract_data_status(self, symbol, has_data, price=None):
        """
        Update the data status for a contract (continuous or underlying).
        
        Args:
            symbol: Contract symbol
            has_data: Whether the contract currently has valid data
            price: Current price if available
        """
        symbol_str = str(symbol)
        
        # Update continuous contract status
        if symbol_str in self.contract_mappings:
            mapping = self.contract_mappings[symbol_str]
            
            if has_data != mapping['has_data']:
                # Data status changed
                if self.config['log_mapping_changes']:
                    self.algorithm.Log(f"ContinuousContractManager: {symbol_str} data status changed: {mapping['has_data']} -> {has_data}")
                
                mapping['has_data'] = has_data
                mapping['last_data_check'] = self.algorithm.Time
                
                if not has_data:
                    # Continuous contract lost data - add to problematic set
                    self.problematic_continuous.add(symbol_str)
                    self._activate_fallback_if_needed(symbol_str)
                else:
                    # Continuous contract regained data
                    self.problematic_continuous.discard(symbol_str)
                    self._deactivate_fallback_if_active(symbol_str)
        
        # Update underlying contract status
        if symbol_str in self.valid_underlying_contracts:
            underlying = self.valid_underlying_contracts[symbol_str]
            underlying['has_data'] = has_data
            underlying['last_update'] = self.algorithm.Time
            
            if price is not None:
                underlying['last_price'] = price
    
    def get_best_symbol_for_data(self, requested_symbol):
        """
        Get the best symbol to use for data requests.
        Returns the continuous contract if it has data, otherwise returns 
        the best underlying contract.
        
        Args:
            requested_symbol: The symbol originally requested (usually continuous)
            
        Returns:
            Symbol to use for data requests, or None if no valid symbol available
        """
        requested_str = str(requested_symbol)
        
        # If this is a continuous contract we manage
        if requested_str in self.contract_mappings:
            mapping = self.contract_mappings[requested_str]
            
            # Check if continuous contract has data
            if mapping['has_data'] and not mapping['fallback_active']:
                return mapping['continuous_symbol']
            
            # Continuous contract has issues - find best underlying
            return self._get_best_underlying_contract(requested_str)
        
        # Not a managed continuous contract - return as-is
        return requested_symbol
    
    def get_best_symbol_for_trading(self, requested_symbol):
        """
        Get the best symbol to use for trading.
        Usually returns the underlying contract that QC will actually trade.
        
        Args:
            requested_symbol: The symbol originally requested
            
        Returns:
            Symbol to use for trading orders
        """
        requested_str = str(requested_symbol)
        
        # For continuous contracts, always use the underlying for trading
        if requested_str in self.contract_mappings:
            # Try to get QC's mapped contract first
            if requested_symbol in self.algorithm.Securities:
                security = self.algorithm.Securities[requested_symbol]
                if hasattr(security, 'Mapped') and security.Mapped:
                    return security.Mapped
            
            # Fallback to our best underlying contract
            return self._get_best_underlying_contract(requested_str)
        
        # Not a continuous contract - return as-is
        return requested_symbol
    
    def _activate_fallback_if_needed(self, continuous_str):
        """Activate fallback mapping for a continuous contract with data issues."""
        if not self.config['enable_fallback_mapping']:
            return
        
        mapping = self.contract_mappings[continuous_str]
        
        # Find best underlying contract
        best_underlying = self._get_best_underlying_contract(continuous_str)
        
        if best_underlying:
            mapping['fallback_active'] = True
            mapping['primary_underlying'] = str(best_underlying)
            
            if self.config['log_mapping_changes']:
                self.algorithm.Log(f"ContinuousContractManager: Activated fallback for {continuous_str} -> {best_underlying}")
    
    def _deactivate_fallback_if_active(self, continuous_str):
        """Deactivate fallback mapping when continuous contract regains data."""
        mapping = self.contract_mappings[continuous_str]
        
        if mapping['fallback_active']:
            mapping['fallback_active'] = False
            mapping['primary_underlying'] = None
            
            if self.config['log_mapping_changes']:
                self.algorithm.Log(f"ContinuousContractManager: Deactivated fallback for {continuous_str}")
    
    def _get_best_underlying_contract(self, continuous_str):
        """
        Find the best underlying contract to use as fallback.
        
        Args:
            continuous_str: String representation of continuous contract
            
        Returns:
            Best underlying contract symbol, or None if none available
        """
        mapping = self.contract_mappings.get(continuous_str)
        if not mapping:
            return None
        
        # Get underlying contracts for this continuous contract
        candidates = []
        
        for underlying_str in mapping['underlying_symbols']:
            if underlying_str in self.valid_underlying_contracts:
                underlying = self.valid_underlying_contracts[underlying_str]
                
                if underlying['has_data'] and underlying['last_price'] is not None:
                    candidates.append((underlying['symbol'], underlying['last_update']))
        
        if not candidates:
            return None
        
        # Sort by most recent data and return the best
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def discover_underlying_contracts_from_securities(self):
        """
        Discover underlying contracts from the algorithm's Securities collection.
        This helps identify the actual tradeable contracts when continuous contracts fail.
        """
        discovered = 0
        
        for symbol in self.algorithm.Securities.Keys:
            symbol_str = str(symbol)
            
            # Look for futures contracts that look like underlying contracts
            if (self.algorithm.Securities[symbol].Type == SecurityType.Future and
                not symbol_str.startswith('/') and  # Not a continuous contract
                not symbol_str.startswith('futures/') and
                len(symbol_str) > 2):  # Has additional identifiers
                
                # Try to match to a continuous contract
                base_ticker = self._extract_base_ticker(symbol_str)
                continuous_key = f"/{base_ticker}"
                
                if continuous_key in self.contract_mappings:
                    # Add this as an underlying contract
                    if symbol_str not in self.valid_underlying_contracts:
                        self.valid_underlying_contracts[symbol_str] = {
                            'symbol': symbol,
                            'continuous_parent': continuous_key,
                            'has_data': True,
                            'last_price': None,
                            'last_update': self.algorithm.Time
                        }
                        
                        # Add to the mapping's underlying symbols
                        if symbol_str not in self.contract_mappings[continuous_key]['underlying_symbols']:
                            self.contract_mappings[continuous_key]['underlying_symbols'].append(symbol_str)
                        
                        discovered += 1
        
        if discovered > 0 and self.config['log_mapping_changes']:
            self.algorithm.Log(f"ContinuousContractManager: Discovered {discovered} new underlying contracts")
    
    def _extract_base_ticker(self, symbol_str):
        """Extract base ticker from underlying contract symbol (e.g., 'ZN VU1EHIDJYKH1' -> 'ZN')."""
        # Split on space and take the first part
        parts = symbol_str.split(' ')
        if len(parts) > 1:
            return parts[0]
        
        # Fallback: look for common patterns
        for ticker in ['ES', 'NQ', 'ZN', 'ZB', 'GC', 'CL', 'NG', '6E', '6J']:
            if symbol_str.startswith(ticker):
                return ticker
        
        return symbol_str[:2]  # Default fallback
    
    def get_status_report(self):
        """Get a status report of continuous contract mappings."""
        report = {
            'total_continuous_contracts': len(self.contract_mappings),
            'problematic_continuous': len(self.problematic_continuous),
            'total_underlying_contracts': len(self.valid_underlying_contracts),
            'active_fallbacks': 0,
            'contracts_with_issues': list(self.problematic_continuous)
        }
        
        for mapping in self.contract_mappings.values():
            if mapping['fallback_active']:
                report['active_fallbacks'] += 1
        
        return report
    
    def validate_all_mappings(self):
        """Validate all contract mappings and update their status."""
        for continuous_str, mapping in self.contract_mappings.items():
            continuous_symbol = mapping['continuous_symbol']
            
            # Check if continuous contract still has data
            has_data = self._check_symbol_has_data(continuous_symbol)
            self.update_contract_data_status(continuous_symbol, has_data)
            
            # Check underlying contracts
            for underlying_str in mapping['underlying_symbols']:
                if underlying_str in self.valid_underlying_contracts:
                    underlying_symbol = self.valid_underlying_contracts[underlying_str]['symbol']
                    underlying_has_data = self._check_symbol_has_data(underlying_symbol)
                    self.update_contract_data_status(underlying_symbol, underlying_has_data)
    
    def _check_symbol_has_data(self, symbol):
        """Check if a symbol currently has valid data."""
        try:
            if symbol not in self.algorithm.Securities:
                return False
            
            security = self.algorithm.Securities[symbol]
            
            # Use QC's built-in properties
            return (hasattr(security, 'HasData') and security.HasData and
                    hasattr(security, 'Price') and security.Price > 0)
        
        except Exception as e:
            return False 