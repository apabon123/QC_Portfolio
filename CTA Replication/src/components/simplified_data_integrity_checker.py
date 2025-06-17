"""
Simplified Data Integrity Checker for QuantConnect Three-Layer CTA Framework

This module provides data validation without redundant caching, leveraging
QuantConnect's native data caching and sharing capabilities.

Key Changes from Original:
- REMOVED: All custom history caching logic
- REMOVED: Custom cache management and cleanup
- REMOVED: Redundant data access patterns
- KEPT: Essential data validation and quarantine logic
- ENHANCED: Uses QC's native data access patterns
"""

from AlgorithmImports import *

class SimplifiedDataIntegrityChecker:
    """
    Simplified Data Integrity Checker - Validation Only
    
    Focuses on what QC doesn't provide:
    - Symbol quarantine management
    - Price range validation
    - Data quality tracking
    - Bad data detection
    
    Leverages QC's native capabilities:
    - Uses Securities[symbol] properties directly
    - No custom history caching (QC handles this)
    - No redundant data storage
    """
    
    def __init__(self, algorithm, config_manager=None):
        """
        Initialize simplified data integrity checker.
        REMOVED: All custom caching logic
        KEPT: Essential validation configuration
        """
        self.algorithm = algorithm
        self.config_manager = config_manager
        
        try:
            if self.config_manager:
                # Get data integrity configuration through centralized manager
                self.config = self.config_manager.get_data_integrity_config()
            else:
                # Minimal fallback config
                self.algorithm.Log("WARNING: No config manager provided to SimplifiedDataIntegrityChecker")
                self.config = {
                    'max_zero_price_streak': 3,
                    'max_no_data_streak': 3,
                    'quarantine_duration_days': 7,
                    'price_ranges': {}
                }
            
            # KEPT: Essential validation tracking
            self.quarantined_symbols = set()
            self.quarantine_reasons = {}
            self.quarantine_timestamps = {}
            self.zero_price_streaks = {}
            self.no_data_streaks = {}
            self.last_valid_prices = {}
            
            # REMOVED: All cache management variables
            # REMOVED: self.history_cache, self.cache_timestamps, self.cache_requests
            # REMOVED: self.cache_stats, self.cache_config
            
            self.algorithm.Log(f"SimplifiedDataIntegrityChecker: Initialized (validation only, QC handles caching)")
            
        except Exception as e:
            error_msg = f"CRITICAL ERROR initializing SimplifiedDataIntegrityChecker: {str(e)}"
            self.algorithm.Error(error_msg)
            raise ValueError(error_msg)
    
    def validate_slice(self, slice):
        """
        Validate slice data using QC's built-in capabilities.
        SIMPLIFIED: Focus on validation, not data manipulation.
        """
        try:
            if not slice or slice.Keys is None:
                return slice
            
            # Let QC handle most validation - only intervene for critical issues
            for symbol in slice.Keys:
                # PRIMARY VALIDATION: Use QC's built-in methods
                if not self._is_symbol_valid_qc_native(symbol):
                    continue
                
                # SECONDARY VALIDATION: Only custom checks QC doesn't provide
                if not self._passes_custom_checks(symbol, slice):
                    continue
            
            # Return original slice (QC handles the data)
            return slice
                
        except Exception as e:
            self.algorithm.Error(f"SimplifiedDataIntegrityChecker: Error validating slice: {str(e)}")
            return slice  # Return original on error
    
    def _is_symbol_valid_qc_native(self, symbol):
        """
        Use QC's native validation properties.
        UNCHANGED: This validation logic is still needed.
        """
        try:
            # Check if symbol exists in securities (QC built-in)
            if symbol not in self.algorithm.Securities:
                self._quarantine_symbol(symbol, "not_in_securities")
                return False
            
            security = self.algorithm.Securities[symbol]
            
            # Check if quarantined by our custom logic
            if symbol in self.quarantined_symbols:
                return False
            
            # QC BUILT-IN VALIDATION:
            
            # 1. HasData property (QC built-in)
            if not security.HasData:
                self._track_no_data(symbol)
                return False
            else:
                self._reset_no_data_streak(symbol)
            
            # 2. IsTradable property (QC built-in)
            if not security.IsTradable:
                self._quarantine_symbol(symbol, "not_tradable")
                return False
            
            # 3. Price property validation (QC built-in)
            if not hasattr(security, 'Price') or security.Price is None:
                self._quarantine_symbol(symbol, "no_price_property")
                return False
            
            # 4. Price sanity check
            if security.Price <= 0:
                self._track_zero_price(symbol)
                return False
            else:
                self._reset_zero_price_streak(symbol)
            
            return True
            
        except Exception as e:
            self.algorithm.Error(f"SimplifiedDataIntegrityChecker: Error in validation for {symbol}: {str(e)}")
            self._quarantine_symbol(symbol, f"validation_error={str(e)[:50]}")
            return False
    
    def _passes_custom_checks(self, symbol, slice):
        """
        Custom checks that QC doesn't provide.
        UNCHANGED: Still needed for business logic.
        """
        try:
            security = self.algorithm.Securities[symbol]
            
            # Custom price range validation
            if not self._is_price_in_reasonable_range(symbol, security.Price):
                self._quarantine_symbol(symbol, f"price_out_of_range={security.Price}")
                return False
            
            # Check quarantine status
            if symbol in self.quarantined_symbols:
                self._check_quarantine_release(symbol)
                return symbol not in self.quarantined_symbols
            
            return True
            
        except Exception as e:
            return True  # Don't block on custom check errors
    
    # =============================================================================
    # VALIDATION TRACKING METHODS (KEPT - Still needed)
    # =============================================================================
    
    def _track_zero_price(self, symbol):
        """Track consecutive zero prices for quarantine decisions."""
        self.zero_price_streaks[symbol] = self.zero_price_streaks.get(symbol, 0) + 1
        
        if self.zero_price_streaks[symbol] >= self.config['max_zero_price_streak']:
            self._quarantine_symbol(symbol, f"zero_price_streak={self.zero_price_streaks[symbol]}")
    
    def _reset_zero_price_streak(self, symbol):
        """Reset zero price streak when valid price is found."""
        if symbol in self.zero_price_streaks:
            del self.zero_price_streaks[symbol]
    
    def _track_no_data(self, symbol):
        """Track consecutive no-data occurrences."""
        self.no_data_streaks[symbol] = self.no_data_streaks.get(symbol, 0) + 1
        
        if self.no_data_streaks[symbol] >= self.config['max_no_data_streak']:
            self._quarantine_symbol(symbol, f"no_data_streak={self.no_data_streaks[symbol]}")
    
    def _reset_no_data_streak(self, symbol):
        """Reset no-data streak when data is found."""
        if symbol in self.no_data_streaks:
            del self.no_data_streaks[symbol]
    
    def _is_price_in_reasonable_range(self, symbol, price):
        """Check if price is within reasonable range for the symbol."""
        try:
            if not price or price <= 0:
                return False
            
            # Get price ranges from config if available
            price_ranges = self.config.get('price_ranges', {})
            symbol_str = str(symbol)
            
            # Check if we have specific ranges for this symbol
            for symbol_pattern, ranges in price_ranges.items():
                if symbol_pattern in symbol_str:
                    min_price = ranges.get('min', 0)
                    max_price = ranges.get('max', float('inf'))
                    
                    if not (min_price <= price <= max_price):
                        return False
            
            return True
            
        except Exception as e:
            return True  # Don't block on range check errors
    
    def _quarantine_symbol(self, symbol, reason):
        """Quarantine a symbol with tracking."""
        if symbol not in self.quarantined_symbols:
            self.quarantined_symbols.add(symbol)
            self.quarantine_reasons[symbol] = reason
            self.quarantine_timestamps[symbol] = self.algorithm.Time
            
            ticker = self._extract_ticker(symbol)
            self.algorithm.Log(f"SimplifiedDataIntegrityChecker: QUARANTINED {ticker} - {reason}")
    
    def _check_quarantine_release(self, symbol):
        """Check if quarantined symbols can be released."""
        try:
            if symbol in self.quarantine_timestamps:
                quarantine_time = self.quarantine_timestamps[symbol]
                days_quarantined = (self.algorithm.Time - quarantine_time).days
                
                if days_quarantined >= self.config['quarantine_duration_days']:
                    self._release_symbol_from_quarantine(symbol)
                    
        except Exception as e:
            self.algorithm.Error(f"SimplifiedDataIntegrityChecker: Error checking quarantine release: {str(e)}")
    
    def _release_symbol_from_quarantine(self, symbol):
        """Release a symbol from quarantine."""
        if symbol in self.quarantined_symbols:
            self.quarantined_symbols.remove(symbol)
            
            # Clean up tracking data
            self.quarantine_reasons.pop(symbol, None)
            self.quarantine_timestamps.pop(symbol, None)
            self.zero_price_streaks.pop(symbol, None)
            self.no_data_streaks.pop(symbol, None)
            
            ticker = self._extract_ticker(symbol)
            self.algorithm.Log(f"SimplifiedDataIntegrityChecker: RELEASED {ticker} from quarantine")
    
    def _extract_ticker(self, symbol):
        """Extract ticker from symbol for logging."""
        try:
            symbol_str = str(symbol)
            # Handle different symbol formats
            if '/' in symbol_str:
                return symbol_str.split('/')[-1]
            return symbol_str
        except:
            return str(symbol)
    
    # =============================================================================
    # PUBLIC INTERFACE METHODS (SIMPLIFIED)
    # =============================================================================
    
    def get_quarantine_status(self):
        """Get current quarantine status."""
        return {
            'quarantined_count': len(self.quarantined_symbols),
            'quarantined_symbols': list(self.quarantined_symbols),
            'quarantine_reasons': dict(self.quarantine_reasons)
        }
    
    def get_safe_symbols(self, symbols):
        """Get symbols that are not quarantined."""
        return [symbol for symbol in symbols if symbol not in self.quarantined_symbols]
    
    def validate_symbol_data_quality(self, symbol):
        """
        Validate data quality for a specific symbol using QC's native properties.
        SIMPLIFIED: Uses QC's native validation, no custom caching.
        """
        try:
            if symbol not in self.algorithm.Securities:
                return {
                    'valid': False,
                    'reason': 'symbol_not_found',
                    'has_data': False,
                    'is_tradable': False,
                    'price_valid': False
                }
            
            security = self.algorithm.Securities[symbol]
            
            # Use QC's native validation properties
            has_data = security.HasData
            is_tradable = security.IsTradable
            price = security.Price
            price_valid = price > 0
            
            # Check quarantine status
            is_quarantined = symbol in self.quarantined_symbols
            
            # Overall validation
            valid = has_data and is_tradable and price_valid and not is_quarantined
            
            return {
                'valid': valid,
                'reason': 'valid' if valid else self._get_validation_failure_reason(has_data, is_tradable, price_valid, is_quarantined),
                'has_data': has_data,
                'is_tradable': is_tradable,
                'price_valid': price_valid,
                'is_quarantined': is_quarantined,
                'current_price': price,
                'validation_time': self.algorithm.Time
            }
            
        except Exception as e:
            return {
                'valid': False,
                'reason': f'validation_error: {str(e)}',
                'has_data': False,
                'is_tradable': False,
                'price_valid': False,
                'is_quarantined': False
            }
    
    def _get_validation_failure_reason(self, has_data: bool, is_tradable: bool, price_valid: bool, is_quarantined: bool) -> str:
        """Get specific reason for validation failure."""
        reasons = []
        
        if not has_data:
            reasons.append('no_data')
        if not is_tradable:
            reasons.append('not_tradable')
        if not price_valid:
            reasons.append('invalid_price')
        if is_quarantined:
            reasons.append('quarantined')
        
        return '_'.join(reasons) if reasons else 'unknown_failure'
    
    def log_status_summary(self):
        """Log simplified status summary."""
        quarantine_status = self.get_quarantine_status()
        
        self.algorithm.Log("=" * 50)
        self.algorithm.Log("SIMPLIFIED DATA INTEGRITY STATUS")
        self.algorithm.Log("=" * 50)
        self.algorithm.Log(f"Quarantined Symbols: {quarantine_status['quarantined_count']}")
        
        if quarantine_status['quarantined_symbols']:
            self.algorithm.Log("Quarantined Details:")
            for symbol in quarantine_status['quarantined_symbols']:
                reason = quarantine_status['quarantine_reasons'].get(symbol, 'unknown')
                ticker = self._extract_ticker(symbol)
                self.algorithm.Log(f"  {ticker}: {reason}")
        
        self.algorithm.Log("Data Access: Using QC's native caching")
        self.algorithm.Log("=" * 50) 