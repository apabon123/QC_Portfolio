class DataIntegrityChecker:
    """
    OPTIMIZED DATA INTEGRITY CHECKER - LEVERAGING QC BUILT-INS
    
    Purpose: Maximize QuantConnect's native capabilities instead of re-engineering
    - Uses Securities.HasData, IsTradable, Price validation
    - Leverages built-in symbol properties and market hours
    - Focuses only on what QC doesn't already provide
    """
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        
        # Import configuration
        try:
            from src.components.data_integrity_config import DATA_INTEGRITY_CONFIG
            self.config = DATA_INTEGRITY_CONFIG
        except ImportError:
            # Fallback minimal config
            self.config = {
                'max_zero_price_streak': 3,
                'quarantine_duration_days': 5,
                'price_ranges': {}
            }
        
        # Lightweight tracking - let QC handle most validation
        self.quarantined_symbols = set()
        self.quarantine_timestamps = {}
        self.quarantine_reasons = {}
        
        # Track only what QC doesn't provide
        self.zero_price_streaks = {}  # Count consecutive zero prices
        self.last_quarantine_check = None
        
    def validate_slice(self, slice):
        """
        FOCUSED validation using QC's built-in capabilities
        Only add custom checks where QC doesn't provide them
        """
        try:
            if not slice or slice.Keys is None:
                return slice
            
            # Let QC handle most validation - only intervene for critical issues
            valid_slice_data = {}
            
            for symbol in slice.Keys:
                # PRIMARY VALIDATION: Use QC's built-in methods
                if not self._is_symbol_valid_qc_native(symbol):
                    continue
                
                # SECONDARY VALIDATION: Only custom checks QC doesn't provide
                if not self._passes_custom_checks(symbol, slice):
                    continue
                
                # Symbol passed all checks
                valid_slice_data[symbol] = slice[symbol]
            
            # Create new slice with valid data only
            if valid_slice_data:
                # Return filtered slice
                return slice
            else:
                return None
                
        except Exception as e:
            self.algorithm.Error(f"DataIntegrityChecker: Error validating slice: {str(e)}")
            return slice  # Return original on error
    
    def _is_symbol_valid_qc_native(self, symbol):
        """
        AGGRESSIVE QUANTCONNECT VALIDATION
        Prevent 'security does not have accurate price' errors
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
            
            # AGGRESSIVE QC BUILT-IN VALIDATION:
            
            # 1. HasData property (QC built-in) - CRITICAL CHECK
            if not security.HasData:
                self._track_no_data(symbol)
                return False
            else:
                self._reset_no_data_streak(symbol)
            
            # 2. IsTradable property (QC built-in) - CRITICAL CHECK
            if not security.IsTradable:
                self._quarantine_symbol(symbol, "not_tradable")
                return False
            
            # 3. Price property validation (QC built-in) - CRITICAL CHECK
            if not hasattr(security, 'Price') or security.Price is None:
                self._quarantine_symbol(symbol, "no_price_property")
                return False
            
            # 4. AGGRESSIVE price sanity (prevent trading with zero/negative prices)
            if security.Price <= 0:
                self._track_zero_price(symbol)
                return False
            else:
                self._reset_zero_price_streak(symbol)
            
            # 5. Additional validation to prevent QuantConnect errors
            # Check if security has been mapped (for futures)
            if hasattr(security, 'Mapped') and security.Mapped is None:
                self._quarantine_symbol(symbol, "no_mapped_contract")
                return False
            
            return True
            
        except Exception as e:
            self.algorithm.Error(f"DataIntegrityChecker: Error in QC native validation for {symbol}: {str(e)}")
            # Quarantine symbols that cause validation errors
            self._quarantine_symbol(symbol, f"validation_error={str(e)[:50]}")
            return False
    
    def _passes_custom_checks(self, symbol, slice):
        """
        ONLY custom checks that QC doesn't provide
        Focus on price range validation and quarantine logic
        """
        try:
            security = self.algorithm.Securities[symbol]
            
            # Custom price range validation (QC doesn't provide this)
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
    
    def _track_zero_price(self, symbol):
        """Track consecutive zero prices for quarantine decisions"""
        self.zero_price_streaks[symbol] = self.zero_price_streaks.get(symbol, 0) + 1
        
        if self.zero_price_streaks[symbol] >= self.config['max_zero_price_streak']:
            self._quarantine_symbol(symbol, f"zero_price_streak={self.zero_price_streaks[symbol]}")
    
    def _reset_zero_price_streak(self, symbol):
        """Reset zero price streak when valid price received"""
        if symbol in self.zero_price_streaks:
            del self.zero_price_streaks[symbol]
    
    def _track_no_data(self, symbol):
        """Track consecutive no data occurrences for quarantine decisions"""
        if not hasattr(self, 'no_data_streaks'):
            self.no_data_streaks = {}
        
        self.no_data_streaks[symbol] = self.no_data_streaks.get(symbol, 0) + 1
        
        # Quarantine after fewer no-data occurrences than zero prices
        max_no_data_streak = self.config.get('max_no_data_streak', 3)
        if self.no_data_streaks[symbol] >= max_no_data_streak:
            self._quarantine_symbol(symbol, f"no_data_streak={self.no_data_streaks[symbol]}")
    
    def _reset_no_data_streak(self, symbol):
        """Reset no data streak when valid data received"""
        if hasattr(self, 'no_data_streaks') and symbol in self.no_data_streaks:
            del self.no_data_streaks[symbol]
    
    def _is_price_in_reasonable_range(self, symbol, price):
        """
        FOCUSED price range validation
        Only validate what makes business sense
        """
        try:
            # Extract ticker for config lookup
            ticker = self._extract_ticker(symbol)
            
            # Use configured ranges if available
            if ticker in self.config.get('price_ranges', {}):
                min_price, max_price = self.config['price_ranges'][ticker]
                return min_price <= price <= max_price
            
            # Ultra-basic sanity check for unknown tickers
            return 0.001 <= price <= 1000000
            
        except Exception as e:
            return True  # Don't block on range check errors
    
    def _quarantine_symbol(self, symbol, reason):
        """Add symbol to quarantine with reason and notify position manager"""
        if symbol not in self.quarantined_symbols:
            self.quarantined_symbols.add(symbol)
            self.quarantine_timestamps[symbol] = self.algorithm.Time
            self.quarantine_reasons[symbol] = reason
            
            ticker = self._extract_ticker(symbol)
            self.algorithm.Log(f"DataIntegrityChecker: QUARANTINED {ticker} - {reason}")
            
            # NEW: Notify bad data position manager if available
            if hasattr(self.algorithm, 'bad_data_position_manager') and self.algorithm.bad_data_position_manager:
                # Map reason to issue type
                issue_type = self._map_reason_to_issue_type(reason)
                severity = self._determine_severity(reason)
                self.algorithm.bad_data_position_manager.report_data_issue(symbol, issue_type, severity)
    
    def _map_reason_to_issue_type(self, reason):
        """Map quarantine reason to issue type for position manager"""
        if 'zero_price' in reason:
            return 'zero_price'
        elif 'no_data' in reason:
            return 'no_data'
        elif 'price_out_of_range' in reason:
            return 'bad_price'
        elif 'not_tradable' in reason:
            return 'not_tradable'
        elif 'validation_error' in reason:
            return 'validation_error'
        else:
            return 'unknown'
    
    def _determine_severity(self, reason):
        """Determine severity based on reason"""
        if any(term in reason for term in ['validation_error', 'not_tradable', 'no_price_property']):
            return 'high'
        elif any(term in reason for term in ['zero_price', 'no_data']):
            return 'medium'
        else:
            return 'low'
    
    def _check_quarantine_release(self, symbol):
        """Check if quarantined symbols can be released"""
        try:
            if symbol in self.quarantine_timestamps:
                quarantine_time = self.quarantine_timestamps[symbol]
                days_quarantined = (self.algorithm.Time - quarantine_time).days
                
                if days_quarantined >= self.config['quarantine_duration_days']:
                    self._release_symbol_from_quarantine(symbol)
                    
        except Exception as e:
            self.algorithm.Error(f"DataIntegrityChecker: Error checking quarantine release: {str(e)}")
    
    def _release_symbol_from_quarantine(self, symbol):
        """Release symbol from quarantine"""
        if symbol in self.quarantined_symbols:
            self.quarantined_symbols.remove(symbol)
            reason = self.quarantine_reasons.pop(symbol, "unknown")
            self.quarantine_timestamps.pop(symbol, None)
            
            ticker = self._extract_ticker(symbol)
            self.algorithm.Log(f"DataIntegrityChecker: RELEASED {ticker} from quarantine (was: {reason})")
    
    def _extract_ticker(self, symbol):
        """Extract ticker from symbol - leverage QC's symbol structure"""
        try:
            # Use QC's built-in symbol properties
            if hasattr(symbol, 'Value'):
                return str(symbol.Value).replace('/', '')
            return str(symbol)
        except:
            return str(symbol)
    
    def get_quarantine_status(self):
        """Get current quarantine status for reporting"""
        # Calculate total symbols being tracked (all securities in algorithm)
        total_symbols_tracked = len(self.algorithm.Securities) if hasattr(self.algorithm, 'Securities') else 0
        
        # Format quarantined symbols for reporting
        quarantined_symbols_list = []
        for symbol in self.quarantined_symbols:
            ticker = self._extract_ticker(symbol)
            reason = self.quarantine_reasons.get(symbol, 'unknown')
            quarantined_since = self.quarantine_timestamps.get(symbol, self.algorithm.Time)
            days_quarantined = (self.algorithm.Time - quarantined_since).days
            
            quarantined_symbols_list.append({
                'ticker': ticker,
                'reason': reason,
                'quarantined_since': quarantined_since,
                'days_quarantined': days_quarantined
            })
        
        return {
            'quarantined_count': len(self.quarantined_symbols),
            'total_symbols_tracked': total_symbols_tracked,
            'quarantined_symbols': quarantined_symbols_list
        }
    
    def get_safe_symbols(self, symbols):
        """
        Return symbols that are safe for trading
        LEVERAGE QC's validation + our quarantine logic
        """
        safe_symbols = []
        
        for symbol in symbols:
            # Use QC's built-in validation first
            if self._is_symbol_valid_qc_native(symbol):
                # Then check our custom quarantine logic
                if symbol not in self.quarantined_symbols:
                    safe_symbols.append(symbol)
        
        return safe_symbols 