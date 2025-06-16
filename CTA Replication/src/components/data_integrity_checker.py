class DataIntegrityChecker:
    """
    ENHANCED DATA INTEGRITY CHECKER WITH CENTRALIZED CACHING
    
    Purpose: 
    - Maximize QuantConnect's native capabilities instead of re-engineering
    - SOLVE CONCURRENCY ISSUES: Centralize all History API calls to prevent multiple strategies from calling simultaneously
    - Cache historical data to ensure consistency across all strategies
    - Uses Securities.HasData, IsTradable, Price validation
    - Leverages built-in symbol properties and market hours
    - Focuses only on what QC doesn't already provide
    """
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        
        # Import configuration
        try:
            from config.data_integrity_config import DATA_INTEGRITY_CONFIG
            self.config = DATA_INTEGRITY_CONFIG
        except ImportError:
            # Fallback minimal config
            self.config = {
                'max_zero_price_streak': 3,
                'quarantine_duration_days': 5,
                'price_ranges': {},
                'cache_max_age_hours': 24,
                'cache_cleanup_frequency_hours': 6,
                'max_cache_entries': 1000
            }
        
        # Lightweight tracking - let QC handle most validation
        self.quarantined_symbols = set()
        self.quarantine_timestamps = {}
        self.quarantine_reasons = {}
        
        # Track only what QC doesn't provide
        self.zero_price_streaks = {}  # Count consecutive zero prices
        self.last_quarantine_check = None
        
        # NEW: CENTRALIZED DATA CACHING to solve concurrency issues
        self.history_cache = {}  # {cache_key: DataFrame}
        self.cache_timestamps = {}  # {cache_key: timestamp}
        self.cache_requests = {}  # {cache_key: request_count} - for debugging
        self.last_cache_cleanup = None
        
        # NEW: Configuration for cache management
        self.cache_config = {
            'max_age_hours': self.config.get('cache_max_age_hours', 24),
            'cleanup_frequency_hours': self.config.get('cache_cleanup_frequency_hours', 6),
            'max_entries': self.config.get('max_cache_entries', 1000)
        }
        
        # NEW: Statistics tracking
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'api_calls_saved': 0,
            'total_requests': 0
        }
        
        self.algorithm.Log(f"DataIntegrityChecker: Enhanced with centralized caching (max_age={self.cache_config['max_age_hours']}h)")
    
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
            # Check if security has been mapped (for futures) - BUT ONLY FOR UNDERLYING CONTRACTS
            # Don't quarantine continuous contracts just because they don't have a .Mapped property
            symbol_str = str(symbol)
            if (hasattr(security, 'Mapped') and security.Mapped is None and 
                not symbol_str.startswith('/') and not symbol_str.startswith('futures/')):
                # Only quarantine underlying contracts that should have a mapped property but don't
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
    
    def get_history(self, symbol, periods, resolution=Resolution.Daily):
        """
        CENTRALIZED HISTORY PROVIDER - SOLVES CONCURRENCY ISSUES
        
        All strategies should call this method instead of algorithm.History directly.
        This prevents multiple strategies from calling the History API simultaneously
        and ensures data consistency across all strategies.
        
        Args:
            symbol: Symbol to get history for
            periods: Number of periods to get
            resolution: Resolution (default: Daily)
            
        Returns:
            pandas.DataFrame: Historical data or None if not available
        """
        try:
            # Create cache key
            cache_key = self._create_cache_key(symbol, periods, resolution)
            
            # Track request
            self.cache_stats['total_requests'] += 1
            self.cache_requests[cache_key] = self.cache_requests.get(cache_key, 0) + 1
            
            # Check cache first
            if self._is_cache_valid(cache_key):
                self.cache_stats['hits'] += 1
                cached_data = self.history_cache[cache_key]
                
                # Debug logging for high-frequency requests
                if self.cache_requests[cache_key] > 1:
                    self.cache_stats['api_calls_saved'] += 1
                    if self.cache_requests[cache_key] % 5 == 0:  # Log every 5th duplicate request
                        ticker = self._extract_ticker(symbol)
                        self.algorithm.Debug(f"DataCache: {ticker} served from cache {self.cache_requests[cache_key]} times (API calls saved: {self.cache_stats['api_calls_saved']})")
                
                return cached_data
            
            # Cache miss - validate symbol first
            if not self._is_symbol_valid_for_history(symbol):
                self.algorithm.Log(f"DataCache: Symbol {self._extract_ticker(symbol)} not valid for history request")
                return None
            
            # Make the actual API call
            self.cache_stats['misses'] += 1
            self.algorithm.Debug(f"DataCache: Fetching history for {self._extract_ticker(symbol)} (periods={periods}, resolution={resolution})")
            
            # Use QC's native History API
            history = self.algorithm.History(symbol, periods, resolution)
            
            if history is not None and not history.empty:
                # Cache successful result
                self.history_cache[cache_key] = history
                self.cache_timestamps[cache_key] = self.algorithm.Time
                
                # Log successful cache
                ticker = self._extract_ticker(symbol)
                self.algorithm.Log(f"DataCache: Cached {len(history)} bars for {ticker}")
                
                # Cleanup cache if needed
                self._cleanup_cache_if_needed()
                
                return history
            else:
                # Log failure but don't cache empty results
                ticker = self._extract_ticker(symbol)
                self.algorithm.Log(f"DataCache: No history data for {ticker} (periods={periods})")
                return None
                
        except Exception as e:
            self.algorithm.Error(f"DataCache: Error getting history for {symbol}: {str(e)}")
            return None
    
    def _create_cache_key(self, symbol, periods, resolution):
        """Create a unique cache key for the request."""
        return f"{str(symbol)}_{periods}_{str(resolution)}"
    
    def _is_cache_valid(self, cache_key):
        """Check if cached data is still valid."""
        if cache_key not in self.history_cache:
            return False
        
        if cache_key not in self.cache_timestamps:
            return False
        
        cache_age_hours = (self.algorithm.Time - self.cache_timestamps[cache_key]).total_seconds() / 3600
        return cache_age_hours < self.cache_config['max_age_hours']
    
    def _is_symbol_valid_for_history(self, symbol):
        """Check if symbol is valid for history requests (lighter validation than full QC native)."""
        try:
            # Check if quarantined
            if symbol in self.quarantined_symbols:
                return False
            
            # Check if symbol exists in securities
            if symbol not in self.algorithm.Securities:
                return False
            
            # Basic validation - don't be too aggressive for history requests
            security = self.algorithm.Securities[symbol]
            
            # Must have basic properties
            if not hasattr(security, 'Price'):
                return False
            
            return True
            
        except Exception as e:
            return False
    
    def _cleanup_cache_if_needed(self):
        """Clean up old cache entries if needed."""
        try:
            # Check if cleanup is needed
            if self.last_cache_cleanup is None:
                self.last_cache_cleanup = self.algorithm.Time
                return
            
            hours_since_cleanup = (self.algorithm.Time - self.last_cache_cleanup).total_seconds() / 3600
            if hours_since_cleanup < self.cache_config['cleanup_frequency_hours']:
                return
            
            # Cleanup is needed
            initial_count = len(self.history_cache)
            
            # Remove expired entries
            expired_keys = []
            for cache_key, timestamp in self.cache_timestamps.items():
                cache_age_hours = (self.algorithm.Time - timestamp).total_seconds() / 3600
                if cache_age_hours >= self.cache_config['max_age_hours']:
                    expired_keys.append(cache_key)
            
            # Remove expired entries
            for key in expired_keys:
                if key in self.history_cache:
                    del self.history_cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
                if key in self.cache_requests:
                    del self.cache_requests[key]
            
            # If still too many entries, remove oldest
            if len(self.history_cache) > self.cache_config['max_entries']:
                # Sort by timestamp and remove oldest
                sorted_keys = sorted(self.cache_timestamps.items(), key=lambda x: x[1])
                keys_to_remove = [key for key, _ in sorted_keys[:len(self.history_cache) - self.cache_config['max_entries']]]
                
                for key in keys_to_remove:
                    if key in self.history_cache:
                        del self.history_cache[key]
                    if key in self.cache_timestamps:
                        del self.cache_timestamps[key]
                    if key in self.cache_requests:
                        del self.cache_requests[key]
            
            final_count = len(self.history_cache)
            removed_count = initial_count - final_count
            
            if removed_count > 0:
                self.algorithm.Log(f"DataCache: Cleaned up {removed_count} expired entries ({final_count} remaining)")
            
            self.last_cache_cleanup = self.algorithm.Time
            
        except Exception as e:
            self.algorithm.Error(f"DataCache: Error during cleanup: {str(e)}")
    
    def get_cache_stats(self):
        """Get cache performance statistics."""
        total_requests = self.cache_stats['total_requests']
        if total_requests == 0:
            hit_rate = 0.0
        else:
            hit_rate = (self.cache_stats['hits'] / total_requests) * 100
        
        return {
            'total_requests': total_requests,
            'cache_hits': self.cache_stats['hits'],
            'cache_misses': self.cache_stats['misses'],
            'hit_rate_percent': round(hit_rate, 1),
            'api_calls_saved': self.cache_stats['api_calls_saved'],
            'cache_entries': len(self.history_cache),
            'max_cache_entries': self.cache_config['max_entries']
        }
    
    def clear_cache(self):
        """Clear all cached data (for testing or memory management)."""
        initial_count = len(self.history_cache)
        self.history_cache.clear()
        self.cache_timestamps.clear()
        self.cache_requests.clear()
        
        # Reset stats
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'api_calls_saved': 0,
            'total_requests': 0
        }
        
        self.algorithm.Log(f"DataCache: Cleared {initial_count} cache entries") 