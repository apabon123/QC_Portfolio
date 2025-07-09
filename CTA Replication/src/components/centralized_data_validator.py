# centralized_data_validator.py - SIMPLIFIED DATA VALIDATION
"""
Centralized Data Validation System - SIMPLE & QC NATIVE

Single source of truth for all data validation in the CTA system.
Uses QC's native methods and handles warmup vs trading periods appropriately.

DESIGN PRINCIPLES:
1. QC Native First - Use HasData, IsTradable, Price properties
2. Simple Logic - Clear warmup vs trading validation
3. Centralized - One method used everywhere
4. Modular - Easy to modify validation rules
5. MINIMAL LOGGING - Only log critical issues and summaries
"""

from AlgorithmImports import *
from datetime import timedelta

class CentralizedDataValidator:
    """
    Simple, centralized data validation using QC's native methods.
    
    Key Features:
    - Warmup: Lenient validation for continuous contracts (indicator building)
    - Trading: Strict validation for both continuous and mapped contracts
    - Portfolio: Validates existing positions to prevent mark-to-market errors
    - Outliers: Simple price range validation
    - MINIMAL LOGGING: Only critical issues and daily summaries
    """
    
    def __init__(self, algorithm, config_manager):
        self.algorithm = algorithm
        self.config_manager = config_manager
        
        # Simple validation configuration
        self.validation_config = {
            'min_price_threshold': 0.001,
            'max_price_multiplier': 10.0,  # Price can't be 10x previous price
            'enable_outlier_detection': True,
            'log_validation_details': False,  # REDUCED: Only log critical issues
            'log_daily_summary': True,       # Keep daily summary
            'log_outliers': True,            # Always log price outliers
            'log_slice_issues': True          # Log slice data issues
        }
        
        # Track last known good prices for outlier detection
        self.last_good_prices = {}  # symbol -> price
        
        # Track validation statistics for daily summary
        self.daily_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'outliers_detected': 0,
            'last_reset': algorithm.Time.date()
        }
        
        # Track first time we encountered each symbol (for fresh-contract grace period)
        self._symbol_first_seen = {}
        
        self.algorithm.Log("CentralizedDataValidator: Initialized (minimal logging mode)")
    
    def validate_symbol_for_trading(self, symbol, slice_data=None):
        """
        Comprehensive validation for trading: warmup-aware with slice data checking.
        
        Args:
            symbol: Symbol to validate
            slice_data: Current slice data (optional, for slice validation)
        
        Returns:
            dict: {'is_valid': bool, 'reason': str, 'trading_symbol': symbol}
        """
        # Update daily statistics
        self._update_daily_stats('total')
        
        # Choose validation method based on warmup status
        if self.algorithm.IsWarmingUp:
            result = self._validate_warmup_data(symbol, slice_data)
        else:
            result = self._validate_trading_data(symbol, slice_data)
        
        # Update success/failure stats
        if result['is_valid']:
            self._update_daily_stats('success')
        else:
            self._update_daily_stats('failed')
        
        return result
    
    def validate_existing_position(self, symbol):
        """
        Validate existing position for mark-to-market accuracy.
        Enhanced to detect OHLC-based portfolio valuation spikes.
        
        Args:
            symbol: Position symbol to validate
            
        Returns:
            dict: {'is_valid': bool, 'reason': str, 'safe_price': float or None}
        """
        # Update daily statistics
        self._update_daily_stats('total')
        
        try:
            if symbol not in self.algorithm.Securities:
                self._update_daily_stats('failed')
                return {'is_valid': False, 'reason': 'symbol_not_in_securities', 'safe_price': None}
            
            # Record first-seen timestamp
            if symbol not in self._symbol_first_seen:
                self._symbol_first_seen[symbol] = self.algorithm.Time
            
            security = self.algorithm.Securities[symbol]
            
            # QC Native: Basic validation
            if not security.HasData:
                self._update_daily_stats('failed')
                return {'is_valid': False, 'reason': 'no_data', 'safe_price': None}
            
            if not security.Price or security.Price <= 0:
                self._update_daily_stats('failed')
                return {'is_valid': False, 'reason': 'invalid_price', 'safe_price': None}
            
            # ENHANCED: Check for mark-to-market spike risks in existing positions
            mtm_validation = self._validate_position_mtm_risk(symbol, security)
            if mtm_validation['spike_detected']:
                # Position has mark-to-market spike risk - provide safe price
                safe_price = mtm_validation.get('safe_price')
                self.algorithm.Log(f"POSITION MTM SPIKE DETECTED: {symbol} - Using safe price ${safe_price:.2f}")
                return {'is_valid': True, 'reason': 'position_spike_detected', 'safe_price': safe_price}
            
            # Standard price outlier check
            if self._is_price_outlier(symbol, security.Price):
                self._update_daily_stats('outliers')
                
                # Try to provide a safe fallback price
                safe_price = self.last_good_prices.get(symbol)
                if safe_price:
                    self.algorithm.Log(f"POSITION PRICE OUTLIER: {symbol} @ ${security.Price:.2f}, using safe price ${safe_price:.2f}")
                    return {'is_valid': True, 'reason': 'price_outlier_with_fallback', 'safe_price': safe_price}
                else:
                    return {'is_valid': False, 'reason': 'price_outlier_no_fallback', 'safe_price': None}
            
            # Update last good price
            self.last_good_prices[symbol] = security.Price
            self._update_daily_stats('success')
            
            return {'is_valid': True, 'reason': 'position_valid', 'safe_price': security.Price}
            
        except Exception as e:
            self._update_daily_stats('failed')
            return {'is_valid': False, 'reason': f'position_validation_error', 'safe_price': None}
    
    def _validate_position_mtm_risk(self, symbol, security):
        """
        Detect mark-to-market spike risks in existing positions.
        Check if High/Low prices could cause portfolio valuation spikes.
        """
        try:
            spike_detected = False
            safe_price = security.Price
            
            # Get all available prices for the position
            prices = {}
            if hasattr(security, 'Open') and security.Open > 0:
                prices['Open'] = security.Open
            if hasattr(security, 'High') and security.High > 0:
                prices['High'] = security.High
            if hasattr(security, 'Low') and security.Low > 0:
                prices['Low'] = security.Low
            if hasattr(security, 'Close') and security.Close > 0:
                prices['Close'] = security.Close
            if security.Price > 0:
                prices['Price'] = security.Price
            
            # Check each price for extreme outliers
            outlier_prices = []
            normal_prices = []
            
            for price_type, price_value in prices.items():
                if self._is_price_outlier(symbol, price_value):
                    outlier_prices.append((price_type, price_value))
                    spike_detected = True
                    
                    # Log the specific outlier for debugging
                    self.algorithm.Log(f"POSITION MTM OUTLIER: {symbol} {price_type} @ ${price_value:.2f}")
                else:
                    normal_prices.append((price_type, price_value))
            
            # If we detected spikes, calculate a safe price from normal prices
            if spike_detected and normal_prices:
                # Use the most recent normal price (prefer Close, then Price)
                price_priority = ['Close', 'Price', 'Open', 'Low', 'High']
                for priority_type in price_priority:
                    for price_type, price_value in normal_prices:
                        if price_type == priority_type:
                            safe_price = price_value
                            break
                    if safe_price != security.Price:  # Found a better price
                        break
                
                # If no normal price found, use last known good price
                if not normal_prices:
                    safe_price = self.last_good_prices.get(symbol, security.Price)
            
            return {
                'spike_detected': spike_detected,
                'safe_price': safe_price,
                'outlier_prices': outlier_prices,
                'normal_prices': normal_prices
            }
            
        except Exception as e:
            return {
                'spike_detected': False,
                'safe_price': security.Price,
                'outlier_prices': [],
                'normal_prices': []
            }
    
    def _validate_warmup_data(self, symbol, slice_data=None):
        """
        Warmup validation: Lenient, focused on continuous contracts for indicators.
        """
        try:
            if symbol not in self.algorithm.Securities:
                return {'is_valid': False, 'reason': 'symbol_not_in_securities', 'trading_symbol': symbol}
            
            # Record first-seen timestamp
            if symbol not in self._symbol_first_seen:
                self._symbol_first_seen[symbol] = self.algorithm.Time
            
            security = self.algorithm.Securities[symbol]
            
            # Warmup: Just need data for indicator building
            has_data = security.HasData
            price_valid = security.Price > 0 if security.Price else False
            
            if has_data and price_valid:
                return {'is_valid': True, 'reason': 'warmup_valid', 'trading_symbol': symbol}
            else:
                return {'is_valid': False, 'reason': f'warmup_no_data', 'trading_symbol': symbol}
                
        except Exception as e:
            return {'is_valid': False, 'reason': f'warmup_error', 'trading_symbol': symbol}
    
    def _validate_trading_data(self, symbol, slice_data=None):
        """
        Trading validation: Strict, checks both continuous and mapped contracts.
        Enhanced with OHLC validation to prevent mark-to-market spikes.
        """
        try:
            if symbol not in self.algorithm.Securities:
                return {'is_valid': False, 'reason': 'symbol_not_in_securities', 'trading_symbol': symbol}
            
            # Record first-seen timestamp
            if symbol not in self._symbol_first_seen:
                self._symbol_first_seen[symbol] = self.algorithm.Time
            
            security = self.algorithm.Securities[symbol]
            
            # QC Native: Basic data validation
            if not security.HasData:
                return {'is_valid': False, 'reason': 'no_data', 'trading_symbol': symbol}
            
            if not security.Price or security.Price <= 0:
                return {'is_valid': False, 'reason': 'invalid_price', 'trading_symbol': symbol}
            
            # ENHANCED: Check ALL OHLC prices for outliers (prevents mark-to-market spikes)
            ohlc_validation = self._validate_ohlc_prices(symbol, security, slice_data)
            if not ohlc_validation['is_valid']:
                return ohlc_validation
            
            # For trading, we need current slice data
            if slice_data and not self._has_slice_data(symbol, slice_data):
                # Get mapped contract for better error message
                mapped_contract = security.Mapped if hasattr(security, 'Mapped') else symbol
                mapped_str = str(mapped_contract).split(' ')[-1] if ' ' in str(mapped_contract) else str(mapped_contract)
                
                # Grace period 1: Newly-added symbols (≤ 1 day)
                age = (self.algorithm.Time - self._symbol_first_seen.get(symbol, self.algorithm.Time)).total_seconds()

                # Grace period 2: Exchange holiday – the previous calendar day produced **no bar at all**.
                # Example: Monday holiday → daily bar for Tuesday will not exist when we run validation at 00:00 Tuesday.
                prev_calendar_day = (self.algorithm.Time - timedelta(days=1)).date()
                is_prev_day_holiday = not security.Exchange.Hours.IsDateOpen(prev_calendar_day)

                if not self.algorithm.IsWarmingUp and age >= 86400 and not is_prev_day_holiday:
                    # Only log missing slice data when it is *unexpected* (i.e., not holiday & not new symbol)
                    self.algorithm.Log(f"SLICE DATA MISSING: {mapped_str} not in current slice data")

                # Treat holiday-related absence as *valid* to avoid downgrading data quality.
                if is_prev_day_holiday:
                    return {'is_valid': True, 'reason': 'holiday_no_slice', 'trading_symbol': symbol}

                return {'is_valid': False, 'reason': 'no_current_slice_data', 'trading_symbol': symbol}
            
            # Use mapped contract for actual trading
            trading_symbol = security.Mapped if hasattr(security, 'Mapped') else symbol
            
            # Final validation: Check if mapped contract is tradeable
            if trading_symbol != symbol and trading_symbol in self.algorithm.Securities:
                mapped_security = self.algorithm.Securities[trading_symbol]
                if not mapped_security.IsTradable:
                    return {'is_valid': False, 'reason': 'mapped_not_tradable', 'trading_symbol': trading_symbol}
            
            return {'is_valid': True, 'reason': 'trading_valid', 'trading_symbol': trading_symbol}
            
        except Exception as e:
            return {'is_valid': False, 'reason': f'validation_error_{str(e)[:20]}', 'trading_symbol': symbol}
    
    def _validate_ohlc_prices(self, symbol, security, slice_data=None):
        """
        Validate Open, High, Low, Close prices to prevent mark-to-market spikes.
        QC can use any of these prices for portfolio valuation during the day.
        """
        try:
            prices_to_check = {}
            
            # Get current OHLC prices from security
            if hasattr(security, 'Open') and security.Open > 0:
                prices_to_check['Open'] = security.Open
            if hasattr(security, 'High') and security.High > 0:
                prices_to_check['High'] = security.High
            if hasattr(security, 'Low') and security.Low > 0:
                prices_to_check['Low'] = security.Low
            if hasattr(security, 'Close') and security.Close > 0:
                prices_to_check['Close'] = security.Close
            if security.Price > 0:
                prices_to_check['Price'] = security.Price
            
            # Also check slice data if available (more current OHLC)
            if slice_data and self._has_slice_data(symbol, slice_data):
                bar_data = self._get_bar_data_from_slice(symbol, slice_data)
                if bar_data:
                    if hasattr(bar_data, 'Open') and bar_data.Open > 0:
                        prices_to_check['SliceOpen'] = bar_data.Open
                    if hasattr(bar_data, 'High') and bar_data.High > 0:
                        prices_to_check['SliceHigh'] = bar_data.High
                    if hasattr(bar_data, 'Low') and bar_data.Low > 0:
                        prices_to_check['SliceLow'] = bar_data.Low
                    if hasattr(bar_data, 'Close') and bar_data.Close > 0:
                        prices_to_check['SliceClose'] = bar_data.Close
            
            # Check each price for outliers
            for price_type, price_value in prices_to_check.items():
                if self._is_price_outlier(symbol, price_value):
                    # CRITICAL: Log the specific price type causing the spike
                    self.algorithm.Log(f"OHLC PRICE OUTLIER: {symbol} {price_type} @ ${price_value:.2f}")
                    self._update_daily_stats('outliers')
                    
                    # For High/Low outliers, this could cause mark-to-market spikes
                    if price_type in ['High', 'SliceHigh', 'Low', 'SliceLow']:
                        self.algorithm.Log(f"MARK-TO-MARKET SPIKE RISK: {symbol} {price_type} could cause portfolio valuation spike")
                    
                    # Don't block trading, but log the issue for monitoring
                    # QC will still use these prices for mark-to-market regardless
            
            # Validate OHLC relationships (High >= Open,Close >= Low, etc.)
            ohlc_relationship_check = self._validate_ohlc_relationships(prices_to_check)
            if not ohlc_relationship_check:
                self.algorithm.Log(f"OHLC RELATIONSHIP ERROR: {symbol} - Invalid OHLC price relationships")
                # This indicates bad data that could cause valuation issues
            
            return {'is_valid': True, 'reason': 'ohlc_valid', 'trading_symbol': symbol}
            
        except Exception as e:
            return {'is_valid': False, 'reason': f'ohlc_validation_error', 'trading_symbol': symbol}
    
    def _get_bar_data_from_slice(self, symbol, slice_data):
        """Get bar data from slice for OHLC validation."""
        try:
            # Check multiple data sources for the symbol
            symbols_to_check = [symbol]
            if symbol in self.algorithm.Securities:
                security = self.algorithm.Securities[symbol]
                if hasattr(security, 'Mapped') and security.Mapped != symbol:
                    symbols_to_check.append(security.Mapped)
            
            for check_symbol in symbols_to_check:
                if hasattr(slice_data, 'Bars') and check_symbol in slice_data.Bars:
                    return slice_data.Bars[check_symbol]
                elif hasattr(slice_data, 'QuoteBars') and check_symbol in slice_data.QuoteBars:
                    return slice_data.QuoteBars[check_symbol]
            
            return None
        except Exception:
            return None
    
    def _validate_ohlc_relationships(self, prices):
        """Validate that OHLC prices have logical relationships."""
        try:
            # Get OHLC values (prefer slice data if available)
            open_price = prices.get('SliceOpen', prices.get('Open'))
            high_price = prices.get('SliceHigh', prices.get('High'))
            low_price = prices.get('SliceLow', prices.get('Low'))
            close_price = prices.get('SliceClose', prices.get('Close'))
            
            if not all([open_price, high_price, low_price, close_price]):
                return True  # Can't validate incomplete data
            
            # Basic OHLC relationship checks
            if high_price < max(open_price, close_price):
                return False  # High should be >= Open and Close
            if low_price > min(open_price, close_price):
                return False  # Low should be <= Open and Close
            if high_price < low_price:
                return False  # High should be >= Low
            
            return True
            
        except Exception:
            return True  # Don't block on validation errors
    
    def _is_price_outlier(self, symbol, current_price):
        """
        Simple outlier detection: Check if price is extremely different from last known good price.
        """
        if not self.validation_config.get('enable_outlier_detection', True):
            return False
        
        if symbol not in self.last_good_prices:
            return False  # No reference price yet
        
        last_price = self.last_good_prices[symbol]
        price_ratio = current_price / last_price
        max_multiplier = self.validation_config.get('max_price_multiplier', 10.0)
        
        # Price changed by more than max_multiplier (e.g., 10x)
        return price_ratio > max_multiplier or price_ratio < (1.0 / max_multiplier)
    
    def _update_daily_stats(self, stat_type):
        """Update daily validation statistics."""
        current_date = self.algorithm.Time.date()
        
        # Reset stats if new day
        if current_date != self.daily_stats['last_reset']:
            self._log_daily_summary()
            self._reset_daily_stats(current_date)
        
        # Update stats
        if stat_type == 'total':
            self.daily_stats['total_validations'] += 1
        elif stat_type == 'success':
            self.daily_stats['successful_validations'] += 1
        elif stat_type == 'failed':
            self.daily_stats['failed_validations'] += 1
        elif stat_type == 'outliers':
            self.daily_stats['outliers_detected'] += 1
    
    def _reset_daily_stats(self, current_date):
        """Reset daily statistics for new day."""
        self.daily_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'outliers_detected': 0,
            'last_reset': current_date
        }
    
    def _log_daily_summary(self):
        """Log daily validation summary - ONLY if there were issues."""
        if not self.validation_config.get('log_daily_summary', True):
            return
        
        stats = self.daily_stats
        
        # Only log if there were failures or outliers
        if stats['failed_validations'] > 0 or stats['outliers_detected'] > 0:
            success_rate = (stats['successful_validations'] / max(stats['total_validations'], 1)) * 100
            
            self.algorithm.Log(f"VALIDATION SUMMARY: {stats['total_validations']} checks, "
                             f"{success_rate:.1f}% success, "
                             f"{stats['failed_validations']} failures, "
                             f"{stats['outliers_detected']} outliers")
    
    def get_validation_summary(self):
        """Get summary of validation activity."""
        return {
            'total_symbols_tracked': len(self.last_good_prices),
            'daily_stats': self.daily_stats,
            'validation_config': self.validation_config
        }
    
    def _has_slice_data(self, symbol, slice_data):
        """
        Check if symbol has current slice data available for trading.
        For futures: Check both continuous contract and mapped contract data.
        """
        if not slice_data:
            return False
        
        try:
            # For futures, we need to check both continuous contract and mapped contract
            symbols_to_check = [symbol]
            
            # If this is a continuous contract, also check the mapped contract
            if symbol in self.algorithm.Securities:
                security = self.algorithm.Securities[symbol]
                if hasattr(security, 'Mapped') and security.Mapped != symbol:
                    symbols_to_check.append(security.Mapped)
            
            # Check if any of the symbols have current data in the slice
            for check_symbol in symbols_to_check:
                if self._symbol_has_slice_data(check_symbol, slice_data):
                    return True
            
            return False
        except Exception:
            return False
    
    def _symbol_has_slice_data(self, symbol, slice_data):
        """Check if a specific symbol has data in the current slice."""
        try:
            return (
                (hasattr(slice_data, 'Bars') and symbol in slice_data.Bars) or
                (hasattr(slice_data, 'QuoteBars') and symbol in slice_data.QuoteBars) or
                (hasattr(slice_data, 'Ticks') and symbol in slice_data.Ticks) or
                (hasattr(slice_data, 'FuturesChains') and symbol in slice_data.FuturesChains)
            )
        except Exception:
            return False 