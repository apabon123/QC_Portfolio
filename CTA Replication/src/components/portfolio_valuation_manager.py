"""
Portfolio Valuation Manager

Handles portfolio valuation issues for existing positions using QuantConnect's native methods.
Prevents "security does not have an accurate price" errors by validating existing positions
and coordinating with BadDataPositionManager for data quality issues.

This addresses the common multi-asset portfolio issue where:
1. We validate data before NEW trades (working correctly)
2. We DON'T validate data for EXISTING positions (causing errors)
3. QC tries to value existing positions with bad/missing data
4. Portfolio valuation fails with "accurate price" errors
"""

from AlgorithmImports import *

class PortfolioValuationManager:
    """
    Manages portfolio valuation for existing positions using QC's native methods.
    
    Key Functions:
    1. Validate existing positions before portfolio valuation
    2. Handle data quality issues for held positions
    3. Coordinate with BadDataPositionManager for position-specific strategies
    4. Use QC's native Portfolio and Securities properties
    """
    
    def __init__(self, algorithm, config_manager, bad_data_manager=None):
        self.algorithm = algorithm
        self.config_manager = config_manager
        self.bad_data_manager = bad_data_manager
        
        # Load configuration
        config = config_manager.get_full_config()
        self.valuation_config = config.get('portfolio_valuation', {
            'validate_positions_before_valuation': True,
            'use_last_good_price_for_valuation': True,
            'max_stale_price_minutes': 60,  # 1 hour
            'min_price_threshold': 0.001,
            'handle_missing_data_gracefully': True,
            'log_valuation_issues': True
        })
        
        # Track position valuation issues
        self.position_valuation_issues = {}  # symbol -> issue_info
        self.last_successful_valuations = {}  # symbol -> {price, timestamp}
        self.valuation_warnings = []
        
        self.algorithm.Log("PortfolioValuationManager: Initialized with QC native validation")
    
    def validate_portfolio_before_valuation(self):
        """
        Validate all existing positions before portfolio valuation.
        This prevents QC's "accurate price" errors during mark-to-market.
        
        Returns:
            dict: Validation results with any issues found
        """
        validation_results = {
            'total_positions': 0,
            'valid_positions': 0,
            'problematic_positions': 0,
            'issues': [],
            'can_proceed_with_valuation': True
        }
        
        try:
            # Use QC's native Portfolio.Values to iterate over all holdings
            for holding in self.algorithm.Portfolio.Values:
                if holding.Invested:  # QC's native property for active positions
                    validation_results['total_positions'] += 1
                    
                    # Validate this position using QC's native methods
                    position_validation = self._validate_single_position(holding)
                    
                    if position_validation['is_valid']:
                        validation_results['valid_positions'] += 1
                    else:
                        validation_results['problematic_positions'] += 1
                        validation_results['issues'].append(position_validation)
                        
                        # Report to bad data manager if available
                        if self.bad_data_manager:
                            self.bad_data_manager.report_data_issue(
                                holding.Symbol,
                                position_validation['issue_type'],
                                position_validation['severity']
                            )
            
            # Determine if we can safely proceed with portfolio valuation
            if validation_results['problematic_positions'] > 0:
                problematic_ratio = validation_results['problematic_positions'] / validation_results['total_positions']
                if problematic_ratio > 0.5:  # More than 50% of positions have issues
                    validation_results['can_proceed_with_valuation'] = False
                    self.algorithm.Error(f"Portfolio valuation blocked: {problematic_ratio:.1%} of positions have data issues")
            
            if self.valuation_config.get('log_valuation_issues', True):
                self._log_validation_summary(validation_results)
            
            return validation_results
            
        except Exception as e:
            self.algorithm.Error(f"PortfolioValuationManager: Validation failed: {str(e)}")
            validation_results['can_proceed_with_valuation'] = False
            return validation_results
    
    def _validate_single_position(self, holding):
        """
        Validate a single position using QC's native methods.
        
        Args:
            holding: QC's native SecurityHolding object
            
        Returns:
            dict: Validation results for this position
        """
        symbol = holding.Symbol
        ticker = self._extract_ticker(symbol)
        
        validation_result = {
            'symbol': symbol,
            'ticker': ticker,
            'is_valid': True,
            'issue_type': None,
            'severity': 'low',
            'current_price': None,
            'last_good_price': None,
            'position_value': float(holding.HoldingsValue),
            'quantity': float(holding.Quantity)
        }
        
        try:
            # Use centralized data validator if available
            if hasattr(self.algorithm, 'data_validator'):
                position_validation = self.algorithm.data_validator.validate_existing_position(symbol)
                
                if position_validation['is_valid']:
                    validation_result['current_price'] = position_validation['safe_price']
                    
                    # Store successful validation
                    self.last_successful_valuations[symbol] = {
                        'price': position_validation['safe_price'],
                        'timestamp': self.algorithm.Time,
                        'holdings_value': float(holding.HoldingsValue)
                    }
                    
                    return validation_result
                else:
                    # Handle validation failure
                    validation_result['is_valid'] = False
                    validation_result['issue_type'] = position_validation['reason']
                    
                    # Set severity based on issue type
                    if position_validation['reason'] in ['symbol_not_in_securities', 'no_data', 'invalid_price']:
                        validation_result['severity'] = 'high'
                    elif position_validation['reason'] in ['price_outlier']:
                        validation_result['severity'] = 'medium'
                        # Use safe price if available
                        if position_validation['safe_price']:
                            validation_result['last_good_price'] = position_validation['safe_price']
                    else:
                        validation_result['severity'] = 'medium'
                    
                    return validation_result
            
            # Fallback to basic QC validation if validator not available
            if symbol not in self.algorithm.Securities:
                validation_result['is_valid'] = False
                validation_result['issue_type'] = 'symbol_not_in_securities'
                validation_result['severity'] = 'high'
                return validation_result
            
            security = self.algorithm.Securities[symbol]
            
            # Use QC's native HasData property
            if not security.HasData:
                validation_result['is_valid'] = False
                validation_result['issue_type'] = 'no_data'
                validation_result['severity'] = 'high'
                return validation_result
            
            # Use QC's native Price property
            current_price = security.Price
            validation_result['current_price'] = float(current_price)
            
            # Validate price is reasonable
            if current_price is None or current_price <= 0:
                validation_result['is_valid'] = False
                validation_result['issue_type'] = 'invalid_price'
                validation_result['severity'] = 'high'
                return validation_result
            
            # Store successful validation
            self.last_successful_valuations[symbol] = {
                'price': current_price,
                'timestamp': self.algorithm.Time,
                'holdings_value': float(holding.HoldingsValue)
            }
            
            return validation_result
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['issue_type'] = 'validation_exception'
            validation_result['severity'] = 'high'
            validation_result['error_message'] = str(e)
            return validation_result
    
    def _validate_futures_position(self, security, holding):
        """Validate futures-specific position issues using QC's native methods"""
        validation_result = {'is_valid': True}
        
        try:
            # Check if continuous contract is properly mapped
            if hasattr(security, 'Mapped') and security.Mapped:
                mapped_contract = security.Mapped
                
                # Check if mapped contract exists and is tradeable
                if mapped_contract in self.algorithm.Securities:
                    mapped_security = self.algorithm.Securities[mapped_contract]
                    
                    # Use QC's native IsTradable property for mapped contract
                    if not mapped_security.IsTradable:
                        validation_result['is_valid'] = False
                        validation_result['issue_type'] = 'mapped_contract_not_tradeable'
                        validation_result['severity'] = 'medium'
                        return validation_result
                    
                    # Use mapped contract's price if continuous contract price is stale
                    if mapped_security.HasData and mapped_security.Price > 0:
                        validation_result['mapped_price'] = float(mapped_security.Price)
                else:
                    validation_result['is_valid'] = False
                    validation_result['issue_type'] = 'mapped_contract_missing'
                    validation_result['severity'] = 'high'
                    return validation_result
            
            # Check for rollover events
            current_slice = getattr(self.algorithm, 'current_slice', None)
            if current_slice and hasattr(current_slice, 'SymbolChangedEvents'):
                for symbol_changed in current_slice.SymbolChangedEvents.Values:
                    if symbol_changed.OldSymbol == holding.Symbol:
                        validation_result['rollover_detected'] = True
                        validation_result['new_symbol'] = symbol_changed.NewSymbol
            
            return validation_result
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['issue_type'] = 'futures_validation_error'
            validation_result['severity'] = 'high'
            validation_result['error_message'] = str(e)
            return validation_result
    
    def _check_price_staleness(self, symbol, current_price):
        """Check if price data is stale using QC's Time property"""
        staleness_info = {'is_fresh': True}
        
        try:
            # Check if we have a recent successful valuation
            if symbol in self.last_successful_valuations:
                last_valuation = self.last_successful_valuations[symbol]
                time_since_last = self.algorithm.Time - last_valuation['timestamp']
                
                max_stale_minutes = self.valuation_config.get('max_stale_price_minutes', 60)
                if time_since_last.total_seconds() > (max_stale_minutes * 60):
                    staleness_info['is_fresh'] = False
                    staleness_info['minutes_since_last'] = time_since_last.total_seconds() / 60
                    staleness_info['last_good_price'] = last_valuation['price']
            
            return staleness_info
            
        except Exception as e:
            staleness_info['is_fresh'] = True  # Default to fresh if check fails
            staleness_info['check_error'] = str(e)
            return staleness_info
    
    def get_safe_position_value(self, symbol):
        """
        Get position value using safe methods that won't trigger QC errors.
        
        Args:
            symbol: The symbol to get value for
            
        Returns:
            float: Position value or 0 if cannot be safely determined
        """
        try:
            # First check if we have the position
            if symbol not in self.algorithm.Portfolio:
                return 0.0
            
            holding = self.algorithm.Portfolio[symbol]
            if not holding.Invested:
                return 0.0
            
            # Try to get current holdings value using QC's native method
            try:
                return float(holding.HoldingsValue)
            except:
                # If HoldingsValue fails, use last good price if available
                if symbol in self.last_successful_valuations:
                    quantity = float(holding.Quantity)
                    last_price = self.last_successful_valuations[symbol]['price']
                    return quantity * last_price
                
                return 0.0
                
        except Exception as e:
            self.algorithm.Log(f"PortfolioValuationManager: Safe value calculation failed for {symbol}: {str(e)}")
            return 0.0
    
    def get_total_portfolio_value_safe(self):
        """
        Calculate total portfolio value using safe methods.
        
        Returns:
            float: Total portfolio value or sum of individual positions if QC method fails
        """
        try:
            # Try QC's native method first
            total_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
            if total_value > 0:
                return total_value
        except:
            pass
        
        # Calculate manually using safe position values
        total_value = float(self.algorithm.Portfolio.Cash)  # Start with cash
        
        for holding in self.algorithm.Portfolio.Values:
            if holding.Invested:
                position_value = self.get_safe_position_value(holding.Symbol)
                total_value += position_value
        
        return total_value
    
    def _is_futures_symbol(self, symbol):
        """Check if symbol is a futures contract"""
        try:
            return hasattr(symbol, 'SecurityType') and symbol.SecurityType == SecurityType.Future
        except:
            return False
    
    def _extract_ticker(self, symbol):
        """Extract ticker from symbol"""
        try:
            if hasattr(symbol, 'Value'):
                return str(symbol.Value).replace('/', '')
            return str(symbol).split()[0] if ' ' in str(symbol) else str(symbol)
        except:
            return str(symbol)
    
    def _log_validation_summary(self, validation_results):
        """Smart validation logging - only log when issues occur or significant changes"""
        # Track state changes to avoid repetitive logging
        if not hasattr(self, '_last_validation_state'):
            self._last_validation_state = {}
        
        current_state = {
            'total_positions': validation_results['total_positions'],
            'problematic_positions': validation_results['problematic_positions'],
            'can_proceed': validation_results['can_proceed_with_valuation']
        }
        
        # Only log if there are issues OR state changed significantly
        should_log = False
        log_reason = ""
        
        # Always log if there are problems
        if validation_results['problematic_positions'] > 0:
            should_log = True
            log_reason = "validation_issues"
        
        # Log if position count changed significantly (new positions or major liquidation)
        elif self._last_validation_state.get('total_positions', 0) != current_state['total_positions']:
            position_change = abs(current_state['total_positions'] - self._last_validation_state.get('total_positions', 0))
            if position_change >= 1:  # At least 1 position change
                should_log = True
                log_reason = "position_change"
        
        # Log if validation status changed (blocked <-> allowed)
        elif self._last_validation_state.get('can_proceed', True) != current_state['can_proceed']:
            should_log = True
            log_reason = "validation_status_change"
        
        # Periodic summary (once per week) for healthy portfolios
        elif not hasattr(self, '_last_summary_log') or (self.algorithm.Time - self._last_summary_log).days >= 7:
            if validation_results['total_positions'] > 0 and validation_results['problematic_positions'] == 0:
                should_log = True
                log_reason = "weekly_healthy_summary"
                self._last_summary_log = self.algorithm.Time
        
        if should_log:
            # Compact logging format
            if log_reason == "validation_issues":
                self.algorithm.Log(f"PORTFOLIO VALIDATION ISSUES: {validation_results['problematic_positions']}/{validation_results['total_positions']} positions problematic")
                for issue in validation_results['issues'][:3]:  # Max 3 issues
                    ticker = issue.get('ticker', 'Unknown')
                    issue_type = issue.get('issue_type', 'Unknown')
                    self.algorithm.Log(f"  {ticker}: {issue_type}")
            
            elif log_reason == "position_change":
                old_count = self._last_validation_state.get('total_positions', 0)
                new_count = current_state['total_positions']
                self.algorithm.Log(f"Portfolio positions: {old_count} -> {new_count} ({validation_results['valid_positions']} valid)")
            
            elif log_reason == "validation_status_change":
                status = "BLOCKED" if not current_state['can_proceed'] else "ALLOWED"
                self.algorithm.Log(f"Portfolio valuation status changed: {status}")
            
            elif log_reason == "weekly_healthy_summary":
                self.algorithm.Log(f"Portfolio healthy: {validation_results['total_positions']} positions, all valid")
        
        # Update state tracking
        self._last_validation_state = current_state
    
    def get_status_report(self):
        """Get status report for debugging"""
        return {
            'total_tracked_positions': len(self.last_successful_valuations),
            'position_issues': len(self.position_valuation_issues),
            'config': self.valuation_config,
            'last_validation_time': getattr(self, 'last_validation_time', None)
        } 