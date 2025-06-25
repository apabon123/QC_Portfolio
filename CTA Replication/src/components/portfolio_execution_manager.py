# portfolio_execution_manager.py - CONFIG-COMPLIANT VERSION
"""
Portfolio Execution Manager for Three-Layer CTA System - CONFIG-COMPLIANT VERSION

This component handles all trade execution, position monitoring, and transaction analysis.
FIXED: Now properly uses all config parameters instead of hardcoded values.
"""

from AlgorithmImports import *

class PortfolioExecutionManager:
    """
    Handles portfolio execution, monitoring, and transaction analysis.
    
    CONFIG-COMPLIANT VERSION:
    - Uses config.execution.min_trade_value ($1,000) instead of hardcoded $100
    - Uses config.execution.min_weight_change (1%) instead of hardcoded 0.1%
    - Uses config.constraints.min_capital ($5M) instead of hardcoded $10k
    - Uses config.execution.max_single_order_value from config
    - Uses config.portfolio_risk parameters for validation
    """
    
    def __init__(self, algorithm, config_manager):
        """
        Initialize Portfolio Execution Manager with comprehensive config support.
        
        Args:
            algorithm: QuantConnect algorithm instance
            config_manager: Configuration manager instance
        """
        self.algorithm = algorithm
        self.config_manager = config_manager
        
        # Load comprehensive configuration
        config = config_manager.get_full_config()
        self.execution_config = config.get('execution', {})
        self.risk_config = config.get('portfolio_risk', {})
        self.constraints_config = config.get('constraints', {})
        
        # NEW: QC Native Features Configuration
        self.qc_config = config.get('qc_native', {})
        self.use_qc_transactions = self.qc_config.get('order_management', {}).get('use_qc_transactions', True)
        self.use_qc_order_events = self.qc_config.get('order_management', {}).get('use_qc_order_events', True)
        self.custom_fill_tracking = self.qc_config.get('order_management', {}).get('custom_fill_tracking', False)
        self.log_order_events = self.qc_config.get('order_management', {}).get('log_order_events', True)
        
        # INTEGRATION: Bad Data Position Manager (injected from main.py)
        self.bad_data_manager = None  # Will be set by main.py
        
        # QC NATIVE: Use QC's built-in order and transaction tracking instead of custom arrays
        # self.trade_history = []  # REMOVED: Use algorithm.Transactions instead
        # self.total_trades_executed = 0  # REMOVED: Use len(algorithm.Transactions.GetOrders()) instead
        self.execution_errors = 0  # Keep for error tracking only
        self.last_portfolio_snapshot = {}
        
        # Log configuration
        self.algorithm.Log(f"ExecutionManager: QC Transactions {'ENABLED' if self.use_qc_transactions else 'DISABLED'}")
        self.algorithm.Log(f"ExecutionManager: QC OrderEvents {'ENABLED' if self.use_qc_order_events else 'DISABLED'}")
        self.algorithm.Log(f"ExecutionManager: Custom Fill Tracking {'ENABLED' if self.custom_fill_tracking else 'DISABLED'}")
        
        # Log configuration parameters
        self._log_config_parameters()
        self._log_capital_diagnostics()
    
    def _extract_ticker_from_symbol(self, symbol):
        """Extract ticker from symbol - simplified since QC handles the heavy lifting."""
        try:
            # For continuous contracts, use the symbol directly
            if hasattr(symbol, 'Value'):
                return symbol.Value
            return str(symbol)
        except:
            return str(symbol)
    
    def _log_config_parameters(self):
        """Log all config parameters being used for transparency."""
        self.algorithm.Log("CONFIG-COMPLIANT EXECUTION PARAMETERS:")
        self.algorithm.Log(f"  Min trade value: ${self.execution_config['min_trade_value']:,} (from config)")
        self.algorithm.Log(f"  Min weight change: {self.execution_config['min_weight_change']:.1%} (from config)")
        self.algorithm.Log(f"  Max single order: ${self.execution_config['max_single_order_value']:,} (from config)")
        
        self.algorithm.Log("CONFIG-COMPLIANT RISK PARAMETERS:")
        self.algorithm.Log(f"  Max single position: {self.risk_config['max_single_position']:.1f} ({self.risk_config['max_single_position']*100:.0f}%) (from config)")
        self.algorithm.Log(f"  Daily stop loss: {self.risk_config['daily_stop_loss']:.1%} (from config)")
        self.algorithm.Log(f"  Target portfolio vol: {self.risk_config['target_portfolio_vol']:.1%} (from config)")
        
        self.algorithm.Log("CONFIG-COMPLIANT CONSTRAINT PARAMETERS:")
        self.algorithm.Log(f"  Min capital: ${self.constraints_config['min_capital']:,} (from config)")
        self.algorithm.Log(f"  Initial capital: ${self.constraints_config['initial_capital']:,} (from config)")
    
    def _log_capital_diagnostics(self):
        """Log capital diagnostics with config comparison."""
        current_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
        current_cash = float(self.algorithm.Portfolio.Cash)
        current_holdings = float(self.algorithm.Portfolio.TotalHoldingsValue)
        
        self.algorithm.Log("CAPITAL DIAGNOSTICS (CONFIG COMPARISON):")
        self.algorithm.Log(f"  Current Portfolio Value: ${current_value:,.0f}")
        self.algorithm.Log(f"  Config Initial Capital: ${self.constraints_config['initial_capital']:,}")
        self.algorithm.Log(f"  Config Min Capital: ${self.constraints_config['min_capital']:,}")
        
        # Check if we're starting with the right amount
        if abs(current_value - self.constraints_config['initial_capital']) > 1000:
            self.algorithm.Log(f"  WARNING: Current value ${current_value:,} differs from config ${self.constraints_config['initial_capital']:,}")
        else:
            self.algorithm.Log(f"  ✓ Capital matches config expectation")
        
        self.algorithm.Log(f"  Cash: ${current_cash:,.0f}")
        self.algorithm.Log(f"  Holdings: ${current_holdings:,.0f}")
    
    def execute_rebalance_result(self, result):
        """Execute the rebalancing result from the orchestrator - CONFIG-COMPLIANT VERSION."""
        self.algorithm.Log("=" * 60)
        self.algorithm.Log("EXECUTION MANAGER: Starting config-compliant rebalance execution")
        self.algorithm.Log("=" * 60)
        
        final_targets = result.get('final_targets', {})
        rebalance_type = result.get('rebalance_type', 'unknown')
        
        # DIAGNOSTIC: Log what we received
        self.algorithm.Log(f"EXECUTION INPUT:")
        self.algorithm.Log(f"  Rebalance Type: {rebalance_type}")
        self.algorithm.Log(f"  Final Targets: {len(final_targets)} positions")
        for symbol, weight in final_targets.items():
            # Extract ticker from symbol string (e.g., "ES ABCDEFG" -> "ES")
            ticker = str(symbol).split()[0] if ' ' in str(symbol) else str(symbol)
            self.algorithm.Log(f"    {ticker}: {weight:.1%}")
        
        if not final_targets:
            self.algorithm.Log(f"EXECUTION: No targets to execute")
            return self._create_empty_execution_summary(rebalance_type)
        
        self.algorithm.Log(f"EXECUTION: Processing {len(final_targets)} target positions...")
        
        # Get rollover information for execution
        rollover_tags = {}
        if hasattr(self.algorithm, 'rollover_manager'):
            try:
                rollover_tags = self.algorithm.rollover_manager.get_rollover_tags_for_rebalance()
            except:
                rollover_tags = {}
        
        # CONFIG-COMPLIANT: Use config-based validation
        validation_result = self._validate_targets_config_compliant(final_targets)
        if not validation_result['valid']:
            self.algorithm.Log(f"EXECUTION BLOCKED BY CONFIG VALIDATION: {validation_result['reason']}")
            return self._create_error_execution_summary(rebalance_type, validation_result['reason'])
        
        # Execute the portfolio changes
        execution_summary = self._execute_final_portfolio(final_targets, rollover_tags)
        execution_summary['rebalance_type'] = rebalance_type
        
        # Mark rollover re-establishments as completed
        if rollover_tags and hasattr(self.algorithm, 'rollover_manager'):
            try:
                self.algorithm.rollover_manager.mark_rollover_reestablishments_complete(rollover_tags.keys())
            except:
                pass
        
        # Post-execution analysis and logging
        self._post_execution_analysis(execution_summary, final_targets)
        
        self.algorithm.Log("=" * 60)
        self.algorithm.Log("EXECUTION MANAGER: Config-compliant rebalance execution complete")
        self.algorithm.Log("=" * 60)
        
        return execution_summary
    
    def _validate_targets_config_compliant(self, final_targets):
        """
        CONFIG-COMPLIANT validation using actual config parameters.
        """
        self.algorithm.Log("EXECUTION VALIDATION: Using CONFIG-COMPLIANT validation")
        
        # Check individual position limits using config
        max_single_position = self.risk_config['max_single_position']
        for symbol, weight in final_targets.items():
            if abs(weight) > max_single_position:
                # Extract ticker from symbol string
                ticker = str(symbol).split()[0] if ' ' in str(symbol) else str(symbol)
                reason = f'{ticker} position {weight:.1%} exceeds config limit {max_single_position:.1%}'
                self.algorithm.Log(f"VALIDATION BLOCK: {reason}")
                return {'valid': False, 'reason': reason}
        
        # Check total exposure against reasonable limits
        total_gross = sum(abs(w) for w in final_targets.values())
        max_reasonable_gross = 50.0  # 5000% still reasonable for futures
        if total_gross > max_reasonable_gross:
            reason = f'Total gross exposure {total_gross:.1%} exceeds reasonable limit {max_reasonable_gross:.1%}'
            self.algorithm.Log(f"VALIDATION BLOCK: {reason}")
            return {'valid': False, 'reason': reason}
        
        # CONFIG-COMPLIANT: Check minimum capital using config
        current_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
        min_capital = self.constraints_config['min_capital']  # $5M from config
        
        if current_value < min_capital:
            reason = f'Portfolio value ${current_value:,} below config minimum ${min_capital:,}'
            self.algorithm.Log(f"VALIDATION BLOCK: {reason}")
            return {'valid': False, 'reason': reason}
        
        # Check maximum number of positions
        position_count = len([w for w in final_targets.values() if abs(w) > 0.001])
        max_total_positions = self.constraints_config['max_total_positions']
        
        if position_count > max_total_positions:
            reason = f'Position count {position_count} exceeds config limit {max_total_positions}'
            self.algorithm.Log(f"VALIDATION BLOCK: {reason}")
            return {'valid': False, 'reason': reason}
        
        self.algorithm.Log(f"CONFIG VALIDATION PASSED:")
        self.algorithm.Log(f"  Positions: {len(final_targets)} (max: {max_total_positions})")
        self.algorithm.Log(f"  Gross exposure: {total_gross:.1%}")
        self.algorithm.Log(f"  Max position: {max([abs(w) for w in final_targets.values()]):.1%} (limit: {max_single_position:.1%})")
        self.algorithm.Log(f"  Capital: ${current_value:,} (min: ${min_capital:,})")
        
        return {'valid': True, 'reason': None}
    
    def _execute_final_portfolio(self, final_targets, rollover_tags=None):
        """Execute the final portfolio targets with config-compliant parameters."""
        if rollover_tags is None:
            rollover_tags = {}
        
        # Initialize execution tracking
        execution_summary = {
            'orders_placed': 0,
            'liquidations': 0,
            'rollover_orders': 0,
            'total_trades': 0,
            'execution_errors': 0,
            'trades_by_ticker': {},
            'total_trade_value': 0,
            'position_changes': {},
            'blocked_trades': [],
            'successful_trades': [],
            'config_parameters_used': {
                'min_trade_value': self.execution_config['min_trade_value'],
                'min_weight_change': self.execution_config['min_weight_change'],
                'max_single_order_value': self.execution_config['max_single_order_value']
            }
        }
        
        total_portfolio_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
        self.algorithm.Log(f"EXECUTION: Portfolio value for calculations: ${total_portfolio_value:,.0f}")
        
        # Pre-filter targets to only include symbols with valid data
        valid_targets = {}
        for symbol, weight in final_targets.items():
            if self._is_symbol_ready_for_execution(symbol):
                valid_targets[symbol] = weight
            else:
                ticker = self._extract_ticker_from_symbol(symbol)
                self.algorithm.Log(f"  SKIPPING {ticker}: No valid data available")
                execution_summary['blocked_trades'].append({
                    'ticker': ticker,
                    'target_weight': weight,
                    'reason': 'No valid data available'
                })
        
        if not valid_targets:
            self.algorithm.Log("  No symbols with valid data - skipping execution")
            return execution_summary
        
        self.algorithm.Log(f"  Executing {len(valid_targets)}/{len(final_targets)} symbols with valid data")
        final_targets = valid_targets
        
        # Execute target positions
        for symbol, target_weight in final_targets.items():
            self.algorithm.Log(f"EXECUTING: {symbol} target {target_weight:.1%}")
            
            trade_result = self._execute_single_position_config_compliant(
                symbol, target_weight, total_portfolio_value, rollover_tags
            )
            
            # Track results with detailed logging
            if trade_result['executed']:
                execution_summary['orders_placed'] += 1
                execution_summary['total_trade_value'] += abs(trade_result['trade_value'])
                execution_summary['successful_trades'].append({
                    'ticker': trade_result['ticker'],
                    'quantity': trade_result['quantity'],
                    'trade_value': trade_result['trade_value']
                })
                
                if trade_result['is_rollover']:
                    execution_summary['rollover_orders'] += 1
                
                # Track by ticker
                ticker = trade_result['ticker']
                execution_summary['trades_by_ticker'][ticker] = {
                    'quantity': trade_result['quantity'],
                    'trade_value': trade_result['trade_value'],
                    'is_rollover': trade_result['is_rollover']
                }
                
                # Track position changes
                execution_summary['position_changes'][symbol] = {
                    'old_weight': trade_result['old_weight'],
                    'new_weight': target_weight,
                    'weight_change': target_weight - trade_result['old_weight']
                }
            elif trade_result.get('blocked_reason'):
                execution_summary['blocked_trades'].append({
                    'ticker': trade_result['ticker'],
                    'target_weight': target_weight,
                    'reason': trade_result['blocked_reason']
                })
            
            if trade_result['error']:
                execution_summary['execution_errors'] += 1
        
        # Liquidate positions not in final targets
        liquidation_result = self._liquidate_excluded_positions(final_targets)
        execution_summary['liquidations'] = liquidation_result['liquidations']
        execution_summary['execution_errors'] += liquidation_result['errors']
        
        # Calculate totals
        execution_summary['total_trades'] = (
            execution_summary['orders_placed'] + execution_summary['liquidations']
        )
        
        # Update tracking
        # self.total_trades_executed += execution_summary['orders_placed']  # REMOVED: Use len(algorithm.Transactions.GetOrders()) instead
        self.execution_errors += execution_summary['execution_errors']
        
        # Log execution results
        self._log_execution_diagnostics(execution_summary)
        
        return execution_summary
    
    def _execute_single_position_config_compliant(self, symbol, target_weight, total_portfolio_value, rollover_tags):
        """
        Execute a single position trade using CONFIG-COMPLIANT parameters.
        """
        result = {
            'executed': False,
            'error': False,
            'ticker': '',
            'quantity': 0,
            'trade_value': 0,
            'old_weight': 0,
            'is_rollover': False,
            'blocked_reason': None
        }
        
        # Get ticker for logging
        try:
            result['ticker'] = self._extract_ticker_from_symbol(symbol)
        except:
            result['ticker'] = symbol
        
        self.algorithm.Log(f"  EXECUTING {result['ticker']} ({symbol}):")
        
        # Check if symbol exists in securities
        if symbol not in self.algorithm.Securities:
            result['error'] = True
            result['blocked_reason'] = f"Symbol {symbol} not in securities"
            self.algorithm.Log(f"    ERROR: {result['blocked_reason']}")
            return result
        
        # Get mapped contract
        try:
            mapped_contract = self.algorithm.Securities[symbol].Mapped
            if mapped_contract is None:
                result['error'] = True
                result['blocked_reason'] = f"No mapped contract for {symbol}"
                self.algorithm.Log(f"    ERROR: {result['blocked_reason']}")
                return result
            
            security = self.algorithm.Securities[mapped_contract]
            
            # ROLLOVER FIX: Use QC's recommended slice.Contains() pattern
            # During rollover, new contract may not have HasData yet but can still be traded
            continuous_security = self.algorithm.Securities[symbol]
            current_slice = getattr(self.algorithm, 'current_slice', None)
            
            if not security.HasData:
                # Check if this is a rollover situation (continuous has data but mapped doesn't)
                if continuous_security.HasData:
                    # Use QC's recommended slice.Contains() pattern for rollover validation
                    if current_slice and current_slice.ContainsKey(mapped_contract):
                        price_info = f"slice price available"
                        if hasattr(security, 'Price') and security.Price and security.Price > 0:
                            price_info = f"price ${security.Price:.2f}"
                        
                        self.algorithm.Log(f"    ROLLOVER: New contract {mapped_contract} ready via slice ({price_info})")
                        # Continue execution - QC slice contains the data we need
                    else:
                        # Check if price is available even without HasData
                        if hasattr(security, 'Price') and security.Price and security.Price > 0:
                            price_info = f"price ${security.Price:.2f}"
                            self.algorithm.Log(f"    ROLLOVER: New contract {mapped_contract} has {price_info}, proceeding")
                            # Continue execution - price is available for trading
                        else:
                            self.algorithm.Log(f"    ROLLOVER: New contract {mapped_contract} not ready yet, skipping")
                            result['error'] = True
                            result['blocked_reason'] = f"Rollover contract {mapped_contract} not ready"
                            return result
                else:
                    result['error'] = True
                    result['blocked_reason'] = f"No data for mapped contract {mapped_contract}"
                    self.algorithm.Log(f"    ERROR: {result['blocked_reason']}")
                    return result
        except Exception as e:
            result['error'] = True
            result['blocked_reason'] = f"Error accessing security: {str(e)}"
            self.algorithm.Log(f"    ERROR: {result['blocked_reason']}")
            return result
        
        # Validate price with proper data checks
        try:
            # Check if security has received data and has a valid price
            if not hasattr(security, 'Price') or security.Price is None:
                result['error'] = True
                result['blocked_reason'] = f"Security {mapped_contract} has no price data"
                self.algorithm.Log(f"    ERROR: {result['blocked_reason']}")
                return result
            
            # REMOVED: Duplicate slice validation - now handled by centralized validator
            # The _is_symbol_ready_for_execution method already validates slice data
            # using our enhanced CentralizedDataValidator
            
            # Additional QC native validation - ROLLOVER-AWARE
            if not security.HasData or security.Price <= 0:
                # During rollover, new contract might not have data yet but price might be available
                if continuous_security.HasData and hasattr(security, 'Price') and security.Price and security.Price > 0:
                    old_price = continuous_security.Price if hasattr(continuous_security, 'Price') else 0
                    price_diff = security.Price - old_price if old_price > 0 else 0
                    self.algorithm.Log(f"    ROLLOVER: Using price ${security.Price:.2f} from new contract {mapped_contract} (vs ${old_price:.2f} old, diff: ${price_diff:+.2f})")
                    # Continue with execution using available price
                else:
                    result['error'] = True
                    result['blocked_reason'] = f"Security {mapped_contract} has invalid price data: {security.Price}"
                    self.algorithm.Log(f"    ERROR: {result['blocked_reason']}")
                    return result
            
            price = float(security.Price)
            
        except Exception as e:
            result['error'] = True
            result['blocked_reason'] = f"Error accessing price for {mapped_contract}: {str(e)}"
            self.algorithm.Log(f"    ERROR: {result['blocked_reason']}")
            return result
        
        self.algorithm.Log(f"    Price: ${price:.2f}")
        
        # Calculate current position
        try:
            holdings_value = float(self.algorithm.Portfolio[mapped_contract].HoldingsValue)
            current_weight = holdings_value / total_portfolio_value if total_portfolio_value != 0 else 0
            result['old_weight'] = current_weight
            
            self.algorithm.Log(f"    Current: {current_weight:.1%} (${holdings_value:,.0f})")
            self.algorithm.Log(f"    Target:  {target_weight:.1%}")
        except Exception as e:
            result['error'] = True
            result['blocked_reason'] = f"Error calculating current position: {str(e)}"
            self.algorithm.Log(f"    ERROR: {result['blocked_reason']}")
            return result
        
        # CONFIG-COMPLIANT: Check if trade is needed using config threshold
        weight_diff = abs(current_weight - target_weight)
        min_weight_change = self.execution_config['min_weight_change']  # 1% from config
        
        self.algorithm.Log(f"    Weight diff: {weight_diff:.3%} (config min: {min_weight_change:.3%})")
        
        if weight_diff <= min_weight_change:
            result['blocked_reason'] = f"Weight change {weight_diff:.3%} below config threshold {min_weight_change:.3%}"
            self.algorithm.Log(f"    SKIPPED: {result['blocked_reason']}")
            return result
        
        # Calculate trade details
        target_value = target_weight * total_portfolio_value
        trade_value = target_value - holdings_value
        result['trade_value'] = trade_value
        
        self.algorithm.Log(f"    Target value: ${target_value:,.0f}")
        self.algorithm.Log(f"    Trade value:  ${trade_value:+,.0f}")
        
        # CONFIG-COMPLIANT: Apply minimum trade value filter from config
        min_trade_value = self.execution_config['min_trade_value']  # $1,000 from config
        if abs(trade_value) <= min_trade_value:
            result['blocked_reason'] = f"Trade value ${abs(trade_value):,.0f} below config minimum ${min_trade_value:,.0f}"
            self.algorithm.Log(f"    BLOCKED: {result['blocked_reason']}")
            return result
        
        # INTEGRATION: Bad Data Position Manager - Check if new trades allowed
        if self.bad_data_manager and not self.bad_data_manager.should_allow_new_trade(symbol):
            result['blocked_reason'] = f"Bad data manager blocked new trade for {result['ticker']}"
            self.algorithm.Log(f"    BLOCKED: {result['blocked_reason']}")
            
            # Log the specific bad data strategy for this symbol
            if symbol in self.bad_data_manager.problematic_positions:
                strategy = self.bad_data_manager.problematic_positions[symbol]['strategy']
                issue_count = self.bad_data_manager.problematic_positions[symbol]['issue_count']
                self.algorithm.Log(f"    BAD DATA: Strategy={strategy}, Issues={issue_count}")
            
            return result
        
        try:
            # Calculate quantity using QC's native properties (no custom futures manager)
            contract_multiplier = security.SymbolProperties.ContractMultiplier
            multiplier = contract_multiplier if contract_multiplier > 0 else 1.0
            
            self.algorithm.Log(f"    Multiplier: {multiplier}")
            
            quantity_diff = int(trade_value / (price * multiplier))
            
            # CRITICAL FIX: Explicit zero quantity check to prevent QC error
            if quantity_diff == 0:
                result['blocked_reason'] = f"Calculated quantity is exactly zero"
                self.algorithm.Log(f"    BLOCKED: {result['blocked_reason']}")
                return result
            
            if abs(quantity_diff) < 1:
                result['blocked_reason'] = f"Quantity {quantity_diff} rounds to less than 1 contract"
                self.algorithm.Log(f"    BLOCKED: {result['blocked_reason']}")
                return result
            
            result['quantity'] = quantity_diff
            self.algorithm.Log(f"    Quantity: {quantity_diff:+d} contracts")
            
            # Check for rollover re-establishment
            trade_tag = "ExecutionManager"
            if symbol in rollover_tags:
                rollover_info = rollover_tags[symbol]
                trade_tag = f"ExecutionManager_{rollover_info['tag']}"
                result['is_rollover'] = True
                self.algorithm.Log(f"    ROLLOVER: {rollover_info['tag']}")
            
            # CONFIG-COMPLIANT: Apply single order size limit from config
            max_order_value = self.execution_config['max_single_order_value']  # $50M from config
            if abs(trade_value) > max_order_value:
                result['blocked_reason'] = f"Trade value ${abs(trade_value):,} exceeds config limit ${max_order_value:,}"
                self.algorithm.Log(f"    BLOCKED: {result['blocked_reason']}")
                return result
            
            # Execute the trade
            self.algorithm.Log(f"    PLACING ORDER: {quantity_diff:+d} contracts of {mapped_contract}")
            order_ticket = self.algorithm.MarketOrder(mapped_contract, quantity_diff, tag=trade_tag)
            
            result['executed'] = True
            
            self.algorithm.Log(f"    ✓ ORDER PLACED: {result['ticker']} {quantity_diff:+d} contracts (${trade_value:+,.0f})")
            
            # Store trade history
            # self.trade_history.append({
            #     'time': self.algorithm.Time,
            #     'symbol': symbol,
            #     'ticker': result['ticker'],
            #     'quantity': quantity_diff,
            #     'trade_value': trade_value,
            #     'price': price,
            #     'is_rollover': result['is_rollover'],
            #     'tag': trade_tag
            # })
            
        except Exception as e:
            self.algorithm.Log(f"    ORDER ERROR: {str(e)}")
            result['error'] = True
            result['blocked_reason'] = f"Order execution error: {str(e)}"
        
        return result
    
    def _liquidate_excluded_positions(self, final_targets):
        """Liquidate positions not in final targets using QC's native futures handling."""
        liquidation_summary = {
            'liquidations': 0,
            'errors': 0,
            'liquidated_tickers': []
        }
        
        # Convert final_targets to continuous contract symbols using QC's native approach
        target_continuous_symbols = set(final_targets.keys())  # Targets should already be continuous symbols
        
        # Check each holding against our target continuous contracts
        for holding in self.algorithm.Portfolio.Values:
            if holding.Invested:
                should_liquidate = True
                
                # Check if this holding belongs to any of our target continuous contracts
                for continuous_symbol in target_continuous_symbols:
                    if continuous_symbol in self.algorithm.Securities:
                        security = self.algorithm.Securities[continuous_symbol]
                        # QC's Mapped property tells us the current active contract
                        if security.Mapped == holding.Symbol:
                            should_liquidate = False
                            self.algorithm.Log(f"  KEEPING: {holding.Symbol} (mapped from {continuous_symbol})")
                            break
                
                if should_liquidate:
                    try:
                        ticker = str(holding.Symbol)
                        self.algorithm.Log(f"  LIQUIDATING EXCLUDED: {ticker}")
                        self.algorithm.Liquidate(holding.Symbol, tag="ExecutionManager_Liquidate")
                        liquidation_summary['liquidations'] += 1
                        liquidation_summary['liquidated_tickers'].append(ticker)
                        
                        # QC NATIVE: Use algorithm.Transactions instead of custom trade history
                        # Liquidation is automatically tracked by QC's Transactions system
                        
                    except Exception as e:
                        self.algorithm.Log(f"  LIQUIDATION ERROR: {str(e)}")
                        liquidation_summary['errors'] += 1
        
        return liquidation_summary
    
    def _log_execution_diagnostics(self, execution_summary):
        """Log comprehensive execution diagnostics with config info."""
        self.algorithm.Log("")
        self.algorithm.Log("CONFIG-COMPLIANT EXECUTION DIAGNOSTICS:")
        self.algorithm.Log(f"  Orders Placed: {execution_summary['orders_placed']}")
        self.algorithm.Log(f"  Liquidations: {execution_summary['liquidations']}")
        self.algorithm.Log(f"  Execution Errors: {execution_summary['execution_errors']}")
        self.algorithm.Log(f"  Total Trade Value: ${execution_summary['total_trade_value']:,.0f}")
        
        # Log config parameters used
        config_params = execution_summary.get('config_parameters_used', {})
        if config_params:
            self.algorithm.Log("CONFIG PARAMETERS USED:")
            self.algorithm.Log(f"  Min trade value: ${config_params.get('min_trade_value', 0):,}")
            self.algorithm.Log(f"  Min weight change: {config_params.get('min_weight_change', 0):.1%}")
        
        # Log successful trades
        if execution_summary['successful_trades']:
            self.algorithm.Log("  SUCCESSFUL TRADES:")
            for trade in execution_summary['successful_trades']:
                self.algorithm.Log(f"    {trade['ticker']}: {trade['quantity']:+d} contracts (${trade['trade_value']:+,.0f})")
        
        # Log blocked trades
        if execution_summary['blocked_trades']:
            self.algorithm.Log("  BLOCKED TRADES:")
            for blocked in execution_summary['blocked_trades']:
                self.algorithm.Log(f"    {blocked['ticker']}: {blocked['target_weight']:.1%} - {blocked['reason']}")
    
    def _post_execution_analysis(self, execution_summary, final_targets):
        """Perform post-execution analysis and logging."""
        current_portfolio_value = self.algorithm.Portfolio.TotalPortfolioValue
        rebalance_type = execution_summary.get('rebalance_type', 'unknown')
        
        # Log execution summary
        self._log_execution_summary(execution_summary, current_portfolio_value, rebalance_type)
        
        # Log top positions
        self._log_top_positions(final_targets)
        
        # Update risk monitoring
        self._update_risk_monitoring_config_compliant(final_targets, current_portfolio_value)
    
    def _log_execution_summary(self, execution_summary, current_portfolio_value, rebalance_type):
        """Log comprehensive execution summary."""
        if execution_summary['total_trades'] > 0:
            summary_label = f"{rebalance_type.upper()} EXECUTION" if rebalance_type != 'unknown' else "EXECUTION"
            
            self.algorithm.Log(f"{summary_label}: {execution_summary['orders_placed']} orders, "
                            f"{execution_summary['liquidations']} liquidations")
            
            if execution_summary['rollover_orders'] > 0:
                self.algorithm.Log(f"  Rollover trades: {execution_summary['rollover_orders']}")
            
            if execution_summary['total_trade_value'] > 0:
                self.algorithm.Log(f"  Total trade value: ${execution_summary['total_trade_value']:,.0f}")
            
            if execution_summary['execution_errors'] > 0:
                self.algorithm.Log(f"  Execution errors: {execution_summary['execution_errors']}")
            
            self.algorithm.Log(f"  Portfolio value: ${current_portfolio_value:,.0f}")
        else:
            self.algorithm.Log(f"EXECUTION SUMMARY: No trades executed")
            
            # If no trades, explain why
            if execution_summary.get('blocked_trades'):
                self.algorithm.Log("  REASONS FOR NO TRADES:")
                reasons = {}
                for blocked in execution_summary['blocked_trades']:
                    reason = blocked['reason']
                    if reason in reasons:
                        reasons[reason] += 1
                    else:
                        reasons[reason] = 1
                
                for reason, count in reasons.items():
                    self.algorithm.Log(f"    {count} position(s): {reason}")
    
    def _log_top_positions(self, final_targets):
        """Log top positions."""
        if not final_targets:
            return
        
        # Sort by absolute weight
        sorted_positions = sorted(final_targets.items(), key=lambda x: abs(x[1]), reverse=True)
        top_positions = []
        
        max_positions_to_show = min(5, len(sorted_positions))
        
        for symbol, weight in sorted_positions[:max_positions_to_show]:
            if abs(weight) > 0.001:  # Only meaningful positions
                ticker = self._extract_ticker_from_symbol(symbol)
                top_positions.append(f"{ticker}: {weight:+.1%}")
        
        if top_positions:
            self.algorithm.Log(f"TOP POSITIONS: {', '.join(top_positions)}")
    
    def _update_risk_monitoring_config_compliant(self, final_targets, current_portfolio_value):
        """Config-compliant risk monitoring update."""
        self.last_risk_check = self.algorithm.Time
        
        # Calculate portfolio metrics
        total_gross = sum(abs(w) for w in final_targets.values()) if final_targets else 0
        total_net = sum(final_targets.values()) if final_targets else 0
        max_position = max([abs(w) for w in final_targets.values()]) if final_targets else 0
        
        # Check against config limits
        max_single_position_limit = self.risk_config['max_single_position']
        min_capital_limit = self.constraints_config['min_capital']
        
        # Store current snapshot with config compliance info
        self.last_portfolio_snapshot = {
            'time': self.algorithm.Time,
            'portfolio_value': current_portfolio_value,
            'gross_exposure': total_gross,
            'net_exposure': total_net,
            'position_count': len([w for w in final_targets.values() if abs(w) > 0.001]) if final_targets else 0,
            'max_position': max_position,
            'config_compliance': {
                'max_position_ok': max_position <= max_single_position_limit,
                'min_capital_ok': float(current_portfolio_value) >= min_capital_limit,
                'limits': {
                    'max_single_position': max_single_position_limit,
                    'min_capital': min_capital_limit
                }
            }
        }
        
        # Log any config violations
        compliance = self.last_portfolio_snapshot['config_compliance']
        if not compliance['max_position_ok']:
            self.algorithm.Log(f"WARNING: CONFIG VIOLATION: Max position {max_position:.1%} exceeds limit {max_single_position_limit:.1%}")
        
        if not compliance['min_capital_ok']:
            self.algorithm.Log(f"WARNING: CONFIG VIOLATION: Portfolio value ${float(current_portfolio_value):,} below minimum ${min_capital_limit:,}")
    
    def _create_empty_execution_summary(self, rebalance_type):
        """Create empty execution summary for no-trade scenarios."""
        return {
            'orders_placed': 0,
            'liquidations': 0,
            'rollover_orders': 0,
            'total_trades': 0,
            'execution_errors': 0,
            'trades_by_ticker': {},
            'total_trade_value': 0,
            'position_changes': {},
            'rebalance_type': rebalance_type,
            'status': 'no_trades_needed',
            'blocked_trades': [],
            'successful_trades': [],
            'config_parameters_used': {
                'min_trade_value': self.execution_config['min_trade_value'],
                'min_weight_change': self.execution_config['min_weight_change']
            }
        }
    
    def _create_error_execution_summary(self, rebalance_type, error_reason):
        """Create error execution summary for failed execution."""
        return {
            'orders_placed': 0,
            'liquidations': 0,
            'rollover_orders': 0,
            'total_trades': 0,
            'execution_errors': 1,
            'trades_by_ticker': {},
            'total_trade_value': 0,
            'position_changes': {},
            'rebalance_type': rebalance_type,
            'status': 'error',
            'error_reason': error_reason,
            'blocked_trades': [],
            'successful_trades': [],
            'config_parameters_used': {
                'min_trade_value': self.execution_config['min_trade_value'],
                'min_weight_change': self.execution_config['min_weight_change']
            }
        }
    
    # Monitoring and tracking methods
    def get_execution_metrics(self):
        """Get comprehensive execution metrics using QC native methods."""
        try:
            # Use QC's native Transactions for trade counting
            total_orders = len(self.algorithm.Transactions.GetOrders())
            filled_orders = len([order for order in self.algorithm.Transactions.GetOrders() if order.Status == OrderStatus.Filled])
            
            return {
                'total_orders': total_orders,
                'filled_orders': filled_orders,
                'execution_errors': self.execution_errors,
                'success_rate': filled_orders / total_orders if total_orders > 0 else 0.0,
                'portfolio_value': float(self.algorithm.Portfolio.TotalPortfolioValue),
                'total_holdings_value': float(self.algorithm.Portfolio.TotalHoldingsValue),
                'cash': float(self.algorithm.Portfolio.Cash),
                'config_parameters': {
                    'execution': self.execution_config,
                    'risk': self.risk_config,
                    'constraints': self.constraints_config
                }
            }
        except Exception as e:
            self.algorithm.Log(f"Error getting execution metrics: {str(e)}")
            return {}
    
    def track_order_execution(self, order_event):
        """
        Track order execution events - Updated to use QC's built-in features when configured
        
        Args:
            order_event: QuantConnect OrderEvent object
        """
        try:
            if order_event.Status == OrderStatus.Filled:
                # Always log filled orders if configured
                if self.log_order_events:
                    self._log_filled_order(order_event)
                
                # Use QC's built-in tracking or custom tracking based on config
                if self.use_qc_transactions:
                    self._process_qc_transaction(order_event)
                elif self.custom_fill_tracking:
                    self._process_custom_fill_tracking(order_event)
                
                # QC NATIVE: Trade count tracked automatically by algorithm.Transactions
                
            elif order_event.Status in [OrderStatus.Canceled, OrderStatus.CancelPending]:
                if self.log_order_events:
                    quantity_int = int(order_event.Quantity) if hasattr(order_event, 'Quantity') else 0
                    self.algorithm.Log(f"ORDER CANCELED: {order_event.Symbol} {quantity_int:+d}")
                self.execution_errors += 1
                
            elif order_event.Status == OrderStatus.Invalid:
                if self.log_order_events:
                    quantity_int = int(order_event.Quantity) if hasattr(order_event, 'Quantity') else 0
                    self.algorithm.Log(f"ORDER INVALID: {order_event.Symbol} {quantity_int:+d} - {order_event.Message}")
                self.execution_errors += 1
                
        except Exception as e:
            self.algorithm.Log(f"ERROR in track_order_execution: {str(e)}")
    
    def _log_filled_order(self, order_event):
        """Log filled order details"""
        fill_quantity_int = int(order_event.FillQuantity)
        fill_value = order_event.FillQuantity * order_event.FillPrice
        
        self.algorithm.Log(f"ORDER FILLED: {order_event.Symbol} "
                         f"{fill_quantity_int:+d} @ ${order_event.FillPrice:.2f} "
                         f"(${fill_value:+,.0f})")
    
    def _process_qc_transaction(self, order_event):
        """Process using QC's built-in transaction tracking"""
        # QC automatically tracks transactions in self.algorithm.Transactions
        # We can access transaction history via self.algorithm.Transactions.GetOrders()
        # and other methods when needed for reporting
        
        # Optional: Still track minimal info for our custom reporting
        if self.custom_fill_tracking:
            self._add_to_custom_history(order_event)
    
    def _process_custom_fill_tracking(self, order_event):
        """Process using our custom fill tracking (fallback)"""
        self._add_to_custom_history(order_event)
    
    def _add_to_custom_history(self, order_event):
        """QC NATIVE: Use algorithm.Transactions instead of custom trade history"""
        # Custom trade history removed - use QC's native Transactions system
        # Access via: self.algorithm.Transactions.GetOrders()
        pass
    
    def get_qc_transaction_summary(self):
        """Get transaction summary using QC's built-in features"""
        if not self.use_qc_transactions:
            return None
        
        try:
            # Get orders from QC's transaction manager
            orders = self.algorithm.Transactions.GetOrders()
            filled_orders = [order for order in orders if order.Status == OrderStatus.Filled]
            
            total_trades = len(filled_orders)
            total_value = sum(abs(order.Quantity * order.Price) for order in filled_orders if hasattr(order, 'Price'))
            
            return {
                'total_orders': len(orders),
                'filled_orders': total_trades,
                'total_trade_value': total_value,
                'open_orders': len([order for order in orders if order.Status in [OrderStatus.Submitted, OrderStatus.PartiallyFilled]]),
                'canceled_orders': len([order for order in orders if order.Status == OrderStatus.Canceled]),
                'last_order_time': max([order.Time for order in orders]) if orders else None
            }
        except Exception as e:
            self.algorithm.Log(f"Error getting QC transaction summary: {str(e)}")
            return None

    def get_position_summary(self):
        """Get current position summary using QC native Portfolio methods."""
        positions = {}
        
        # Use QC's native Portfolio properties
        total_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
        
        if total_value <= 0:
            return positions
        
        # Use QC's native Portfolio.Values to get all holdings
        for holding in self.algorithm.Portfolio.Values:
            if holding.Invested:
                # Find corresponding futures symbol
                ticker = self._extract_ticker_from_symbol(holding.Symbol)
                
                # Use QC's native holding properties
                weight = float(holding.HoldingsValue) / total_value
                max_position_limit = self.risk_config['max_single_position']
                
                positions[ticker] = {
                    'symbol': holding.Symbol,
                    'quantity': int(holding.Quantity),
                    'market_value': float(holding.HoldingsValue),
                    'weight': weight,
                    'unrealized_pnl': float(holding.UnrealizedProfit),
                    'average_price': float(holding.AveragePrice),
                    'config_compliance': {
                        'within_limits': abs(weight) <= max_position_limit,
                        'limit': max_position_limit
                    }
                }
        
        return positions

    def get_config_summary(self):
        """Get summary of all config parameters being used."""
        return {
            'execution_config': self.execution_config,
            'risk_config': self.risk_config,
            'constraints_config': self.constraints_config,
            'config_compliance_status': self.last_portfolio_snapshot.get('config_compliance', {})
        }

    def track_rollover_event(self, old_symbol, new_symbol):
        """
        Track rollover events from QC's OnSymbolChangedEvents.
        QC NATIVE: Rollover events are automatically tracked by QC's SymbolChangedEvents.
        """
        try:
            self.algorithm.Log(f"EXECUTION MANAGER: Rollover tracked {old_symbol} → {new_symbol}")
            
            # QC NATIVE: Rollover events are automatically handled by QC's continuous contract system
            # No need for custom tracking - positions automatically transfer to new contract
            # Access rollover history via slice.SymbolChangedEvents in OnData()
            
            # Update any internal tracking if needed
            if hasattr(self, 'rollover_events'):
                self.rollover_events.append({
                    'date': self.algorithm.Time,
                    'old_symbol': str(old_symbol),
                    'new_symbol': str(new_symbol)
                })
            else:
                self.rollover_events = [{
                    'date': self.algorithm.Time,
                    'old_symbol': str(old_symbol),
                    'new_symbol': str(new_symbol)
                }]
                
        except Exception as e:
            self.algorithm.Log(f"ERROR tracking rollover event: {str(e)}")
    
    def _is_symbol_ready_for_execution(self, symbol):
        """Check if a symbol is ready for execution using centralized data validator."""
        try:
            # Get current slice data if available
            current_slice = getattr(self.algorithm, 'current_slice', None)
            
            # Use centralized data validator if available
            if hasattr(self.algorithm, 'data_validator'):
                validation_result = self.algorithm.data_validator.validate_symbol_for_trading(symbol, current_slice)
                
                if validation_result['is_valid']:
                    trading_symbol = validation_result.get('trading_symbol', symbol)
                    
                    if trading_symbol != symbol:
                        self.algorithm.Log(f"EXECUTION: {symbol} -> {trading_symbol} for execution")
                    
                    return True
                else:
                    # Log why validation failed
                    reason = validation_result.get('reason', 'Unknown')
                    self.algorithm.Log(f"EXECUTION: Validation failed for {symbol}: {reason}")
                    return False
            
            # Fallback to basic QC validation if validator not available
            if symbol not in self.algorithm.Securities:
                return False
            
            security = self.algorithm.Securities[symbol]
            
            # QC RECOMMENDED PATTERN: Use slice.Contains() first, then HasData
            if current_slice and current_slice.ContainsKey(symbol):
                # Symbol is in current slice - QC's recommended validation approach
                if security.IsTradable:
                    # Try to get price from slice or security
                    if hasattr(security, 'Price') and security.Price > 0:
                        return True
                    else:
                        self.algorithm.Log(f"EXECUTION: {symbol} in slice but no valid price")
                        return False
                else:
                    return False
            
            # Fallback to traditional HasData validation
            if not security.HasData:
                self.algorithm.Log(f"EXECUTION: {symbol} not in slice and no HasData")
                return False
            
            if not security.Price or security.Price <= 0:
                self.algorithm.Log(f"EXECUTION: {symbol} has invalid price: {security.Price}")
                return False
            
            # For futures, check if tradeable (mapped contract) - ROLLOVER-AWARE
            if hasattr(security, 'Mapped') and security.Mapped:
                mapped_contract = security.Mapped
                if mapped_contract in self.algorithm.Securities:
                    mapped_security = self.algorithm.Securities[mapped_contract]
                    
                    # ROLLOVER FIX: During rollover, new contract may not have HasData yet
                    # Check if this is a rollover situation by seeing if the continuous contract has data
                    # but the mapped contract doesn't
                    if security.HasData and not mapped_security.HasData:
                        # This is likely a rollover - be more lenient
                        # Allow execution if the continuous contract is tradeable
                        self.algorithm.Log(f"EXECUTION: Rollover detected for {symbol} -> {mapped_contract}, using lenient validation")
                        return security.IsTradable
                    
                    # Normal case: both should have data
                    return mapped_security.IsTradable and mapped_security.HasData
            
            return security.IsTradable
            
        except Exception as e:
            self.algorithm.Error(f"EXECUTION: Error validating symbol {symbol}: {str(e)}")
            return False

    def set_bad_data_manager(self, bad_data_manager):
        """Set the bad data position manager for integration."""
        self.bad_data_manager = bad_data_manager
        self.algorithm.Log("ExecutionManager: Bad Data Position Manager integrated")
