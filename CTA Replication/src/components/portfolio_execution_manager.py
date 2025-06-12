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
        """Initialize the portfolio execution manager with full config compliance."""
        self.algorithm = algorithm
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        
        # Execution tracking
        self.total_trades_executed = 0
        self.total_liquidations = 0
        self.total_rollover_trades = 0
        self.execution_errors = 0
        
        # Performance tracking
        self.trade_history = []
        self.position_history = []
        self.last_portfolio_snapshot = {}
        
        # CONFIG-COMPLIANT: Load execution parameters from config
        execution_config = self.config.get('execution', {})
        self.execution_config = {
            'min_trade_value': execution_config.get('min_trade_value', 1000),      # Config: $1,000
            'min_weight_change': execution_config.get('min_weight_change', 0.01),  # Config: 1%
            'max_single_order_value': execution_config.get('max_single_order_value', 50000000),
            'order_type': execution_config.get('order_type', 'market'),
            'max_portfolio_turnover': execution_config.get('max_portfolio_turnover', 2.0)
        }
        
        # CONFIG-COMPLIANT: Load risk parameters from config
        risk_config = self.config.get('portfolio_risk', {})
        self.risk_config = {
            'max_single_position': risk_config.get('max_single_position', 1.0),    # Config: 10.0 (1000%)
            'daily_stop_loss': risk_config.get('daily_stop_loss', 0.2),            # Config: 20%
            'max_drawdown_stop': risk_config.get('max_drawdown_stop', 0.75),       # Config: 75%
            'target_portfolio_vol': risk_config.get('target_portfolio_vol', 0.5),  # Config: 50%
            'max_leverage_multiplier': risk_config.get('max_leverage_multiplier', 100) # Config: 100x
        }
        
        # CONFIG-COMPLIANT: Load constraint parameters from config
        constraints_config = self.config.get('constraints', {})
        self.constraints_config = {
            'min_capital': constraints_config.get('min_capital', 5000000),         # Config: $5M
            'initial_capital': constraints_config.get('initial_capital', 10000000), # Config: $10M
            'max_positions_per_strategy': constraints_config.get('max_positions_per_strategy', 10),
            'max_total_positions': constraints_config.get('max_total_positions', 20)
        }
        
        # Risk monitoring
        self.last_risk_check = None
        self.risk_alerts = []
        
        self.algorithm.Log("Portfolio Execution Manager initialized - CONFIG-COMPLIANT VERSION")
        self._log_config_parameters()
        self._log_capital_diagnostics()
    
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
            self.algorithm.Log(f"  ⚠️  WARNING: Current value ${current_value:,} differs from config ${self.constraints_config['initial_capital']:,}")
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
            ticker = self.algorithm.futures_manager.futures_data.get(symbol, {}).get('ticker', symbol)
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
                ticker = self.algorithm.futures_manager.futures_data.get(symbol, {}).get('ticker', symbol)
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
        self.total_trades_executed += execution_summary['orders_placed']
        self.total_liquidations += execution_summary['liquidations']
        self.total_rollover_trades += execution_summary['rollover_orders']
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
            result['ticker'] = self.algorithm.futures_manager.futures_data[symbol]['ticker']
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
            if not security.HasData:
                result['error'] = True
                result['blocked_reason'] = f"No data for mapped contract {mapped_contract}"
                self.algorithm.Log(f"    ERROR: {result['blocked_reason']}")
                return result
        except Exception as e:
            result['error'] = True
            result['blocked_reason'] = f"Error accessing security: {str(e)}"
            self.algorithm.Log(f"    ERROR: {result['blocked_reason']}")
            return result
        
        # Validate price
        price = float(security.Price)
        if price <= 0:
            result['error'] = True
            result['blocked_reason'] = f"Invalid price: {price}"
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
        
        try:
            # Calculate quantity using futures manager
            if hasattr(self.algorithm.futures_manager, 'get_effective_multiplier'):
                multiplier = self.algorithm.futures_manager.get_effective_multiplier(mapped_contract)
            else:
                # Fallback multiplier calculation
                contract_multiplier = security.SymbolProperties.ContractMultiplier
                point_value = security.SymbolProperties.MinimumPriceVariation
                multiplier = contract_multiplier if contract_multiplier > 0 else 1.0
            
            self.algorithm.Log(f"    Multiplier: {multiplier}")
            
            quantity_diff = int(trade_value / (price * multiplier))
            
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
            self.trade_history.append({
                'time': self.algorithm.Time,
                'symbol': symbol,
                'ticker': result['ticker'],
                'quantity': quantity_diff,
                'trade_value': trade_value,
                'price': price,
                'is_rollover': result['is_rollover'],
                'tag': trade_tag
            })
            
        except Exception as e:
            self.algorithm.Log(f"    ORDER ERROR: {str(e)}")
            result['error'] = True
            result['blocked_reason'] = f"Order execution error: {str(e)}"
        
        return result
    
    def _liquidate_excluded_positions(self, final_targets):
        """Liquidate positions not in final targets."""
        liquidation_summary = {
            'liquidations': 0,
            'errors': 0,
            'liquidated_tickers': []
        }
        
        current_symbols = set(final_targets.keys())
        
        for holding in self.algorithm.Portfolio.Values:
            if holding.Invested:
                # Find corresponding symbol
                holding_symbol = None
                for symbol in self.algorithm.futures_manager.futures_data.keys():
                    if (symbol in self.algorithm.Securities and 
                        self.algorithm.Securities[symbol].Mapped == holding.Symbol):
                        holding_symbol = symbol
                        break
                
                if holding_symbol and holding_symbol not in current_symbols:
                    try:
                        ticker = self.algorithm.futures_manager.futures_data[holding_symbol]['ticker']
                        self.algorithm.Liquidate(holding.Symbol, tag="ExecutionManager_Liquidate")
                        liquidation_summary['liquidations'] += 1
                        liquidation_summary['liquidated_tickers'].append(ticker)
                        self.algorithm.Log(f"  LIQUIDATED: {ticker}")
                        
                        # Store liquidation in trade history
                        self.trade_history.append({
                            'time': self.algorithm.Time,
                            'symbol': holding_symbol,
                            'ticker': ticker,
                            'quantity': -holding.Quantity,
                            'trade_value': -holding.HoldingsValue,
                            'price': holding.Price,
                            'is_rollover': False,
                            'tag': 'ExecutionManager_Liquidate'
                        })
                        
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
                ticker = self.algorithm.futures_manager.futures_data.get(symbol, {}).get('ticker', symbol)
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
            self.algorithm.Log(f"⚠️  CONFIG VIOLATION: Max position {max_position:.1%} exceeds limit {max_single_position_limit:.1%}")
        
        if not compliance['min_capital_ok']:
            self.algorithm.Log(f"⚠️  CONFIG VIOLATION: Portfolio value ${float(current_portfolio_value):,} below minimum ${min_capital_limit:,}")
    
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
        """Get comprehensive execution metrics with config compliance."""
        return {
            'total_trades_executed': self.total_trades_executed,
            'execution_errors': self.execution_errors,
            'last_portfolio_snapshot': self.last_portfolio_snapshot,
            'config_parameters': {
                'execution': self.execution_config,
                'risk': self.risk_config,
                'constraints': self.constraints_config
            }
        }
    
    def track_order_execution(self, order_event):
        """
        Track order execution events - called from OnOrderEvent.
        FIXED: Corrected float formatting issue.
        
        Args:
            order_event: QuantConnect OrderEvent object
        """
        try:
            if order_event.Status == OrderStatus.Filled:
                # FIXED: Use proper format for FillQuantity (which can be a float)
                # Convert to int for display since futures quantities should be whole numbers
                fill_quantity_int = int(order_event.FillQuantity)
                
                # Log the filled order with corrected formatting
                self.algorithm.Log(f"ORDER FILLED: {order_event.Symbol} "
                                 f"{fill_quantity_int:+d} @ ${order_event.FillPrice:.2f} "
                                 f"(${order_event.FillQuantity * order_event.FillPrice:+,.0f})")
                
                # Update our tracking
                fill_data = {
                    'time': self.algorithm.Time,
                    'symbol': str(order_event.Symbol),
                    'quantity': order_event.FillQuantity,
                    'fill_price': order_event.FillPrice,
                    'fill_value': order_event.FillQuantity * order_event.FillPrice,
                    'order_id': order_event.OrderId,
                    'tag': order_event.Tag if hasattr(order_event, 'Tag') else 'Unknown'
                }
                
                # Add to trade history if not already there
                existing_trade = None
                for trade in self.trade_history:
                    if (trade['time'] == self.algorithm.Time and 
                        abs(trade['quantity'] - order_event.FillQuantity) < 0.1):
                        existing_trade = trade
                        break
                
                if not existing_trade:
                    # Find ticker name
                    ticker = "Unknown"
                    for symbol in self.algorithm.futures_manager.futures_data.keys():
                        if (symbol in self.algorithm.Securities and 
                            self.algorithm.Securities[symbol].Mapped == order_event.Symbol):
                            ticker = self.algorithm.futures_manager.futures_data[symbol]['ticker']
                            break
                    
                    self.trade_history.append({
                        'time': self.algorithm.Time,
                        'symbol': str(order_event.Symbol),
                        'ticker': ticker,
                        'quantity': order_event.FillQuantity,
                        'trade_value': fill_data['fill_value'],
                        'price': order_event.FillPrice,
                        'is_rollover': 'Rollover' in fill_data['tag'],
                        'tag': fill_data['tag']
                    })
            
            elif order_event.Status in [OrderStatus.Canceled, OrderStatus.CancelPending]:
                # FIXED: Use proper format for Quantity (convert to int for display)
                quantity_int = int(order_event.Quantity) if hasattr(order_event, 'Quantity') else 0
                self.algorithm.Log(f"ORDER CANCELED: {order_event.Symbol} {quantity_int:+d}")
                self.execution_errors += 1
                
            elif order_event.Status == OrderStatus.Invalid:
                # FIXED: Use proper format for Quantity (convert to int for display)
                quantity_int = int(order_event.Quantity) if hasattr(order_event, 'Quantity') else 0
                self.algorithm.Log(f"ORDER INVALID: {order_event.Symbol} {quantity_int:+d} - {order_event.Message}")
                self.execution_errors += 1
                
        except Exception as e:
            self.algorithm.Log(f"ERROR in track_order_execution: {str(e)}")

    def get_position_summary(self):
        """Get current position summary with config compliance info."""
        positions = {}
        total_value = self.algorithm.Portfolio.TotalPortfolioValue
        
        if total_value <= 0:
            return positions
        
        for holding in self.algorithm.Portfolio.Values:
            if holding.Invested:
                # Find symbol and ticker
                symbol = None
                ticker = "Unknown"
                
                for sym in self.algorithm.futures_manager.futures_data.keys():
                    if (sym in self.algorithm.Securities and 
                        self.algorithm.Securities[sym].Mapped == holding.Symbol):
                        symbol = sym
                        ticker = self.algorithm.futures_manager.futures_data[sym]['ticker']
                        break
                
                if symbol:
                    weight = holding.HoldingsValue / total_value
                    max_position_limit = self.risk_config['max_single_position']
                    
                    positions[ticker] = {
                        'symbol': symbol,
                        'quantity': holding.Quantity,
                        'market_value': holding.HoldingsValue,
                        'weight': weight,
                        'unrealized_pnl': holding.UnrealizedProfit,
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
