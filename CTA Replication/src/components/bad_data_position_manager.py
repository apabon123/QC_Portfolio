"""
Bad Data Position Manager

Handles positions when symbols develop data quality issues.
Instead of aggressive filtering, this provides strategies for managing existing positions
when their underlying data becomes unreliable.
"""

from AlgorithmImports import *

class BadDataPositionManager:
    """
    Manages positions when symbols develop data quality issues.
    
    Strategies:
    1. HOLD - Keep position, ignore bad data for mark-to-market
    2. LIQUIDATE - Close position immediately 
    3. HEDGE - Reduce position size gradually
    4. FREEZE - Stop new trades but keep existing position
    """
    
    def __init__(self, algorithm, config_manager):
        self.algorithm = algorithm
        self.config_manager = config_manager
        
        # Load configuration
        config = config_manager.get_full_config()
        self.bad_data_config = config.get('bad_data_management', {
            'default_strategy': 'FREEZE',
            'liquidation_threshold_days': 3,
            'hedge_reduction_rate': 0.25,  # Reduce by 25% each rebalance
            'freeze_new_trades': True,
            'use_last_good_price': True,
            'max_stale_price_days': 5
        })
        
        # Track positions with data issues
        self.problematic_positions = {}  # symbol -> issue_info
        self.last_good_prices = {}       # symbol -> last_good_price
        self.position_strategies = {}    # symbol -> strategy
        
        # Default strategies by symbol type
        self.default_strategies = {
            # Core positions (ES, NQ, ZN) - more conservative
            'ES': 'HOLD',
            'NQ': 'HOLD', 
            'ZN': 'HOLD',
            
            # Priority 2 futures - more aggressive management
            '6E': 'FREEZE',  # Euro FX - freeze new trades
            '6J': 'FREEZE',  # Japanese Yen - freeze new trades
            'CL': 'HEDGE',   # Crude Oil - reduce gradually
            'GC': 'HOLD',    # Gold - hold through data issues
            'YM': 'FREEZE',  # Dow - freeze new trades
            'ZB': 'HOLD',    # 30Y Treasury - hold
            
            # VIX futures - special handling
            'VX': 'LIQUIDATE'  # VIX - liquidate on data issues
        }
        
        self.algorithm.Log("BadDataPositionManager: Initialized with targeted position management")
        self._log_strategies()
    
    def _log_strategies(self):
        """Log the strategies for each symbol type"""
        self.algorithm.Log("Bad Data Position Strategies:")
        for ticker, strategy in self.default_strategies.items():
            self.algorithm.Log(f"  {ticker}: {strategy}")
    
    def report_data_issue(self, symbol, issue_type, severity='medium'):
        """
        Report a data quality issue for a symbol.
        
        Args:
            symbol: The symbol with data issues
            issue_type: Type of issue ('no_data', 'bad_price', 'extreme_move', etc.)
            severity: 'low', 'medium', 'high'
        """
        ticker = self._extract_ticker(symbol)
        
        # Check if we have a position in this symbol
        if not self._has_position(symbol):
            # No position, no need to manage
            return
        
        # Record the issue
        if symbol not in self.problematic_positions:
            self.problematic_positions[symbol] = {
                'first_issue_time': self.algorithm.Time,
                'issue_count': 0,
                'issues': [],
                'strategy': self._get_strategy_for_symbol(ticker),
                'last_action': None
            }
        
        issue_info = self.problematic_positions[symbol]
        issue_info['issue_count'] += 1
        issue_info['issues'].append({
            'time': self.algorithm.Time,
            'type': issue_type,
            'severity': severity
        })
        
        self.algorithm.Log(f"BadDataManager: {ticker} data issue #{issue_info['issue_count']} - {issue_type} ({severity})")
        
        # Take action based on strategy
        self._execute_strategy(symbol, issue_info)
    
    def _has_position(self, symbol):
        """Check if we have a position in this symbol"""
        try:
            if symbol in self.algorithm.Portfolio:
                holding = self.algorithm.Portfolio[symbol]
                return holding.Invested and abs(holding.Quantity) > 0
            return False
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
    
    def _get_strategy_for_symbol(self, ticker):
        """Get the management strategy for a ticker"""
        return self.default_strategies.get(ticker, self.bad_data_config['default_strategy'])
    
    def _execute_strategy(self, symbol, issue_info):
        """Execute the management strategy for a problematic position"""
        strategy = issue_info['strategy']
        ticker = self._extract_ticker(symbol)
        days_since_first_issue = (self.algorithm.Time - issue_info['first_issue_time']).days
        
        if strategy == 'HOLD':
            self._execute_hold_strategy(symbol, issue_info)
        elif strategy == 'LIQUIDATE':
            self._execute_liquidate_strategy(symbol, issue_info)
        elif strategy == 'HEDGE':
            self._execute_hedge_strategy(symbol, issue_info)
        elif strategy == 'FREEZE':
            self._execute_freeze_strategy(symbol, issue_info)
        
        issue_info['last_action'] = self.algorithm.Time
    
    def _execute_hold_strategy(self, symbol, issue_info):
        """HOLD: Keep position, use last good price for valuation if possible"""
        ticker = self._extract_ticker(symbol)
        
        # Store last good price if we have one
        if symbol in self.algorithm.Securities:
            security = self.algorithm.Securities[symbol]
            if hasattr(security, 'Price') and security.Price > 0:
                self.last_good_prices[symbol] = security.Price
        
        self.algorithm.Log(f"BadDataManager: HOLDING {ticker} position despite data issues")
    
    def _execute_liquidate_strategy(self, symbol, issue_info):
        """LIQUIDATE: Close position immediately"""
        ticker = self._extract_ticker(symbol)
        
        try:
            if self._has_position(symbol):
                self.algorithm.Liquidate(symbol, tag=f"BadData_Liquidate_{ticker}")
                self.algorithm.Log(f"BadDataManager: LIQUIDATED {ticker} due to data issues")
                
                # Remove from tracking since position is closed
                if symbol in self.problematic_positions:
                    del self.problematic_positions[symbol]
        except Exception as e:
            self.algorithm.Error(f"BadDataManager: Failed to liquidate {ticker}: {str(e)}")
    
    def _execute_hedge_strategy(self, symbol, issue_info):
        """HEDGE: Gradually reduce position size"""
        ticker = self._extract_ticker(symbol)
        
        try:
            if self._has_position(symbol):
                holding = self.algorithm.Portfolio[symbol]
                current_quantity = holding.Quantity
                
                # Reduce position by configured rate
                reduction_rate = self.bad_data_config['hedge_reduction_rate']
                new_quantity = current_quantity * (1 - reduction_rate)
                quantity_to_close = current_quantity - new_quantity
                
                if abs(quantity_to_close) >= 1:  # Only if meaningful reduction
                    # Close partial position
                    order_ticket = self.algorithm.MarketOrder(symbol, -quantity_to_close, 
                                                            tag=f"BadData_Hedge_{ticker}")
                    self.algorithm.Log(f"BadDataManager: HEDGED {ticker} - reduced by {reduction_rate:.1%}")
                else:
                    self.algorithm.Log(f"BadDataManager: {ticker} position too small to hedge further")
        except Exception as e:
            self.algorithm.Error(f"BadDataManager: Failed to hedge {ticker}: {str(e)}")
    
    def _execute_freeze_strategy(self, symbol, issue_info):
        """FREEZE: Stop new trades but keep existing position"""
        ticker = self._extract_ticker(symbol)
        self.algorithm.Log(f"BadDataManager: FROZEN {ticker} - no new trades allowed")
        # The actual freezing logic will be implemented in the execution manager
    
    def should_allow_new_trade(self, symbol):
        """
        Check if new trades should be allowed for this symbol.
        Returns False if symbol is frozen due to data issues.
        """
        if symbol not in self.problematic_positions:
            return True
        
        strategy = self.problematic_positions[symbol]['strategy']
        
        # FREEZE and HEDGE strategies block new trades
        if strategy in ['FREEZE', 'HEDGE']:
            return False
        
        # LIQUIDATE blocks new trades (position should be closed anyway)
        if strategy == 'LIQUIDATE':
            return False
        
        # HOLD allows new trades
        return True
    
    def get_position_override_price(self, symbol):
        """
        Get override price for position valuation if data is bad.
        Returns None if no override needed.
        """
        if symbol not in self.problematic_positions:
            return None
        
        strategy = self.problematic_positions[symbol]['strategy']
        
        # For HOLD strategy, use last good price if configured
        if strategy == 'HOLD' and self.bad_data_config.get('use_last_good_price', False):
            if symbol in self.last_good_prices:
                days_since_good_price = (self.algorithm.Time - 
                    self.problematic_positions[symbol]['first_issue_time']).days
                
                max_stale_days = self.bad_data_config.get('max_stale_price_days', 5)
                if days_since_good_price <= max_stale_days:
                    return self.last_good_prices[symbol]
        
        return None
    
    def get_status_report(self):
        """Get status report of all problematic positions"""
        if not self.problematic_positions:
            return "No positions with data issues"
        
        report = []
        report.append(f"Positions with data issues: {len(self.problematic_positions)}")
        
        for symbol, info in self.problematic_positions.items():
            ticker = self._extract_ticker(symbol)
            days_since_issue = (self.algorithm.Time - info['first_issue_time']).days
            report.append(f"  {ticker}: {info['strategy']} strategy, {info['issue_count']} issues, {days_since_issue} days")
        
        return "\n".join(report)
    
    def cleanup_resolved_issues(self):
        """Remove symbols that no longer have positions or issues resolved"""
        symbols_to_remove = []
        
        for symbol in self.problematic_positions:
            # Remove if no longer have position
            if not self._has_position(symbol):
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            ticker = self._extract_ticker(symbol)
            self.algorithm.Log(f"BadDataManager: Cleaned up resolved issue for {ticker}")
            del self.problematic_positions[symbol]
            if symbol in self.last_good_prices:
                del self.last_good_prices[symbol] 