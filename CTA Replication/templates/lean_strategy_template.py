"""
LEAN-Aware CTA Strategy Template
Always refer to docs/lean-cheatsheet.md before implementing custom solutions

This template demonstrates proper LEAN API usage for CTA strategies.
Copy this template when creating new strategies.
"""

from AlgorithmImports import *

class LEANAwareCTAStrategy(QCAlgorithm):
    """
    Template that demonstrates proper LEAN API usage for CTA strategies.
    Shows how to use LEAN's built-in capabilities instead of custom implementations.
    """
    
    def Initialize(self):
        """Standard LEAN initialization pattern"""
        # Use LEAN's built-in setup methods - NEVER hardcode these
        self.SetStartDate(2020, 1, 1)
        self.SetCash(1000000)
        self.SetBenchmark("SPY")
        
        # Use LEAN's built-in warm-up system - NEVER build custom warm-up
        self.SetWarmUp(timedelta(days=60))
        self.Settings.AutomaticIndicatorWarmUp = True  # Auto-warm indicators
        
        # Use LEAN's built-in brokerage models
        self.SetBrokerageModel(BrokerageName.InteractiveBrokers)
        
        # Initialize components using LEAN methods
        self._setup_universe()
        self._setup_indicators() 
        self._setup_scheduling()
        self._setup_risk_management()
        
        self.Log("LEAN-Aware CTA Strategy initialized")
    
    def _setup_universe(self):
        """Setup futures universe using LEAN's AddFuture - NEVER custom universe logic"""
        # CTA futures universe
        self.futures_symbols = {}
        futures_tickers = ["ES", "NQ", "YM", "ZN", "ZB", "6E", "6J", "CL", "GC"]
        
        for ticker in futures_tickers:
            # Use LEAN's AddFuture for continuous contracts - NEVER AddFutureContract
            future = self.AddFuture(
                ticker, 
                Resolution.Daily,
                dataMappingMode=DataMappingMode.OpenInterest,      # LEAN's native mapping
                dataNormalizationMode=DataNormalizationMode.BackwardsRatio,  # LEAN's native normalization
                contractDepthOffset=0
            )
            
            # Store symbol for later use
            self.futures_symbols[ticker] = future.Symbol
            
        self.Log(f"Added {len(self.futures_symbols)} futures contracts using LEAN's AddFuture")
    
    def _setup_indicators(self):
        """Setup indicators using LEAN's built-in indicators - NEVER custom indicators"""
        self.indicators = {}
        
        for ticker, symbol in self.futures_symbols.items():
            # Use LEAN's built-in indicators - NEVER implement custom ones
            self.indicators[ticker] = {
                # Trend indicators
                'sma_fast': self.SMA(symbol, 20, Resolution.Daily),    # LEAN's SMA
                'sma_slow': self.SMA(symbol, 50, Resolution.Daily),    # LEAN's SMA
                'ema_fast': self.EMA(symbol, 20, Resolution.Daily),    # LEAN's EMA
                
                # Momentum indicators  
                'rsi': self.RSI(symbol, 14, Resolution.Daily),         # LEAN's RSI
                'macd': self.MACD(symbol, 12, 26, 9, Resolution.Daily), # LEAN's MACD
                
                # Volatility indicators
                'atr': self.ATR(symbol, 20, Resolution.Daily),         # LEAN's ATR
                'bb': self.BB(symbol, 20, 2, Resolution.Daily),        # LEAN's Bollinger Bands
                
                # Volume indicators
                'obv': self.OBV(symbol, Resolution.Daily)              # LEAN's OBV
            }
            
        self.Log("Indicators initialized using LEAN's built-in indicator library")
    
    def _setup_scheduling(self):
        """Setup scheduling using LEAN's Schedule.On - NEVER custom schedulers"""
        
        # Daily signal updates using LEAN's scheduling
        self.Schedule.On(
            self.DateRules.EveryDay("ES"),                    # LEAN's DateRules
            self.TimeRules.AfterMarketOpen("ES", 30),         # LEAN's TimeRules
            self.UpdateDailySignals                           # Our custom method
        )
        
        # Weekly rebalancing using LEAN's scheduling
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Friday),           # LEAN's weekly scheduling
            self.TimeRules.BeforeMarketClose("ES", 30),       # LEAN's market timing
            self.WeeklyRebalance                              # Our custom method
        )
        
        # Monthly allocation review using LEAN's scheduling
        self.Schedule.On(
            self.DateRules.MonthStart("ES"),                  # LEAN's monthly scheduling
            self.TimeRules.At(10, 0),                         # LEAN's specific time
            self.MonthlyAllocationReview                      # Our custom method
        )
        
        self.Log("Scheduling setup using LEAN's Schedule.On methods")
    
    def _setup_risk_management(self):
        """Setup risk management using LEAN's built-in controls"""
        # Use LEAN's built-in risk settings - NEVER custom risk systems
        self.Settings.MaximumOrderValue = 10000000           # LEAN's order value limit
        
        # Risk parameters (would typically come from configuration)
        self.risk_config = {
            'max_leverage': 2.0,
            'max_drawdown': 0.15,
            'risk_per_trade': 0.02,
            'max_position_size': 0.10
        }
        
        # Track high water mark for drawdown calculation
        self.high_water_mark = self.Portfolio.TotalPortfolioValue
    
    def OnData(self, slice):
        """Process market data using LEAN's slice structure"""
        try:
            # Use LEAN's IsWarmingUp property - NEVER custom warm-up logic
            if self.IsWarmingUp:
                return
            
            # Handle LEAN's automatic rollover events - NEVER custom rollover logic
            self._handle_rollover_events(slice)
            
            # Process futures chains using LEAN's FuturesChains - NEVER custom chain logic
            self._process_futures_chains(slice)
            
            # Update strategy state
            self._update_strategy_state(slice)
            
        except Exception as e:
            # Use LEAN's error logging - NEVER custom error handling
            self.Error(f"OnData error: {str(e)}")
    
    def _handle_rollover_events(self, slice):
        """Handle rollover events using LEAN's SymbolChangedEvents"""
        # Use LEAN's native rollover detection - NEVER custom rollover detection
        for changed in slice.SymbolChangedEvents.Values:
            old_symbol = changed.OldSymbol
            new_symbol = changed.NewSymbol
            
            # Log rollover using LEAN's logging
            self.Log(f"Rollover detected: {old_symbol} -> {new_symbol}")
            
            # Positions automatically transfer in LEAN - just track for analytics
            self._track_rollover_cost(old_symbol, new_symbol)
    
    def _process_futures_chains(self, slice):
        """Process futures chains using LEAN's FuturesChains property"""
        liquid_contracts = {}
        
        for ticker, symbol in self.futures_symbols.items():
            # Use LEAN's FuturesChains - NEVER custom chain processing
            if symbol in slice.FuturesChains:
                chain = slice.FuturesChains[symbol]
                
                if len(chain) > 0:
                    # Find most liquid contract using LEAN's chain data
                    most_liquid = max(chain, key=lambda x: x.Volume)
                    
                    liquid_contracts[ticker] = {
                        'symbol': most_liquid.Symbol,
                        'volume': most_liquid.Volume,
                        'open_interest': most_liquid.OpenInterest,
                        'expiry': most_liquid.Expiry,
                        'price': most_liquid.LastPrice
                    }
        
        # Store for strategy use
        self.liquid_contracts = liquid_contracts
    
    def _update_strategy_state(self, slice):
        """Update strategy state using LEAN's indicator values"""
        # Only trade during market hours using LEAN's market hours detection
        if not self.IsMarketOpen("ES"):
            return
        
        # Generate signals using LEAN's indicator values - NEVER custom calculations
        signals = self._generate_signals()
        
        # Calculate position sizes using LEAN's portfolio methods
        position_targets = self._calculate_position_sizes(signals)
        
        # Apply risk management using LEAN's portfolio properties
        risk_adjusted_targets = self._apply_risk_management(position_targets)
        
        # Execute trades using LEAN's order methods
        self._execute_trades(risk_adjusted_targets)
    
    def _generate_signals(self):
        """Generate trading signals using LEAN's indicator values"""
        signals = {}
        
        for ticker in self.futures_symbols.keys():
            indicators = self.indicators[ticker]
            
            # Check if indicators are ready using LEAN's IsReady property
            if not (indicators['sma_fast'].IsReady and 
                   indicators['sma_slow'].IsReady and 
                   indicators['rsi'].IsReady):
                signals[ticker] = 0
                continue
            
            # Use LEAN's indicator values - NEVER custom indicator calculations
            sma_fast_value = indicators['sma_fast'].Current.Value
            sma_slow_value = indicators['sma_slow'].Current.Value
            rsi_value = indicators['rsi'].Current.Value
            atr_value = indicators['atr'].Current.Value if indicators['atr'].IsReady else 0
            
            # Simple momentum strategy logic
            trend_signal = 1 if sma_fast_value > sma_slow_value else -1
            rsi_filter = 1 if 30 < rsi_value < 70 else 0  # Avoid overbought/oversold
            
            # Position size based on volatility (ATR)
            volatility_adjustment = self._calculate_volatility_adjustment(atr_value)
            
            signals[ticker] = trend_signal * rsi_filter * volatility_adjustment
        
        return signals
    
    def _calculate_position_sizes(self, signals):
        """Calculate position sizes using LEAN's portfolio methods"""
        position_targets = {}
        
        # Use LEAN's Portfolio properties - NEVER custom portfolio tracking
        total_value = self.Portfolio.TotalPortfolioValue
        available_cash = self.Portfolio.Cash
        
        for ticker, signal in signals.items():
            if signal == 0:
                position_targets[ticker] = 0
                continue
            
            symbol = self.futures_symbols[ticker]
            
            # Use LEAN's Securities for current price - NEVER custom price tracking
            if symbol in self.Securities and self.Securities[symbol].HasData:
                current_price = self.Securities[symbol].Price
                
                # Risk-based position sizing
                risk_amount = total_value * self.risk_config['risk_per_trade']
                max_position_value = total_value * self.risk_config['max_position_size']
                
                # Calculate target position size
                position_value = min(risk_amount / abs(signal), max_position_value)
                target_quantity = (position_value / current_price) * np.sign(signal)
                
                position_targets[ticker] = target_quantity
            else:
                position_targets[ticker] = 0
        
        return position_targets
    
    def _apply_risk_management(self, position_targets):
        """Apply risk management using LEAN's portfolio properties"""
        # Use LEAN's Portfolio properties for risk calculations
        total_value = self.Portfolio.TotalPortfolioValue
        margin_used = self.Portfolio.TotalMarginUsed
        
        # Check leverage limits
        projected_margin = sum(
            abs(qty) * self.Securities[self.futures_symbols[ticker]].Price
            for ticker, qty in position_targets.items()
            if qty != 0 and self.futures_symbols[ticker] in self.Securities
        )
        
        projected_leverage = projected_margin / total_value
        max_leverage = self.risk_config['max_leverage']
        
        if projected_leverage > max_leverage:
            # Scale down positions to meet leverage limit
            scale_factor = max_leverage / projected_leverage
            position_targets = {ticker: qty * scale_factor 
                              for ticker, qty in position_targets.items()}
            
            self.Log(f"Scaling positions by {scale_factor:.2f} due to leverage limit")
        
        # Check drawdown limits using LEAN's portfolio tracking
        self._check_drawdown_limits()
        
        return position_targets
    
    def _check_drawdown_limits(self):
        """Check drawdown limits using LEAN's portfolio properties"""
        # Use LEAN's Portfolio.TotalPortfolioValue - NEVER custom portfolio tracking
        current_value = self.Portfolio.TotalPortfolioValue
        
        # Update high water mark
        self.high_water_mark = max(self.high_water_mark, current_value)
        
        # Calculate drawdown
        drawdown = (self.high_water_mark - current_value) / self.high_water_mark
        max_drawdown = self.risk_config['max_drawdown']
        
        if drawdown > max_drawdown:
            # Emergency liquidation using LEAN's Liquidate method
            self.Liquidate()  # LEAN's built-in liquidation
            self.Error(f"Emergency liquidation: Drawdown {drawdown:.2%} exceeds limit")
            
            # Halt algorithm using LEAN's Quit method
            self.Quit("Maximum drawdown exceeded")  # LEAN's built-in halt
    
    def _execute_trades(self, position_targets):
        """Execute trades using LEAN's order methods"""
        for ticker, target_quantity in position_targets.items():
            symbol = self.futures_symbols[ticker]
            
            # Use LEAN's Portfolio for current position - NEVER custom position tracking
            current_quantity = self.Portfolio[symbol].Quantity
            
            # Calculate required trade
            trade_quantity = target_quantity - current_quantity
            
            if abs(trade_quantity) > 1:  # Minimum trade size
                try:
                    # Use LEAN's order methods - NEVER custom order management
                    if abs(target_quantity) < 1:
                        # Close position using LEAN's Liquidate
                        ticket = self.Liquidate(symbol, "Position too small")
                    else:
                        # Market order using LEAN's MarketOrder
                        ticket = self.MarketOrder(symbol, trade_quantity)
                    
                    # Log using LEAN's logging
                    if ticket:
                        self.Log(f"Trade executed: {ticker} {trade_quantity} shares")
                    
                except Exception as e:
                    # Use LEAN's error logging
                    self.Error(f"Trade execution failed for {ticker}: {str(e)}")
    
    def _calculate_volatility_adjustment(self, atr_value):
        """Calculate volatility-based position adjustment"""
        if atr_value > 0:
            # Simple inverse volatility sizing
            base_volatility = 0.02  # 2% base volatility assumption
            return min(base_volatility / atr_value, 2.0)  # Cap at 2x
        return 1.0
    
    def _track_rollover_cost(self, old_symbol, new_symbol):
        """Track rollover costs for analytics"""
        # Simple rollover cost tracking
        self.Debug(f"Rollover cost tracking: {old_symbol} -> {new_symbol}")
    
    def UpdateDailySignals(self):
        """Scheduled daily signal update"""
        self.Debug("Daily signal update")
        # Update any daily calculations here
    
    def WeeklyRebalance(self):
        """Scheduled weekly rebalancing"""
        self.Log("Weekly rebalancing")
        # Main rebalancing logic handled in OnData
    
    def MonthlyAllocationReview(self):
        """Scheduled monthly allocation review"""
        self.Log("Monthly allocation review")
        # Review and adjust strategy allocations
    
    def OnWarmupFinished(self):
        """LEAN's native warm-up completion event"""
        self.Log("Warm-up completed - strategy ready for trading")
        
        # Validate that indicators are ready using LEAN's IsReady property
        all_ready = True
        for ticker, indicators in self.indicators.items():
            for name, indicator in indicators.items():
                if not indicator.IsReady:
                    self.Error(f"Indicator {name} for {ticker} not ready after warm-up")
                    all_ready = False
        
        if all_ready:
            self.Log("All indicators warmed up successfully")
        else:
            self.Error("Some indicators not ready - check warm-up period")

# Template Usage Instructions:
"""
TO USE THIS TEMPLATE:

1. Copy this file to create your new strategy
2. Replace 'LEANAwareCTAStrategy' with your strategy name
3. Modify the signal generation logic in _generate_signals()
4. Adjust risk parameters in _setup_risk_management()
5. Customize indicators in _setup_indicators() as needed

REMEMBER:
- Always use LEAN's built-in methods (marked with comments)
- Never implement custom versions of portfolio tracking, indicators, etc.
- Use LEAN's scheduling, order management, and data access
- Refer to docs/lean-cheatsheet.md before adding any functionality

TESTING:
- Test with small amounts first
- Validate all indicators warm up correctly
- Monitor drawdown and leverage limits
- Check rollover handling works as expected
""" 