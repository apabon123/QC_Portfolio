# universe.py - UPDATED WITH CONFIG INTEGRATION AND ROLLOVER SUPPORT
from AlgorithmImports import *
import numpy as np

class AssetFilterManager:
    """
    Asset filtering manager for multi-strategy systems.
    Provides clean separation between strategy logic and asset universe management.
    NOW FULLY CONFIG-DRIVEN.
    """
    
    def __init__(self, algorithm, config_manager):
        self.algorithm = algorithm
        self.config_manager = config_manager
        
        # Load asset categories and filters from config
        self.asset_categories = config_manager.config.get('asset_categories', {})
        self.strategy_filters = config_manager.config.get('strategy_asset_filters', {})
        
        self.algorithm.Log(f"AssetFilterManager: Initialized with {len(self.asset_categories)} categories, {len(self.strategy_filters)} strategy filters")
    
    def get_symbols_for_strategy(self, strategy_name, all_symbols):
        """
        Filter symbols for a specific strategy using config-defined rules.
        
        Args:
            strategy_name: Name of strategy (e.g., 'KestnerCTA')
            all_symbols: List of all available symbols
            
        Returns:
            List of symbols this strategy should trade
        """
        if strategy_name not in self.strategy_filters:
            # If no filter defined, return all symbols (backwards compatible)
            self.algorithm.Log(f"ASSET FILTER: {strategy_name} - no filtering applied")
            return all_symbols
        
        filter_config = self.strategy_filters[strategy_name]
        allowed_symbols = set()
        
        # 1. Add symbols from allowed categories
        for category in filter_config.get('allowed_categories', []):
            if category in self.asset_categories:
                allowed_symbols.update(self.asset_categories[category])
        
        # 2. Add specifically allowed symbols
        allowed_symbols.update(filter_config.get('allowed_symbols', []))
        
        # 3. Remove symbols from excluded categories
        for category in filter_config.get('excluded_categories', []):
            if category.endswith('*'):
                # Handle wildcard exclusions like 'futures_*'
                prefix = category[:-1]
                for cat_name, symbols in self.asset_categories.items():
                    if cat_name.startswith(prefix):
                        allowed_symbols -= set(symbols)
            elif category in self.asset_categories:
                allowed_symbols -= set(self.asset_categories[category])
        
        # 4. Remove specifically excluded symbols
        allowed_symbols -= set(filter_config.get('excluded_symbols', []))
        
        # 5. Convert symbols to tickers for comparison (handle Symbol objects)
        all_tickers = []
        for sym in all_symbols:
            if hasattr(sym, 'Value'):
                # It's a Symbol object, extract the ticker
                ticker = self._extract_ticker_from_symbol(sym)
                all_tickers.append((ticker, sym))
            else:
                # It's already a string ticker
                all_tickers.append((sym, sym))
        
        # 6. Only return symbols that exist in universe and are allowed
        final_symbols = []
        for ticker, original_symbol in all_tickers:
            if ticker in allowed_symbols:
                final_symbols.append(original_symbol)
        
        return final_symbols
    
    def _extract_ticker_from_symbol(self, symbol):
        """Extract ticker from a QuantConnect Symbol object."""
        try:
            # For futures, the ID.Symbol gives us the base ticker
            if hasattr(symbol, 'ID') and hasattr(symbol.ID, 'Symbol'):
                return symbol.ID.Symbol
            # Fallback to string representation
            return str(symbol).split(' ')[0]
        except:
            return str(symbol)
    
    def get_filter_reason(self, strategy_name):
        """Get the reasoning behind a strategy's asset filter."""
        if strategy_name in self.strategy_filters:
            return self.strategy_filters[strategy_name].get('reason', 'No reason specified')
        return 'No filter applied'
    
    def log_filtering_results(self, strategy_name, all_symbols, filtered_symbols):
        """Log filtering results for debugging."""
        if len(filtered_symbols) < len(all_symbols):
            all_tickers = [self._extract_ticker_from_symbol(s) for s in all_symbols]
            filtered_tickers = [self._extract_ticker_from_symbol(s) for s in filtered_symbols]
            removed_tickers = set(all_tickers) - set(filtered_tickers)
            
            self.algorithm.Log(f"ASSET FILTER: {strategy_name} excluded {removed_tickers}")
            self.algorithm.Log(f"ASSET FILTER: {strategy_name} will trade {filtered_tickers}")
            self.algorithm.Log(f"ASSET FILTER: Reason - {self.get_filter_reason(strategy_name)}")
        else:
            self.algorithm.Log(f"ASSET FILTER: {strategy_name} - no filtering applied")

class FuturesManager:
    """
    OPTIMIZED FUTURES MANAGER - LEVERAGING QC BUILT-INS
    
    Purpose: Maximize QuantConnect's native capabilities for futures management
    - Uses QC's built-in Securities properties (HasData, IsTradable, Price)
    - Leverages native SymbolProperties for lot sizes and multipliers  
    - Uses QC's built-in market hours and exchange properties
    - Only adds custom logic where QC doesn't provide it
    """
    
    def __init__(self, algorithm, config_manager):
        self.algorithm = algorithm
        self.config_manager = config_manager
        
        # Get universe configuration
        self.universe_config = config_manager.get_universe_config()
        
        # Storage for securities - let QC manage the securities themselves
        self.futures_symbols = []
        
        # Track priority groupings for our custom logic
        self.priority_groups = {}
        self._initialize_priority_groups()
        
        # Track rollover state - QC doesn't handle this automatically
        self.rollover_state = {}
        self.last_rollover_check = None
        
        self.algorithm.Log("FuturesManager: Initialized with QC-optimized approach")
    
    def _initialize_priority_groups(self):
        """Initialize priority groups from configuration"""
        try:
            for priority, symbols in self.universe_config.items():
                if str(priority).isdigit():
                    self.priority_groups[int(priority)] = symbols
                    self.algorithm.Log(f"FuturesManager: Priority {priority} - {len(symbols)} symbols")
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Error initializing priority groups: {str(e)}")
    
    def initialize_universe(self):
        """
        Initialize the futures universe - main entry point called by algorithm
        This method orchestrates the complete universe setup process
        """
        try:
            self.algorithm.Log("FuturesManager: Starting universe initialization...")
            
            # Add all futures contracts to the universe
            self.add_futures_universe()
            
            # Log initialization results
            info = self.get_futures_info()
            self.algorithm.Log(f"FuturesManager: Universe initialized - {info['total_symbols']} total symbols, {info['liquid_symbols']} liquid")
            
            # Log priority breakdown
            for priority, count in info['priority_breakdown'].items():
                self.algorithm.Log(f"FuturesManager: Priority {priority}: {count} symbols")
            
            self.algorithm.Log("FuturesManager: Universe initialization complete")
            
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Error in initialize_universe: {str(e)}")
            raise

    def add_futures_universe(self):
        """
        Add futures universe using QC's native methods
        Leverage QC's built-in futures chain and contract management
        FIXED: Check if symbols are already added before trying to add them again
        """
        try:
            added_count = 0
            found_existing_count = 0
            
            # Process each priority group
            for priority in sorted(self.priority_groups.keys()):
                symbols = self.priority_groups[priority]
                self.algorithm.Log(f"Processing Priority {priority} futures ({len(symbols)} symbols)...")
                
                for symbol_config in symbols:
                    # CRITICAL FIX: Check if symbol is already in Securities first
                    if isinstance(symbol_config, dict):
                        ticker = symbol_config.get('ticker')
                    else:
                        ticker = str(symbol_config)
                    
                    # Look for existing symbols in the algorithm's Securities
                    existing_symbol = self._find_existing_symbol(ticker)
                    
                    if existing_symbol:
                        # Use the existing symbol instead of adding a new one
                        self.algorithm.Log(f"FuturesManager: Found existing symbol for {ticker}: {existing_symbol}")
                        self.futures_symbols.append(existing_symbol)
                        
                        # Initialize our custom tracking for existing symbol
                        self.rollover_state[existing_symbol] = {
                            'priority': priority,
                            'ticker': ticker,
                            'market': 'CME',
                            'last_contract': None,
                            'rollover_count': 0
                        }
                        
                        found_existing_count += 1
                    else:
                        # Symbol doesn't exist, try to add it
                        symbol_added = self._add_single_future_qc_native(symbol_config, priority)
                        if symbol_added:
                            added_count += 1
            
            self.algorithm.Log(f"FuturesManager: Found {found_existing_count} existing symbols, added {added_count} new symbols")
            self.algorithm.Log(f"FuturesManager: Total symbols managed: {len(self.futures_symbols)}")
            
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Error adding futures universe: {str(e)}")
    
    def _find_existing_symbol(self, ticker):
        """
        Find existing symbol in algorithm's Securities collection.
        Look for both continuous contract formats (e.g., /ES, futures/ES)
        """
        try:
            # Check different symbol formats that might exist
            possible_symbols = [
                f"/{ticker}",  # e.g., /ES
                f"futures/{ticker}",  # e.g., futures/ES  
                ticker,  # e.g., ES
            ]
            
            for symbol_str in possible_symbols:
                for existing_symbol in self.algorithm.Securities.Keys:
                    if str(existing_symbol) == symbol_str or str(existing_symbol).endswith(f"/{ticker}"):
                        self.algorithm.Log(f"FuturesManager: Matched {ticker} -> {existing_symbol}")
                        return existing_symbol
            
            return None
            
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Error finding existing symbol for {ticker}: {str(e)}")
            return None
    
    def _add_single_future_qc_native(self, symbol_config, priority):
        """
        Add single future using QC's native AddFuture method
        Leverage all of QC's built-in futures handling
        """
        try:
            # Extract symbol info
            if isinstance(symbol_config, dict):
                ticker = symbol_config.get('ticker')
                market = symbol_config.get('market', 'CME')  # Default to CME
            else:
                ticker = str(symbol_config)
                market = 'CME'  # Default market
            
            if not ticker:
                return False
            
            # Use QC's native AddFuture with optimal settings
            future = self.algorithm.AddFuture(
                ticker=ticker,
                resolution=Resolution.DAILY,  # Use daily for longer-term CTA
                market=market,
                fillForward=True,  # Fill forward for cleaner data
                leverage=1.0,  # Conservative leverage
                extendedMarketHours=False,  # Standard hours for most futures
                dataMappingMode=DataMappingMode.LAST_TRADING_DAY,  # Standard rollover
                dataNormalizationMode=DataNormalizationMode.RAW,  # Raw prices for futures
                contractDepthOffset=0  # Front month
            )
            
            if future:
                # Store the continuous symbol - QC handles contract mapping
                self.futures_symbols.append(future.Symbol)
                
                # Initialize our custom tracking
                self.rollover_state[future.Symbol] = {
                    'priority': priority,
                    'ticker': ticker,
                    'market': market,
                    'last_contract': None,
                    'rollover_count': 0
                }
                
                return True
            
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Error adding {ticker}: {str(e)}")
            
        return False
    
    def get_liquid_symbols(self):
        """
        Return tradable symbols using QC's built-in validation
        Leverage Securities properties instead of custom checks
        """
        liquid_symbols = []
        
        try:
            self.algorithm.Log(f"FuturesManager: Checking {len(self.futures_symbols)} symbols for liquidity")
            
            for symbol in self.futures_symbols:
                symbol_str = str(symbol)
                
                # Use QC's built-in validation methods
                if self._is_symbol_tradable_qc_native(symbol):
                    liquid_symbols.append(symbol)
                    self.algorithm.Log(f"FuturesManager: {symbol_str} is LIQUID")
                else:
                    # Log why it's not liquid for debugging
                    if symbol in self.algorithm.Securities:
                        security = self.algorithm.Securities[symbol]
                        has_data = getattr(security, 'HasData', False)
                        is_tradable = getattr(security, 'IsTradable', False)
                        price = getattr(security, 'Price', None)
                        self.algorithm.Log(f"FuturesManager: {symbol_str} NOT liquid - HasData:{has_data}, IsTradable:{is_tradable}, Price:{price}")
                    else:
                        self.algorithm.Log(f"FuturesManager: {symbol_str} NOT in Securities")
        
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Error getting liquid symbols: {str(e)}")
        
        self.algorithm.Log(f"FuturesManager: Found {len(liquid_symbols)} liquid symbols out of {len(self.futures_symbols)} total")
        return liquid_symbols
    
    def _is_symbol_tradable_qc_native(self, symbol):
        """
        LEVERAGE QC's built-in tradability checks
        Use Securities properties instead of rebuilding validation
        """
        try:
            # Check if symbol exists in securities (QC managed)
            if symbol not in self.algorithm.Securities:
                return False
            
            security = self.algorithm.Securities[symbol]
            
            # LEVERAGE QC'S BUILT-IN PROPERTIES:
            
            # 1. IsTradable property (QC built-in)
            if not security.IsTradable:
                return False
            
            # 2. HasData property (QC built-in)  
            if not security.HasData:
                return False
            
            # 3. Price validation (QC built-in Price property)
            if not hasattr(security, 'Price') or security.Price is None or security.Price <= 0:
                return False
            
            # 4. Use QC's SymbolProperties for contract validation
            if hasattr(security, 'SymbolProperties'):
                # QC provides lot size, minimum price variation, etc.
                symbol_props = security.SymbolProperties
                
                # Validate lot size exists (QC built-in)
                if hasattr(symbol_props, 'LotSize') and symbol_props.LotSize <= 0:
                    return False
            
            # 5. Market hours validation (QC built-in via Exchange)
            if hasattr(security, 'Exchange') and security.Exchange:
                # QC handles market hours automatically
                # We don't need to reimplement this
                pass
            
            return True
            
        except Exception as e:
            # If validation fails, assume not tradable
            return False
    
    def validate_price(self, symbol, price):
        """
        Basic price validation - QC doesn't provide range checking
        Only implement what QC doesn't already handle
        """
        try:
            # QC already validates Price > 0 in Securities.Price
            # We only need to add business logic validation
            
            if price <= 0:
                return False
                
            # Get ticker for range checking
            ticker = self._get_ticker_from_symbol(symbol)
            
            # Basic sanity ranges for different asset classes
            # This is business logic QC doesn't provide
            ranges = {
                'ES': (500, 10000),     # E-mini S&P 500
                'NQ': (1000, 30000),    # E-mini Nasdaq
                'GC': (800, 3000),      # Gold
                'CL': (10, 200),        # Crude Oil
                'ZN': (50, 200),        # 10-Year Treasury
                'ZB': (70, 200),        # 30-Year Treasury
                '6E': (0.8, 1.5),       # Euro
                '6J': (0.004, 0.02),    # Japanese Yen
                'YM': (10000, 50000),   # Dow Mini
            }
            
            if ticker in ranges:
                min_price, max_price = ranges[ticker]
                return min_price <= price <= max_price
            
            # Default wide range for unknown contracts
            return 0.001 <= price <= 1000000
            
        except Exception as e:
            return True  # Don't block on validation errors
    
    def validate_multiplier(self, symbol):
        """
        Leverage QC's SymbolProperties for contract specifications
        QC provides this information - we don't need to rebuild it
        """
        try:
            if symbol not in self.algorithm.Securities:
                return False
            
            security = self.algorithm.Securities[symbol]
            
            # Use QC's built-in SymbolProperties
            if hasattr(security, 'SymbolProperties'):
                symbol_props = security.SymbolProperties
                
                # QC provides LotSize (contract multiplier)
                if hasattr(symbol_props, 'LotSize'):
                    return symbol_props.LotSize > 0
            
            return True  # If we can't validate, assume it's okay
            
        except Exception as e:
            return True
    
    def get_contract_multiplier(self, symbol):
        """
        Get contract multiplier using QC's SymbolProperties
        Leverage QC's built-in contract specifications
        """
        try:
            if symbol in self.algorithm.Securities:
                security = self.algorithm.Securities[symbol]
                
                # Use QC's built-in SymbolProperties
                if hasattr(security, 'SymbolProperties'):
                    symbol_props = security.SymbolProperties
                    
                    # QC provides LotSize which is the contract multiplier
                    if hasattr(symbol_props, 'LotSize'):
                        return float(symbol_props.LotSize)
            
            # Fallback values for major contracts
            ticker = self._get_ticker_from_symbol(symbol)
            fallback_multipliers = {
                'ES': 50, 'NQ': 20, 'YM': 5, 'GC': 100, 'CL': 1000,
                'ZN': 1000, 'ZB': 1000, '6E': 125000, '6J': 12500000
            }
            
            return fallback_multipliers.get(ticker, 1.0)
            
        except Exception as e:
            return 1.0
    
    def check_rollover_needed(self):
        """
        Check if any contracts need rollover
        QC doesn't handle automatic rollover - this is our custom logic
        """
        rollover_events = []
        
        try:
            for symbol in self.futures_symbols:
                if self._needs_rollover_qc_check(symbol):
                    rollover_info = self._prepare_rollover_info(symbol)
                    if rollover_info:
                        rollover_events.append(rollover_info)
        
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Error checking rollover: {str(e)}")
        
        return rollover_events
    
    def _needs_rollover_qc_check(self, symbol):
        """
        Check rollover using QC's contract information
        Leverage QC's expiry and contract data
        """
        try:
            if symbol not in self.algorithm.Securities:
                return False
            
            security = self.algorithm.Securities[symbol]
            
            # QC provides Mapped property for current contract
            if hasattr(security, 'Mapped') and security.Mapped:
                mapped_contract = security.Mapped
                
                # Check if mapped contract has changed (QC handles this)
                current_contract = str(mapped_contract)
                last_contract = self.rollover_state[symbol].get('last_contract')
                
                if last_contract and current_contract != last_contract:
                    return True
                
                # Use QC's built-in expiry checking
                if hasattr(mapped_contract, 'ID') and hasattr(mapped_contract.ID, 'Date'):
                    expiry = mapped_contract.ID.Date
                    days_to_expiry = (expiry - self.algorithm.Time).days
                    
                    # Rollover 5 days before expiry
                    return days_to_expiry <= 5
            
            return False
            
        except Exception as e:
            return False
    
    def _prepare_rollover_info(self, symbol):
        """Prepare rollover information using QC's contract data"""
        try:
            if symbol in self.algorithm.Securities:
                security = self.algorithm.Securities[symbol]
                
                rollover_info = {
                    'symbol': symbol,
                    'ticker': self.rollover_state[symbol]['ticker'],
                    'priority': self.rollover_state[symbol]['priority'],
                    'old_contract': self.rollover_state[symbol].get('last_contract'),
                    'new_contract': str(security.Mapped) if hasattr(security, 'Mapped') else None
                }
                
                return rollover_info
            
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Error preparing rollover info: {str(e)}")
        
        return None
    
    def update_rollover_state(self, symbol, new_contract):
        """Update rollover tracking state"""
        if symbol in self.rollover_state:
            self.rollover_state[symbol]['last_contract'] = str(new_contract)
            self.rollover_state[symbol]['rollover_count'] += 1
    
    def _get_ticker_from_symbol(self, symbol):
        """Extract ticker from QC Symbol using built-in properties"""
        try:
            # Use QC's Symbol.Value property
            if hasattr(symbol, 'Value'):
                return str(symbol.Value).replace('/', '')
            return str(symbol)
        except:
            return str(symbol)
    
    def get_priority_symbols(self, priority):
        """Get symbols for a specific priority level"""
        priority_symbols = []
        
        for symbol in self.futures_symbols:
            if symbol in self.rollover_state:
                if self.rollover_state[symbol]['priority'] == priority:
                    priority_symbols.append(symbol)
        
        return priority_symbols
    
    def get_futures_info(self):
        """Get comprehensive futures information using QC properties"""
        info = {
            'total_symbols': len(self.futures_symbols),
            'liquid_symbols': len(self.get_liquid_symbols()),
            'priority_breakdown': {},
            'contract_details': {}
        }
        
        # Priority breakdown
        for priority in self.priority_groups.keys():
            priority_symbols = self.get_priority_symbols(priority)
            info['priority_breakdown'][priority] = len(priority_symbols)
        
        # Contract details using QC properties
        for symbol in self.futures_symbols[:5]:  # Sample first 5
            try:
                if symbol in self.algorithm.Securities:
                    security = self.algorithm.Securities[symbol]
                    ticker = self._get_ticker_from_symbol(symbol)
                    
                    details = {
                        'has_data': security.HasData if hasattr(security, 'HasData') else False,
                        'is_tradable': security.IsTradable if hasattr(security, 'IsTradable') else False,
                        'price': security.Price if hasattr(security, 'Price') else 0,
                        'multiplier': self.get_contract_multiplier(symbol)
                    }
                    
                    # Add QC's SymbolProperties info
                    if hasattr(security, 'SymbolProperties'):
                        props = security.SymbolProperties
                        if hasattr(props, 'LotSize'):
                            details['lot_size'] = props.LotSize
                        if hasattr(props, 'MinimumPriceVariation'):
                            details['min_price_variation'] = props.MinimumPriceVariation
                    
                    info['contract_details'][ticker] = details
                    
            except Exception as e:
                continue
        
        return info
