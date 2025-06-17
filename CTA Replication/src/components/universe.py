# universe.py - UPDATED WITH CONFIG INTEGRATION AND ROLLOVER SUPPORT
from AlgorithmImports import *
import numpy as np

# Defensive import for QuantConnect compatibility
try:
    from asset_filter_manager import AssetFilterManager
except ImportError:
    # Fallback: Define a minimal AssetFilterManager inline if import fails
    class AssetFilterManager:
        def __init__(self, algorithm, config_manager):
            self.algorithm = algorithm
            self.config_manager = config_manager
            algorithm.Log("AssetFilterManager: Using fallback implementation")
        
        def get_symbols_for_strategy(self, strategy_name, all_symbols):
            return all_symbols

class FuturesManager:
    """
    OPTIMIZED FUTURES MANAGER - LEVERAGING QC BUILT-INS
    
    Purpose: Maximize QuantConnect's native capabilities for futures management
    - Uses QC's built-in Securities properties (HasData, IsTradable, Price)
    - Leverages native SymbolProperties for lot sizes and multipliers  
    - Uses QC's built-in market hours and exchange properties
    - Only adds custom logic where QC doesn't provide it
    """
    
    def __init__(self, algorithm, config_manager, shared_symbols=None):
        try:
            self.algorithm = algorithm
            self.algorithm.Log("FuturesManager: Starting initialization...")
            
            self.config_manager = config_manager
            self.algorithm.Log("FuturesManager: Config manager set")
            
            self.shared_symbols = shared_symbols or {}  # Shared symbols from OptimizedSymbolManager
            self.algorithm.Log(f"FuturesManager: Shared symbols set ({len(self.shared_symbols)} symbols)")
            
            # Get universe configuration
            self.algorithm.Log("FuturesManager: Getting universe configuration...")
            self.universe_config = config_manager.get_universe_config()
            self.algorithm.Log("FuturesManager: Universe configuration retrieved")
            
            # Storage for securities - let QC manage the securities themselves
            self.futures_symbols = []
            self.algorithm.Log("FuturesManager: Futures symbols list initialized")
            
            # Track priority groupings for our custom logic
            self.priority_groups = {}
            self.algorithm.Log("FuturesManager: About to initialize priority groups...")
            self._initialize_priority_groups()
            self.algorithm.Log("FuturesManager: Priority groups initialized successfully")
            
            # Track rollover state - QC doesn't handle this automatically
            self.rollover_state = {}
            self.last_rollover_check = None
            self.algorithm.Log("FuturesManager: Rollover state initialized")
            
            self.algorithm.Log("FuturesManager: Initialized with QC-optimized approach")
            
        except Exception as e:
            algorithm.Error(f"CRITICAL ERROR in FuturesManager.__init__: {str(e)}")
            import traceback
            algorithm.Error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def _initialize_priority_groups(self):
        """Initialize priority groups from configuration"""
        try:
            self.algorithm.Log("_initialize_priority_groups: Starting...")
            self.priority_groups = {}
            
            # Parse futures section
            self.algorithm.Log("_initialize_priority_groups: Getting futures config...")
            futures_config = self.universe_config.get('futures', {})
            self.algorithm.Log(f"_initialize_priority_groups: Futures config type: {type(futures_config)}, keys: {list(futures_config.keys()) if isinstance(futures_config, dict) else 'Not a dict'}")
            
            for category_name, category_symbols in futures_config.items():
                self.algorithm.Log(f"_initialize_priority_groups: Processing category {category_name}, type: {type(category_symbols)}")
                if isinstance(category_symbols, dict):
                    for ticker, symbol_config in category_symbols.items():
                        self.algorithm.Log(f"_initialize_priority_groups: Processing ticker {ticker}, config type: {type(symbol_config)}")
                        if isinstance(symbol_config, dict):
                            priority = symbol_config.get('priority', 1)
                            if priority not in self.priority_groups:
                                self.priority_groups[priority] = []
                            
                            # Create symbol config with ticker
                            symbol_entry = {
                                'ticker': ticker,
                                'name': symbol_config.get('name', ticker),
                                'category': symbol_config.get('category', 'unknown'),
                                'priority': priority,
                                'min_volume': symbol_config.get('min_volume', 0)
                            }
                            self.priority_groups[priority].append(symbol_entry)
                            self.algorithm.Log(f"_initialize_priority_groups: Added {ticker} to priority {priority}")
            
            # Parse expansion candidates section
            self.algorithm.Log("_initialize_priority_groups: Getting expansion candidates config...")
            expansion_config = self.universe_config.get('expansion_candidates', {})
            self.algorithm.Log(f"_initialize_priority_groups: Expansion config type: {type(expansion_config)}, keys: {list(expansion_config.keys()) if isinstance(expansion_config, dict) else 'Not a dict'}")
            
            for ticker, symbol_config in expansion_config.items():
                self.algorithm.Log(f"_initialize_priority_groups: Processing expansion ticker {ticker}, config type: {type(symbol_config)}")
                if isinstance(symbol_config, dict):
                    priority = symbol_config.get('priority', 2)
                    if priority not in self.priority_groups:
                        self.priority_groups[priority] = []
                    
                    # Create symbol config with ticker
                    symbol_entry = {
                        'ticker': ticker,
                        'name': symbol_config.get('name', ticker),
                        'category': symbol_config.get('category', 'unknown'),
                        'priority': priority,
                        'min_volume': symbol_config.get('min_volume', 0)
                    }
                    self.priority_groups[priority].append(symbol_entry)
                    self.algorithm.Log(f"_initialize_priority_groups: Added expansion {ticker} to priority {priority}")
            
            # Log results
            self.algorithm.Log("_initialize_priority_groups: Logging final results...")
            for priority in sorted(self.priority_groups.keys()):
                symbols = self.priority_groups[priority]
                tickers = [s['ticker'] for s in symbols]
                self.algorithm.Log(f"FuturesManager: Priority {priority} - {len(symbols)} symbols: {tickers}")
            
            self.algorithm.Log("_initialize_priority_groups: Completed successfully")
                
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Error initializing priority groups: {str(e)}")
            import traceback
            self.algorithm.Error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def initialize_universe(self):
        """
        Initialize the futures universe using shared symbols from OptimizedSymbolManager.
        This method now leverages shared symbol subscriptions instead of creating new ones.
        """
        try:
            self.algorithm.Log("FuturesManager: Starting universe initialization with shared symbols...")
            
            if self.shared_symbols:
                # Use shared symbols from OptimizedSymbolManager
                self.futures_symbols = list(self.shared_symbols.values())
                self.algorithm.Log(f"FuturesManager: Using {len(self.futures_symbols)} shared symbols from OptimizedSymbolManager")
                
                # Initialize rollover state for shared symbols
                for ticker, symbol in self.shared_symbols.items():
                    self._initialize_rollover_state(symbol, {'ticker': ticker, 'priority': 1})
                
            else:
                # Fallback: Add futures contracts manually (should not happen with optimized approach)
                self.algorithm.Log("FuturesManager: WARNING - No shared symbols provided, falling back to manual universe creation")
                self.add_futures_universe()
            
            # Log initialization results
            info = self.get_futures_info()
            self.algorithm.Log(f"FuturesManager: Universe initialized - {info['total_symbols']} total symbols")
            
            self.algorithm.Log("FuturesManager: Universe initialization complete")
            
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Error in initialize_universe: {str(e)}")
            raise

    def add_futures_universe(self):
        """
        Add futures universe using QC's native methods
        Leverage QC's built-in futures chain and contract management
        CRITICAL: No fallback logic - fail fast if configuration is wrong
        """
        try:
            added_count = 0
            found_existing_count = 0
            
            # VALIDATION: Ensure we have valid priority groups from configuration
            if not self.priority_groups:
                error_msg = "CRITICAL ERROR: No priority groups found in configuration"
                self.algorithm.Error(error_msg)
                raise ValueError(error_msg)
            
            total_configured_symbols = sum(len(symbols) for symbols in self.priority_groups.values())
            if total_configured_symbols == 0:
                error_msg = "CRITICAL ERROR: No symbols found in any priority group"
                self.algorithm.Error(error_msg)
                raise ValueError(error_msg)
            
            self.algorithm.Log(f"FuturesManager: Processing {total_configured_symbols} configured symbols")
            
            # Process symbols by priority (1 = highest priority)
            for priority in sorted(self.priority_groups.keys()):
                symbols = self.priority_groups[priority]
                self.algorithm.Log(f"Processing priority {priority}: {len(symbols)} symbols")
                
                for symbol_config in symbols:
                    ticker = symbol_config['ticker']
                    
                    try:
                        # Check if continuous contract already exists in algorithm
                        continuous_symbol = f"/{ticker}"
                        existing_symbol = None
                        
                        for symbol in self.algorithm.Securities.Keys:
                            if str(symbol) == continuous_symbol:
                                existing_symbol = symbol
                                break
                        
                        if existing_symbol:
                            # Symbol already added - validate it's properly configured
                            security = self.algorithm.Securities[existing_symbol]
                            if security.Type == SecurityType.Future:
                                self.futures_symbols.append(existing_symbol)
                                self._initialize_rollover_state(existing_symbol, symbol_config)
                                found_existing_count += 1
                                self.algorithm.Log(f"Found existing futures contract: {continuous_symbol}")
                            else:
                                self.algorithm.Error(f"Symbol {continuous_symbol} exists but is not a Future")
                        else:
                            # Need to add the futures contract
                            self.algorithm.Log(f"Adding new futures contract: {ticker}")
                            
                            # Get futures configuration from centralized config manager
                            try:
                                execution_config = self.config_manager.get_execution_config()
                                futures_params = execution_config.get('futures_config', {}).get('add_future_params', {})
                                filter_params = execution_config.get('futures_config', {}).get('contract_filter', {})
                            except Exception as config_error:
                                self.algorithm.Error(f"CRITICAL: Failed to get futures configuration: {str(config_error)}")
                                raise ValueError(f"Cannot add futures without valid configuration: {str(config_error)}")
                            
                            # Map string values to QuantConnect enums
                            resolution = getattr(Resolution, futures_params.get('resolution', 'Daily'))
                            data_mapping_mode = getattr(DataMappingMode, futures_params.get('data_mapping_mode', 'OpenInterest'))
                            data_normalization_mode = getattr(DataNormalizationMode, futures_params.get('data_normalization_mode', 'BackwardsRatio'))
                            
                            # SIMPLIFIED APPROACH: Use AddFuture for continuous contracts
                            # This creates a futures universe but we'll use the continuous contract for trading
                            future = self.algorithm.AddFuture(
                                ticker=ticker,
                                resolution=resolution,
                                fillForward=futures_params.get('fill_forward', True),
                                leverage=futures_params.get('leverage', 1.0),
                                extendedMarketHours=futures_params.get('extended_market_hours', False),
                                dataMappingMode=data_mapping_mode,
                                dataNormalizationMode=data_normalization_mode,
                                contractDepthOffset=futures_params.get('contract_depth_offset', 0)
                            )
                            
                            # Set contract filter - this determines which contracts are available in the chain
                            min_days = filter_params.get('min_days_out', 0)
                            max_days = filter_params.get('max_days_out', 182)
                            future.SetFilter(timedelta(days=min_days), timedelta(days=max_days))
                            
                            self.algorithm.Log(f"FUTURES SETUP: {ticker} - Created futures universe with continuous contract")
                            
                            # Track the symbol and log detailed information
                            self.futures_symbols.append(future.Symbol)
                            self._initialize_rollover_state(future.Symbol, symbol_config)
                            added_count += 1
                            
                            # DEBUG: Log detailed symbol information
                            self.algorithm.Log(f"FUTURES DEBUG: Added {ticker}")
                            self.algorithm.Log(f"  - Symbol: {future.Symbol}")
                            self.algorithm.Log(f"  - Symbol Type: {future.Symbol.SecurityType}")
                            self.algorithm.Log(f"  - Symbol Value: {future.Symbol.Value}")
                            self.algorithm.Log(f"  - Resolution: {resolution}")
                            self.algorithm.Log(f"  - DataMappingMode: {data_mapping_mode}")
                            self.algorithm.Log(f"  - DataNormalizationMode: {data_normalization_mode}")
                            
                    except Exception as e:
                        error_msg = f"CRITICAL ERROR adding futures contract {ticker}: {str(e)}"
                        self.algorithm.Error(error_msg)
                        # Don't continue with partial configuration - this could lead to wrong trades
                        raise ValueError(error_msg)
            
            # FINAL VALIDATION: Ensure we have the expected number of symbols
            expected_count = total_configured_symbols
            actual_count = len(self.futures_symbols)
            
            if actual_count != expected_count:
                error_msg = (f"CRITICAL ERROR: Symbol count mismatch. "
                           f"Expected {expected_count}, got {actual_count}. "
                           f"This indicates configuration or setup failure.")
                self.algorithm.Error(error_msg)
                raise ValueError(error_msg)
            
            self.algorithm.Log(f"FuturesManager: Successfully configured {actual_count} futures contracts")
            self.algorithm.Log(f"  - Added new: {added_count}")
            self.algorithm.Log(f"  - Found existing: {found_existing_count}")
            
        except Exception as e:
            error_msg = f"CRITICAL FAILURE in add_futures_universe: {str(e)}"
            self.algorithm.Error(error_msg)
            # Clear any partial state to prevent trading with wrong configuration
            self.futures_symbols = []
            self.rollover_state = {}
            # Re-raise to stop algorithm execution
            raise ValueError(error_msg)
    
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
    
    def _initialize_rollover_state(self, symbol, symbol_config):
        """
        Initialize rollover tracking state for a futures symbol.
        This tracks our custom rollover logic separate from QC's built-in handling.
        """
        try:
            # Extract symbol information
            if isinstance(symbol_config, dict):
                ticker = symbol_config.get('ticker', self._get_ticker_from_symbol(symbol))
                priority = symbol_config.get('priority', 1)
                market = symbol_config.get('market', 'CME')
                category = symbol_config.get('category', 'unknown')
            else:
                # Fallback if symbol_config is just a string
                ticker = str(symbol_config)
                priority = 1
                market = 'CME'
                category = 'unknown'
            
            # Initialize rollover state tracking
            self.rollover_state[symbol] = {
                'priority': priority,
                'ticker': ticker,
                'market': market,
                'category': category,
                'last_contract': None,
                'rollover_count': 0,
                'initialized_time': self.algorithm.Time
            }
            
            self.algorithm.Log(f"FuturesManager: Initialized rollover state for {ticker} (priority {priority})")
            
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Error initializing rollover state for {symbol}: {str(e)}")
            # Initialize with minimal state to prevent further errors
            self.rollover_state[symbol] = {
                'priority': 1,
                'ticker': str(symbol),
                'market': 'CME',
                'category': 'unknown',
                'last_contract': None,
                'rollover_count': 0,
                'initialized_time': self.algorithm.Time
            }
    
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
    
    def get_liquid_symbols(self, slice=None):
        """
        Get currently liquid symbols with ENHANCED warm-up awareness and futures chain analysis.
        
        CRITICAL: Now takes slice parameter to properly access FuturesChains data.
        
        During warm-up: Uses chain analysis or fallback to major liquid contracts
        Post warm-up: Full liquidity validation with IsTradable + HasData + chain data
        
        Args:
            slice: Current data slice with FuturesChains data
        
        Returns:
            list: List of liquid symbols
        """
        liquid_symbols = []
        
        try:
            is_warming_up = getattr(self.algorithm, 'IsWarmingUp', False)
            chain_config = self.config_manager.get_futures_chain_config()
            
            self.algorithm.Log(f"FuturesManager: Checking {len(self.futures_symbols)} symbols for liquidity (WarmingUp: {is_warming_up})")
            
            if is_warming_up:
                # WARM-UP LIQUIDITY CHECKING (BASED ON QC PRIMER)
                liquid_symbols = self._get_liquid_symbols_during_warmup(slice, chain_config)
            else:
                # POST WARM-UP LIQUIDITY CHECKING
                liquid_symbols = self._get_liquid_symbols_post_warmup(slice, chain_config)
        
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Error getting liquid symbols: {str(e)}")
        
        self.algorithm.Log(f"FuturesManager: Found {len(liquid_symbols)} liquid symbols out of {len(self.futures_symbols)} total")
        return liquid_symbols

    def _get_liquid_symbols_during_warmup(self, slice, chain_config):
        """
        Get liquid symbols during warm-up using futures chain analysis or fallback.
        
        CRITICAL: Uses slice parameter directly for FuturesChains access.
        
        Based on QC primer: Chain data IS available during warm-up, but IsTradable may be False.
        
        Args:
            slice: Current data slice with FuturesChains data
            chain_config: Futures chain configuration
            
        Returns:
            list: Liquid symbols during warm-up
        """
        liquid_symbols = []
        warmup_config = chain_config.get('liquidity_during_warmup', {})
        
        if not warmup_config.get('enabled', True):
            self.algorithm.Log("FuturesManager: Warm-up liquidity checking disabled")
            return []
        
        for symbol in self.futures_symbols:
            symbol_str = str(symbol)
            
            try:
                # CRITICAL: Try futures chain analysis first using slice parameter
                if slice and hasattr(slice, 'FuturesChains') and symbol in slice.FuturesChains:
                    chain = slice.FuturesChains[symbol]
                    if self._analyze_chain_liquidity(chain, warmup_config):
                        liquid_symbols.append(symbol)
                        ticker = self._get_ticker_from_symbol(symbol)
                        self.algorithm.Log(f"FuturesManager: {symbol_str} LIQUID via chain analysis (ticker: {ticker})")
                        continue
                
                # Fallback to major liquid contracts list
                if warmup_config.get('fallback_to_major_list', True):
                    if self.config_manager.should_assume_liquid_during_warmup(symbol_str):
                        liquid_symbols.append(symbol)
                        ticker = self._get_ticker_from_symbol(symbol)
                        self.algorithm.Log(f"FuturesManager: {symbol_str} assumed LIQUID during warmup (major contract: {ticker})")
                    else:
                        ticker = self._get_ticker_from_symbol(symbol)
                        self.algorithm.Log(f"FuturesManager: {symbol_str} NOT in major liquid list during warmup (ticker: {ticker})")
                
            except Exception as e:
                self.algorithm.Error(f"FuturesManager: Error checking warmup liquidity for {symbol_str}: {str(e)}")
        
        return liquid_symbols

    def _get_liquid_symbols_post_warmup(self, slice, chain_config):
        """
        Get liquid symbols after warm-up using full validation.
        
        CRITICAL: Uses slice parameter directly for FuturesChains access.
        
        Args:
            slice: Current data slice with FuturesChains data
            chain_config: Futures chain configuration
            
        Returns:
            list: Liquid symbols after warm-up
        """
        liquid_symbols = []
        post_warmup_config = chain_config.get('post_warmup_liquidity', {})
        
        for symbol in self.futures_symbols:
            symbol_str = str(symbol)
            
            try:
                # Full validation after warm-up
                if self._is_symbol_tradable_qc_native(symbol):
                    # Additional chain-based validation if enabled
                    if post_warmup_config.get('use_mapped_contract', True):
                        if self._validate_mapped_contract_liquidity(symbol, slice, post_warmup_config):
                            liquid_symbols.append(symbol)
                            self.algorithm.Log(f"FuturesManager: {symbol_str} is LIQUID (full validation passed)")
                        else:
                            self.algorithm.Log(f"FuturesManager: {symbol_str} failed mapped contract validation")
                    else:
                        liquid_symbols.append(symbol)
                        self.algorithm.Log(f"FuturesManager: {symbol_str} is LIQUID (basic validation)")
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
                self.algorithm.Error(f"FuturesManager: Error checking post-warmup liquidity for {symbol_str}: {str(e)}")
        
        return liquid_symbols

    def _analyze_chain_liquidity(self, chain, warmup_config):
        """
        Analyze a futures chain for liquidity indicators.
        
        CRITICAL: Uses chain parameter directly (from slice.FuturesChains).
        
        Args:
            chain: FuturesChain object from slice.FuturesChains[symbol]
            warmup_config: Warm-up configuration
            
        Returns:
            bool: True if chain shows liquidity
        """
        try:
            if not chain or not hasattr(chain, 'Contracts'):
                return False
            
            # Analyze the chain for liquidity indicators
            liquid_contracts = 0
            total_volume = 0
            total_open_interest = 0
            
            for contract_symbol, contract in chain.Contracts.items():
                try:
                    # Check if contract has valid data
                    if not hasattr(contract, 'LastPrice') or contract.LastPrice <= 0:
                        continue
                    
                    # Get volume and open interest (if available)
                    volume = getattr(contract, 'Volume', 0)
                    open_interest = getattr(contract, 'OpenInterest', 0)
                    
                    total_volume += volume
                    total_open_interest += open_interest
                    
                    # Count as liquid if it has reasonable volume or open interest
                    min_volume = warmup_config.get('min_volume', 100)  # Lower threshold during warmup
                    min_oi = warmup_config.get('min_open_interest', 10000)  # Lower threshold during warmup
                    
                    if volume >= min_volume or open_interest >= min_oi:
                        liquid_contracts += 1
                
                except Exception as contract_e:
                    self.algorithm.Debug(f"Error analyzing contract {contract_symbol}: {str(contract_e)}")
                    continue
            
            # Consider liquid if we have at least one liquid contract
            is_liquid = liquid_contracts > 0
            
            if warmup_config.get('log_chain_analysis', True) and is_liquid:
                self.algorithm.Log(f"FuturesManager: Chain analysis - "
                                 f"LiquidContracts:{liquid_contracts}, TotalVol:{total_volume}, TotalOI:{total_open_interest}")
            
            return is_liquid
            
        except Exception as e:
            if warmup_config.get('log_chain_analysis', True):
                self.algorithm.Debug(f"Chain analysis failed: {str(e)}")
            return False

    def _is_symbol_tradable_qc_native(self, symbol):
        """
        Check if symbol is tradable using QC's native validation with warmup awareness
        During warmup, continuous contracts may not have data yet - this is normal
        """
        try:
            if symbol not in self.algorithm.Securities:
                self.algorithm.Log(f"FuturesManager: {symbol} NOT in Securities collection")
                return False
            
            security = self.algorithm.Securities[symbol]
            
            # Get basic properties
            has_data = security.HasData
            is_tradable = security.IsTradable
            price = security.Price
            
            # Check if we're still warming up
            is_warming_up = self.algorithm.IsWarmingUp
            
            # Additional debugging information
            market_hours_info = ""
            try:
                if hasattr(security, 'Exchange') and hasattr(security.Exchange, 'Hours'):
                    is_market_open = security.Exchange.Hours.IsOpen(self.algorithm.Time, False)
                    market_hours_info = f", MarketOpen:{is_market_open}"
                else:
                    market_hours_info = ", MarketHours:Unknown"
            except:
                market_hours_info = ", MarketHours:Error"
            
            # Check if it's a continuous contract vs underlying
            symbol_str = str(symbol)
            contract_type = "Continuous" if symbol_str.startswith('/') else "Underlying"
            
            # WARMUP-AWARE LOGIC: During warmup, continuous contracts are expected to be valid
            # even if HasData=False temporarily
            if is_warming_up and contract_type == "Continuous":
                # During warmup, assume major liquid contracts are valid if they're in Securities
                ticker = self._get_ticker_from_symbol(symbol)
                major_liquid_contracts = ['ES', 'NQ', 'YM', 'GC', 'CL', 'ZN', 'ZB', '6E', '6J']
                
                if ticker in major_liquid_contracts:
                    self.algorithm.Log(f"FuturesManager: {symbol} ASSUMED liquid during warmup - {contract_type}, Ticker:{ticker}")
                    return True
                else:
                    self.algorithm.Log(f"FuturesManager: {symbol} NOT in major liquid contracts during warmup - {contract_type}, Ticker:{ticker}")
                    return False
            
            # POST-WARMUP LOGIC: Full validation required
            if is_tradable and has_data and price > 0:
                self.algorithm.Log(f"FuturesManager: {symbol} IS liquid - {contract_type}, HasData:{has_data}, IsTradable:{is_tradable}, Price:{price}{market_hours_info}")
                return True
            else:
                self.algorithm.Log(f"FuturesManager: {symbol} NOT liquid - {contract_type}, HasData:{has_data}, IsTradable:{is_tradable}, Price:{price}{market_hours_info}, WarmingUp:{is_warming_up}")
                
                # Additional debugging for why it's not tradable
                if not is_tradable:
                    try:
                        # Check if it's a settlement issue
                        if hasattr(security, 'IsDelisted'):
                            self.algorithm.Log(f"  DEBUG: {symbol} IsDelisted: {security.IsDelisted}")
                        if hasattr(security, 'IsTradable'):
                            self.algorithm.Log(f"  DEBUG: {symbol} IsTradable reason: Check market hours, settlement, or contract expiry")
                    except Exception as debug_e:
                        self.algorithm.Log(f"  DEBUG ERROR for {symbol}: {str(debug_e)}")
                
                return False
                
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Error checking tradability for {symbol}: {str(e)}")
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

    def _validate_mapped_contract_liquidity(self, symbol, slice, post_warmup_config):
        """
        Validate liquidity of the currently mapped contract for a continuous future.
        
        CRITICAL: Uses slice parameter directly for FuturesChains access.
        
        Args:
            symbol: Continuous futures symbol
            slice: Current data slice with FuturesChains data
            post_warmup_config: Post warm-up liquidity configuration
            
        Returns:
            bool: True if mapped contract is liquid
        """
        try:
            # Get the currently mapped contract from QC
            if symbol not in self.algorithm.Securities:
                return False
            
            security = self.algorithm.Securities[symbol]
            
            # Try to get the mapped contract (QC's continuous -> specific contract mapping)
            if hasattr(security, 'Mapped'):
                mapped_contract = security.Mapped
                
                if mapped_contract and mapped_contract in self.algorithm.Securities:
                    mapped_security = self.algorithm.Securities[mapped_contract]
                    
                    # Validate mapped contract liquidity
                    has_data = getattr(mapped_security, 'HasData', False)
                    is_tradable = getattr(mapped_security, 'IsTradable', False)
                    price = getattr(mapped_security, 'Price', 0)
                    
                    if has_data and is_tradable and price > 0:
                        # Additional chain-based validation if available
                        return self._validate_contract_chain_liquidity(mapped_contract, slice, post_warmup_config)
                    else:
                        symbol_str = str(symbol)
                        mapped_str = str(mapped_contract)
                        self.algorithm.Log(f"FuturesManager: {symbol_str} mapped contract {mapped_str} not liquid - "
                                         f"HasData:{has_data}, IsTradable:{is_tradable}, Price:{price}")
                        return False
            
            # Fallback to basic security validation if no mapped contract available
            has_data = getattr(security, 'HasData', False)
            is_tradable = getattr(security, 'IsTradable', False)
            price = getattr(security, 'Price', 0)
            
            return has_data and is_tradable and price > 0
            
        except Exception as e:
            self.algorithm.Error(f"Error validating mapped contract liquidity for {symbol}: {str(e)}")
            return False

    def _validate_contract_chain_liquidity(self, contract_symbol, slice, post_warmup_config):
        """
        Validate liquidity using chain data for a specific contract.
        
        CRITICAL: Uses slice parameter directly for FuturesChains access.
        
        Args:
            contract_symbol: Specific futures contract symbol
            slice: Current data slice with FuturesChains data
            post_warmup_config: Post warm-up configuration
            
        Returns:
            bool: True if contract meets liquidity requirements
        """
        try:
            if not slice or not hasattr(slice, 'FuturesChains'):
                return True  # Default to True if no chain data available
            
            # Find the chain that contains this contract
            for chain_symbol, chain in slice.FuturesChains.items():
                if hasattr(chain, 'Contracts') and contract_symbol in chain.Contracts:
                    contract = chain.Contracts[contract_symbol]
                    
                    # Check volume requirements
                    min_volume = post_warmup_config.get('min_volume', 1000)
                    volume = getattr(contract, 'Volume', 0)
                    
                    if volume < min_volume:
                        return False
                    
                    # Check open interest requirements
                    min_oi = post_warmup_config.get('min_open_interest', 50000)
                    open_interest = getattr(contract, 'OpenInterest', 0)
                    
                    if open_interest < min_oi:
                        return False
                    
                    # Check bid-ask spread if available
                    max_spread = post_warmup_config.get('max_bid_ask_spread', 0.001)
                    bid_price = getattr(contract, 'BidPrice', 0)
                    ask_price = getattr(contract, 'AskPrice', 0)
                    
                    if bid_price > 0 and ask_price > 0:
                        spread_ratio = (ask_price - bid_price) / bid_price
                        if spread_ratio > max_spread:
                            return False
                    
                    return True
            
            # Contract not found in any chain, default to True
            return True
            
        except Exception as e:
            self.algorithm.Debug(f"Error validating contract chain liquidity: {str(e)}")
            return True  # Default to True on error

    def update_during_warmup(self, slice):
        """
        Update universe manager during warm-up period.
        
        CRITICAL: Uses slice parameter directly for FuturesChains access.
        
        During warm-up, we can analyze futures chains and build liquidity profiles.
        
        Args:
            slice: Current data slice with FuturesChains data
        """
        try:
            # Analyze futures chains during warm-up (if available)
            if hasattr(slice, 'FuturesChains') and slice.FuturesChains:
                self._analyze_chains_during_warmup(slice.FuturesChains)
            
            # Track rollover events during warm-up
            if hasattr(slice, 'SymbolChangedEvents') and slice.SymbolChangedEvents:
                self._track_rollover_events_during_warmup(slice.SymbolChangedEvents)
                
        except Exception as e:
            self.algorithm.Error(f"Error updating universe during warmup: {str(e)}")

    def update_with_slice(self, slice):
        """
        Update universe manager with current slice for normal trading.
        
        CRITICAL: Uses slice parameter directly for FuturesChains access.
        
        Args:
            slice: Current data slice with FuturesChains data
        """
        try:
            if self.algorithm.IsWarmingUp:
                self.update_during_warmup(slice)
            else:
                self._analyze_chains_during_trading(slice.FuturesChains if hasattr(slice, 'FuturesChains') else {})
                self._track_rollover_events_during_trading(slice.SymbolChangedEvents if hasattr(slice, 'SymbolChangedEvents') else [])
                
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Error in update_with_slice: {str(e)}")

    def update_with_unified_data(self, unified_data, slice):
        """
        PHASE 3: Update with unified data interface.
        Leverages standardized chain analysis and symbol data.
        """
        try:
            # Validate unified data structure
            if not unified_data or not unified_data.get('valid', False):
                self.algorithm.Log("FuturesManager: Invalid unified data received")
                return
            
            # Extract symbols and chain data from unified data
            symbols_data = unified_data.get('symbols', {})
            if not symbols_data:
                return
            
            # Process unified chain data if available
            chains_data = {}
            liquid_symbols = []
            
            for symbol, symbol_data in symbols_data.items():
                if not symbol_data.get('valid', False):
                    continue
                
                # Check if symbol has chain data
                if 'chain' in symbol_data.get('data', {}):
                    chain_info = symbol_data['data']['chain']
                    if chain_info.get('valid', False):
                        chains_data[symbol] = chain_info
                
                # Check if symbol is liquid based on unified data
                security_data = symbol_data.get('data', {}).get('security', {})
                if (security_data.get('is_tradable', False) and 
                    security_data.get('has_data', False) and
                    security_data.get('market_hours_open', False)):
                    liquid_symbols.append(symbol)
            
            # Update analysis based on algorithm state
            if self.algorithm.IsWarmingUp:
                self._analyze_unified_chains_during_warmup(chains_data, liquid_symbols)
            else:
                self._analyze_unified_chains_during_trading(chains_data, liquid_symbols)
                
            # Track rollover events from slice if available
            if hasattr(slice, 'SymbolChangedEvents') and slice.SymbolChangedEvents:
                self._track_rollover_events_during_trading(slice.SymbolChangedEvents)
            
            # Log unified data processing
            metadata = unified_data.get('metadata', {})
            self.algorithm.Debug(f"FuturesManager: Processed {len(liquid_symbols)} liquid symbols, "
                               f"{len(chains_data)} chains from unified data")
                
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Error in update_with_unified_data: {str(e)}")
            # Fallback to traditional method
            self.update_with_slice(slice)

    def _analyze_unified_chains_during_warmup(self, chains_data, liquid_symbols):
        """Analyze unified chain data during warm-up period."""
        try:
            # Get warm-up configuration
            warmup_config = self.universe_config.get('liquidity_during_warmup', {
                'min_volume_warmup': 100,
                'min_open_interest_warmup': 10000,
                'major_liquid_contracts': ['ES', 'NQ', 'YM', 'ZN', 'ZB', '6E', '6J', 'CL', 'GC']
            })
            
            for symbol, chain_info in chains_data.items():
                # Analyze liquidity using unified chain data
                total_volume = chain_info.get('total_volume', 0)
                total_open_interest = chain_info.get('total_open_interest', 0)
                most_liquid = chain_info.get('most_liquid_contract')
                
                # Update rollover state based on chain analysis
                if most_liquid:
                    self._update_rollover_tracking_from_unified_data(symbol, most_liquid, chain_info)
                
                # Log detailed analysis if symbol is in liquid list
                if symbol in liquid_symbols:
                    self.algorithm.Debug(f"FuturesManager: {symbol} - Vol: {total_volume}, "
                                       f"OI: {total_open_interest}, Contracts: {chain_info.get('total_contracts', 0)}")
                    
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Error in unified chain analysis during warmup: {str(e)}")

    def _analyze_unified_chains_during_trading(self, chains_data, liquid_symbols):
        """Analyze unified chain data during normal trading."""
        try:
            # Get post-warmup configuration
            trading_config = self.universe_config.get('post_warmup_liquidity', {
                'min_volume': 1000,
                'min_open_interest': 50000,
                'max_bid_ask_spread': 0.001
            })
            
            for symbol, chain_info in chains_data.items():
                # More stringent liquidity analysis for trading
                if symbol in liquid_symbols:
                    # Validate chain meets trading requirements
                    if (chain_info.get('total_volume', 0) >= trading_config['min_volume'] and
                        chain_info.get('total_open_interest', 0) >= trading_config['min_open_interest']):
                        
                        # Update rollover tracking
                        most_liquid = chain_info.get('most_liquid_contract')
                        if most_liquid:
                            self._update_rollover_tracking_from_unified_data(symbol, most_liquid, chain_info)
                            
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Error in unified chain analysis during trading: {str(e)}")

    def _update_rollover_tracking_from_unified_data(self, symbol, most_liquid_contract, chain_info):
        """Update rollover tracking using unified chain data."""
        try:
            if not most_liquid_contract:
                return
                
            # Extract contract information from unified data
            contract_symbol = most_liquid_contract.get('symbol')
            expiry = most_liquid_contract.get('expiry')
            volume = most_liquid_contract.get('volume', 0)
            open_interest = most_liquid_contract.get('open_interest', 0)
            
            # Update rollover state
            ticker = self._get_ticker_from_symbol(symbol)
            if ticker not in self.rollover_state:
                self.rollover_state[ticker] = {}
                
            self.rollover_state[ticker].update({
                'current_contract': contract_symbol,
                'expiry': expiry,
                'volume': volume,
                'open_interest': open_interest,
                'last_update': self.algorithm.Time,
                'chain_liquidity': chain_info.get('total_volume', 0)
            })
            
        except Exception as e:
            self.algorithm.Error(f"FuturesManager: Error updating rollover tracking: {str(e)}")

    def _analyze_chains_during_warmup(self, futures_chains):
        """
        Analyze futures chains during warm-up to build liquidity profiles.
        
        Args:
            futures_chains: FuturesChains object from slice
        """
        try:
            for chain_symbol, chain in futures_chains.items():
                if not hasattr(chain, 'Contracts'):
                    continue
                
                contract_count = len(chain.Contracts)
                total_volume = 0
                total_oi = 0
                
                for contract_symbol, contract in chain.Contracts.items():
                    try:
                        volume = getattr(contract, 'Volume', 0)
                        oi = getattr(contract, 'OpenInterest', 0)
                        total_volume += volume
                        total_oi += oi
                    except:
                        continue
                
                # Log chain analysis periodically (avoid spam)
                if self.algorithm.Time.day == 1:  # First of month
                    chain_str = str(chain_symbol)
                    ticker = self._get_ticker_from_symbol(chain_symbol)
                    self.algorithm.Log(f"WARMUP CHAIN ANALYSIS: {ticker} - "
                                     f"Contracts:{contract_count}, Vol:{total_volume}, OI:{total_oi}")
                    
        except Exception as e:
            self.algorithm.Debug(f"Error analyzing chains during warmup: {str(e)}")

    def _analyze_chains_during_trading(self, futures_chains):
        """
        Analyze futures chains during normal trading.
        
        Args:
            futures_chains: FuturesChains object from slice
        """
        try:
            # Similar to warmup analysis but less frequent logging
            for chain_symbol, chain in futures_chains.items():
                if not hasattr(chain, 'Contracts'):
                    continue
                
                # Only log on first day of month to avoid spam
                if self.algorithm.Time.day == 1:
                    contract_count = len(chain.Contracts)
                    total_volume = 0
                    total_oi = 0
                    
                    for contract_symbol, contract in chain.Contracts.items():
                        try:
                            volume = getattr(contract, 'Volume', 0)
                            oi = getattr(contract, 'OpenInterest', 0)
                            total_volume += volume
                            total_oi += oi
                        except:
                            continue
                    
                    chain_str = str(chain_symbol)
                    ticker = self._get_ticker_from_symbol(chain_symbol)
                    self.algorithm.Log(f"TRADING CHAIN ANALYSIS: {ticker} - "
                                     f"Contracts:{contract_count}, Vol:{total_volume}, OI:{total_oi}")
                    
        except Exception as e:
            self.algorithm.Debug(f"Error analyzing chains during trading: {str(e)}")

    def _track_rollover_events_during_warmup(self, symbol_changed_events):
        """
        Track rollover events during warm-up for learning purposes.
        
        Args:
            symbol_changed_events: SymbolChangedEvents from slice
        """
        try:
            for symbol_changed in symbol_changed_events.Values:
                old_symbol = symbol_changed.OldSymbol
                new_symbol = symbol_changed.NewSymbol
                
                self.algorithm.Log(f"WARMUP ROLLOVER EVENT: {old_symbol} -> {new_symbol}")
                
                # Track for post-warmup analysis
                if not hasattr(self, '_warmup_rollover_events'):
                    self._warmup_rollover_events = []
                
                self._warmup_rollover_events.append({
                    'date': self.algorithm.Time,
                    'old_symbol': old_symbol,
                    'new_symbol': new_symbol
                })
                
        except Exception as e:
            self.algorithm.Debug(f"Error tracking rollover events during warmup: {str(e)}")

    def _track_rollover_events_during_trading(self, symbol_changed_events):
        """
        Track rollover events during normal trading.
        
        Args:
            symbol_changed_events: SymbolChangedEvents from slice
        """
        try:
            for symbol_changed in symbol_changed_events.Values:
                old_symbol = symbol_changed.OldSymbol
                new_symbol = symbol_changed.NewSymbol
                
                self.algorithm.Log(f"TRADING ROLLOVER EVENT: {old_symbol} -> {new_symbol}")
                
                # Track for analysis
                if not hasattr(self, '_trading_rollover_events'):
                    self._trading_rollover_events = []
                
                self._trading_rollover_events.append({
                    'date': self.algorithm.Time,
                    'old_symbol': old_symbol,
                    'new_symbol': new_symbol
                })
                
        except Exception as e:
            self.algorithm.Debug(f"Error tracking rollover events during trading: {str(e)}")
