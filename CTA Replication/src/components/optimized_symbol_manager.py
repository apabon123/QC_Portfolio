"""
Optimized Symbol Manager for QuantConnect Three-Layer CTA Framework

This module implements centralized symbol management that:
1. Ensures single subscription per symbol across ALL strategies
2. Leverages QuantConnect's native data caching and sharing
3. Provides efficient symbol distribution to multiple strategies
4. Handles configuration-driven symbol requirements

Key Benefits:
- No duplicate subscriptions (cost optimization)
- Automatic QC data sharing and caching
- Scalable architecture for adding new strategies
- Configuration-driven symbol management
"""

from AlgorithmImports import *
from typing import Dict, List, Set, Optional, Tuple
import sys
import os

class OptimizedSymbolManager:
    """
    Centralized symbol management leveraging QC's native data sharing.
    
    Ensures each symbol (like ES futures) is subscribed to only once,
    then efficiently distributes that cached data to all strategies.
    """
    
    def __init__(self, algorithm, config_manager):
        """
        Initialize the optimized symbol manager.
        
        Args:
            algorithm: QCAlgorithm instance
            config_manager: AlgorithmConfigManager instance
        """
        self.algorithm = algorithm
        self.config_manager = config_manager
        
        # Symbol tracking
        self.shared_symbols = {}                    # {ticker_str: Symbol} - QC Symbol objects
        self.symbol_subscribers = {}                # {ticker_str: [strategy_names]} - Who uses what
        self.symbol_metadata = {}                   # {ticker_str: metadata_dict} - Symbol info
        self.subscription_stats = {}                # {ticker_str: subscription_info} - Performance tracking
        
        # Strategy requirements
        self.strategy_requirements = {}             # {strategy_name: [required_tickers]}
        self.enabled_strategies = {}                # {strategy_name: strategy_config}
        
        # Performance tracking
        self.total_subscriptions_created = 0
        self.total_subscriptions_reused = 0
        self.initialization_time = None
        
        self.algorithm.Log("OptimizedSymbolManager: Initialized for centralized symbol management")
    
    def setup_shared_subscriptions(self) -> Dict[str, object]:
        """
        Setup shared symbol subscriptions for ALL enabled strategies.
        
        This is the core optimization - each symbol is subscribed to only once,
        regardless of how many strategies need it.
        
        Returns:
            Dict[str, object]: Dictionary of ticker -> QC Symbol mappings
        """
        try:
            self.initialization_time = self.algorithm.Time
            
            # Step 1: Analyze strategy requirements
            self._analyze_strategy_requirements()
            
            # Step 2: Get unique symbols across all strategies
            all_required_symbols = self._get_unique_required_symbols()
            
            if not all_required_symbols:
                self.algorithm.Log("OptimizedSymbolManager: No symbols required by enabled strategies")
                return {}
            
            # Step 3: Create single subscription per unique symbol
            self._create_optimized_subscriptions(all_required_symbols)
            
            # Step 4: Log optimization results
            self._log_optimization_results()
            
            return self.shared_symbols.copy()
            
        except Exception as e:
            self.algorithm.Error(f"OptimizedSymbolManager: Failed to setup subscriptions: {str(e)}")
            raise
    
    def _analyze_strategy_requirements(self):
        """Analyze what symbols each enabled strategy requires."""
        try:
            self.enabled_strategies = self.config_manager.get_enabled_strategies()
            
            for strategy_name, strategy_config in self.enabled_strategies.items():
                # Get symbols this strategy is allowed to trade
                universe_config = self.config_manager.get_universe_config()
                all_available_symbols = self._get_all_available_symbols(universe_config)
                
                # Apply strategy-specific filtering
                allowed_symbols = self._get_strategy_allowed_symbols(strategy_name, all_available_symbols)
                
                # Store requirements
                self.strategy_requirements[strategy_name] = allowed_symbols
                
                self.algorithm.Log(f"OptimizedSymbolManager: {strategy_name} requires {len(allowed_symbols)} symbols: {allowed_symbols}")
                
        except Exception as e:
            self.algorithm.Error(f"Error analyzing strategy requirements: {str(e)}")
            raise
    
    def _get_all_available_symbols(self, universe_config) -> List[str]:
        """Get all available symbols from universe configuration."""
        all_symbols = []
        
        try:
            # Get symbols from futures configuration
            futures_config = universe_config.get('futures', {})
            
            for category_name, category_symbols in futures_config.items():
                if isinstance(category_symbols, dict):
                    all_symbols.extend(category_symbols.keys())
            
            # Get symbols from expansion candidates
            expansion_candidates = universe_config.get('expansion_candidates', {})
            all_symbols.extend(expansion_candidates.keys())
            
            # Remove duplicates and sort
            unique_symbols = sorted(list(set(all_symbols)))
            
            self.algorithm.Log(f"OptimizedSymbolManager: Found {len(unique_symbols)} total available symbols")
            return unique_symbols
            
        except Exception as e:
            self.algorithm.Error(f"Error getting available symbols: {str(e)}")
            return []
    
    def _get_strategy_allowed_symbols(self, strategy_name: str, all_symbols: List[str]) -> List[str]:
        """Get symbols allowed for a specific strategy based on filtering rules."""
        try:
            # Use existing filtering logic from config_market_strategy.py
            from config_market_strategy import get_strategy_allowed_symbols
            return get_strategy_allowed_symbols(strategy_name, all_symbols)
            
        except Exception as e:
            self.algorithm.Error(f"Error filtering symbols for {strategy_name}: {str(e)}")
            return all_symbols  # Fallback to all symbols if filtering fails
    
    def _get_unique_required_symbols(self) -> Set[str]:
        """Get unique set of symbols required across ALL enabled strategies."""
        all_required = set()
        
        for strategy_name, required_symbols in self.strategy_requirements.items():
            all_required.update(required_symbols)
            
            # Track which strategies use which symbols
            for symbol in required_symbols:
                if symbol not in self.symbol_subscribers:
                    self.symbol_subscribers[symbol] = []
                self.symbol_subscribers[symbol].append(strategy_name)
        
        self.algorithm.Log(f"OptimizedSymbolManager: {len(all_required)} unique symbols needed across {len(self.enabled_strategies)} strategies")
        
        # Log sharing statistics
        for symbol, subscribers in self.symbol_subscribers.items():
            if len(subscribers) > 1:
                self.algorithm.Log(f"  SHARED: {symbol} -> {subscribers} ({len(subscribers)} strategies)")
        
        return all_required
    
    def _create_optimized_subscriptions(self, required_symbols: Set[str]):
        """Create optimized subscriptions - one per unique symbol."""
        algo_config = self.config_manager.get_algorithm_config()
        resolution = getattr(Resolution, algo_config.get('resolution', 'Daily'))
        
        for ticker in required_symbols:
            try:
                # Check if already subscribed (shouldn't happen, but defensive)
                if ticker in self.shared_symbols:
                    self.total_subscriptions_reused += 1
                    self.algorithm.Log(f"OptimizedSymbolManager: {ticker} already subscribed - reusing")
                    continue
                
                # Create single subscription using QC's AddFuture
                future = self.algorithm.AddFuture(
                    ticker=ticker,
                    resolution=resolution,
                    dataMappingMode=DataMappingMode.OpenInterest,  # Use open interest for rollover
                    dataNormalizationMode=DataNormalizationMode.BackwardsRatio,  # Backwards ratio for CTA
                    contractDepthOffset=0  # Front month contract
                )
                
                if future and future.Symbol:
                    # Store the continuous symbol
                    self.shared_symbols[ticker] = future.Symbol
                    
                    # Store metadata
                    self.symbol_metadata[ticker] = {
                        'symbol': future.Symbol,
                        'ticker': ticker,
                        'resolution': resolution,
                        'subscribers': self.symbol_subscribers.get(ticker, []),
                        'subscription_time': self.algorithm.Time,
                        'data_mapping_mode': DataMappingMode.OpenInterest,
                        'normalization_mode': DataNormalizationMode.BackwardsRatio
                    }
                    
                    # Track subscription stats
                    self.subscription_stats[ticker] = {
                        'created_at': self.algorithm.Time,
                        'subscriber_count': len(self.symbol_subscribers.get(ticker, [])),
                        'data_requests': 0,
                        'last_access': None
                    }
                    
                    self.total_subscriptions_created += 1
                    
                    subscribers = self.symbol_subscribers.get(ticker, [])
                    self.algorithm.Log(f"SUBSCRIBED: {ticker} -> {future.Symbol} (serves {len(subscribers)} strategies: {subscribers})")
                    
                else:
                    self.algorithm.Error(f"Failed to subscribe to {ticker} - AddFuture returned None")
                    
            except Exception as e:
                self.algorithm.Error(f"Error subscribing to {ticker}: {str(e)}")
                continue
    
    def _log_optimization_results(self):
        """Log the results of symbol optimization."""
        total_strategies = len(self.enabled_strategies)
        total_unique_symbols = len(self.shared_symbols)
        total_strategy_symbol_pairs = sum(len(symbols) for symbols in self.strategy_requirements.values())
        
        # Calculate efficiency metrics
        if total_strategy_symbol_pairs > 0:
            efficiency_ratio = total_unique_symbols / total_strategy_symbol_pairs
            savings_ratio = 1.0 - efficiency_ratio
        else:
            efficiency_ratio = 1.0
            savings_ratio = 0.0
        
        self.algorithm.Log("=" * 80)
        self.algorithm.Log("OPTIMIZED SYMBOL MANAGEMENT RESULTS")
        self.algorithm.Log("=" * 80)
        self.algorithm.Log(f"Enabled Strategies: {total_strategies}")
        self.algorithm.Log(f"Unique Symbols Subscribed: {total_unique_symbols}")
        self.algorithm.Log(f"Total Strategy-Symbol Pairs: {total_strategy_symbol_pairs}")
        self.algorithm.Log(f"Efficiency Ratio: {efficiency_ratio:.2%} (lower is better)")
        self.algorithm.Log(f"Subscription Savings: {savings_ratio:.2%}")
        self.algorithm.Log(f"New Subscriptions Created: {self.total_subscriptions_created}")
        self.algorithm.Log(f"Subscriptions Reused: {self.total_subscriptions_reused}")
        
        # Log sharing details
        shared_symbols = {k: v for k, v in self.symbol_subscribers.items() if len(v) > 1}
        if shared_symbols:
            self.algorithm.Log(f"SHARED SYMBOLS ({len(shared_symbols)} symbols serving multiple strategies):")
            for symbol, subscribers in shared_symbols.items():
                self.algorithm.Log(f"  {symbol}: {subscribers} ({len(subscribers)} strategies)")
        
        self.algorithm.Log("=" * 80)
    
    def get_shared_symbols(self) -> Dict[str, object]:
        """Get the shared symbol dictionary for use by strategies."""
        return self.shared_symbols.copy()
    
    def get_symbols_for_strategy(self, strategy_name: str) -> Dict[str, object]:
        """Get symbols that a specific strategy should use."""
        if strategy_name not in self.strategy_requirements:
            return {}
        
        strategy_symbols = {}
        required_tickers = self.strategy_requirements[strategy_name]
        
        for ticker in required_tickers:
            if ticker in self.shared_symbols:
                strategy_symbols[ticker] = self.shared_symbols[ticker]
                
                # Track access for performance monitoring
                if ticker in self.subscription_stats:
                    self.subscription_stats[ticker]['data_requests'] += 1
                    self.subscription_stats[ticker]['last_access'] = self.algorithm.Time
        
        return strategy_symbols
    
    def get_optimization_stats(self) -> Dict:
        """Get optimization performance statistics."""
        return {
            'total_strategies': len(self.enabled_strategies),
            'total_unique_symbols': len(self.shared_symbols),
            'total_subscriptions_created': self.total_subscriptions_created,
            'total_subscriptions_reused': self.total_subscriptions_reused,
            'symbol_subscribers': dict(self.symbol_subscribers),
            'subscription_stats': dict(self.subscription_stats),
            'initialization_time': self.initialization_time
        }
    
    def validate_symbol_access(self, strategy_name: str, ticker: str) -> bool:
        """Validate that a strategy has access to a specific symbol."""
        if strategy_name not in self.strategy_requirements:
            return False
        
        return ticker in self.strategy_requirements[strategy_name]
    
    def get_symbol_metadata(self, ticker: str) -> Optional[Dict]:
        """Get metadata for a specific symbol."""
        return self.symbol_metadata.get(ticker)
    
    def log_symbol_usage_report(self):
        """Log a detailed symbol usage report for performance monitoring."""
        if not self.subscription_stats:
            return
        
        self.algorithm.Log("=" * 60)
        self.algorithm.Log("SYMBOL USAGE REPORT")
        self.algorithm.Log("=" * 60)
        
        for ticker, stats in self.subscription_stats.items():
            subscribers = self.symbol_subscribers.get(ticker, [])
            self.algorithm.Log(f"{ticker}: {stats['subscriber_count']} strategies, "
                             f"{stats['data_requests']} requests, "
                             f"last access: {stats['last_access']}")
        
        self.algorithm.Log("=" * 60) 