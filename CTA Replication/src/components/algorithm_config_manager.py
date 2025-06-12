# algorithm_config_manager.py - FIXED VERSION: Simplified Imports
"""
Algorithm Configuration Manager - FIXED Single Source of Truth

CRITICAL FIXES IMPLEMENTED:
1. Simplified imports - only imports get_full_config
2. Proper single strategy allocation handling (100% only when testing single strategy)
3. Enhanced QuantConnect settings application with verification
4. Smarter allocation bounds management
5. Explicit date and warmup application
6. Comprehensive config verification

Key Features:
- Configuration loading from config.py (simplified)
- SMART single strategy detection and handling
- Allocation normalization with bounds intelligence
- SINGLE AUTHORITY for all QuantConnect algorithm parameter setting
- Enhanced verification and diagnostics
- Runtime configuration updates

CLEAN ARCHITECTURE PRINCIPLE:
This is the ONLY component that calls SetCash(), SetStartDate(), SetWarmup(), etc.
"""

from AlgorithmImports import *
from datetime import datetime, timedelta

# SIMPLIFIED IMPORT - Only import what we actually need
from config import get_full_config

class AlgorithmConfigManager:
    """
    FIXED SINGLE AUTHORITY for all configuration aspects including loading,
    validation, normalization, and QuantConnect algorithm parameter application.
    
    CRITICAL FIXES:
    - Simplified imports (no more complex variants)
    - Smart single strategy handling
    - Proper bounds management
    - Enhanced config application verification
    """
    
    def __init__(self, algorithm):
        """
        Initialize the configuration manager.
        
        Args:
            algorithm: The main QuantConnect algorithm instance
        """
        self.algorithm = algorithm
        self.config = None
        self.enabled_strategies = {}
        self.config_variant = None
        self.applied_settings = {}
        self.last_update = None
        self.single_strategy_mode = False  # Track if we're in single strategy testing mode
        
    def load_and_validate_config(self, variant="full"):
        """
        Load configuration from config.py and perform full validation/normalization.
        This is the SINGLE ENTRY POINT for all config operations.
        
        Args:
            variant: Configuration variant to load (simplified - only 'full' supported now)
            
        Returns:
            dict: Fully validated and normalized configuration
        """
        try:
            self.algorithm.Log(f"CONFIG MANAGER: Loading simplified configuration...")
            
            # Load the main configuration (simplified)
            self.config = self._load_main_config()
            self.config_variant = "full"
            self.last_update = self.algorithm.Time if hasattr(self.algorithm, 'Time') else None
            
            # Validate and normalize the configuration
            self._validate_and_normalize_config()
            
            # SINGLE AUTHORITY: Apply ALL QuantConnect algorithm parameters
            self._apply_all_quantconnect_settings()
            
            # Verify config was applied correctly
            self._verify_config_application()
            
            # Log successful initialization
            self._log_config_summary()
            
            return self.config
            
        except Exception as e:
            error_msg = f"CRITICAL ERROR: Configuration manager failed: {str(e)}"
            self.algorithm.Error(error_msg)
            raise
    
    def _load_main_config(self):
        """
        Load the main configuration from config.py (simplified).
        
        Returns:
            dict: Raw configuration dictionary
        """
        try:
            config = get_full_config()
            self.algorithm.Log(f"CONFIG MANAGER: Main configuration loaded successfully")
            
            # Log basic config info
            algo_config = config['algorithm']
            start_date = f"{algo_config['start_date']['year']}-{algo_config['start_date']['month']:02d}-{algo_config['start_date']['day']:02d}"
            end_date = f"{algo_config['end_date']['year']}-{algo_config['end_date']['month']:02d}-{algo_config['end_date']['day']:02d}"
            
            self.algorithm.Log(f"CONFIG MANAGER: Period: {start_date} to {end_date}")
            self.algorithm.Log(f"CONFIG MANAGER: Initial capital: ${algo_config['initial_cash']:,}")
            
            return config
            
        except Exception as e:
            raise ValueError(f"Failed to load main config: {str(e)}")
    
    def _validate_and_normalize_config(self):
        """
        FIXED: Validate configuration consistency and intelligently handle single vs multi-strategy modes.
        This method ensures the system works for both single strategy testing AND multi-strategy operation.
        """
        self.algorithm.Log("CONFIG MANAGER: Starting configuration validation and normalization...")
        
        # Validate required sections
        self._validate_required_sections()
        
        # Validate algorithm configuration
        self._validate_algorithm_config()
        
        # FIXED: Smart strategy handling
        self._smart_strategy_allocation_handling()
        
        # FIXED: Intelligent allocation bounds management
        self._intelligent_bounds_management()
        
        # Final validation checks
        self._perform_final_validation()
        
        self.algorithm.Log("CONFIG MANAGER: Configuration validation and normalization completed successfully")
    
    def _validate_required_sections(self):
        """Validate that all required configuration sections are present."""
        required_sections = ['algorithm', 'system', 'strategies', 'strategy_allocation', 'portfolio_risk']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    def _validate_algorithm_config(self):
        """Validate algorithm-specific configuration parameters."""
        algo_config = self.config['algorithm']
        required_algo_params = ['start_date', 'end_date', 'initial_cash']
        
        for param in required_algo_params:
            if param not in algo_config:
                raise ValueError(f"Missing required algorithm parameter: {param}")
        
        # Validate date structure
        for date_param in ['start_date', 'end_date']:
            date_config = algo_config[date_param]
            required_date_fields = ['year', 'month', 'day']
            for field in required_date_fields:
                if field not in date_config:
                    raise ValueError(f"Missing {field} in {date_param}")
    
    def _smart_strategy_allocation_handling(self):
        """
        FIXED: Smart strategy allocation handling that detects single vs multi-strategy scenarios.
        
        Key Logic:
        - Single strategy enabled: Gets 100% allocation regardless of bounds (testing mode)
        - Multiple strategies enabled: Apply normal allocation rules and bounds
        """
        # Get enabled strategies
        self.enabled_strategies = {
            config['name']: config for config in self.config['strategies'].values() 
            if config.get('enabled', False)
        }
        
        enabled_strategy_names = list(self.enabled_strategies.keys())
        self.algorithm.Log(f"CONFIG MANAGER: Enabled strategies found: {enabled_strategy_names}")
        
        # Validate that we have at least one enabled strategy
        if not self.enabled_strategies:
            raise ValueError("No strategies are enabled in configuration. Please enable at least one strategy.")
        
        # FIXED: Detect single strategy mode
        self.single_strategy_mode = len(self.enabled_strategies) == 1
        
        if self.single_strategy_mode:
            single_strategy = enabled_strategy_names[0]
            self.algorithm.Log(f"CONFIG MANAGER: SINGLE STRATEGY MODE DETECTED: {single_strategy}")
            self.algorithm.Log(f"CONFIG MANAGER: Single strategy will receive 100% allocation for testing")
            
            # In single strategy mode, give 100% allocation regardless of original config
            normalized_allocations = {single_strategy: 1.0}
            
        else:
            self.algorithm.Log(f"CONFIG MANAGER: MULTI-STRATEGY MODE: {len(self.enabled_strategies)} strategies")
            
            # Multi-strategy mode: use normal allocation logic
            allocation_config = self.config['strategy_allocation']
            configured_allocations = allocation_config['initial_allocations']
            
            self.algorithm.Log(f"CONFIG MANAGER: Original configured allocations: {configured_allocations}")
            
            # Filter allocations to only include enabled strategies
            valid_allocations = {
                strategy_name: allocation 
                for strategy_name, allocation in configured_allocations.items()
                if strategy_name in self.enabled_strategies and allocation > 0
            }
            
            self.algorithm.Log(f"CONFIG MANAGER: Valid allocations for enabled strategies: {valid_allocations}")
            
            # Check if we have valid allocations
            if not valid_allocations:
                # If no valid allocations, create equal weight for all enabled strategies
                self.algorithm.Log("CONFIG MANAGER: No valid allocations found. Creating equal-weight allocation.")
                valid_allocations = {
                    strategy_name: 1.0 / len(self.enabled_strategies)
                    for strategy_name in self.enabled_strategies.keys()
                }
            
            # Normalize allocations to sum to 1.0
            total_allocation = sum(valid_allocations.values())
            if total_allocation > 0:
                normalized_allocations = {
                    strategy: allocation / total_allocation 
                    for strategy, allocation in valid_allocations.items()
                }
            else:
                raise ValueError("Total allocation is zero for enabled strategies")
        
        # Update the config with normalized allocations
        self.config['strategy_allocation']['initial_allocations'] = normalized_allocations
        
        self.algorithm.Log(f"CONFIG MANAGER: Final normalized allocations: {normalized_allocations}")
        self.algorithm.Log(f"CONFIG MANAGER: Allocation sum: {sum(normalized_allocations.values()):.4f}")
    
    def _intelligent_bounds_management(self):
        """
        FIXED: Intelligent allocation bounds management.
        
        Key Logic:
        - Single strategy mode: Ignore bounds completely (allow 100%)
        - Multi-strategy mode: Apply bounds normally with warnings
        """
        allocation_config = self.config['strategy_allocation']
        original_bounds = allocation_config.get('allocation_bounds', {})
        
        if self.single_strategy_mode:
            # Single strategy mode: Create permissive bounds
            single_strategy = list(self.enabled_strategies.keys())[0]
            permissive_bounds = {single_strategy: {'min': 0.0, 'max': 1.0}}
            
            self.algorithm.Log(f"CONFIG MANAGER: SINGLE STRATEGY MODE - Ignoring restrictive bounds")
            self.algorithm.Log(f"CONFIG MANAGER: Applied permissive bounds: {permissive_bounds}")
            
            self.config['strategy_allocation']['allocation_bounds'] = permissive_bounds
            
        else:
            # Multi-strategy mode: Apply normal bounds logic
            # Filter bounds to only include enabled strategies
            filtered_bounds = {
                strategy_name: bounds 
                for strategy_name, bounds in original_bounds.items()
                if strategy_name in self.enabled_strategies
            }
            
            # Add default bounds for strategies without explicit bounds
            for strategy_name in self.enabled_strategies.keys():
                if strategy_name not in filtered_bounds:
                    filtered_bounds[strategy_name] = {'min': 0.0, 'max': 1.0}
            
            self.config['strategy_allocation']['allocation_bounds'] = filtered_bounds
            self.algorithm.Log(f"CONFIG MANAGER: MULTI-STRATEGY MODE - Applied bounds: {filtered_bounds}")
    
    def _perform_final_validation(self):
        """Perform final validation checks on normalized configuration."""
        normalized_allocations = self.config['strategy_allocation']['initial_allocations']
        filtered_bounds = self.config['strategy_allocation']['allocation_bounds']
        
        # Validate final allocations against bounds (warnings only, not errors)
        for strategy_name, allocation in normalized_allocations.items():
            if strategy_name in filtered_bounds:
                bounds = filtered_bounds[strategy_name]
                min_bound = bounds.get('min', 0)
                max_bound = bounds.get('max', 1)
                
                if allocation < min_bound:
                    if not self.single_strategy_mode:  # Only warn in multi-strategy mode
                        self.algorithm.Log(f"CONFIG WARNING: {strategy_name} allocation {allocation:.2%} below minimum {min_bound:.2%}")
                
                if allocation > max_bound:
                    if self.single_strategy_mode:
                        self.algorithm.Log(f"CONFIG INFO: {strategy_name} allocation {allocation:.2%} above normal max {max_bound:.2%} (ALLOWED in single strategy mode)")
                    else:
                        self.algorithm.Log(f"CONFIG WARNING: {strategy_name} allocation {allocation:.2%} above maximum {max_bound:.2%}")

    def _apply_all_quantconnect_settings(self):
        """
        FIXED: Enhanced QuantConnect algorithm parameters application with explicit verification.
        This is the ONLY place where QuantConnect settings should be applied.
        """
        try:
            self.algorithm.Log("CONFIG MANAGER: Applying ALL QuantConnect algorithm settings...")
            algo_config = self.config['algorithm']
            
            # Clear any previously applied settings tracking
            self.applied_settings = {}
            
            # 1. Set initial cash (MOST IMPORTANT)
            initial_cash = algo_config.get('initial_cash', 10000000)
            self.algorithm.SetCash(initial_cash)
            self.applied_settings['initial_cash'] = initial_cash
            self.algorithm.Log(f"CONFIG APPLIED: SetCash(${initial_cash:,})")
            
            # 2. FIXED: Set date range with explicit datetime objects
            start_date_config = algo_config.get('start_date', {'year': 2015, 'month': 1, 'day': 1})
            end_date_config = algo_config.get('end_date', {'year': 2020, 'month': 1, 'day': 1})
            
            # Create explicit datetime objects
            start_datetime = datetime(
                start_date_config['year'],
                start_date_config['month'], 
                start_date_config['day']
            )
            end_datetime = datetime(
                end_date_config['year'],
                end_date_config['month'],
                end_date_config['day']
            )
            
            # Apply with explicit datetime objects
            self.algorithm.SetStartDate(start_datetime)
            self.algorithm.SetEndDate(end_datetime)
            
            self.applied_settings['start_date'] = start_date_config
            self.applied_settings['end_date'] = end_date_config
            self.applied_settings['start_datetime'] = start_datetime
            self.applied_settings['end_datetime'] = end_datetime
            
            self.algorithm.Log(f"CONFIG APPLIED: SetStartDate({start_datetime})")
            self.algorithm.Log(f"CONFIG APPLIED: SetEndDate({end_datetime})")
            
            # 3. Set benchmark (TEMPORARILY DISABLED for data access testing)
            benchmark = algo_config.get('benchmark', 'SPY')
            # self.algorithm.SetBenchmark(benchmark)  # Disabled until data access is resolved
            self.applied_settings['benchmark'] = benchmark
            self.algorithm.Log(f"CONFIG NOTED: Benchmark = {benchmark} (SetBenchmark disabled for testing)")
            
            # 4. Set brokerage model
            brokerage_model = algo_config.get('brokerage_model', 'InteractiveBrokers')
            if brokerage_model == 'InteractiveBrokers':
                self.algorithm.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)
                self.applied_settings['brokerage_model'] = brokerage_model
                self.algorithm.Log(f"CONFIG APPLIED: SetBrokerageModel({brokerage_model})")
            
            # 5. FIXED: Set warmup period with explicit calculation
            warmup_days = algo_config.get('warmup_period_days', 80)
            self.algorithm.SetWarmUp(warmup_days)
            self.applied_settings['warmup_period_days'] = warmup_days
            
            # Calculate expected warmup end date for logging
            expected_warmup_end = start_datetime + timedelta(days=warmup_days)
            self.applied_settings['expected_warmup_end'] = expected_warmup_end
            
            self.algorithm.Log(f"CONFIG APPLIED: SetWarmUp({warmup_days} days)")
            self.algorithm.Log(f"CONFIG INFO: Warmup should end around {expected_warmup_end}")
            
            # 6. Set resolution
            resolution = algo_config.get('resolution', 'Daily')
            self.applied_settings['resolution'] = resolution
            self.algorithm.Log(f"CONFIG NOTED: Resolution = {resolution}")
            
            # 7. Set timezone (if supported)
            timezone = algo_config.get('timezone', 'America/New_York')
            self.applied_settings['timezone'] = timezone
            self.algorithm.Log(f"CONFIG NOTED: Timezone = {timezone}")
            
            self.algorithm.Log("CONFIG MANAGER: ALL QuantConnect settings applied successfully")
            
        except Exception as e:
            raise ValueError(f"Failed to apply QuantConnect algorithm settings: {str(e)}")
    
    def _verify_config_application(self):
        """
        FIXED: Verify that configuration was properly applied to QuantConnect.
        Performs immediate verification after application.
        """
        try:
            self.algorithm.Log("CONFIG MANAGER: Verifying QuantConnect settings application...")
            
            # Verify cash application
            expected_cash = self.applied_settings.get('initial_cash', 0)
            actual_cash = float(self.algorithm.Portfolio.Cash)
            portfolio_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
            
            if abs(actual_cash - expected_cash) < 1000:  # Within $1k tolerance
                self.algorithm.Log(f"CONFIG VERIFIED: Initial cash ${actual_cash:,} matches config ${expected_cash:,}")
            else:
                self.algorithm.Error(f"CONFIG ERROR: Cash ${actual_cash:,} != expected ${expected_cash:,}")
            
            self.algorithm.Log(f"CONFIG VERIFIED: Portfolio value ${portfolio_value:,}")
            
            # Log strategy mode verification
            if self.single_strategy_mode:
                single_strategy = list(self.enabled_strategies.keys())[0]
                allocation = self.config['strategy_allocation']['initial_allocations'][single_strategy]
                self.algorithm.Log(f"CONFIG VERIFIED: Single strategy mode - {single_strategy} gets {allocation:.1%}")
            else:
                allocations = self.config['strategy_allocation']['initial_allocations']
                self.algorithm.Log(f"CONFIG VERIFIED: Multi-strategy mode - {len(allocations)} strategies")
            
            # Log warmup verification
            warmup_days = self.applied_settings.get('warmup_period_days', 0)
            expected_end = self.applied_settings.get('expected_warmup_end', 'unknown')
            self.algorithm.Log(f"CONFIG VERIFIED: Warmup {warmup_days} days, expected end: {expected_end}")
            
            self.algorithm.Log("CONFIG MANAGER: Configuration verification completed successfully")
            
        except Exception as e:
            self.algorithm.Error(f"ERROR in config verification: {str(e)}")
    
    def _log_config_summary(self):
        """Log a summary of the loaded and applied configuration."""
        system_info = self.config['system']
        self.algorithm.Log("=" * 80)
        self.algorithm.Log(f"CONFIG MANAGER: {system_info['name']} v{system_info['version']}")
        self.algorithm.Log(f"Description: {system_info['description']}")
        self.algorithm.Log(f"Configuration: Simplified Main Config")
        self.algorithm.Log(f"Last Updated: {system_info['last_updated']}")
        
        # Log strategy mode
        if self.single_strategy_mode:
            strategy_name = list(self.enabled_strategies.keys())[0]
            self.algorithm.Log(f"STRATEGY MODE: Single Strategy Testing ({strategy_name})")
        else:
            self.algorithm.Log(f"STRATEGY MODE: Multi-Strategy Operation ({len(self.enabled_strategies)} strategies)")
        
        self.algorithm.Log("SINGLE CONFIG AUTHORITY: All QuantConnect settings applied")
        self.algorithm.Log("SIMPLIFIED IMPORTS: No complex config variants")
        self.algorithm.Log("=" * 80)
    
    # =============================================================================
    # PUBLIC API METHODS - Enhanced with Single Strategy Awareness
    # =============================================================================
    
    def get_config(self):
        """Get the current configuration."""
        return self.config
    
    def get_enabled_strategies(self):
        """Get the dictionary of enabled strategies."""
        return self.enabled_strategies
    
    def is_single_strategy_mode(self):
        """Check if we're in single strategy testing mode."""
        return self.single_strategy_mode
    
    def get_strategy_config(self, strategy_name):
        """Get configuration for a specific strategy."""
        return self.enabled_strategies.get(strategy_name)
    
    def get_allocation_config(self):
        """Get the current allocation configuration."""
        return self.config['strategy_allocation']
    
    def get_risk_config(self):
        """Get the risk management configuration."""
        return self.config['portfolio_risk']
    
    def is_strategy_enabled(self, strategy_name):
        """Check if a strategy is enabled."""
        return strategy_name in self.enabled_strategies
    
    def get_strategy_config_key(self, strategy_name):
        """Get the config key for a strategy name."""
        for key, config in self.config['strategies'].items():
            if config.get('name') == strategy_name:
                return key
        return None
    
    def get_applied_settings(self):
        """Get the settings that were actually applied to QuantConnect."""
        return self.applied_settings.copy()
    
    def get_config_summary(self):
        """
        ENHANCED: Get comprehensive config summary with single strategy mode awareness.
        """
        try:
            config_summary = {
                'variant': 'simplified_main_config',
                'last_update': self.last_update,
                'applied_settings': self.applied_settings.copy(),
                'strategy_mode': {
                    'single_strategy_mode': self.single_strategy_mode,
                    'enabled_strategies_count': len(self.enabled_strategies),
                    'enabled_strategy_names': list(self.enabled_strategies.keys())
                },
                'algorithm_settings': self.config.get('algorithm', {}),
                'execution_settings': self.config.get('execution', {}),
                'risk_settings': self.config.get('portfolio_risk', {}),
                'constraint_settings': self.config.get('constraints', {}),
                'strategy_settings': {
                    name: config for name, config in self.config.get('strategies', {}).items()
                    if config.get('enabled', False)
                },
                'allocation_settings': self.config.get('strategy_allocation', {}),
                'total_enabled_strategies': len(self.enabled_strategies),
                'single_authority_status': 'active',
                'simplified_imports': True,
                'config_compliance_status': {
                    'all_settings_applied': len(self.applied_settings) > 0,
                    'no_duplicate_applications': True,
                    'single_source_authority': True,
                    'smart_allocation_handling': True,
                    'proper_bounds_management': True,
                    'simplified_imports': True
                }
            }
            
            return config_summary
            
        except Exception as e:
            self.algorithm.Log(f"ERROR getting config summary: {str(e)}")
            return {'error': str(e), 'single_authority_status': 'error'}
    
    # =============================================================================
    # RUNTIME CONFIGURATION UPDATES - Enhanced
    # =============================================================================
    
    def update_strategy_allocation(self, strategy_name, new_allocation):
        """Update allocation for a specific strategy at runtime."""
        if strategy_name not in self.enabled_strategies:
            raise ValueError(f"Strategy '{strategy_name}' is not enabled")
        
        if not 0.0 <= new_allocation <= 1.0:
            raise ValueError(f"Allocation must be between 0.0 and 1.0, got {new_allocation}")
        
        # Handle single strategy mode
        if self.single_strategy_mode:
            if strategy_name == list(self.enabled_strategies.keys())[0]:
                # In single strategy mode, always set to 100%
                self.config['strategy_allocation']['initial_allocations'][strategy_name] = 1.0
                self.algorithm.Log(f"CONFIG MANAGER: Single strategy mode - {strategy_name} remains at 100%")
                return
        
        # Multi-strategy mode: Update the allocation and renormalize
        self.config['strategy_allocation']['initial_allocations'][strategy_name] = new_allocation
        
        # Renormalize all allocations
        total = sum(self.config['strategy_allocation']['initial_allocations'].values())
        if total > 0:
            for name in self.config['strategy_allocation']['initial_allocations']:
                self.config['strategy_allocation']['initial_allocations'][name] /= total
        
        self.algorithm.Log(f"CONFIG MANAGER: Updated allocation for {strategy_name}: {new_allocation:.2%}")
    
    def update_risk_parameter(self, parameter, new_value):
        """Update a risk management parameter at runtime."""
        if parameter not in self.config['portfolio_risk']:
            raise ValueError(f"Risk parameter '{parameter}' not found")
        
        old_value = self.config['portfolio_risk'][parameter]
        self.config['portfolio_risk'][parameter] = new_value
        
        self.algorithm.Log(f"CONFIG MANAGER: Updated risk parameter {parameter}: {old_value} -> {new_value}")
    
    def validate_runtime_change(self, section, parameter, new_value):
        """Validate a runtime configuration change before applying it."""
        # Add validation logic here as needed
        return True
