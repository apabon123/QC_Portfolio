# strategy_data_processor.py - UNIFIED DATA PROCESSING UTILITIES

from AlgorithmImports import *

class StrategyDataProcessor:
    """
    Utility class for processing unified data for strategies.
    Extracted from BaseStrategy to reduce file size.
    """
    
    def __init__(self, algorithm, strategy_name):
        self.algorithm = algorithm
        self.strategy_name = strategy_name
        self.unified_data_stats = {
            'total_updates': 0,
            'symbols_processed': 0,
            'data_efficiency': 0.0
        }
    
    def process_unified_data_for_strategy(self, symbols_data, unified_data):
        """Process unified data into strategy-friendly format."""
        try:
            processed_data = {
                'timestamp': unified_data.get('timestamp', self.algorithm.Time),
                'bars': {},
                'chains': {},
                'security_info': {},
                'liquid_symbols': []
            }
            
            # Process each symbol's data
            for symbol, symbol_data in symbols_data.items():
                if not symbol_data.get('valid', False):
                    continue
                
                data = symbol_data.get('data', {})
                
                # Extract bar data
                if 'bar' in data:
                    bar_data = data['bar']
                    processed_data['bars'][symbol] = {
                        'open': bar_data.get('open', 0),
                        'high': bar_data.get('high', 0),
                        'low': bar_data.get('low', 0),
                        'close': bar_data.get('close', 0),
                        'volume': bar_data.get('volume', 0),
                        'time': bar_data.get('time', self.algorithm.Time)
                    }
                
                # Extract chain data
                if 'chain' in data and data['chain'].get('valid', False):
                    processed_data['chains'][symbol] = data['chain']
                
                # Extract security information
                if 'security' in data:
                    security_data = data['security']
                    processed_data['security_info'][symbol] = {
                        'price': security_data.get('price', 0),
                        'has_data': security_data.get('has_data', False),
                        'is_tradable': security_data.get('is_tradable', False),
                        'mapped_symbol': security_data.get('mapped_symbol'),
                        'market_hours_open': security_data.get('market_hours', False)
                    }
                    
                    # Add to liquid symbols if tradable
                    if (security_data.get('is_tradable', False) and 
                        security_data.get('has_data', False)):
                        processed_data['liquid_symbols'].append(symbol)
            
            return processed_data
            
        except Exception as e:
            self.algorithm.Error(f"{self.strategy_name}: Error processing unified data: {str(e)}")
            return {'timestamp': self.algorithm.Time, 'bars': {}, 'chains': {}, 'security_info': {}, 'liquid_symbols': []}
    
    def update_indicators_with_unified_data(self, processed_data, indicators):
        """Update indicators using unified data."""
        try:
            bars_data = processed_data.get('bars', {})
            
            for symbol, bar_data in bars_data.items():
                if symbol in indicators:
                    indicator = indicators[symbol]
                    
                    # Create IndicatorDataPoint for QC indicators
                    try:
                        data_point = IndicatorDataPoint(
                            bar_data['time'],
                            bar_data['close']
                        )
                        
                        if hasattr(indicator, 'Update'):
                            indicator.Update(data_point)
                            
                    except Exception as e:
                        self.algorithm.Debug(f"{self.strategy_name}: Error updating indicator for {symbol}: {str(e)}")
                        
        except Exception as e:
            self.algorithm.Error(f"{self.strategy_name}: Error updating indicators with unified data: {str(e)}")
    
    def track_unified_data_usage(self, unified_data, processed_data):
        """Track unified data usage statistics for strategy performance."""
        try:
            # Update statistics
            self.unified_data_stats['total_updates'] += 1
            self.unified_data_stats['symbols_processed'] += len(processed_data.get('liquid_symbols', []))
            
            # Calculate running efficiency
            metadata = unified_data.get('metadata', {})
            current_efficiency = len(processed_data.get('liquid_symbols', [])) / max(metadata.get('total_symbols', 1), 1)
            
            updates = self.unified_data_stats['total_updates']
            self.unified_data_stats['data_efficiency'] = (
                (self.unified_data_stats['data_efficiency'] * (updates - 1) + current_efficiency) / updates
            )
            
        except Exception as e:
            self.algorithm.Debug(f"{self.strategy_name}: Error tracking unified data usage: {str(e)}")
    
    def get_unified_data_performance(self):
        """Get unified data performance statistics for this strategy."""
        try:
            return {
                'strategy_name': self.strategy_name,
                'total_unified_updates': self.unified_data_stats.get('total_updates', 0),
                'avg_symbols_processed': self.unified_data_stats.get('symbols_processed', 0) / max(self.unified_data_stats.get('total_updates', 1), 1),
                'data_efficiency': round(self.unified_data_stats.get('data_efficiency', 0.0) * 100, 1),
                'integration_status': 'active'
            }
        except Exception as e:
            self.algorithm.Error(f"{self.strategy_name}: Error getting unified data performance: {str(e)}")
            return {'strategy_name': self.strategy_name, 'integration_status': 'error'} 