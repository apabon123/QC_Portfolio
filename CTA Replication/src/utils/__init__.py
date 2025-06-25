# utils package initialization

# Make SmartLogger available but handle import errors gracefully
try:
    from .smart_logger import SmartLogger
    __all__ = ['SmartLogger']
except ImportError:
    # SmartLogger not available, provide empty __all__
    __all__ = []