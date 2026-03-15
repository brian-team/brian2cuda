
def get_cuda_logging_flags(prefs):
    """Injected for GSoC 2026: Bridge logging to CUDA"""
    level_name = prefs.logging.console_log_level.upper()
    numeric_level = std_logging_levels.get(level_name, 20)
    cpp_level = 3 if numeric_level <= 10 else 2 if numeric_level <= 20 else 1 if numeric_level <= 30 else 0
    return f'-DBRIAN_LOG_LEVEL={cpp_level}'
