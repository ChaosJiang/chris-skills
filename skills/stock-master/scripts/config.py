"""
Configuration settings for the Financial Report Analyzer.

All settings can be overridden via environment variables.
"""

import os
from pathlib import Path

# ============================================================================
# API Keys (Optional - some data sources may require these)
# ============================================================================

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")

# ============================================================================
# Logging Configuration
# ============================================================================

# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Enable file logging
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"

# Log directory (relative to project root)
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))

# ============================================================================
# Data Validation Configuration
# ============================================================================

# Balance sheet equation tolerance (percentage)
# Assets should equal Liabilities + Equity within this tolerance
BALANCE_SHEET_TOLERANCE = float(os.getenv("BALANCE_TOLERANCE", "0.01"))  # 1%

# Enable strict validation mode (fail on validation errors vs. just warn)
ENABLE_STRICT_VALIDATION = os.getenv("STRICT_VALIDATION", "false").lower() == "true"

# Quarterly data frequency tolerance (days)
# Expected quarterly interval is ~90 days ± this tolerance
QUARTERLY_FREQUENCY_TOLERANCE = int(os.getenv("QUARTERLY_TOLERANCE_DAYS", "10"))

# ============================================================================
# Field Matching Configuration
# ============================================================================

# Enable fuzzy field matching
ENABLE_FUZZY_MATCHING = os.getenv("FUZZY_MATCHING", "true").lower() == "true"

# Log fuzzy matches as warnings
LOG_FUZZY_MATCHES = os.getenv("LOG_FUZZY_MATCHES", "true").lower() == "true"

# Minimum confidence threshold for fuzzy matches (0.0 - 1.0)
FUZZY_MATCH_MIN_CONFIDENCE = float(os.getenv("FUZZY_MATCH_CONFIDENCE", "0.5"))

# ============================================================================
# Cache Configuration
# ============================================================================

# Default cache age in hours
DEFAULT_CACHE_AGE_HOURS = float(os.getenv("CACHE_AGE_HOURS", "24"))

# Output directory for all generated files
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))

# ============================================================================
# Currency Conversion Configuration
# ============================================================================

# Retry attempts for fetching exchange rates
CURRENCY_FETCH_RETRIES = int(os.getenv("CURRENCY_RETRIES", "3"))

# Delay between retries (seconds)
CURRENCY_FETCH_RETRY_DELAY = float(os.getenv("CURRENCY_RETRY_DELAY", "1.0"))

# ============================================================================
# Analysis Configuration
# ============================================================================

# Number of years of historical data to fetch by default
DEFAULT_YEARS = int(os.getenv("DEFAULT_YEARS", "1"))

# Number of years of price history to fetch
DEFAULT_PRICE_YEARS = int(os.getenv("DEFAULT_PRICE_YEARS", "6"))

# ============================================================================
# Valuation Configuration
# ============================================================================

# DCF model default parameters
DCF_DISCOUNT_RATE = float(os.getenv("DCF_DISCOUNT_RATE", "0.10"))  # 10%
DCF_GROWTH_RATE = float(os.getenv("DCF_GROWTH_RATE", "0.05"))  # 5%
DCF_TERMINAL_GROWTH = float(os.getenv("DCF_TERMINAL_GROWTH", "0.02"))  # 2%
DCF_FORECAST_YEARS = int(os.getenv("DCF_FORECAST_YEARS", "5"))

# ============================================================================
# Reporting Configuration
# ============================================================================

# Maximum number of validation warnings to show in report
MAX_VALIDATION_WARNINGS_IN_REPORT = int(os.getenv("MAX_VALIDATION_WARNINGS", "5"))

# Maximum number of fuzzy matches to show in report
MAX_FUZZY_MATCHES_IN_REPORT = int(os.getenv("MAX_FUZZY_MATCHES", "5"))

# ============================================================================
# Helper Functions
# ============================================================================


def get_validation_config() -> dict:
    """Get validation configuration as a dictionary."""
    return {
        "balance_sheet_tolerance": BALANCE_SHEET_TOLERANCE,
        "enable_strict_validation": ENABLE_STRICT_VALIDATION,
        "quarterly_frequency_tolerance": QUARTERLY_FREQUENCY_TOLERANCE,
    }


def get_field_matching_config() -> dict:
    """Get field matching configuration as a dictionary."""
    return {
        "enable_fuzzy_matching": ENABLE_FUZZY_MATCHING,
        "log_fuzzy_matches": LOG_FUZZY_MATCHES,
        "min_confidence": FUZZY_MATCH_MIN_CONFIDENCE,
    }


def get_logging_config() -> dict:
    """Get logging configuration as a dictionary."""
    return {
        "log_level": LOG_LEVEL,
        "log_to_file": LOG_TO_FILE,
        "log_dir": LOG_DIR,
    }


def get_dcf_config() -> dict:
    """Get DCF model configuration as a dictionary."""
    return {
        "discount_rate": DCF_DISCOUNT_RATE,
        "growth_rate": DCF_GROWTH_RATE,
        "terminal_growth": DCF_TERMINAL_GROWTH,
        "forecast_years": DCF_FORECAST_YEARS,
    }
