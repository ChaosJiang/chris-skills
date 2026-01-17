"""
Structured logging system for the Financial Report Analyzer.

Provides:
- Multi-level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Console and file output
- Data quality dedicated logger
- Contextual information in all logs
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        if sys.stdout.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = (
                    f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
                )
        return super().format(record)


class DataQualityLogger:
    """Specialized logger for tracking data quality issues."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.fuzzy_matches = []
        self.missing_fields = []
        self.validation_warnings = []

    def log_fuzzy_match(self, field: str, matched: str, confidence: float = 0.0):
        """Log when a fuzzy field match is used."""
        self.fuzzy_matches.append(
            {"field": field, "matched": matched, "confidence": confidence}
        )
        self.logger.warning(
            f"Fuzzy field match: '{field}' -> '{matched}' "
            f"(confidence: {confidence:.2f})"
        )

    def log_missing_field(self, field: str, context: str = ""):
        """Log when a required field is missing."""
        self.missing_fields.append({"field": field, "context": context})
        self.logger.warning(f"Missing field: '{field}' {context}")

    def log_validation_warning(self, message: str, details: dict = None):
        """Log a data validation warning."""
        self.validation_warnings.append({"message": message, "details": details or {}})
        self.logger.warning(f"Validation: {message}")
        if details:
            self.logger.debug(f"Validation details: {details}")

    def get_summary(self) -> dict:
        """Get a summary of all data quality issues."""
        return {
            "fuzzy_matches": len(self.fuzzy_matches),
            "missing_fields": len(self.missing_fields),
            "validation_warnings": len(self.validation_warnings),
            "fuzzy_matches_detail": self.fuzzy_matches,
            "missing_fields_detail": self.missing_fields,
            "validation_warnings_detail": self.validation_warnings,
        }

    def reset(self):
        """Clear all logged issues."""
        self.fuzzy_matches.clear()
        self.missing_fields.clear()
        self.validation_warnings.clear()


def setup_logging(
    log_level: str = "INFO", log_to_file: bool = True, log_dir: Path | None = None
) -> tuple[logging.Logger, DataQualityLogger]:
    """
    Set up the logging system.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to a file
        log_dir: Directory for log files (defaults to project_root/logs)

    Returns:
        Tuple of (main_logger, data_quality_logger)
    """
    # Create root logger
    logger = logging.getLogger("financial_report_analyzer")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear any existing handlers
    logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter("%(levelname)-8s %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if enabled)
    if log_to_file:
        if log_dir is None:
            # Default to project_root/logs
            project_root = Path(__file__).parent.parent
            log_dir = project_root / "logs"

        log_dir.mkdir(exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"financial_report_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    # Create data quality logger
    dq_logger = logging.getLogger("financial_report_analyzer.data_quality")
    data_quality_logger = DataQualityLogger(dq_logger)

    return logger, data_quality_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name (will be prefixed with 'financial_report_analyzer.')

    Returns:
        Logger instance
    """
    return logging.getLogger(f"financial_report_analyzer.{name}")


# Module-level loggers cache
_loggers = {}


def get_module_logger() -> logging.Logger:
    """
    Get a logger for the calling module.

    Returns:
        Logger instance for the calling module
    """
    import inspect

    frame = inspect.currentframe().f_back
    module = inspect.getmodule(frame)
    module_name = module.__name__ if module else "unknown"

    if module_name not in _loggers:
        # Extract just the module name without 'scripts.' prefix
        short_name = module_name.split(".")[-1]
        _loggers[module_name] = get_logger(short_name)

    return _loggers[module_name]
