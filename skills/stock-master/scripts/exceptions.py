"""
Custom exception hierarchy for the Financial Report Analyzer.

Provides specific exception types for different failure modes,
enabling better error handling and user messaging.
"""


class FinancialReportError(Exception):
    """Base exception for all financial report analyzer errors."""

    def __init__(self, message: str, details: dict = None):
        """
        Initialize exception.

        Args:
            message: Human-readable error message
            details: Optional dict with additional error context
        """
        super().__init__(message)
        self.details = details or {}


class DataFetchError(FinancialReportError):
    """Error occurred while fetching data from external sources."""


class APIError(DataFetchError):
    """Error from API call (HTTP errors, rate limiting, etc.)."""

    def __init__(self, message: str, status_code: int = None, details: dict = None):
        """
        Initialize API error.

        Args:
            message: Error message
            status_code: HTTP status code if applicable
            details: Additional error details
        """
        super().__init__(message, details)
        self.status_code = status_code


class SymbolNotFoundError(DataFetchError):
    """Stock symbol not found in the data source."""

    def __init__(self, symbol: str, market: str = None):
        """
        Initialize symbol not found error.

        Args:
            symbol: The stock symbol that wasn't found
            market: The market being searched (US, CN, HK, JP)
        """
        market_info = f" in {market} market" if market else ""
        super().__init__(
            f"Symbol '{symbol}' not found{market_info}",
            details={"symbol": symbol, "market": market},
        )
        self.symbol = symbol
        self.market = market


class RateLimitError(APIError):
    """API rate limit exceeded."""

    def __init__(
        self, message: str = "API rate limit exceeded", retry_after: int = None
    ):
        """
        Initialize rate limit error.

        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
        """
        super().__init__(message, status_code=429, details={"retry_after": retry_after})
        self.retry_after = retry_after


class DataValidationError(FinancialReportError):
    """Data failed validation checks."""

    def __init__(self, message: str, validation_type: str, details: dict = None):
        """
        Initialize validation error.

        Args:
            message: Error message
            validation_type: Type of validation that failed
            details: Validation details
        """
        super().__init__(message, details)
        self.validation_type = validation_type


class BalanceSheetValidationError(DataValidationError):
    """Balance sheet failed accounting equation validation."""

    def __init__(
        self, assets: float, liabilities: float, equity: float, tolerance: float
    ):
        """
        Initialize balance sheet validation error.

        Args:
            assets: Total assets value
            liabilities: Total liabilities value
            equity: Total equity value
            tolerance: Validation tolerance used
        """
        diff = assets - (liabilities + equity)
        message = (
            f"Balance sheet equation failed: "
            f"Assets ({assets:,.2f}) ≠ Liabilities ({liabilities:,.2f}) + "
            f"Equity ({equity:,.2f}). Difference: {diff:,.2f}"
        )
        super().__init__(
            message,
            validation_type="balance_sheet_equation",
            details={
                "assets": assets,
                "liabilities": liabilities,
                "equity": equity,
                "difference": diff,
                "tolerance": tolerance,
            },
        )


class CurrencyConversionError(FinancialReportError):
    """Error during currency conversion."""

    def __init__(self, from_currency: str, to_currency: str, reason: str = None):
        """
        Initialize currency conversion error.

        Args:
            from_currency: Source currency code
            to_currency: Target currency code
            reason: Optional reason for failure
        """
        message = f"Failed to convert {from_currency} to {to_currency}"
        if reason:
            message += f": {reason}"

        super().__init__(
            message,
            details={
                "from_currency": from_currency,
                "to_currency": to_currency,
                "reason": reason,
            },
        )
        self.from_currency = from_currency
        self.to_currency = to_currency


class FieldNotFoundError(FinancialReportError):
    """Required financial field not found in data."""

    def __init__(self, field_name: str, available_fields: list = None):
        """
        Initialize field not found error.

        Args:
            field_name: Name of the missing field
            available_fields: List of fields that are available
        """
        message = f"Required field '{field_name}' not found"
        if available_fields:
            message += f". Available fields: {', '.join(available_fields[:5])}"
            if len(available_fields) > 5:
                message += f" (and {len(available_fields) - 5} more)"

        super().__init__(
            message,
            details={"field_name": field_name, "available_fields": available_fields},
        )
        self.field_name = field_name


class DataQualityWarning(Warning):
    """Warning for non-critical data quality issues."""


class ConfigurationError(FinancialReportError):
    """Error in configuration or environment setup."""


class ReportGenerationError(FinancialReportError):
    """Error during report generation."""


def format_error_for_user(error: Exception) -> str:
    """
    Format an exception into a user-friendly error message.

    Args:
        error: The exception to format

    Returns:
        Formatted error message with actionable guidance
    """
    if isinstance(error, SymbolNotFoundError):
        msg = f"❌ {error}\n"
        msg += "\nSuggestions:\n"
        msg += "  • Check that the symbol is correct\n"
        if error.market:
            msg += f"  • Verify the symbol exists in the {error.market} market\n"
        msg += "  • Try searching for the company on finance.yahoo.com\n"
        return msg

    if isinstance(error, RateLimitError):
        msg = f"❌ {error}\n"
        if error.retry_after:
            msg += f"\nPlease wait {error.retry_after} seconds before trying again.\n"
        else:
            msg += "\nSuggestions:\n"
            msg += "  • Wait a few minutes before retrying\n"
            msg += "  • Use cached data with --use-cache if available\n"
        return msg

    if isinstance(error, CurrencyConversionError):
        msg = f"❌ {error}\n"
        msg += "\nSuggestions:\n"
        msg += "  • Check your internet connection (currency rates require network access)\n"
        msg += "  • The analysis will continue but values may be in original currency\n"
        return msg

    if isinstance(error, FieldNotFoundError):
        msg = f"❌ {error}\n"
        msg += "\nThis may indicate:\n"
        msg += "  • The company's financial statements use different field names\n"
        msg += "  • The data source has changed its format\n"
        msg += "  • The company hasn't filed this particular statement\n"
        return msg

    if isinstance(error, DataValidationError):
        msg = f"⚠️  Data validation warning: {error}\n"
        msg += "\nThis doesn't prevent analysis but indicates potential data quality issues.\n"
        return msg

    if isinstance(error, APIError):
        msg = f"❌ API Error: {error}\n"
        if error.status_code:
            msg += f"Status code: {error.status_code}\n"
        msg += "\nSuggestions:\n"
        msg += "  • Check your internet connection\n"
        msg += "  • Try again in a few minutes\n"
        msg += "  • Check if the data source is experiencing issues\n"
        return msg

    # Generic error formatting
    return f"❌ Error: {error}\n"
