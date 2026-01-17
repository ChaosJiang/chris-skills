"""
Data validation layer for financial data quality checks.

Provides validators for:
- Balance sheet accounting equations
- Margin consistency
- Time series regularity
- Data range validation
"""

from datetime import datetime

from logging_config import get_module_logger

logger = get_module_logger()


class ValidationResult:
    """Result of a validation check."""

    def __init__(
        self,
        passed: bool,
        message: str,
        severity: str = "warning",
        details: dict = None,
    ):
        """
        Initialize validation result.

        Args:
            passed: Whether validation passed
            message: Description of the result
            severity: 'info', 'warning', or 'error'
            details: Additional details about the validation
        """
        self.passed = passed
        self.message = message
        self.severity = severity
        self.details = details or {}

    def __repr__(self):
        status = "✓" if self.passed else "✗"
        return f"{status} [{self.severity.upper()}] {self.message}"


class FinancialValidator:
    """Validator for financial data quality."""

    def __init__(self, config: dict = None):
        """
        Initialize validator with configuration.

        Args:
            config: Configuration dict with tolerance settings
        """
        self.config = config or {}
        self.results: list[ValidationResult] = []

    def validate_balance_sheet_equation(
        self,
        assets: float | None,
        liabilities: float | None,
        equity: float | None,
        tolerance: float = 0.01,
    ) -> ValidationResult:
        """
        Validate: Assets = Liabilities + Equity.

        Args:
            assets: Total assets
            liabilities: Total liabilities
            equity: Total equity
            tolerance: Acceptable percentage difference (default 1%)

        Returns:
            ValidationResult
        """
        if assets is None or liabilities is None or equity is None:
            result = ValidationResult(
                passed=True,
                message="Balance sheet equation check skipped (missing data)",
                severity="info",
                details={
                    "assets": assets,
                    "liabilities": liabilities,
                    "equity": equity,
                },
            )
            self.results.append(result)
            return result

        # Skip validation if any value is zero (likely incomplete data)
        if assets == 0 or (liabilities == 0 and equity == 0):
            result = ValidationResult(
                passed=True,
                message="Balance sheet equation check skipped (zero values)",
                severity="info",
            )
            self.results.append(result)
            return result

        expected = liabilities + equity
        difference = abs(assets - expected)
        relative_diff = difference / abs(assets) if assets != 0 else 0

        passed = relative_diff <= tolerance

        if passed:
            result = ValidationResult(
                passed=True,
                message=f"Balance sheet equation validated (diff: {relative_diff:.2%})",
                severity="info",
                details={
                    "assets": assets,
                    "liabilities": liabilities,
                    "equity": equity,
                    "difference": difference,
                    "relative_difference": relative_diff,
                },
            )
        else:
            result = ValidationResult(
                passed=False,
                message=(
                    f"Balance sheet equation mismatch: Assets ({assets:,.0f}) ≠ "
                    f"Liabilities ({liabilities:,.0f}) + Equity ({equity:,.0f}). "
                    f"Difference: {difference:,.0f} ({relative_diff:.2%})"
                ),
                severity="warning",
                details={
                    "assets": assets,
                    "liabilities": liabilities,
                    "equity": equity,
                    "difference": difference,
                    "relative_difference": relative_diff,
                    "tolerance": tolerance,
                },
            )

        self.results.append(result)
        logger.log(
            logger.level if passed else 30,  # WARNING if failed
            result.message,
        )
        return result

    def validate_margin_consistency(
        self,
        gross_margin: float | None,
        operating_margin: float | None,
        net_margin: float | None,
    ) -> ValidationResult:
        """
        Validate: Gross Margin >= Operating Margin >= Net Margin.

        Args:
            gross_margin: Gross profit margin
            operating_margin: Operating profit margin
            net_margin: Net profit margin

        Returns:
            ValidationResult
        """
        margins = {
            "gross": gross_margin,
            "operating": operating_margin,
            "net": net_margin,
        }

        # Filter out None values
        available_margins = {k: v for k, v in margins.items() if v is not None}

        if len(available_margins) < 2:
            result = ValidationResult(
                passed=True,
                message="Margin consistency check skipped (insufficient data)",
                severity="info",
            )
            self.results.append(result)
            return result

        issues = []

        # Check gross >= operating
        if gross_margin is not None and operating_margin is not None:
            if gross_margin < operating_margin:
                issues.append(
                    f"Gross margin ({gross_margin:.2%}) < Operating margin ({operating_margin:.2%})"
                )

        # Check operating >= net
        if operating_margin is not None and net_margin is not None:
            if operating_margin < net_margin:
                issues.append(
                    f"Operating margin ({operating_margin:.2%}) < Net margin ({net_margin:.2%})"
                )

        # Check gross >= net
        if gross_margin is not None and net_margin is not None:
            if gross_margin < net_margin:
                issues.append(
                    f"Gross margin ({gross_margin:.2%}) < Net margin ({net_margin:.2%})"
                )

        passed = len(issues) == 0

        if passed:
            result = ValidationResult(
                passed=True,
                message="Margin consistency validated",
                severity="info",
                details=margins,
            )
        else:
            result = ValidationResult(
                passed=False,
                message="Margin consistency issues: " + "; ".join(issues),
                severity="warning",
                details=margins,
            )

        self.results.append(result)
        logger.log(
            logger.level if passed else 30,  # WARNING if failed
            result.message,
        )
        return result

    def validate_time_series_frequency(
        self,
        dates: list[datetime],
        expected_frequency: str = "quarterly",
        tolerance_days: int = 10,
    ) -> ValidationResult:
        """
        Validate that time series has consistent frequency.

        Args:
            dates: List of dates in the time series
            expected_frequency: 'quarterly' or 'annual'
            tolerance_days: Acceptable deviation in days

        Returns:
            ValidationResult
        """
        if len(dates) < 2:
            result = ValidationResult(
                passed=True,
                message="Time series frequency check skipped (insufficient data points)",
                severity="info",
            )
            self.results.append(result)
            return result

        # Expected intervals
        if expected_frequency == "quarterly":
            expected_days = 90  # ~3 months
            min_days = 85 - tolerance_days
            max_days = 95 + tolerance_days
        elif expected_frequency == "annual":
            expected_days = 365  # ~1 year
            min_days = 360 - tolerance_days
            max_days = 370 + tolerance_days
        else:
            raise ValueError(f"Unknown frequency: {expected_frequency}")

        # Sort dates
        sorted_dates = sorted(dates)

        # Calculate intervals
        intervals = []
        irregular_intervals = []

        for i in range(1, len(sorted_dates)):
            diff = (sorted_dates[i] - sorted_dates[i - 1]).days
            intervals.append(diff)

            if not (min_days <= diff <= max_days):
                irregular_intervals.append(
                    {
                        "from": sorted_dates[i - 1].isoformat(),
                        "to": sorted_dates[i].isoformat(),
                        "days": diff,
                    }
                )

        if not intervals:
            result = ValidationResult(
                passed=True,
                message="Time series frequency check skipped (no intervals)",
                severity="info",
            )
            self.results.append(result)
            return result

        avg_interval = sum(intervals) / len(intervals)
        passed = len(irregular_intervals) == 0

        if passed:
            result = ValidationResult(
                passed=True,
                message=f"Time series frequency is regular ({expected_frequency}, avg: {avg_interval:.0f} days)",
                severity="info",
                details={
                    "expected_frequency": expected_frequency,
                    "average_interval_days": avg_interval,
                    "intervals": intervals,
                },
            )
        else:
            result = ValidationResult(
                passed=False,
                message=(
                    f"{len(irregular_intervals)} irregular intervals detected "
                    f"(expected {expected_frequency}: {expected_days}±{tolerance_days} days)"
                ),
                severity="warning",
                details={
                    "expected_frequency": expected_frequency,
                    "average_interval_days": avg_interval,
                    "irregular_intervals": irregular_intervals,
                },
            )

        self.results.append(result)
        logger.log(
            logger.level if passed else 30,  # WARNING if failed
            result.message,
        )
        return result

    def validate_data_range(
        self,
        value: float | None,
        min_value: float | None = None,
        max_value: float | None = None,
        field_name: str = "value",
    ) -> ValidationResult:
        """
        Validate that a value is within expected range.

        Args:
            value: Value to check
            min_value: Minimum acceptable value
            max_value: Maximum acceptable value
            field_name: Name of the field being validated

        Returns:
            ValidationResult
        """
        if value is None:
            result = ValidationResult(
                passed=True,
                message=f"Range check skipped for {field_name} (no value)",
                severity="info",
            )
            self.results.append(result)
            return result

        issues = []

        if min_value is not None and value < min_value:
            issues.append(f"{field_name} ({value}) < minimum ({min_value})")

        if max_value is not None and value > max_value:
            issues.append(f"{field_name} ({value}) > maximum ({max_value})")

        passed = len(issues) == 0

        if passed:
            result = ValidationResult(
                passed=True,
                message=f"{field_name} is within valid range",
                severity="info",
                details={"value": value, "min": min_value, "max": max_value},
            )
        else:
            result = ValidationResult(
                passed=False,
                message="Range validation failed: " + "; ".join(issues),
                severity="warning",
                details={"value": value, "min": min_value, "max": max_value},
            )

        self.results.append(result)
        logger.log(
            logger.level if passed else 30,  # WARNING if failed
            result.message,
        )
        return result

    def get_summary(self) -> dict:
        """
        Get summary of all validation results.

        Returns:
            Dict with counts and details of validation results
        """
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        by_severity = {
            "info": [r for r in self.results if r.severity == "info"],
            "warning": [r for r in self.results if r.severity == "warning"],
            "error": [r for r in self.results if r.severity == "error"],
        }

        return {
            "total_checks": total,
            "passed": passed,
            "failed": failed,
            "by_severity": {
                "info": len(by_severity["info"]),
                "warning": len(by_severity["warning"]),
                "error": len(by_severity["error"]),
            },
            "results": [
                {
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity,
                    "details": r.details,
                }
                for r in self.results
            ],
        }

    def reset(self):
        """Clear all validation results."""
        self.results.clear()


def validate_financial_data(
    balance_sheet: dict,
    income_statement: dict,
    config: dict = None,
) -> FinancialValidator:
    """
    Run all financial data validations.

    Args:
        balance_sheet: Balance sheet data
        income_statement: Income statement data
        config: Validation configuration

    Returns:
        FinancialValidator with results
    """
    validator = FinancialValidator(config)

    # Extract latest values from balance sheet
    assets = balance_sheet.get("total_assets")
    liabilities = balance_sheet.get("total_liabilities")
    equity = balance_sheet.get("total_equity")

    # Validate balance sheet equation
    tolerance = config.get("balance_sheet_tolerance", 0.01) if config else 0.01
    validator.validate_balance_sheet_equation(assets, liabilities, equity, tolerance)

    # Extract margins from income statement
    gross_margin = income_statement.get("gross_margin")
    operating_margin = income_statement.get("operating_margin")
    net_margin = income_statement.get("net_margin")

    # Validate margin consistency
    validator.validate_margin_consistency(gross_margin, operating_margin, net_margin)

    # Validate margin ranges (shouldn't exceed 100% or be less than -100%)
    if gross_margin is not None:
        validator.validate_data_range(
            gross_margin, min_value=-1.0, max_value=1.0, field_name="gross_margin"
        )
    if operating_margin is not None:
        validator.validate_data_range(
            operating_margin,
            min_value=-1.0,
            max_value=1.0,
            field_name="operating_margin",
        )
    if net_margin is not None:
        validator.validate_data_range(
            net_margin, min_value=-1.0, max_value=1.0, field_name="net_margin"
        )

    return validator
