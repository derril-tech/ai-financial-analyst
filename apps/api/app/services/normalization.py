"""Data normalization services."""

import re
from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

from app.core.observability import trace_function


class CurrencyNormalizer:
    """Service for currency normalization."""
    
    def __init__(self) -> None:
        """Initialize currency normalizer."""
        # Exchange rates (in production, fetch from API)
        self.exchange_rates = {
            "USD": 1.0,
            "EUR": 1.08,
            "GBP": 1.25,
            "JPY": 0.0067,
            "CAD": 0.74,
            "AUD": 0.66,
        }
        
        # Currency symbols and patterns
        self.currency_patterns = {
            "USD": [r"\$", r"USD", r"US\$"],
            "EUR": [r"€", r"EUR"],
            "GBP": [r"£", r"GBP"],
            "JPY": [r"¥", r"JPY"],
            "CAD": [r"CAD", r"C\$"],
            "AUD": [r"AUD", r"A\$"],
        }
    
    @trace_function("currency_normalizer.normalize_amount")
    def normalize_amount(
        self, 
        amount: Any, 
        from_currency: str, 
        to_currency: str = "USD"
    ) -> Optional[Decimal]:
        """Normalize amount to target currency."""
        try:
            # Parse amount if it's a string
            if isinstance(amount, str):
                amount = self._parse_amount_string(amount)
            
            if amount is None:
                return None
            
            # Convert to Decimal for precision
            amount_decimal = Decimal(str(amount))
            
            # Get exchange rates
            from_rate = self.exchange_rates.get(from_currency.upper(), 1.0)
            to_rate = self.exchange_rates.get(to_currency.upper(), 1.0)
            
            # Convert to USD first, then to target currency
            usd_amount = amount_decimal / Decimal(str(from_rate))
            target_amount = usd_amount * Decimal(str(to_rate))
            
            return target_amount
            
        except (ValueError, TypeError, KeyError):
            return None
    
    def _parse_amount_string(self, amount_str: str) -> Optional[float]:
        """Parse amount from string, handling various formats."""
        if not amount_str:
            return None
        
        # Remove whitespace
        amount_str = amount_str.strip()
        
        # Remove currency symbols
        for currency, patterns in self.currency_patterns.items():
            for pattern in patterns:
                amount_str = re.sub(pattern, "", amount_str, flags=re.IGNORECASE)
        
        # Remove common formatting
        amount_str = amount_str.replace(",", "").replace(" ", "")
        
        # Handle parentheses (negative numbers)
        if amount_str.startswith("(") and amount_str.endswith(")"):
            amount_str = "-" + amount_str[1:-1]
        
        try:
            return float(amount_str)
        except ValueError:
            return None
    
    def detect_currency(self, text: str) -> Optional[str]:
        """Detect currency from text."""
        text_upper = text.upper()
        
        for currency, patterns in self.currency_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_upper):
                    return currency
        
        return None


class UnitNormalizer:
    """Service for unit normalization."""
    
    def __init__(self) -> None:
        """Initialize unit normalizer."""
        # Scale factors
        self.scale_factors = {
            "ones": 1,
            "thousands": 1_000,
            "millions": 1_000_000,
            "billions": 1_000_000_000,
            "trillions": 1_000_000_000_000,
        }
        
        # Unit patterns
        self.unit_patterns = {
            "thousands": [r"thousands?", r"\(000s?\)", r"k\b", r"000s?"],
            "millions": [r"millions?", r"\(000,000s?\)", r"m\b", r"mm\b"],
            "billions": [r"billions?", r"\(000,000,000s?\)", r"b\b", r"bn\b"],
            "trillions": [r"trillions?", r"t\b", r"tn\b"],
        }
    
    @trace_function("unit_normalizer.normalize_value")
    def normalize_value(
        self, 
        value: Any, 
        unit_text: str, 
        target_scale: str = "ones"
    ) -> Optional[Decimal]:
        """Normalize value to target scale."""
        try:
            # Parse value
            if isinstance(value, str):
                value = float(value.replace(",", ""))
            
            if value is None:
                return None
            
            # Detect scale from unit text
            detected_scale = self.detect_scale(unit_text)
            
            # Get scale factors
            from_factor = self.scale_factors.get(detected_scale, 1)
            to_factor = self.scale_factors.get(target_scale, 1)
            
            # Convert
            normalized_value = Decimal(str(value)) * Decimal(str(from_factor)) / Decimal(str(to_factor))
            
            return normalized_value
            
        except (ValueError, TypeError):
            return None
    
    def detect_scale(self, text: str) -> str:
        """Detect scale from text."""
        text_lower = text.lower()
        
        for scale, patterns in self.unit_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return scale
        
        return "ones"


class FiscalCalendarAligner:
    """Service for fiscal calendar alignment."""
    
    def __init__(self) -> None:
        """Initialize fiscal calendar aligner."""
        # Common fiscal year end months by industry
        self.common_fy_ends = {
            "technology": 12,  # December
            "retail": 1,       # January
            "healthcare": 12,  # December
            "financial": 12,   # December
            "energy": 12,      # December
        }
    
    @trace_function("fiscal_calendar_aligner.align_period")
    def align_period(
        self, 
        fiscal_period: str, 
        fiscal_year: int, 
        fy_end_month: int = 12
    ) -> Tuple[date, date]:
        """Align fiscal period to calendar dates."""
        try:
            if fiscal_period.upper() == "FY":
                # Full fiscal year
                if fy_end_month == 12:
                    start_date = date(fiscal_year - 1, 1, 1)
                    end_date = date(fiscal_year - 1, 12, 31)
                else:
                    start_date = date(fiscal_year - 1, fy_end_month + 1, 1)
                    end_date = date(fiscal_year, fy_end_month, self._last_day_of_month(fiscal_year, fy_end_month))
            
            elif fiscal_period.upper().startswith("Q"):
                # Quarterly period
                quarter = int(fiscal_period[1])
                quarter_months = self._get_quarter_months(quarter, fy_end_month)
                
                start_month, end_month = quarter_months[0], quarter_months[-1]
                
                if fy_end_month == 12:
                    year = fiscal_year - 1
                else:
                    year = fiscal_year - 1 if start_month > fy_end_month else fiscal_year
                
                start_date = date(year, start_month, 1)
                end_date = date(
                    year if end_month >= start_month else year + 1,
                    end_month,
                    self._last_day_of_month(year, end_month)
                )
            
            else:
                raise ValueError(f"Unknown fiscal period: {fiscal_period}")
            
            return start_date, end_date
            
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to align fiscal period {fiscal_period}: {e}")
    
    def _get_quarter_months(self, quarter: int, fy_end_month: int) -> list[int]:
        """Get months for a fiscal quarter."""
        # Calculate quarter start month
        q1_start = (fy_end_month + 1) % 12
        if q1_start == 0:
            q1_start = 12
        
        quarter_start = (q1_start + (quarter - 1) * 3 - 1) % 12 + 1
        
        months = []
        for i in range(3):
            month = (quarter_start + i - 1) % 12 + 1
            months.append(month)
        
        return months
    
    def _last_day_of_month(self, year: int, month: int) -> int:
        """Get last day of month."""
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        
        last_day = next_month - timedelta(days=1)
        return last_day.day
    
    def detect_fy_end(self, company_name: str, industry: str = None) -> int:
        """Detect fiscal year end month."""
        # Use industry default if available
        if industry and industry.lower() in self.common_fy_ends:
            return self.common_fy_ends[industry.lower()]
        
        # Default to December
        return 12


class DataValidator:
    """Service for data validation and cross-checks."""
    
    @trace_function("data_validator.validate_financial_data")
    def validate_financial_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate financial data for consistency."""
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
        }
        
        # Check for negative revenue (warning)
        if "revenue" in data and data["revenue"] and data["revenue"] < 0:
            validation_results["warnings"].append("Negative revenue detected")
        
        # Check assets = liabilities + equity
        if all(key in data for key in ["total_assets", "total_liabilities", "total_equity"]):
            assets = data["total_assets"]
            liabilities = data["total_liabilities"]
            equity = data["total_equity"]
            
            if assets and liabilities and equity:
                balance_diff = abs(assets - (liabilities + equity))
                tolerance = max(assets * 0.01, 1000)  # 1% or $1000
                
                if balance_diff > tolerance:
                    validation_results["errors"].append(
                        f"Balance sheet doesn't balance: Assets ({assets}) != Liabilities + Equity ({liabilities + equity})"
                    )
                    validation_results["is_valid"] = False
        
        # Check cash flow consistency
        if all(key in data for key in ["net_income", "operating_cash_flow"]):
            net_income = data["net_income"]
            operating_cf = data["operating_cash_flow"]
            
            if net_income and operating_cf:
                # Operating cash flow should generally be close to net income
                diff_ratio = abs(operating_cf - net_income) / max(abs(net_income), 1)
                
                if diff_ratio > 2.0:  # More than 200% difference
                    validation_results["warnings"].append(
                        f"Large difference between net income ({net_income}) and operating cash flow ({operating_cf})"
                    )
        
        return validation_results
