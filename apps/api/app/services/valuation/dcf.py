"""DCF (Discounted Cash Flow) valuation models."""

import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from app.core.observability import trace_function


class DCFModel(Enum):
    """DCF model types."""
    THREE_STAGE = "three_stage"
    H_MODEL = "h_model"
    TWO_STAGE = "two_stage"


class TerminalMethod(Enum):
    """Terminal value calculation methods."""
    GORDON_GROWTH = "gordon_growth"
    EXIT_MULTIPLE = "exit_multiple"


@dataclass
class DCFInputs:
    """DCF model inputs."""
    # Financial data
    revenue: float
    operating_margin: float
    tax_rate: float
    capex_percent_revenue: float
    depreciation_percent_revenue: float
    working_capital_percent_revenue: float
    
    # Growth assumptions
    revenue_growth_years_1_5: List[float]  # 5 years of growth rates
    revenue_growth_years_6_10: Optional[List[float]] = None  # For 3-stage model
    terminal_growth_rate: float = 0.025  # 2.5% long-term growth
    
    # Discount rate
    wacc: float = 0.10  # 10% WACC
    
    # Terminal value
    terminal_method: TerminalMethod = TerminalMethod.GORDON_GROWTH
    exit_multiple: Optional[float] = None  # EV/Revenue or EV/EBITDA
    
    # Shares outstanding
    shares_outstanding: float = 1.0
    net_debt: float = 0.0  # Net debt (debt - cash)


@dataclass
class DCFOutput:
    """DCF model output."""
    enterprise_value: float
    equity_value: float
    share_price: float
    
    # Detailed projections
    projections: pd.DataFrame
    terminal_value: float
    pv_terminal_value: float
    pv_explicit_period: float
    
    # Sensitivity analysis
    sensitivity_table: Optional[pd.DataFrame] = None
    
    # Model metadata
    model_type: DCFModel = DCFModel.THREE_STAGE
    assumptions: Dict[str, Any] = None


class DCFCalculator:
    """DCF valuation calculator."""
    
    def __init__(self) -> None:
        """Initialize DCF calculator."""
        pass
    
    @trace_function("dcf_calculator.calculate_three_stage")
    def calculate_three_stage_dcf(self, inputs: DCFInputs) -> DCFOutput:
        """Calculate 3-stage DCF model."""
        # Stage 1: High growth (years 1-5)
        stage1_projections = self._project_stage1(inputs)
        
        # Stage 2: Declining growth (years 6-10)
        stage2_projections = self._project_stage2(inputs, stage1_projections)
        
        # Stage 3: Terminal value
        terminal_value = self._calculate_terminal_value(inputs, stage2_projections)
        
        # Combine projections
        all_projections = pd.concat([stage1_projections, stage2_projections], ignore_index=True)
        
        # Calculate present values
        pv_explicit = self._calculate_pv_explicit_period(all_projections, inputs.wacc)
        pv_terminal = terminal_value / ((1 + inputs.wacc) ** 10)
        
        # Enterprise and equity value
        enterprise_value = pv_explicit + pv_terminal
        equity_value = enterprise_value - inputs.net_debt
        share_price = equity_value / inputs.shares_outstanding
        
        return DCFOutput(
            enterprise_value=enterprise_value,
            equity_value=equity_value,
            share_price=share_price,
            projections=all_projections,
            terminal_value=terminal_value,
            pv_terminal_value=pv_terminal,
            pv_explicit_period=pv_explicit,
            model_type=DCFModel.THREE_STAGE,
            assumptions=self._get_assumptions_dict(inputs),
        )
    
    @trace_function("dcf_calculator.calculate_h_model")
    def calculate_h_model_dcf(self, inputs: DCFInputs) -> DCFOutput:
        """Calculate H-model DCF (declining growth model)."""
        # H-model parameters
        initial_growth = inputs.revenue_growth_years_1_5[0]
        terminal_growth = inputs.terminal_growth_rate
        decline_period = 10  # years for growth to decline to terminal
        
        projections = []
        
        for year in range(1, 11):  # 10 year projection
            # Calculate declining growth rate
            if year <= decline_period:
                growth_rate = terminal_growth + (initial_growth - terminal_growth) * (decline_period - year) / decline_period
            else:
                growth_rate = terminal_growth
            
            # Project financials
            if year == 1:
                revenue = inputs.revenue * (1 + growth_rate)
            else:
                revenue = projections[-1]["revenue"] * (1 + growth_rate)
            
            projection = self._calculate_year_projection(revenue, inputs, year)
            projection["growth_rate"] = growth_rate
            projections.append(projection)
        
        projections_df = pd.DataFrame(projections)
        
        # Terminal value using Gordon Growth
        final_fcf = projections_df.iloc[-1]["free_cash_flow"]
        terminal_value = final_fcf * (1 + inputs.terminal_growth_rate) / (inputs.wacc - inputs.terminal_growth_rate)
        
        # Present values
        pv_explicit = self._calculate_pv_explicit_period(projections_df, inputs.wacc)
        pv_terminal = terminal_value / ((1 + inputs.wacc) ** 10)
        
        # Enterprise and equity value
        enterprise_value = pv_explicit + pv_terminal
        equity_value = enterprise_value - inputs.net_debt
        share_price = equity_value / inputs.shares_outstanding
        
        return DCFOutput(
            enterprise_value=enterprise_value,
            equity_value=equity_value,
            share_price=share_price,
            projections=projections_df,
            terminal_value=terminal_value,
            pv_terminal_value=pv_terminal,
            pv_explicit_period=pv_explicit,
            model_type=DCFModel.H_MODEL,
            assumptions=self._get_assumptions_dict(inputs),
        )
    
    def _project_stage1(self, inputs: DCFInputs) -> pd.DataFrame:
        """Project Stage 1 (high growth) financials."""
        projections = []
        
        for year in range(1, 6):  # Years 1-5
            growth_rate = inputs.revenue_growth_years_1_5[year - 1]
            
            if year == 1:
                revenue = inputs.revenue * (1 + growth_rate)
            else:
                revenue = projections[-1]["revenue"] * (1 + growth_rate)
            
            projection = self._calculate_year_projection(revenue, inputs, year)
            projection["growth_rate"] = growth_rate
            projection["stage"] = 1
            projections.append(projection)
        
        return pd.DataFrame(projections)
    
    def _project_stage2(self, inputs: DCFInputs, stage1_df: pd.DataFrame) -> pd.DataFrame:
        """Project Stage 2 (declining growth) financials."""
        projections = []
        
        # Get final revenue from stage 1
        final_stage1_revenue = stage1_df.iloc[-1]["revenue"]
        
        # Calculate declining growth rates
        initial_growth = inputs.revenue_growth_years_1_5[-1]  # Last year of stage 1
        terminal_growth = inputs.terminal_growth_rate
        
        for year in range(6, 11):  # Years 6-10
            # Linear decline from initial to terminal growth
            decline_factor = (year - 6) / 4  # 4 years to decline
            growth_rate = initial_growth - (initial_growth - terminal_growth) * decline_factor
            
            if year == 6:
                revenue = final_stage1_revenue * (1 + growth_rate)
            else:
                revenue = projections[-1]["revenue"] * (1 + growth_rate)
            
            projection = self._calculate_year_projection(revenue, inputs, year)
            projection["growth_rate"] = growth_rate
            projection["stage"] = 2
            projections.append(projection)
        
        return pd.DataFrame(projections)
    
    def _calculate_year_projection(self, revenue: float, inputs: DCFInputs, year: int) -> Dict[str, float]:
        """Calculate financial projection for a single year."""
        # Operating income
        operating_income = revenue * inputs.operating_margin
        
        # Taxes
        taxes = operating_income * inputs.tax_rate
        nopat = operating_income - taxes
        
        # Capital expenditures
        capex = revenue * inputs.capex_percent_revenue
        
        # Depreciation
        depreciation = revenue * inputs.depreciation_percent_revenue
        
        # Working capital change
        if year == 1:
            wc_change = revenue * inputs.working_capital_percent_revenue
        else:
            # Assume working capital grows with revenue
            wc_change = (revenue * inputs.working_capital_percent_revenue) * inputs.revenue_growth_years_1_5[min(year-1, 4)]
        
        # Free cash flow
        free_cash_flow = nopat + depreciation - capex - wc_change
        
        return {
            "year": year,
            "revenue": revenue,
            "operating_income": operating_income,
            "nopat": nopat,
            "taxes": taxes,
            "capex": capex,
            "depreciation": depreciation,
            "wc_change": wc_change,
            "free_cash_flow": free_cash_flow,
        }
    
    def _calculate_terminal_value(self, inputs: DCFInputs, projections_df: pd.DataFrame) -> float:
        """Calculate terminal value."""
        final_fcf = projections_df.iloc[-1]["free_cash_flow"]
        
        if inputs.terminal_method == TerminalMethod.GORDON_GROWTH:
            # Gordon Growth Model
            terminal_fcf = final_fcf * (1 + inputs.terminal_growth_rate)
            terminal_value = terminal_fcf / (inputs.wacc - inputs.terminal_growth_rate)
        
        elif inputs.terminal_method == TerminalMethod.EXIT_MULTIPLE:
            # Exit Multiple Method
            final_revenue = projections_df.iloc[-1]["revenue"]
            terminal_value = final_revenue * inputs.exit_multiple
        
        else:
            raise ValueError(f"Unknown terminal method: {inputs.terminal_method}")
        
        return terminal_value
    
    def _calculate_pv_explicit_period(self, projections_df: pd.DataFrame, wacc: float) -> float:
        """Calculate present value of explicit forecast period."""
        pv_total = 0.0
        
        for _, row in projections_df.iterrows():
            year = row["year"]
            fcf = row["free_cash_flow"]
            pv = fcf / ((1 + wacc) ** year)
            pv_total += pv
        
        return pv_total
    
    def _get_assumptions_dict(self, inputs: DCFInputs) -> Dict[str, Any]:
        """Get assumptions dictionary for output."""
        return {
            "wacc": inputs.wacc,
            "terminal_growth_rate": inputs.terminal_growth_rate,
            "tax_rate": inputs.tax_rate,
            "operating_margin": inputs.operating_margin,
            "capex_percent_revenue": inputs.capex_percent_revenue,
            "terminal_method": inputs.terminal_method.value,
        }
    
    @trace_function("dcf_calculator.sensitivity_analysis")
    def sensitivity_analysis(
        self, 
        inputs: DCFInputs, 
        wacc_range: Tuple[float, float] = (0.08, 0.12),
        growth_range: Tuple[float, float] = (0.02, 0.04),
        steps: int = 5
    ) -> pd.DataFrame:
        """Perform sensitivity analysis on WACC and terminal growth rate."""
        wacc_values = np.linspace(wacc_range[0], wacc_range[1], steps)
        growth_values = np.linspace(growth_range[0], growth_range[1], steps)
        
        sensitivity_matrix = []
        
        for wacc in wacc_values:
            row = []
            for growth in growth_values:
                # Create modified inputs
                modified_inputs = DCFInputs(
                    revenue=inputs.revenue,
                    operating_margin=inputs.operating_margin,
                    tax_rate=inputs.tax_rate,
                    capex_percent_revenue=inputs.capex_percent_revenue,
                    depreciation_percent_revenue=inputs.depreciation_percent_revenue,
                    working_capital_percent_revenue=inputs.working_capital_percent_revenue,
                    revenue_growth_years_1_5=inputs.revenue_growth_years_1_5,
                    terminal_growth_rate=growth,
                    wacc=wacc,
                    shares_outstanding=inputs.shares_outstanding,
                    net_debt=inputs.net_debt,
                )
                
                # Calculate DCF
                result = self.calculate_three_stage_dcf(modified_inputs)
                row.append(result.share_price)
            
            sensitivity_matrix.append(row)
        
        # Create DataFrame
        sensitivity_df = pd.DataFrame(
            sensitivity_matrix,
            index=[f"{wacc:.1%}" for wacc in wacc_values],
            columns=[f"{growth:.1%}" for growth in growth_values]
        )
        
        return sensitivity_df


class WACCCalculator:
    """WACC (Weighted Average Cost of Capital) calculator."""
    
    @trace_function("wacc_calculator.calculate_wacc")
    def calculate_wacc(
        self,
        market_value_equity: float,
        market_value_debt: float,
        cost_of_equity: float,
        cost_of_debt: float,
        tax_rate: float,
    ) -> float:
        """Calculate WACC."""
        total_value = market_value_equity + market_value_debt
        
        equity_weight = market_value_equity / total_value
        debt_weight = market_value_debt / total_value
        
        after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
        
        wacc = (equity_weight * cost_of_equity) + (debt_weight * after_tax_cost_of_debt)
        
        return wacc
    
    @trace_function("wacc_calculator.calculate_cost_of_equity_capm")
    def calculate_cost_of_equity_capm(
        self,
        risk_free_rate: float,
        beta: float,
        market_risk_premium: float,
    ) -> float:
        """Calculate cost of equity using CAPM."""
        return risk_free_rate + (beta * market_risk_premium)
    
    @trace_function("wacc_calculator.calculate_cost_of_debt")
    def calculate_cost_of_debt(
        self,
        interest_expense: float,
        total_debt: float,
    ) -> float:
        """Calculate cost of debt from financial statements."""
        if total_debt == 0:
            return 0.0
        
        return interest_expense / total_debt
