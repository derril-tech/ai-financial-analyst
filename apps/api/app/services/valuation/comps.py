"""Comparable company analysis (Comps) engine."""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats

from app.core.observability import trace_function


class Multiple(Enum):
    """Valuation multiples."""
    EV_REVENUE = "ev_revenue"
    EV_EBITDA = "ev_ebitda"
    EV_EBIT = "ev_ebit"
    PE_RATIO = "pe_ratio"
    PB_RATIO = "pb_ratio"
    PS_RATIO = "ps_ratio"
    PEG_RATIO = "peg_ratio"
    EV_FCF = "ev_fcf"


@dataclass
class CompanyData:
    """Company financial data for comps analysis."""
    symbol: str
    name: str
    market_cap: float
    enterprise_value: float
    
    # Income statement
    revenue: float
    ebitda: float
    ebit: float
    net_income: float
    
    # Balance sheet
    total_assets: float
    book_value: float
    
    # Cash flow
    free_cash_flow: float
    
    # Per share data
    shares_outstanding: float
    book_value_per_share: float
    
    # Growth rates
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    
    # Other metrics
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap_category: Optional[str] = None  # large, mid, small cap


@dataclass
class CompsResult:
    """Comparable company analysis result."""
    target_company: str
    peer_companies: List[str]
    
    # Valuation ranges
    valuation_summary: Dict[str, Dict[str, float]]  # multiple -> {min, max, median, mean}
    
    # Detailed multiples
    multiples_table: pd.DataFrame
    
    # Statistical analysis
    statistics: Dict[str, Any]
    
    # Peer selection criteria
    selection_criteria: Dict[str, Any]


class CompsEngine:
    """Comparable company analysis engine."""
    
    def __init__(self) -> None:
        """Initialize comps engine."""
        self.winsorization_percentile = 0.05  # Remove top/bottom 5%
    
    @trace_function("comps_engine.analyze_comps")
    def analyze_comps(
        self,
        target_company: CompanyData,
        peer_companies: List[CompanyData],
        multiples: List[Multiple],
        selection_criteria: Optional[Dict[str, Any]] = None,
    ) -> CompsResult:
        """Perform comparable company analysis."""
        # Filter and select peers
        selected_peers = self._select_peers(target_company, peer_companies, selection_criteria)
        
        if not selected_peers:
            raise ValueError("No suitable peer companies found")
        
        # Calculate multiples for all companies
        all_companies = [target_company] + selected_peers
        multiples_data = self._calculate_multiples(all_companies, multiples)
        
        # Create multiples table
        multiples_df = pd.DataFrame(multiples_data)
        
        # Winsorize peer multiples (exclude target)
        peer_multiples = multiples_df[multiples_df['symbol'] != target_company.symbol]
        winsorized_multiples = self._winsorize_multiples(peer_multiples, multiples)
        
        # Calculate valuation statistics
        valuation_summary = self._calculate_valuation_summary(winsorized_multiples, multiples)
        
        # Statistical analysis
        statistics = self._calculate_statistics(winsorized_multiples, multiples)
        
        return CompsResult(
            target_company=target_company.symbol,
            peer_companies=[peer.symbol for peer in selected_peers],
            valuation_summary=valuation_summary,
            multiples_table=multiples_df,
            statistics=statistics,
            selection_criteria=selection_criteria or {},
        )
    
    def _select_peers(
        self,
        target: CompanyData,
        candidates: List[CompanyData],
        criteria: Optional[Dict[str, Any]] = None,
    ) -> List[CompanyData]:
        """Select peer companies based on criteria."""
        if not criteria:
            # Default selection criteria
            criteria = {
                "same_sector": True,
                "market_cap_range": (0.5, 2.0),  # 0.5x to 2x target market cap
                "revenue_range": (0.3, 3.0),     # 0.3x to 3x target revenue
                "min_peers": 5,
                "max_peers": 15,
            }
        
        selected_peers = []
        
        for candidate in candidates:
            # Skip if same company
            if candidate.symbol == target.symbol:
                continue
            
            # Sector filter
            if criteria.get("same_sector", False):
                if candidate.sector != target.sector:
                    continue
            
            # Market cap filter
            if "market_cap_range" in criteria:
                min_ratio, max_ratio = criteria["market_cap_range"]
                ratio = candidate.market_cap / target.market_cap
                if not (min_ratio <= ratio <= max_ratio):
                    continue
            
            # Revenue filter
            if "revenue_range" in criteria:
                min_ratio, max_ratio = criteria["revenue_range"]
                ratio = candidate.revenue / target.revenue
                if not (min_ratio <= ratio <= max_ratio):
                    continue
            
            # Additional filters can be added here
            selected_peers.append(candidate)
        
        # Sort by similarity (market cap proximity)
        selected_peers.sort(
            key=lambda x: abs(x.market_cap - target.market_cap)
        )
        
        # Apply min/max peer limits
        min_peers = criteria.get("min_peers", 5)
        max_peers = criteria.get("max_peers", 15)
        
        if len(selected_peers) < min_peers:
            # Relax criteria if not enough peers
            return self._select_peers_relaxed(target, candidates, min_peers)
        
        return selected_peers[:max_peers]
    
    def _select_peers_relaxed(
        self,
        target: CompanyData,
        candidates: List[CompanyData],
        min_peers: int,
    ) -> List[CompanyData]:
        """Select peers with relaxed criteria."""
        # Sort all candidates by market cap similarity
        candidates_sorted = sorted(
            [c for c in candidates if c.symbol != target.symbol],
            key=lambda x: abs(x.market_cap - target.market_cap)
        )
        
        return candidates_sorted[:min_peers]
    
    def _calculate_multiples(
        self,
        companies: List[CompanyData],
        multiples: List[Multiple],
    ) -> List[Dict[str, Any]]:
        """Calculate valuation multiples for all companies."""
        multiples_data = []
        
        for company in companies:
            row = {
                "symbol": company.symbol,
                "name": company.name,
                "market_cap": company.market_cap,
                "enterprise_value": company.enterprise_value,
                "revenue": company.revenue,
                "ebitda": company.ebitda,
                "sector": company.sector,
            }
            
            # Calculate each multiple
            for multiple in multiples:
                value = self._calculate_single_multiple(company, multiple)
                row[multiple.value] = value
            
            multiples_data.append(row)
        
        return multiples_data
    
    def _calculate_single_multiple(self, company: CompanyData, multiple: Multiple) -> Optional[float]:
        """Calculate a single valuation multiple."""
        try:
            if multiple == Multiple.EV_REVENUE:
                return company.enterprise_value / company.revenue if company.revenue > 0 else None
            
            elif multiple == Multiple.EV_EBITDA:
                return company.enterprise_value / company.ebitda if company.ebitda > 0 else None
            
            elif multiple == Multiple.EV_EBIT:
                return company.enterprise_value / company.ebit if company.ebit > 0 else None
            
            elif multiple == Multiple.PE_RATIO:
                eps = company.net_income / company.shares_outstanding
                price_per_share = company.market_cap / company.shares_outstanding
                return price_per_share / eps if eps > 0 else None
            
            elif multiple == Multiple.PB_RATIO:
                return company.market_cap / company.book_value if company.book_value > 0 else None
            
            elif multiple == Multiple.PS_RATIO:
                return company.market_cap / company.revenue if company.revenue > 0 else None
            
            elif multiple == Multiple.EV_FCF:
                return company.enterprise_value / company.free_cash_flow if company.free_cash_flow > 0 else None
            
            elif multiple == Multiple.PEG_RATIO:
                pe_ratio = self._calculate_single_multiple(company, Multiple.PE_RATIO)
                if pe_ratio and company.earnings_growth and company.earnings_growth > 0:
                    return pe_ratio / (company.earnings_growth * 100)
                return None
            
            else:
                return None
                
        except (ZeroDivisionError, TypeError):
            return None
    
    def _winsorize_multiples(
        self,
        multiples_df: pd.DataFrame,
        multiples: List[Multiple],
    ) -> pd.DataFrame:
        """Winsorize multiples to remove outliers."""
        winsorized_df = multiples_df.copy()
        
        for multiple in multiples:
            column = multiple.value
            if column in winsorized_df.columns:
                values = winsorized_df[column].dropna()
                if len(values) > 2:
                    # Calculate winsorization bounds
                    lower_bound = values.quantile(self.winsorization_percentile)
                    upper_bound = values.quantile(1 - self.winsorization_percentile)
                    
                    # Apply winsorization
                    winsorized_df[column] = winsorized_df[column].clip(
                        lower=lower_bound, 
                        upper=upper_bound
                    )
        
        return winsorized_df
    
    def _calculate_valuation_summary(
        self,
        multiples_df: pd.DataFrame,
        multiples: List[Multiple],
    ) -> Dict[str, Dict[str, float]]:
        """Calculate valuation summary statistics."""
        summary = {}
        
        for multiple in multiples:
            column = multiple.value
            if column in multiples_df.columns:
                values = multiples_df[column].dropna()
                
                if len(values) > 0:
                    summary[multiple.value] = {
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "median": float(values.median()),
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "count": len(values),
                        "25th_percentile": float(values.quantile(0.25)),
                        "75th_percentile": float(values.quantile(0.75)),
                    }
        
        return summary
    
    def _calculate_statistics(
        self,
        multiples_df: pd.DataFrame,
        multiples: List[Multiple],
    ) -> Dict[str, Any]:
        """Calculate statistical analysis of multiples."""
        statistics = {}
        
        # Correlation matrix
        multiple_columns = [m.value for m in multiples if m.value in multiples_df.columns]
        if len(multiple_columns) > 1:
            correlation_matrix = multiples_df[multiple_columns].corr()
            statistics["correlation_matrix"] = correlation_matrix.to_dict()
        
        # Distribution tests
        for multiple in multiples:
            column = multiple.value
            if column in multiples_df.columns:
                values = multiples_df[column].dropna()
                
                if len(values) >= 3:
                    # Shapiro-Wilk test for normality
                    try:
                        shapiro_stat, shapiro_p = stats.shapiro(values)
                        statistics[f"{column}_normality_test"] = {
                            "statistic": float(shapiro_stat),
                            "p_value": float(shapiro_p),
                            "is_normal": shapiro_p > 0.05,
                        }
                    except Exception:
                        pass
        
        return statistics
    
    @trace_function("comps_engine.value_target_company")
    def value_target_company(
        self,
        target_company: CompanyData,
        comps_result: CompsResult,
        multiple: Multiple,
        method: str = "median",  # median, mean, 25th, 75th
    ) -> Dict[str, float]:
        """Value target company using peer multiples."""
        if multiple.value not in comps_result.valuation_summary:
            raise ValueError(f"Multiple {multiple.value} not available in comps result")
        
        summary = comps_result.valuation_summary[multiple.value]
        
        # Get the multiple value to apply
        if method == "median":
            peer_multiple = summary["median"]
        elif method == "mean":
            peer_multiple = summary["mean"]
        elif method == "25th":
            peer_multiple = summary["25th_percentile"]
        elif method == "75th":
            peer_multiple = summary["75th_percentile"]
        else:
            raise ValueError(f"Unknown valuation method: {method}")
        
        # Calculate implied valuation
        if multiple == Multiple.EV_REVENUE:
            implied_ev = target_company.revenue * peer_multiple
            implied_equity_value = implied_ev - (target_company.enterprise_value - target_company.market_cap)
            
        elif multiple == Multiple.EV_EBITDA:
            implied_ev = target_company.ebitda * peer_multiple
            implied_equity_value = implied_ev - (target_company.enterprise_value - target_company.market_cap)
            
        elif multiple == Multiple.PS_RATIO:
            implied_equity_value = target_company.revenue * peer_multiple
            implied_ev = implied_equity_value + (target_company.enterprise_value - target_company.market_cap)
            
        elif multiple == Multiple.PE_RATIO:
            implied_equity_value = target_company.net_income * peer_multiple
            implied_ev = implied_equity_value + (target_company.enterprise_value - target_company.market_cap)
            
        else:
            # Add other multiples as needed
            raise ValueError(f"Valuation not implemented for multiple: {multiple.value}")
        
        # Calculate per share values
        implied_share_price = implied_equity_value / target_company.shares_outstanding
        current_share_price = target_company.market_cap / target_company.shares_outstanding
        
        return {
            "multiple_used": peer_multiple,
            "implied_enterprise_value": implied_ev,
            "implied_equity_value": implied_equity_value,
            "implied_share_price": implied_share_price,
            "current_share_price": current_share_price,
            "upside_downside": (implied_share_price - current_share_price) / current_share_price,
            "method": method,
        }
    
    @trace_function("comps_engine.create_football_field")
    def create_football_field(
        self,
        target_company: CompanyData,
        comps_result: CompsResult,
        multiples: List[Multiple],
    ) -> pd.DataFrame:
        """Create football field chart data showing valuation ranges."""
        football_field_data = []
        
        for multiple in multiples:
            if multiple.value not in comps_result.valuation_summary:
                continue
            
            # Calculate valuations using different percentiles
            methods = ["25th", "median", "75th"]
            valuations = {}
            
            for method in methods:
                try:
                    valuation = self.value_target_company(
                        target_company, comps_result, multiple, method
                    )
                    valuations[method] = valuation["implied_share_price"]
                except Exception as e:
                    print(f"Failed to calculate {method} valuation for {multiple.value}: {e}")
                    continue
            
            if len(valuations) >= 2:
                football_field_data.append({
                    "multiple": multiple.value,
                    "low": min(valuations.values()),
                    "mid": valuations.get("median", np.mean(list(valuations.values()))),
                    "high": max(valuations.values()),
                })
        
        return pd.DataFrame(football_field_data)
