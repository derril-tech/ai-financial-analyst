"""Portfolio API for batch ticker analysis and portfolio management reports."""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures

import pandas as pd
import numpy as np
from scipy import stats
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.observability import trace_function
from app.services.valuation.dcf import DCFValuationService
from app.services.valuation.comps import ComparableCompanyService
from app.services.analytics.risk import RiskAnalyticsService
from app.services.market_data import MarketDataService


class PortfolioType(Enum):
    """Types of portfolios."""
    EQUITY_LONG_ONLY = "equity_long_only"
    EQUITY_LONG_SHORT = "equity_long_short"
    SECTOR_FOCUSED = "sector_focused"
    GROWTH_FOCUSED = "growth_focused"
    VALUE_FOCUSED = "value_focused"
    DIVIDEND_FOCUSED = "dividend_focused"
    ESG_FOCUSED = "esg_focused"
    FACTOR_BASED = "factor_based"


class ReportType(Enum):
    """Types of portfolio reports."""
    PERFORMANCE_ATTRIBUTION = "performance_attribution"
    RISK_ANALYSIS = "risk_analysis"
    SECTOR_ALLOCATION = "sector_allocation"
    VALUATION_SUMMARY = "valuation_summary"
    ESG_SCORECARD = "esg_scorecard"
    FACTOR_EXPOSURE = "factor_exposure"
    PEER_COMPARISON = "peer_comparison"
    STRESS_TEST = "stress_test"


@dataclass
class PortfolioHolding:
    """Individual portfolio holding."""
    ticker: str
    company_name: str
    shares: float
    market_value: float
    weight: float  # Portfolio weight
    
    # Pricing
    current_price: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    
    # Fundamentals
    market_cap: float
    sector: str
    industry: str
    
    # Valuation metrics
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    dividend_yield: Optional[float] = None
    
    # Risk metrics
    beta: Optional[float] = None
    volatility: Optional[float] = None
    
    # ESG scores
    esg_score: Optional[float] = None
    
    # Analysis results
    dcf_valuation: Optional[float] = None
    target_price: Optional[float] = None
    recommendation: Optional[str] = None


@dataclass
class PortfolioMetrics:
    """Portfolio-level metrics."""
    total_value: float
    total_cost: float
    total_pnl: float
    total_pnl_pct: float
    
    # Performance
    daily_return: float
    mtd_return: float
    qtd_return: float
    ytd_return: float
    
    # Risk
    portfolio_beta: float
    portfolio_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk
    
    # Allocation
    sector_allocation: Dict[str, float]
    market_cap_allocation: Dict[str, float]  # Large, Mid, Small cap
    
    # Concentration
    top_10_weight: float
    herfindahl_index: float  # Concentration measure
    
    # Factor exposures
    factor_exposures: Dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioReport:
    """Complete portfolio analysis report."""
    portfolio_id: str
    report_type: ReportType
    generated_at: datetime
    
    # Portfolio info
    portfolio_name: str
    portfolio_type: PortfolioType
    benchmark: str
    
    # Holdings and metrics
    holdings: List[PortfolioHolding]
    metrics: PortfolioMetrics
    
    # Analysis sections
    executive_summary: str
    key_insights: List[str]
    recommendations: List[str]
    
    # Detailed analysis
    performance_analysis: Dict[str, Any] = field(default_factory=dict)
    risk_analysis: Dict[str, Any] = field(default_factory=dict)
    attribution_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Charts and visualizations
    charts: List[Dict[str, Any]] = field(default_factory=list)


class PortfolioAnalyzer:
    """Core portfolio analysis engine."""
    
    def __init__(self) -> None:
        """Initialize portfolio analyzer."""
        self.market_data_service = MarketDataService()
        self.dcf_service = DCFValuationService()
        self.comps_service = ComparableCompanyService()
        self.risk_service = RiskAnalyticsService()
    
    @trace_function("portfolio_analyzer.analyze_holdings")
    async def analyze_holdings(
        self,
        db: AsyncSession,
        tickers: List[str],
        weights: Optional[List[float]] = None,
    ) -> List[PortfolioHolding]:
        """Analyze individual portfolio holdings."""
        if weights is None:
            weights = [1.0 / len(tickers)] * len(tickers)
        
        holdings = []
        
        # Process holdings in parallel
        tasks = []
        for ticker, weight in zip(tickers, weights):
            task = self._analyze_single_holding(db, ticker, weight)
            tasks.append(task)
        
        holding_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in holding_results:
            if isinstance(result, Exception):
                print(f"Error analyzing holding: {result}")
            else:
                holdings.append(result)
        
        return holdings
    
    async def _analyze_single_holding(
        self,
        db: AsyncSession,
        ticker: str,
        weight: float,
    ) -> PortfolioHolding:
        """Analyze a single portfolio holding."""
        # Get market data
        market_data = await self.market_data_service.get_current_data(ticker)
        
        # Get fundamental data
        fundamentals = await self.market_data_service.get_fundamentals(ticker)
        
        # Calculate valuation metrics
        dcf_result = await self.dcf_service.calculate_dcf_valuation(
            db, ticker, fundamentals
        )
        
        # Get risk metrics
        risk_metrics = await self.risk_service.calculate_stock_risk_metrics(
            ticker, market_data.get("price_history", [])
        )
        
        # Mock some values for demonstration
        shares = 100  # Would come from portfolio data
        current_price = market_data.get("price", 100.0)
        cost_basis = current_price * 0.9  # Mock 10% gain
        
        market_value = shares * current_price
        unrealized_pnl = shares * (current_price - cost_basis)
        unrealized_pnl_pct = (current_price - cost_basis) / cost_basis * 100
        
        holding = PortfolioHolding(
            ticker=ticker,
            company_name=fundamentals.get("company_name", ticker),
            shares=shares,
            market_value=market_value,
            weight=weight,
            current_price=current_price,
            cost_basis=cost_basis,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            market_cap=fundamentals.get("market_cap", 0),
            sector=fundamentals.get("sector", "Unknown"),
            industry=fundamentals.get("industry", "Unknown"),
            pe_ratio=fundamentals.get("pe_ratio"),
            pb_ratio=fundamentals.get("pb_ratio"),
            ev_ebitda=fundamentals.get("ev_ebitda"),
            dividend_yield=fundamentals.get("dividend_yield"),
            beta=risk_metrics.get("beta"),
            volatility=risk_metrics.get("volatility"),
            esg_score=fundamentals.get("esg_score"),
            dcf_valuation=dcf_result.get("fair_value"),
            target_price=dcf_result.get("target_price"),
            recommendation=dcf_result.get("recommendation"),
        )
        
        return holding
    
    @trace_function("portfolio_analyzer.calculate_portfolio_metrics")
    def calculate_portfolio_metrics(
        self,
        holdings: List[PortfolioHolding],
        benchmark_data: Optional[Dict[str, Any]] = None,
    ) -> PortfolioMetrics:
        """Calculate portfolio-level metrics."""
        if not holdings:
            return self._empty_portfolio_metrics()
        
        # Basic portfolio values
        total_value = sum(h.market_value for h in holdings)
        total_cost = sum(h.shares * h.cost_basis for h in holdings)
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        # Portfolio weights (normalize)
        weights = np.array([h.weight for h in holdings])
        weights = weights / weights.sum()  # Ensure weights sum to 1
        
        # Risk metrics
        portfolio_beta = self._calculate_portfolio_beta(holdings, weights)
        portfolio_volatility = self._calculate_portfolio_volatility(holdings, weights)
        
        # Performance metrics (mock for now)
        daily_return = 0.5  # Would calculate from price history
        mtd_return = 2.1
        qtd_return = 8.5
        ytd_return = 15.2
        
        sharpe_ratio = self._calculate_sharpe_ratio(ytd_return, portfolio_volatility)
        max_drawdown = self._calculate_max_drawdown(holdings)
        var_95 = self._calculate_var(holdings, weights)
        
        # Allocation analysis
        sector_allocation = self._calculate_sector_allocation(holdings)
        market_cap_allocation = self._calculate_market_cap_allocation(holdings)
        
        # Concentration metrics
        top_10_weight = sum(sorted([h.weight for h in holdings], reverse=True)[:10])
        herfindahl_index = sum(w**2 for w in weights)
        
        # Factor exposures
        factor_exposures = self._calculate_factor_exposures(holdings, weights)
        
        return PortfolioMetrics(
            total_value=total_value,
            total_cost=total_cost,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            daily_return=daily_return,
            mtd_return=mtd_return,
            qtd_return=qtd_return,
            ytd_return=ytd_return,
            portfolio_beta=portfolio_beta,
            portfolio_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            sector_allocation=sector_allocation,
            market_cap_allocation=market_cap_allocation,
            top_10_weight=top_10_weight,
            herfindahl_index=herfindahl_index,
            factor_exposures=factor_exposures,
        )
    
    def _empty_portfolio_metrics(self) -> PortfolioMetrics:
        """Return empty portfolio metrics."""
        return PortfolioMetrics(
            total_value=0,
            total_cost=0,
            total_pnl=0,
            total_pnl_pct=0,
            daily_return=0,
            mtd_return=0,
            qtd_return=0,
            ytd_return=0,
            portfolio_beta=0,
            portfolio_volatility=0,
            sharpe_ratio=0,
            max_drawdown=0,
            var_95=0,
            sector_allocation={},
            market_cap_allocation={},
            top_10_weight=0,
            herfindahl_index=0,
        )
    
    def _calculate_portfolio_beta(self, holdings: List[PortfolioHolding], weights: np.ndarray) -> float:
        """Calculate portfolio beta."""
        betas = np.array([h.beta or 1.0 for h in holdings])
        return np.sum(weights * betas)
    
    def _calculate_portfolio_volatility(self, holdings: List[PortfolioHolding], weights: np.ndarray) -> float:
        """Calculate portfolio volatility."""
        # Simplified calculation - in production, use covariance matrix
        volatilities = np.array([h.volatility or 0.2 for h in holdings])
        
        # Assume average correlation of 0.3
        avg_correlation = 0.3
        
        # Portfolio variance
        individual_var = np.sum((weights * volatilities)**2)
        correlation_var = 2 * avg_correlation * np.sum(
            np.outer(weights * volatilities, weights * volatilities)[np.triu_indices_from(np.outer(weights, weights), k=1)]
        )
        
        portfolio_variance = individual_var + correlation_var
        return np.sqrt(portfolio_variance)
    
    def _calculate_sharpe_ratio(self, annual_return: float, volatility: float, risk_free_rate: float = 4.5) -> float:
        """Calculate Sharpe ratio."""
        if volatility == 0:
            return 0
        return (annual_return - risk_free_rate) / volatility
    
    def _calculate_max_drawdown(self, holdings: List[PortfolioHolding]) -> float:
        """Calculate maximum drawdown."""
        # Simplified calculation - would need historical portfolio values
        unrealized_losses = [h.unrealized_pnl_pct for h in holdings if h.unrealized_pnl_pct < 0]
        
        if unrealized_losses:
            return min(unrealized_losses)
        return 0.0
    
    def _calculate_var(self, holdings: List[PortfolioHolding], weights: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        # Simplified VaR calculation
        portfolio_volatility = self._calculate_portfolio_volatility(holdings, weights)
        z_score = stats.norm.ppf(1 - confidence)
        
        total_value = sum(h.market_value for h in holdings)
        var = total_value * portfolio_volatility * z_score / np.sqrt(252)  # Daily VaR
        
        return abs(var)
    
    def _calculate_sector_allocation(self, holdings: List[PortfolioHolding]) -> Dict[str, float]:
        """Calculate sector allocation."""
        sector_weights = {}
        total_weight = sum(h.weight for h in holdings)
        
        for holding in holdings:
            sector = holding.sector
            if sector not in sector_weights:
                sector_weights[sector] = 0
            sector_weights[sector] += holding.weight / total_weight * 100
        
        return sector_weights
    
    def _calculate_market_cap_allocation(self, holdings: List[PortfolioHolding]) -> Dict[str, float]:
        """Calculate market cap allocation."""
        cap_weights = {"Large Cap": 0, "Mid Cap": 0, "Small Cap": 0}
        total_weight = sum(h.weight for h in holdings)
        
        for holding in holdings:
            market_cap = holding.market_cap
            
            if market_cap > 10e9:  # > $10B
                cap_category = "Large Cap"
            elif market_cap > 2e9:  # > $2B
                cap_category = "Mid Cap"
            else:
                cap_category = "Small Cap"
            
            cap_weights[cap_category] += holding.weight / total_weight * 100
        
        return cap_weights
    
    def _calculate_factor_exposures(self, holdings: List[PortfolioHolding], weights: np.ndarray) -> Dict[str, float]:
        """Calculate factor exposures."""
        # Simplified factor calculation
        exposures = {}
        
        # Value factor (based on P/E ratios)
        pe_ratios = np.array([h.pe_ratio or 15.0 for h in holdings])
        value_score = np.sum(weights * (1 / pe_ratios))  # Lower P/E = higher value
        exposures["Value"] = value_score
        
        # Growth factor (mock calculation)
        growth_scores = np.array([np.random.normal(0, 1) for _ in holdings])  # Mock growth scores
        exposures["Growth"] = np.sum(weights * growth_scores)
        
        # Quality factor (based on various metrics)
        quality_scores = np.array([
            (h.pe_ratio or 15) / 15 + (h.pb_ratio or 2) / 2  # Simplified quality
            for h in holdings
        ])
        exposures["Quality"] = np.sum(weights * quality_scores)
        
        # Size factor (market cap)
        log_market_caps = np.array([np.log(h.market_cap) if h.market_cap > 0 else 20 for h in holdings])
        exposures["Size"] = np.sum(weights * log_market_caps)
        
        return exposures


class PortfolioReportGenerator:
    """Generate comprehensive portfolio reports."""
    
    def __init__(self) -> None:
        """Initialize report generator."""
        self.analyzer = PortfolioAnalyzer()
    
    @trace_function("report_generator.generate_report")
    async def generate_report(
        self,
        db: AsyncSession,
        portfolio_id: str,
        portfolio_name: str,
        tickers: List[str],
        weights: Optional[List[float]] = None,
        report_type: ReportType = ReportType.PERFORMANCE_ATTRIBUTION,
        benchmark: str = "SPY",
    ) -> PortfolioReport:
        """Generate comprehensive portfolio report."""
        # Analyze holdings
        holdings = await self.analyzer.analyze_holdings(db, tickers, weights)
        
        # Calculate portfolio metrics
        metrics = self.analyzer.calculate_portfolio_metrics(holdings)
        
        # Generate report sections based on type
        if report_type == ReportType.PERFORMANCE_ATTRIBUTION:
            analysis_sections = await self._generate_performance_attribution(holdings, metrics)
        elif report_type == ReportType.RISK_ANALYSIS:
            analysis_sections = await self._generate_risk_analysis(holdings, metrics)
        elif report_type == ReportType.VALUATION_SUMMARY:
            analysis_sections = await self._generate_valuation_summary(holdings, metrics)
        else:
            analysis_sections = await self._generate_default_analysis(holdings, metrics)
        
        # Generate executive summary and insights
        executive_summary = self._generate_executive_summary(metrics, analysis_sections)
        key_insights = self._generate_key_insights(holdings, metrics)
        recommendations = self._generate_recommendations(holdings, metrics)
        
        # Generate charts
        charts = self._generate_charts(holdings, metrics)
        
        report = PortfolioReport(
            portfolio_id=portfolio_id,
            report_type=report_type,
            generated_at=datetime.utcnow(),
            portfolio_name=portfolio_name,
            portfolio_type=PortfolioType.EQUITY_LONG_ONLY,  # Default
            benchmark=benchmark,
            holdings=holdings,
            metrics=metrics,
            executive_summary=executive_summary,
            key_insights=key_insights,
            recommendations=recommendations,
            performance_analysis=analysis_sections.get("performance", {}),
            risk_analysis=analysis_sections.get("risk", {}),
            attribution_analysis=analysis_sections.get("attribution", {}),
            charts=charts,
        )
        
        return report
    
    async def _generate_performance_attribution(
        self,
        holdings: List[PortfolioHolding],
        metrics: PortfolioMetrics,
    ) -> Dict[str, Any]:
        """Generate performance attribution analysis."""
        # Sector attribution
        sector_returns = {}
        for holding in holdings:
            sector = holding.sector
            if sector not in sector_returns:
                sector_returns[sector] = []
            sector_returns[sector].append(holding.unrealized_pnl_pct * holding.weight)
        
        sector_attribution = {
            sector: sum(returns) for sector, returns in sector_returns.items()
        }
        
        # Stock selection vs allocation effects
        stock_selection_effect = sum(
            h.unrealized_pnl_pct * h.weight for h in holdings
        ) / 100  # Convert to decimal
        
        return {
            "performance": {
                "sector_attribution": sector_attribution,
                "stock_selection_effect": stock_selection_effect,
                "allocation_effect": 0.02,  # Mock allocation effect
                "interaction_effect": 0.005,  # Mock interaction effect
                "total_excess_return": stock_selection_effect + 0.02 + 0.005,
            }
        }
    
    async def _generate_risk_analysis(
        self,
        holdings: List[PortfolioHolding],
        metrics: PortfolioMetrics,
    ) -> Dict[str, Any]:
        """Generate risk analysis."""
        # Concentration risk
        concentration_risk = {
            "top_10_concentration": metrics.top_10_weight,
            "herfindahl_index": metrics.herfindahl_index,
            "concentration_level": "High" if metrics.herfindahl_index > 0.15 else "Moderate" if metrics.herfindahl_index > 0.10 else "Low",
        }
        
        # Sector risk
        max_sector_weight = max(metrics.sector_allocation.values()) if metrics.sector_allocation else 0
        sector_risk = {
            "max_sector_weight": max_sector_weight,
            "sector_concentration": "High" if max_sector_weight > 30 else "Moderate" if max_sector_weight > 20 else "Low",
            "sector_diversification": len(metrics.sector_allocation),
        }
        
        # Market risk
        market_risk = {
            "portfolio_beta": metrics.portfolio_beta,
            "market_sensitivity": "High" if metrics.portfolio_beta > 1.2 else "Moderate" if metrics.portfolio_beta > 0.8 else "Low",
            "systematic_risk": metrics.portfolio_beta * 0.16,  # Assuming market vol of 16%
        }
        
        return {
            "risk": {
                "concentration_risk": concentration_risk,
                "sector_risk": sector_risk,
                "market_risk": market_risk,
                "var_analysis": {
                    "daily_var_95": metrics.var_95,
                    "monthly_var_95": metrics.var_95 * np.sqrt(21),
                    "annual_var_95": metrics.var_95 * np.sqrt(252),
                },
            }
        }
    
    async def _generate_valuation_summary(
        self,
        holdings: List[PortfolioHolding],
        metrics: PortfolioMetrics,
    ) -> Dict[str, Any]:
        """Generate valuation summary."""
        # Portfolio valuation metrics
        weighted_pe = sum(
            (h.pe_ratio or 0) * h.weight for h in holdings if h.pe_ratio
        ) / sum(h.weight for h in holdings if h.pe_ratio)
        
        weighted_pb = sum(
            (h.pb_ratio or 0) * h.weight for h in holdings if h.pb_ratio
        ) / sum(h.weight for h in holdings if h.pb_ratio)
        
        # DCF analysis
        dcf_upside = []
        for holding in holdings:
            if holding.dcf_valuation and holding.current_price:
                upside = (holding.dcf_valuation - holding.current_price) / holding.current_price * 100
                dcf_upside.append(upside * holding.weight)
        
        portfolio_dcf_upside = sum(dcf_upside) if dcf_upside else 0
        
        return {
            "valuation": {
                "portfolio_pe": weighted_pe,
                "portfolio_pb": weighted_pb,
                "dcf_upside": portfolio_dcf_upside,
                "overvalued_positions": len([h for h in holdings if h.dcf_valuation and h.current_price > h.dcf_valuation]),
                "undervalued_positions": len([h for h in holdings if h.dcf_valuation and h.current_price < h.dcf_valuation]),
            }
        }
    
    async def _generate_default_analysis(
        self,
        holdings: List[PortfolioHolding],
        metrics: PortfolioMetrics,
    ) -> Dict[str, Any]:
        """Generate default analysis sections."""
        performance = await self._generate_performance_attribution(holdings, metrics)
        risk = await self._generate_risk_analysis(holdings, metrics)
        valuation = await self._generate_valuation_summary(holdings, metrics)
        
        return {**performance, **risk, **valuation}
    
    def _generate_executive_summary(
        self,
        metrics: PortfolioMetrics,
        analysis_sections: Dict[str, Any],
    ) -> str:
        """Generate executive summary."""
        return_performance = "strong" if metrics.ytd_return > 10 else "moderate" if metrics.ytd_return > 0 else "weak"
        risk_level = "high" if metrics.portfolio_volatility > 0.25 else "moderate" if metrics.portfolio_volatility > 0.15 else "low"
        
        summary = (
            f"Portfolio delivered {return_performance} performance with {metrics.ytd_return:.1f}% YTD returns. "
            f"Risk profile is {risk_level} with {metrics.portfolio_volatility*100:.1f}% volatility and "
            f"{metrics.portfolio_beta:.2f} beta. Portfolio is diversified across {len(metrics.sector_allocation)} "
            f"sectors with {metrics.top_10_weight:.1f}% concentration in top 10 holdings."
        )
        
        return summary
    
    def _generate_key_insights(
        self,
        holdings: List[PortfolioHolding],
        metrics: PortfolioMetrics,
    ) -> List[str]:
        """Generate key insights."""
        insights = []
        
        # Performance insights
        if metrics.ytd_return > 15:
            insights.append(f"Strong YTD performance of {metrics.ytd_return:.1f}% outpacing market averages")
        
        # Risk insights
        if metrics.portfolio_beta > 1.2:
            insights.append("High beta portfolio with elevated market sensitivity")
        elif metrics.portfolio_beta < 0.8:
            insights.append("Defensive portfolio with below-market volatility")
        
        # Concentration insights
        if metrics.herfindahl_index > 0.15:
            insights.append("High concentration risk - consider diversification")
        
        # Sector insights
        if metrics.sector_allocation:
            max_sector = max(metrics.sector_allocation.items(), key=lambda x: x[1])
            if max_sector[1] > 30:
                insights.append(f"Heavy allocation to {max_sector[0]} sector ({max_sector[1]:.1f}%)")
        
        # Valuation insights
        overvalued_count = len([h for h in holdings if h.dcf_valuation and h.current_price > h.dcf_valuation * 1.1])
        if overvalued_count > len(holdings) * 0.3:
            insights.append("Significant portion of holdings appear overvalued based on DCF analysis")
        
        return insights[:5]  # Limit to top 5 insights
    
    def _generate_recommendations(
        self,
        holdings: List[PortfolioHolding],
        metrics: PortfolioMetrics,
    ) -> List[str]:
        """Generate portfolio recommendations."""
        recommendations = []
        
        # Concentration recommendations
        if metrics.herfindahl_index > 0.15:
            recommendations.append("Reduce concentration by trimming largest positions")
        
        # Sector recommendations
        if metrics.sector_allocation:
            max_sector_weight = max(metrics.sector_allocation.values())
            if max_sector_weight > 35:
                recommendations.append("Rebalance sector allocation to reduce concentration risk")
        
        # Risk recommendations
        if metrics.portfolio_beta > 1.3:
            recommendations.append("Consider adding defensive positions to reduce portfolio beta")
        
        # Performance recommendations
        if metrics.sharpe_ratio < 0.5:
            recommendations.append("Focus on risk-adjusted returns - current Sharpe ratio is suboptimal")
        
        # Valuation recommendations
        overvalued_holdings = [h for h in holdings if h.dcf_valuation and h.current_price > h.dcf_valuation * 1.2]
        if overvalued_holdings:
            top_overvalued = sorted(overvalued_holdings, key=lambda x: x.weight, reverse=True)[:3]
            tickers = [h.ticker for h in top_overvalued]
            recommendations.append(f"Consider reducing positions in overvalued holdings: {', '.join(tickers)}")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _generate_charts(
        self,
        holdings: List[PortfolioHolding],
        metrics: PortfolioMetrics,
    ) -> List[Dict[str, Any]]:
        """Generate chart specifications for visualization."""
        charts = []
        
        # Sector allocation pie chart
        if metrics.sector_allocation:
            charts.append({
                "type": "pie",
                "title": "Sector Allocation",
                "data": metrics.sector_allocation,
            })
        
        # Top holdings bar chart
        top_holdings = sorted(holdings, key=lambda x: x.weight, reverse=True)[:10]
        charts.append({
            "type": "bar",
            "title": "Top 10 Holdings",
            "data": {h.ticker: h.weight for h in top_holdings},
        })
        
        # Risk-return scatter plot
        charts.append({
            "type": "scatter",
            "title": "Risk-Return Profile",
            "data": {
                "x": [h.volatility or 0.2 for h in holdings],
                "y": [h.unrealized_pnl_pct for h in holdings],
                "labels": [h.ticker for h in holdings],
            },
        })
        
        # Performance attribution waterfall
        if metrics.sector_allocation:
            charts.append({
                "type": "waterfall",
                "title": "Sector Performance Attribution",
                "data": {
                    sector: weight * 0.1  # Mock attribution
                    for sector, weight in metrics.sector_allocation.items()
                },
            })
        
        return charts


class PortfolioAPIService:
    """Main Portfolio API service for batch analysis and reporting."""
    
    def __init__(self) -> None:
        """Initialize Portfolio API service."""
        self.report_generator = PortfolioReportGenerator()
        self.analyzer = PortfolioAnalyzer()
    
    @trace_function("portfolio_api.batch_analyze_tickers")
    async def batch_analyze_tickers(
        self,
        db: AsyncSession,
        tickers: List[str],
        analysis_types: List[str] = None,
    ) -> Dict[str, Any]:
        """Perform batch analysis on multiple tickers."""
        if analysis_types is None:
            analysis_types = ["valuation", "risk", "fundamentals"]
        
        results = {}
        
        # Process tickers in parallel batches
        batch_size = 10
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            
            tasks = []
            for ticker in batch_tickers:
                task = self._analyze_single_ticker(db, ticker, analysis_types)
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for ticker, result in zip(batch_tickers, batch_results):
                if isinstance(result, Exception):
                    results[ticker] = {"error": str(result)}
                else:
                    results[ticker] = result
        
        return {
            "analysis_results": results,
            "summary": self._generate_batch_summary(results),
            "generated_at": datetime.utcnow().isoformat(),
        }
    
    async def _analyze_single_ticker(
        self,
        db: AsyncSession,
        ticker: str,
        analysis_types: List[str],
    ) -> Dict[str, Any]:
        """Analyze a single ticker."""
        result = {"ticker": ticker}
        
        # Get market data
        market_data_service = MarketDataService()
        market_data = await market_data_service.get_current_data(ticker)
        fundamentals = await market_data_service.get_fundamentals(ticker)
        
        # Valuation analysis
        if "valuation" in analysis_types:
            dcf_service = DCFValuationService()
            dcf_result = await dcf_service.calculate_dcf_valuation(db, ticker, fundamentals)
            result["valuation"] = dcf_result
        
        # Risk analysis
        if "risk" in analysis_types:
            risk_service = RiskAnalyticsService()
            risk_metrics = await risk_service.calculate_stock_risk_metrics(
                ticker, market_data.get("price_history", [])
            )
            result["risk"] = risk_metrics
        
        # Fundamental analysis
        if "fundamentals" in analysis_types:
            result["fundamentals"] = fundamentals
        
        return result
    
    def _generate_batch_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of batch analysis results."""
        successful_analyses = {k: v for k, v in results.items() if "error" not in v}
        failed_analyses = {k: v for k, v in results.items() if "error" in v}
        
        # Aggregate metrics
        valuations = []
        for ticker, result in successful_analyses.items():
            if "valuation" in result and "fair_value" in result["valuation"]:
                valuations.append(result["valuation"]["fair_value"])
        
        summary = {
            "total_tickers": len(results),
            "successful_analyses": len(successful_analyses),
            "failed_analyses": len(failed_analyses),
            "average_valuation": np.mean(valuations) if valuations else 0,
            "valuation_range": {
                "min": min(valuations) if valuations else 0,
                "max": max(valuations) if valuations else 0,
            },
        }
        
        if failed_analyses:
            summary["failed_tickers"] = list(failed_analyses.keys())
        
        return summary
    
    @trace_function("portfolio_api.generate_pm_report")
    async def generate_pm_report(
        self,
        db: AsyncSession,
        portfolio_config: Dict[str, Any],
    ) -> PortfolioReport:
        """Generate portfolio manager report."""
        portfolio_id = portfolio_config.get("portfolio_id", "default")
        portfolio_name = portfolio_config.get("name", "Portfolio")
        tickers = portfolio_config["tickers"]
        weights = portfolio_config.get("weights")
        report_type = ReportType(portfolio_config.get("report_type", "performance_attribution"))
        benchmark = portfolio_config.get("benchmark", "SPY")
        
        report = await self.report_generator.generate_report(
            db=db,
            portfolio_id=portfolio_id,
            portfolio_name=portfolio_name,
            tickers=tickers,
            weights=weights,
            report_type=report_type,
            benchmark=benchmark,
        )
        
        return report
    
    @trace_function("portfolio_api.compare_portfolios")
    async def compare_portfolios(
        self,
        db: AsyncSession,
        portfolio_configs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare multiple portfolios."""
        portfolio_reports = []
        
        for config in portfolio_configs:
            report = await self.generate_pm_report(db, config)
            portfolio_reports.append(report)
        
        # Generate comparison analysis
        comparison = self._generate_portfolio_comparison(portfolio_reports)
        
        return {
            "portfolios": [
                {
                    "name": report.portfolio_name,
                    "total_return": report.metrics.ytd_return,
                    "volatility": report.metrics.portfolio_volatility,
                    "sharpe_ratio": report.metrics.sharpe_ratio,
                    "max_drawdown": report.metrics.max_drawdown,
                }
                for report in portfolio_reports
            ],
            "comparison_analysis": comparison,
            "generated_at": datetime.utcnow().isoformat(),
        }
    
    def _generate_portfolio_comparison(self, reports: List[PortfolioReport]) -> Dict[str, Any]:
        """Generate portfolio comparison analysis."""
        if not reports:
            return {}
        
        # Performance comparison
        returns = [r.metrics.ytd_return for r in reports]
        volatilities = [r.metrics.portfolio_volatility for r in reports]
        sharpe_ratios = [r.metrics.sharpe_ratio for r in reports]
        
        # Risk-adjusted performance ranking
        performance_ranking = sorted(
            enumerate(reports),
            key=lambda x: x[1].metrics.sharpe_ratio,
            reverse=True
        )
        
        # Sector allocation comparison
        all_sectors = set()
        for report in reports:
            all_sectors.update(report.metrics.sector_allocation.keys())
        
        sector_comparison = {}
        for sector in all_sectors:
            sector_comparison[sector] = [
                report.metrics.sector_allocation.get(sector, 0)
                for report in reports
            ]
        
        return {
            "performance_summary": {
                "best_performer": reports[performance_ranking[0][0]].portfolio_name,
                "worst_performer": reports[performance_ranking[-1][0]].portfolio_name,
                "avg_return": np.mean(returns),
                "avg_volatility": np.mean(volatilities),
                "avg_sharpe": np.mean(sharpe_ratios),
            },
            "risk_analysis": {
                "lowest_risk": min(volatilities),
                "highest_risk": max(volatilities),
                "risk_spread": max(volatilities) - min(volatilities),
            },
            "sector_analysis": sector_comparison,
            "recommendations": [
                "Focus on risk-adjusted returns rather than absolute performance",
                "Consider portfolio diversification benefits",
                "Monitor correlation between portfolios",
            ],
        }
