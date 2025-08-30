"""Advanced features endpoints for backlog capabilities."""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional

from app.core.database import get_db
from app.core.observability import trace_function
from app.services.guidance_tracker import GuidanceConsistencyService
from app.services.footnote_time_machine import FootnoteTimeMachineService
from app.services.counterfactual_explainer import (
    CounterfactualExplainerService, 
    ScenarioType, 
    ImpactMagnitude
)
from app.services.portfolio_api import PortfolioAPIService, ReportType
from app.services.alt_data_pack import AltDataPackService, AltDataSource

router = APIRouter()

# Initialize services
guidance_service = GuidanceConsistencyService()
footnote_service = FootnoteTimeMachineService()
counterfactual_service = CounterfactualExplainerService()
portfolio_service = PortfolioAPIService()
alt_data_service = AltDataPackService()


@router.get("/guidance-consistency/{ticker}")
@trace_function("advanced_features.get_guidance_consistency")
async def get_guidance_consistency(
    ticker: str,
    periods: int = 8,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Get guidance consistency analysis for ticker."""
    try:
        analysis = await guidance_service.analyze_ticker(db, ticker, periods)
        return analysis
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Guidance consistency analysis failed: {str(e)}"
        )


@router.post("/guidance-consistency/compare")
@trace_function("advanced_features.compare_guidance_consistency")
async def compare_guidance_consistency(
    tickers: List[str],
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Compare guidance consistency across peer companies."""
    try:
        if len(tickers) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 tickers allowed for comparison"
            )
        
        comparison = await guidance_service.compare_peers(db, tickers)
        return comparison
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Guidance comparison failed: {str(e)}"
        )


@router.get("/footnote-time-machine/{ticker}")
@trace_function("advanced_features.get_footnote_timeline")
async def get_footnote_timeline(
    ticker: str,
    periods: int = 8,
    footnote_types: Optional[List[str]] = None,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Get footnote change timeline for ticker."""
    try:
        analysis = await footnote_service.analyze_historical_changes(db, ticker, periods)
        return analysis
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Footnote timeline analysis failed: {str(e)}"
        )


@router.post("/footnote-time-machine/compare")
@trace_function("advanced_features.compare_disclosure_practices")
async def compare_disclosure_practices(
    tickers: List[str],
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Compare footnote disclosure practices across companies."""
    try:
        if len(tickers) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 tickers allowed for comparison"
            )
        
        comparison = await footnote_service.compare_disclosure_practices(db, tickers)
        return comparison
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Disclosure practices comparison failed: {str(e)}"
        )


@router.post("/counterfactual-analysis")
@trace_function("advanced_features.create_counterfactual_scenario")
async def create_counterfactual_scenario(
    request_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Create and analyze counterfactual scenario."""
    try:
        ticker = request_data["ticker"]
        scenario_type = ScenarioType(request_data["scenario_type"])
        impact_magnitude = ImpactMagnitude(request_data["impact_magnitude"])
        base_financials = request_data["base_financials"]
        custom_parameters = request_data.get("custom_parameters")
        
        result = await counterfactual_service.create_and_analyze_scenario(
            db=db,
            ticker=ticker,
            scenario_type=scenario_type,
            impact_magnitude=impact_magnitude,
            base_financials=base_financials,
            custom_parameters=custom_parameters,
        )
        
        return {
            "scenario_id": result.scenario_id,
            "ticker": result.ticker,
            "valuation_impact": {
                "base_valuation": result.base_case_valuation,
                "scenario_valuation": result.scenario_valuation,
                "change_amount": result.valuation_change,
                "change_percentage": result.valuation_change_pct,
            },
            "risk_assessment": {
                "downside_risk": result.downside_risk,
                "upside_potential": result.upside_potential,
                "volatility_impact": result.volatility_impact,
            },
            "analysis": {
                "executive_summary": result.executive_summary,
                "key_drivers": result.key_drivers,
                "mitigation_strategies": result.mitigation_strategies,
                "confidence_score": result.confidence_score,
            },
            "sensitivity_analysis": result.parameter_sensitivities,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Counterfactual analysis failed: {str(e)}"
        )


@router.post("/counterfactual-analysis/compare-scenarios")
@trace_function("advanced_features.compare_counterfactual_scenarios")
async def compare_counterfactual_scenarios(
    request_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Compare multiple counterfactual scenarios."""
    try:
        ticker = request_data["ticker"]
        scenario_configs = request_data["scenario_configs"]
        base_financials = request_data["base_financials"]
        
        if len(scenario_configs) > 5:
            raise HTTPException(
                status_code=400,
                detail="Maximum 5 scenarios allowed for comparison"
            )
        
        comparison = await counterfactual_service.compare_scenarios(
            db=db,
            ticker=ticker,
            scenario_configs=scenario_configs,
            base_financials=base_financials,
        )
        
        return comparison
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Scenario comparison failed: {str(e)}"
        )


@router.post("/portfolio/batch-analysis")
@trace_function("advanced_features.batch_analyze_tickers")
async def batch_analyze_tickers(
    request_data: Dict[str, Any],
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Perform batch analysis on multiple tickers."""
    try:
        tickers = request_data["tickers"]
        analysis_types = request_data.get("analysis_types", ["valuation", "risk", "fundamentals"])
        
        if len(tickers) > 50:
            raise HTTPException(
                status_code=400,
                detail="Maximum 50 tickers allowed for batch analysis"
            )
        
        if background_tasks and len(tickers) > 20:
            # Run in background for large batches
            background_tasks.add_task(
                portfolio_service.batch_analyze_tickers,
                db, tickers, analysis_types
            )
            
            return {
                "status": "started",
                "message": f"Batch analysis of {len(tickers)} tickers running in background",
                "tickers": tickers,
            }
        else:
            # Run synchronously for smaller batches
            results = await portfolio_service.batch_analyze_tickers(db, tickers, analysis_types)
            return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch analysis failed: {str(e)}"
        )


@router.post("/portfolio/generate-report")
@trace_function("advanced_features.generate_portfolio_report")
async def generate_portfolio_report(
    portfolio_config: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Generate comprehensive portfolio report."""
    try:
        report = await portfolio_service.generate_pm_report(db, portfolio_config)
        
        return {
            "portfolio_id": report.portfolio_id,
            "portfolio_name": report.portfolio_name,
            "report_type": report.report_type.value,
            "generated_at": report.generated_at.isoformat(),
            "metrics": {
                "total_value": report.metrics.total_value,
                "total_return": report.metrics.ytd_return,
                "volatility": report.metrics.portfolio_volatility,
                "sharpe_ratio": report.metrics.sharpe_ratio,
                "max_drawdown": report.metrics.max_drawdown,
                "var_95": report.metrics.var_95,
            },
            "allocation": {
                "sector_allocation": report.metrics.sector_allocation,
                "market_cap_allocation": report.metrics.market_cap_allocation,
                "top_10_weight": report.metrics.top_10_weight,
            },
            "analysis": {
                "executive_summary": report.executive_summary,
                "key_insights": report.key_insights,
                "recommendations": report.recommendations,
            },
            "holdings_count": len(report.holdings),
            "charts": report.charts,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Portfolio report generation failed: {str(e)}"
        )


@router.post("/portfolio/compare")
@trace_function("advanced_features.compare_portfolios")
async def compare_portfolios(
    portfolio_configs: List[Dict[str, Any]],
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Compare multiple portfolios."""
    try:
        if len(portfolio_configs) > 5:
            raise HTTPException(
                status_code=400,
                detail="Maximum 5 portfolios allowed for comparison"
            )
        
        comparison = await portfolio_service.compare_portfolios(db, portfolio_configs)
        return comparison
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Portfolio comparison failed: {str(e)}"
        )


@router.post("/alt-data/collect")
@trace_function("advanced_features.collect_alt_data")
async def collect_alt_data(
    request_data: Dict[str, Any],
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Collect comprehensive alternative data for company."""
    try:
        ticker = request_data["ticker"]
        company_name = request_data["company_name"]
        data_sources = request_data.get("data_sources", [
            "web_scraping", "job_postings", "app_rankings", "satellite_imagery"
        ])
        config = request_data.get("config", {})
        
        # Convert string sources to enum
        source_enums = []
        for source in data_sources:
            try:
                source_enums.append(AltDataSource(source))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid data source: {source}"
                )
        
        if background_tasks and len(source_enums) > 2:
            # Run in background for comprehensive collection
            background_tasks.add_task(
                alt_data_service.collect_comprehensive_data,
                db, ticker, company_name, source_enums, config
            )
            
            return {
                "status": "started",
                "message": f"Alternative data collection for {ticker} running in background",
                "ticker": ticker,
                "data_sources": data_sources,
            }
        else:
            # Run synchronously for smaller collections
            results = await alt_data_service.collect_comprehensive_data(
                db, ticker, company_name, source_enums, config
            )
            return results
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Alternative data collection failed: {str(e)}"
        )


@router.get("/alt-data/sources")
@trace_function("advanced_features.list_alt_data_sources")
async def list_alt_data_sources() -> Dict[str, Any]:
    """List available alternative data sources."""
    return {
        "available_sources": [
            {
                "name": source.value,
                "description": {
                    "web_scraping": "News sentiment and web mentions analysis",
                    "job_postings": "Hiring trends and skill demand analysis",
                    "app_rankings": "Mobile app performance and user engagement",
                    "satellite_imagery": "Facility activity and construction monitoring",
                    "social_sentiment": "Social media sentiment analysis",
                    "patent_filings": "Innovation and R&D activity tracking",
                    "executive_movements": "Leadership changes and talent flow",
                    "supply_chain": "Supply chain disruption monitoring",
                    "esg_metrics": "Environmental and social governance tracking",
                    "foot_traffic": "Physical location traffic analysis",
                }.get(source.value, "Alternative data source"),
                "data_types": {
                    "web_scraping": ["sentiment_score", "mention_volume", "news_velocity"],
                    "job_postings": ["hiring_velocity", "skill_demand", "geographic_expansion"],
                    "app_rankings": ["app_ranking", "app_rating", "download_velocity"],
                    "satellite_imagery": ["parking_occupancy", "construction_activity", "logistics_activity"],
                }.get(source.value, ["metric_data"]),
            }
            for source in AltDataSource
        ],
        "collection_frequencies": [freq.value for freq in DataFrequency],
        "supported_config_options": {
            "app_rankings": ["app_names"],
            "satellite_imagery": ["facility_locations"],
            "web_scraping": ["news_sources", "lookback_days"],
            "job_postings": ["lookback_months"],
        },
    }


@router.get("/features/summary")
@trace_function("advanced_features.get_features_summary")
async def get_features_summary() -> Dict[str, Any]:
    """Get summary of all advanced features."""
    return {
        "advanced_features": {
            "guidance_consistency_map": {
                "description": "Track and analyze management guidance changes over time",
                "capabilities": [
                    "Guidance extraction from earnings calls and filings",
                    "Consistency scoring and trend analysis",
                    "Peer company comparison",
                    "Investor confidence impact assessment",
                ],
                "endpoints": ["/guidance-consistency/{ticker}", "/guidance-consistency/compare"],
            },
            "footnote_time_machine": {
                "description": "Historical context tracking for footnote and disclosure changes",
                "capabilities": [
                    "Footnote change detection and classification",
                    "Materiality and complexity scoring",
                    "Temporal change analysis",
                    "Regulatory compliance tracking",
                ],
                "endpoints": ["/footnote-time-machine/{ticker}", "/footnote-time-machine/compare"],
            },
            "counterfactual_explainer": {
                "description": "Scenario analysis and what-if modeling for investment decisions",
                "capabilities": [
                    "Economic shock scenario modeling",
                    "Competitive pressure analysis",
                    "Regulatory change impact assessment",
                    "Valuation sensitivity analysis",
                ],
                "endpoints": ["/counterfactual-analysis", "/counterfactual-analysis/compare-scenarios"],
            },
            "portfolio_api": {
                "description": "Batch ticker analysis and portfolio management reports",
                "capabilities": [
                    "Batch ticker analysis (up to 50 tickers)",
                    "Comprehensive portfolio reporting",
                    "Risk-return analysis and attribution",
                    "Portfolio comparison and benchmarking",
                ],
                "endpoints": ["/portfolio/batch-analysis", "/portfolio/generate-report", "/portfolio/compare"],
            },
            "alt_data_pack": {
                "description": "Alternative data integration for enhanced investment insights",
                "capabilities": [
                    "Web scraping and sentiment analysis",
                    "Job postings and hiring trend analysis",
                    "Mobile app rankings and performance tracking",
                    "Satellite imagery and facility monitoring",
                ],
                "endpoints": ["/alt-data/collect", "/alt-data/sources"],
            },
        },
        "integration_notes": [
            "All features support both synchronous and background processing",
            "Rate limiting and data quality controls are built-in",
            "Results include confidence scores and uncertainty factors",
            "APIs are designed for institutional-grade analysis workflows",
        ],
    }
