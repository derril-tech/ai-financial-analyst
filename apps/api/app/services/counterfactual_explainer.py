"""Counterfactual Explainer for scenario analysis and what-if modeling."""

import json
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import copy

import pandas as pd
import numpy as np
from scipy import stats
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.observability import trace_function


class ScenarioType(Enum):
    """Types of counterfactual scenarios."""
    ECONOMIC_SHOCK = "economic_shock"
    COMPETITIVE_PRESSURE = "competitive_pressure"
    REGULATORY_CHANGE = "regulatory_change"
    TECHNOLOGY_DISRUPTION = "technology_disruption"
    MARKET_EXPANSION = "market_expansion"
    COST_INFLATION = "cost_inflation"
    DEMAND_SHIFT = "demand_shift"
    SUPPLY_CHAIN_DISRUPTION = "supply_chain_disruption"
    MANAGEMENT_CHANGE = "management_change"
    CAPITAL_STRUCTURE_CHANGE = "capital_structure_change"


class ImpactMagnitude(Enum):
    """Magnitude of scenario impact."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"


@dataclass
class ScenarioParameter:
    """Parameter for scenario modeling."""
    name: str
    description: str
    base_value: float
    scenario_value: float
    unit: str
    confidence_interval: Tuple[float, float]
    
    # Impact modeling
    impact_function: str  # "linear", "exponential", "logarithmic", "step"
    elasticity: float = 1.0  # How sensitive outcomes are to this parameter
    
    # Constraints
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    # Metadata
    data_source: str = ""
    last_updated: datetime = None


@dataclass
class CounterfactualScenario:
    """Complete counterfactual scenario definition."""
    id: str
    name: str
    description: str
    scenario_type: ScenarioType
    impact_magnitude: ImpactMagnitude
    
    # Parameters
    parameters: List[ScenarioParameter]
    
    # Timeline
    start_date: datetime
    duration_months: int
    
    # Probability
    probability: float  # 0-1 scale
    
    # Context
    triggers: List[str]
    assumptions: List[str]
    limitations: List[str]
    
    # Results (populated after analysis)
    financial_impact: Dict[str, float] = field(default_factory=dict)
    operational_impact: Dict[str, Any] = field(default_factory=dict)
    strategic_implications: List[str] = field(default_factory=list)


@dataclass
class CounterfactualResult:
    """Result of counterfactual analysis."""
    scenario_id: str
    ticker: str
    analysis_date: datetime
    
    # Financial projections
    base_case_metrics: Dict[str, float]
    scenario_metrics: Dict[str, float]
    delta_metrics: Dict[str, float]
    
    # Valuation impact
    base_case_valuation: float
    scenario_valuation: float
    valuation_change: float
    valuation_change_pct: float
    
    # Risk assessment
    downside_risk: float
    upside_potential: float
    volatility_impact: float
    
    # Sensitivity analysis
    parameter_sensitivities: Dict[str, float]
    
    # Narrative explanation
    executive_summary: str
    key_drivers: List[str]
    mitigation_strategies: List[str]
    
    # Confidence metrics
    confidence_score: float  # 0-1 scale
    uncertainty_factors: List[str]


class ScenarioBuilder:
    """Builder for creating counterfactual scenarios."""
    
    def __init__(self) -> None:
        """Initialize scenario builder."""
        self.scenario_templates = self._load_scenario_templates()
        self.economic_indicators = self._load_economic_indicators()
    
    def _load_scenario_templates(self) -> Dict[ScenarioType, Dict[str, Any]]:
        """Load predefined scenario templates."""
        return {
            ScenarioType.ECONOMIC_SHOCK: {
                "parameters": [
                    {
                        "name": "gdp_growth_rate",
                        "description": "GDP growth rate change",
                        "base_value": 2.5,
                        "scenario_values": {"mild": 1.0, "moderate": -0.5, "severe": -2.0, "extreme": -4.0},
                        "unit": "percentage",
                        "impact_function": "linear",
                        "elasticity": 1.2,
                    },
                    {
                        "name": "interest_rates",
                        "description": "Federal funds rate",
                        "base_value": 5.0,
                        "scenario_values": {"mild": 6.0, "moderate": 7.5, "severe": 9.0, "extreme": 12.0},
                        "unit": "percentage",
                        "impact_function": "exponential",
                        "elasticity": 0.8,
                    },
                    {
                        "name": "unemployment_rate",
                        "description": "Unemployment rate",
                        "base_value": 4.0,
                        "scenario_values": {"mild": 6.0, "moderate": 8.0, "severe": 12.0, "extreme": 16.0},
                        "unit": "percentage",
                        "impact_function": "linear",
                        "elasticity": -0.6,
                    },
                ],
                "duration_months": 18,
                "probability_range": (0.05, 0.15),
            },
            
            ScenarioType.COMPETITIVE_PRESSURE: {
                "parameters": [
                    {
                        "name": "market_share_loss",
                        "description": "Market share erosion",
                        "base_value": 0.0,
                        "scenario_values": {"mild": -2.0, "moderate": -5.0, "severe": -10.0, "extreme": -20.0},
                        "unit": "percentage_points",
                        "impact_function": "linear",
                        "elasticity": 1.5,
                    },
                    {
                        "name": "pricing_pressure",
                        "description": "Average selling price decline",
                        "base_value": 0.0,
                        "scenario_values": {"mild": -3.0, "moderate": -7.0, "severe": -15.0, "extreme": -25.0},
                        "unit": "percentage",
                        "impact_function": "linear",
                        "elasticity": 2.0,
                    },
                    {
                        "name": "customer_acquisition_cost",
                        "description": "CAC increase",
                        "base_value": 0.0,
                        "scenario_values": {"mild": 15.0, "moderate": 35.0, "severe": 60.0, "extreme": 100.0},
                        "unit": "percentage",
                        "impact_function": "exponential",
                        "elasticity": 0.7,
                    },
                ],
                "duration_months": 24,
                "probability_range": (0.15, 0.30),
            },
            
            ScenarioType.REGULATORY_CHANGE: {
                "parameters": [
                    {
                        "name": "compliance_costs",
                        "description": "Additional compliance costs",
                        "base_value": 0.0,
                        "scenario_values": {"mild": 2.0, "moderate": 5.0, "severe": 12.0, "extreme": 25.0},
                        "unit": "percentage_of_revenue",
                        "impact_function": "step",
                        "elasticity": 1.0,
                    },
                    {
                        "name": "market_access_restriction",
                        "description": "Revenue impact from restricted markets",
                        "base_value": 0.0,
                        "scenario_values": {"mild": -5.0, "moderate": -15.0, "severe": -30.0, "extreme": -50.0},
                        "unit": "percentage",
                        "impact_function": "step",
                        "elasticity": 1.0,
                    },
                ],
                "duration_months": 36,
                "probability_range": (0.20, 0.40),
            },
        }
    
    def _load_economic_indicators(self) -> Dict[str, float]:
        """Load current economic indicators for baseline."""
        # In production, fetch from economic data APIs
        return {
            "gdp_growth": 2.5,
            "inflation_rate": 3.2,
            "unemployment_rate": 3.8,
            "fed_funds_rate": 5.25,
            "10y_treasury": 4.5,
            "vix": 18.5,
            "oil_price": 85.0,
            "usd_index": 103.2,
        }
    
    @trace_function("scenario_builder.create_scenario")
    def create_scenario(
        self,
        scenario_type: ScenarioType,
        impact_magnitude: ImpactMagnitude,
        ticker: str,
        custom_parameters: Optional[Dict[str, Any]] = None,
    ) -> CounterfactualScenario:
        """Create a counterfactual scenario."""
        template = self.scenario_templates.get(scenario_type)
        
        if not template:
            raise ValueError(f"No template found for scenario type: {scenario_type}")
        
        # Build parameters
        parameters = []
        for param_template in template["parameters"]:
            scenario_value = param_template["scenario_values"][impact_magnitude.value]
            
            # Apply custom overrides
            if custom_parameters and param_template["name"] in custom_parameters:
                scenario_value = custom_parameters[param_template["name"]]
            
            parameter = ScenarioParameter(
                name=param_template["name"],
                description=param_template["description"],
                base_value=param_template["base_value"],
                scenario_value=scenario_value,
                unit=param_template["unit"],
                confidence_interval=self._calculate_confidence_interval(scenario_value),
                impact_function=param_template["impact_function"],
                elasticity=param_template["elasticity"],
                last_updated=datetime.utcnow(),
            )
            
            parameters.append(parameter)
        
        # Calculate probability
        prob_range = template["probability_range"]
        magnitude_multiplier = {
            ImpactMagnitude.MILD: 1.0,
            ImpactMagnitude.MODERATE: 0.7,
            ImpactMagnitude.SEVERE: 0.4,
            ImpactMagnitude.EXTREME: 0.1,
        }
        
        base_probability = (prob_range[0] + prob_range[1]) / 2
        probability = base_probability * magnitude_multiplier[impact_magnitude]
        
        scenario_id = f"{ticker}_{scenario_type.value}_{impact_magnitude.value}_{int(datetime.now().timestamp())}"
        
        scenario = CounterfactualScenario(
            id=scenario_id,
            name=f"{scenario_type.value.replace('_', ' ').title()} - {impact_magnitude.value.title()}",
            description=self._generate_scenario_description(scenario_type, impact_magnitude),
            scenario_type=scenario_type,
            impact_magnitude=impact_magnitude,
            parameters=parameters,
            start_date=datetime.utcnow(),
            duration_months=template["duration_months"],
            probability=probability,
            triggers=self._generate_triggers(scenario_type),
            assumptions=self._generate_assumptions(scenario_type, impact_magnitude),
            limitations=self._generate_limitations(),
        )
        
        return scenario
    
    def _calculate_confidence_interval(self, value: float, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for parameter value."""
        # Assume 20% standard deviation for uncertainty
        std_dev = abs(value) * 0.2
        
        # Calculate confidence interval
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin = z_score * std_dev
        
        return (value - margin, value + margin)
    
    def _generate_scenario_description(self, scenario_type: ScenarioType, magnitude: ImpactMagnitude) -> str:
        """Generate scenario description."""
        descriptions = {
            ScenarioType.ECONOMIC_SHOCK: {
                ImpactMagnitude.MILD: "Economic slowdown with modest GDP decline and rising unemployment",
                ImpactMagnitude.MODERATE: "Recession with significant economic contraction and market volatility",
                ImpactMagnitude.SEVERE: "Deep recession with major GDP decline and financial market stress",
                ImpactMagnitude.EXTREME: "Economic crisis with severe contraction and systemic financial risks",
            },
            ScenarioType.COMPETITIVE_PRESSURE: {
                ImpactMagnitude.MILD: "Increased competition leading to modest market share erosion",
                ImpactMagnitude.MODERATE: "Significant competitive threats impacting pricing and market position",
                ImpactMagnitude.SEVERE: "Intense competition causing major market share loss and margin compression",
                ImpactMagnitude.EXTREME: "Disruptive competition fundamentally altering industry dynamics",
            },
            ScenarioType.REGULATORY_CHANGE: {
                ImpactMagnitude.MILD: "New regulations requiring modest compliance investments",
                ImpactMagnitude.MODERATE: "Significant regulatory changes impacting operations and costs",
                ImpactMagnitude.SEVERE: "Major regulatory overhaul requiring substantial business model changes",
                ImpactMagnitude.EXTREME: "Transformative regulation fundamentally reshaping industry structure",
            },
        }
        
        return descriptions.get(scenario_type, {}).get(magnitude, "Scenario analysis")
    
    def _generate_triggers(self, scenario_type: ScenarioType) -> List[str]:
        """Generate potential triggers for scenario."""
        triggers = {
            ScenarioType.ECONOMIC_SHOCK: [
                "Federal Reserve aggressive rate hikes",
                "Geopolitical tensions escalation",
                "Banking sector stress",
                "Commodity price spikes",
                "Trade war intensification",
            ],
            ScenarioType.COMPETITIVE_PRESSURE: [
                "New market entrant with disruptive technology",
                "Incumbent competitor price war",
                "Substitute product adoption acceleration",
                "Customer preference shifts",
                "Regulatory barriers reduction",
            ],
            ScenarioType.REGULATORY_CHANGE: [
                "New legislation passage",
                "Regulatory agency policy shift",
                "International standard adoption",
                "Court ruling precedent",
                "Political administration change",
            ],
        }
        
        return triggers.get(scenario_type, ["External market forces"])
    
    def _generate_assumptions(self, scenario_type: ScenarioType, magnitude: ImpactMagnitude) -> List[str]:
        """Generate key assumptions for scenario."""
        base_assumptions = [
            "Company maintains current business model",
            "No major strategic pivots during scenario period",
            "Management team remains stable",
            "Current capital structure maintained",
        ]
        
        scenario_specific = {
            ScenarioType.ECONOMIC_SHOCK: [
                "Economic indicators move in correlated fashion",
                "Consumer behavior follows historical recession patterns",
                "Government fiscal response is limited",
            ],
            ScenarioType.COMPETITIVE_PRESSURE: [
                "Competitors have sufficient resources to sustain pressure",
                "Market size remains relatively stable",
                "Technology adoption follows predictable curves",
            ],
        }
        
        return base_assumptions + scenario_specific.get(scenario_type, [])
    
    def _generate_limitations(self) -> List[str]:
        """Generate analysis limitations."""
        return [
            "Based on historical patterns which may not predict future outcomes",
            "Assumes linear relationships between variables",
            "Does not account for management's adaptive responses",
            "External factors beyond modeled parameters may influence results",
            "Scenario probabilities are subjective estimates",
        ]


class CounterfactualAnalyzer:
    """Analyzer for running counterfactual scenarios."""
    
    def __init__(self) -> None:
        """Initialize counterfactual analyzer."""
        self.scenario_builder = ScenarioBuilder()
    
    @trace_function("counterfactual_analyzer.analyze_scenario")
    async def analyze_scenario(
        self,
        db: AsyncSession,
        ticker: str,
        scenario: CounterfactualScenario,
        base_financials: Dict[str, float],
    ) -> CounterfactualResult:
        """Analyze counterfactual scenario impact."""
        # Calculate scenario impact on key metrics
        scenario_metrics = self._calculate_scenario_metrics(base_financials, scenario)
        
        # Calculate deltas
        delta_metrics = {
            key: scenario_metrics[key] - base_financials[key]
            for key in base_financials.keys()
            if key in scenario_metrics
        }
        
        # Valuation impact
        base_valuation = await self._calculate_valuation(db, ticker, base_financials)
        scenario_valuation = await self._calculate_valuation(db, ticker, scenario_metrics)
        
        valuation_change = scenario_valuation - base_valuation
        valuation_change_pct = (valuation_change / base_valuation) * 100 if base_valuation != 0 else 0
        
        # Risk assessment
        risk_metrics = self._assess_risk_impact(scenario, delta_metrics)
        
        # Sensitivity analysis
        sensitivities = self._calculate_parameter_sensitivities(scenario, base_financials)
        
        # Generate narrative
        narrative = self._generate_narrative_explanation(scenario, delta_metrics, valuation_change_pct)
        
        # Calculate confidence
        confidence_score = self._calculate_confidence_score(scenario, base_financials)
        
        result = CounterfactualResult(
            scenario_id=scenario.id,
            ticker=ticker,
            analysis_date=datetime.utcnow(),
            base_case_metrics=base_financials,
            scenario_metrics=scenario_metrics,
            delta_metrics=delta_metrics,
            base_case_valuation=base_valuation,
            scenario_valuation=scenario_valuation,
            valuation_change=valuation_change,
            valuation_change_pct=valuation_change_pct,
            downside_risk=risk_metrics["downside_risk"],
            upside_potential=risk_metrics["upside_potential"],
            volatility_impact=risk_metrics["volatility_impact"],
            parameter_sensitivities=sensitivities,
            executive_summary=narrative["executive_summary"],
            key_drivers=narrative["key_drivers"],
            mitigation_strategies=narrative["mitigation_strategies"],
            confidence_score=confidence_score,
            uncertainty_factors=narrative["uncertainty_factors"],
        )
        
        return result
    
    def _calculate_scenario_metrics(
        self,
        base_financials: Dict[str, float],
        scenario: CounterfactualScenario,
    ) -> Dict[str, float]:
        """Calculate financial metrics under scenario conditions."""
        scenario_metrics = copy.deepcopy(base_financials)
        
        # Apply parameter impacts
        for parameter in scenario.parameters:
            impact = self._calculate_parameter_impact(parameter, base_financials)
            
            # Apply impact to relevant metrics
            affected_metrics = self._get_affected_metrics(parameter.name)
            
            for metric in affected_metrics:
                if metric in scenario_metrics:
                    if parameter.impact_function == "linear":
                        scenario_metrics[metric] *= (1 + impact)
                    elif parameter.impact_function == "exponential":
                        scenario_metrics[metric] *= np.exp(impact)
                    elif parameter.impact_function == "logarithmic":
                        scenario_metrics[metric] *= (1 + np.log(1 + abs(impact)))
                    elif parameter.impact_function == "step":
                        scenario_metrics[metric] *= (1 + impact) if abs(impact) > 0.01 else 1.0
        
        return scenario_metrics
    
    def _calculate_parameter_impact(self, parameter: ScenarioParameter, base_financials: Dict[str, float]) -> float:
        """Calculate impact of a single parameter change."""
        # Calculate percentage change from base to scenario
        if parameter.base_value == 0:
            pct_change = parameter.scenario_value
        else:
            pct_change = (parameter.scenario_value - parameter.base_value) / parameter.base_value
        
        # Apply elasticity
        impact = pct_change * parameter.elasticity
        
        return impact
    
    def _get_affected_metrics(self, parameter_name: str) -> List[str]:
        """Get financial metrics affected by parameter."""
        impact_mapping = {
            "gdp_growth_rate": ["revenue", "operating_income", "net_income"],
            "interest_rates": ["interest_expense", "net_income", "free_cash_flow"],
            "unemployment_rate": ["revenue", "operating_expenses"],
            "market_share_loss": ["revenue", "gross_profit", "operating_income"],
            "pricing_pressure": ["revenue", "gross_profit", "gross_margin"],
            "customer_acquisition_cost": ["marketing_expenses", "operating_expenses"],
            "compliance_costs": ["operating_expenses", "operating_income"],
            "market_access_restriction": ["revenue", "operating_income"],
        }
        
        return impact_mapping.get(parameter_name, ["revenue", "operating_income"])
    
    async def _calculate_valuation(
        self,
        db: AsyncSession,
        ticker: str,
        financials: Dict[str, float],
    ) -> float:
        """Calculate company valuation based on financial metrics."""
        # Simplified DCF calculation
        # In production, use full valuation service
        
        revenue = financials.get("revenue", 0)
        operating_margin = financials.get("operating_margin", 0.15)
        tax_rate = financials.get("tax_rate", 0.25)
        wacc = financials.get("wacc", 0.10)
        growth_rate = financials.get("growth_rate", 0.03)
        
        # Calculate NOPAT
        operating_income = revenue * operating_margin
        nopat = operating_income * (1 - tax_rate)
        
        # Terminal value (simplified)
        terminal_value = nopat * (1 + growth_rate) / (wacc - growth_rate)
        
        # Present value (assuming immediate cash flow for simplicity)
        valuation = terminal_value / (1 + wacc)
        
        return valuation
    
    def _assess_risk_impact(
        self,
        scenario: CounterfactualScenario,
        delta_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """Assess risk impact of scenario."""
        # Calculate downside risk
        negative_impacts = [abs(delta) for delta in delta_metrics.values() if delta < 0]
        downside_risk = np.mean(negative_impacts) if negative_impacts else 0.0
        
        # Calculate upside potential
        positive_impacts = [delta for delta in delta_metrics.values() if delta > 0]
        upside_potential = np.mean(positive_impacts) if positive_impacts else 0.0
        
        # Volatility impact based on scenario magnitude
        volatility_multipliers = {
            ImpactMagnitude.MILD: 1.2,
            ImpactMagnitude.MODERATE: 1.5,
            ImpactMagnitude.SEVERE: 2.0,
            ImpactMagnitude.EXTREME: 3.0,
        }
        
        volatility_impact = volatility_multipliers.get(scenario.impact_magnitude, 1.0)
        
        return {
            "downside_risk": downside_risk,
            "upside_potential": upside_potential,
            "volatility_impact": volatility_impact,
        }
    
    def _calculate_parameter_sensitivities(
        self,
        scenario: CounterfactualScenario,
        base_financials: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate sensitivity of outcomes to each parameter."""
        sensitivities = {}
        
        for parameter in scenario.parameters:
            # Calculate impact of 1% change in parameter
            original_value = parameter.scenario_value
            parameter.scenario_value = original_value * 1.01
            
            # Recalculate metrics
            new_metrics = self._calculate_scenario_metrics(base_financials, scenario)
            
            # Calculate sensitivity (change in revenue per 1% parameter change)
            base_revenue = base_financials.get("revenue", 0)
            new_revenue = new_metrics.get("revenue", 0)
            
            if base_revenue != 0:
                sensitivity = ((new_revenue - base_revenue) / base_revenue) / 0.01
            else:
                sensitivity = 0.0
            
            sensitivities[parameter.name] = sensitivity
            
            # Restore original value
            parameter.scenario_value = original_value
        
        return sensitivities
    
    def _generate_narrative_explanation(
        self,
        scenario: CounterfactualScenario,
        delta_metrics: Dict[str, float],
        valuation_change_pct: float,
    ) -> Dict[str, Any]:
        """Generate narrative explanation of scenario impact."""
        # Executive summary
        impact_direction = "positive" if valuation_change_pct > 0 else "negative"
        impact_magnitude = "significant" if abs(valuation_change_pct) > 10 else "moderate"
        
        executive_summary = (
            f"The {scenario.name} scenario results in a {impact_magnitude} {impact_direction} "
            f"impact on company valuation ({valuation_change_pct:.1f}% change). "
            f"Key drivers include {', '.join([p.name.replace('_', ' ') for p in scenario.parameters[:3]])}."
        )
        
        # Key drivers
        key_drivers = []
        for parameter in scenario.parameters:
            change = parameter.scenario_value - parameter.base_value
            direction = "increase" if change > 0 else "decrease"
            key_drivers.append(f"{parameter.description}: {abs(change):.1f}{parameter.unit} {direction}")
        
        # Mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(scenario)
        
        # Uncertainty factors
        uncertainty_factors = [
            "Parameter correlation assumptions may not hold",
            "Management response not modeled",
            "External intervention possibilities",
            "Second-order effects not captured",
        ]
        
        return {
            "executive_summary": executive_summary,
            "key_drivers": key_drivers,
            "mitigation_strategies": mitigation_strategies,
            "uncertainty_factors": uncertainty_factors,
        }
    
    def _generate_mitigation_strategies(self, scenario: CounterfactualScenario) -> List[str]:
        """Generate potential mitigation strategies."""
        strategies = {
            ScenarioType.ECONOMIC_SHOCK: [
                "Implement cost reduction programs",
                "Diversify revenue streams geographically",
                "Build cash reserves and credit facilities",
                "Focus on recession-resilient product lines",
            ],
            ScenarioType.COMPETITIVE_PRESSURE: [
                "Accelerate product innovation and differentiation",
                "Strengthen customer relationships and loyalty programs",
                "Optimize pricing strategy and value proposition",
                "Consider strategic partnerships or acquisitions",
            ],
            ScenarioType.REGULATORY_CHANGE: [
                "Engage proactively with regulators",
                "Invest in compliance infrastructure early",
                "Explore regulatory arbitrage opportunities",
                "Build industry coalition for favorable outcomes",
            ],
        }
        
        return strategies.get(scenario.scenario_type, ["Monitor situation closely", "Maintain strategic flexibility"])
    
    def _calculate_confidence_score(
        self,
        scenario: CounterfactualScenario,
        base_financials: Dict[str, float],
    ) -> float:
        """Calculate confidence score for analysis."""
        confidence = 0.8  # Base confidence
        
        # Adjust for scenario probability
        confidence *= scenario.probability * 2  # Scale probability impact
        
        # Adjust for parameter uncertainty
        avg_uncertainty = np.mean([
            abs(p.confidence_interval[1] - p.confidence_interval[0]) / abs(p.scenario_value)
            for p in scenario.parameters
            if p.scenario_value != 0
        ])
        
        confidence *= (1 - min(avg_uncertainty, 0.5))
        
        # Adjust for scenario magnitude (extreme scenarios less certain)
        magnitude_adjustment = {
            ImpactMagnitude.MILD: 1.0,
            ImpactMagnitude.MODERATE: 0.9,
            ImpactMagnitude.SEVERE: 0.7,
            ImpactMagnitude.EXTREME: 0.5,
        }
        
        confidence *= magnitude_adjustment.get(scenario.impact_magnitude, 0.8)
        
        return min(max(confidence, 0.1), 1.0)


class CounterfactualExplainerService:
    """Main service for counterfactual explanation and scenario analysis."""
    
    def __init__(self) -> None:
        """Initialize counterfactual explainer service."""
        self.analyzer = CounterfactualAnalyzer()
        self.scenario_cache = {}
    
    @trace_function("counterfactual_service.create_and_analyze_scenario")
    async def create_and_analyze_scenario(
        self,
        db: AsyncSession,
        ticker: str,
        scenario_type: ScenarioType,
        impact_magnitude: ImpactMagnitude,
        base_financials: Dict[str, float],
        custom_parameters: Optional[Dict[str, Any]] = None,
    ) -> CounterfactualResult:
        """Create and analyze a counterfactual scenario."""
        # Create scenario
        scenario = self.analyzer.scenario_builder.create_scenario(
            scenario_type, impact_magnitude, ticker, custom_parameters
        )
        
        # Analyze scenario
        result = await self.analyzer.analyze_scenario(db, ticker, scenario, base_financials)
        
        # Cache for future reference
        self.scenario_cache[scenario.id] = scenario
        
        return result
    
    @trace_function("counterfactual_service.compare_scenarios")
    async def compare_scenarios(
        self,
        db: AsyncSession,
        ticker: str,
        scenario_configs: List[Dict[str, Any]],
        base_financials: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compare multiple counterfactual scenarios."""
        results = []
        
        for config in scenario_configs:
            result = await self.create_and_analyze_scenario(
                db=db,
                ticker=ticker,
                scenario_type=ScenarioType(config["scenario_type"]),
                impact_magnitude=ImpactMagnitude(config["impact_magnitude"]),
                base_financials=base_financials,
                custom_parameters=config.get("custom_parameters"),
            )
            results.append(result)
        
        # Generate comparison insights
        comparison = self._generate_scenario_comparison(results)
        
        return {
            "scenario_results": [
                {
                    "scenario_id": r.scenario_id,
                    "scenario_name": self.scenario_cache[r.scenario_id].name,
                    "valuation_change_pct": r.valuation_change_pct,
                    "confidence_score": r.confidence_score,
                    "downside_risk": r.downside_risk,
                    "executive_summary": r.executive_summary,
                }
                for r in results
            ],
            "comparison_insights": comparison,
        }
    
    def _generate_scenario_comparison(self, results: List[CounterfactualResult]) -> Dict[str, Any]:
        """Generate insights from scenario comparison."""
        if not results:
            return {}
        
        # Rank by valuation impact
        sorted_by_impact = sorted(results, key=lambda x: x.valuation_change_pct)
        
        # Risk-return analysis
        risk_return = [
            {
                "scenario_id": r.scenario_id,
                "return": r.valuation_change_pct,
                "risk": r.downside_risk,
                "confidence": r.confidence_score,
            }
            for r in results
        ]
        
        # Key insights
        worst_case = sorted_by_impact[0]
        best_case = sorted_by_impact[-1]
        
        insights = {
            "worst_case_scenario": {
                "name": self.scenario_cache[worst_case.scenario_id].name,
                "impact": worst_case.valuation_change_pct,
            },
            "best_case_scenario": {
                "name": self.scenario_cache[best_case.scenario_id].name,
                "impact": best_case.valuation_change_pct,
            },
            "risk_return_profile": risk_return,
            "key_vulnerabilities": self._identify_key_vulnerabilities(results),
            "strategic_recommendations": self._generate_strategic_recommendations(results),
        }
        
        return insights
    
    def _identify_key_vulnerabilities(self, results: List[CounterfactualResult]) -> List[str]:
        """Identify key vulnerabilities from scenario analysis."""
        vulnerabilities = []
        
        # High-impact scenarios
        high_impact_scenarios = [r for r in results if abs(r.valuation_change_pct) > 15]
        
        if high_impact_scenarios:
            vulnerabilities.append("Significant exposure to external economic shocks")
        
        # High sensitivity parameters
        all_sensitivities = {}
        for result in results:
            for param, sensitivity in result.parameter_sensitivities.items():
                if param not in all_sensitivities:
                    all_sensitivities[param] = []
                all_sensitivities[param].append(abs(sensitivity))
        
        high_sensitivity_params = [
            param for param, sensitivities in all_sensitivities.items()
            if np.mean(sensitivities) > 0.5
        ]
        
        if high_sensitivity_params:
            vulnerabilities.append(f"High sensitivity to: {', '.join(high_sensitivity_params)}")
        
        return vulnerabilities
    
    def _generate_strategic_recommendations(self, results: List[CounterfactualResult]) -> List[str]:
        """Generate strategic recommendations based on scenario analysis."""
        recommendations = []
        
        # Analyze common mitigation strategies
        all_strategies = []
        for result in results:
            all_strategies.extend(result.mitigation_strategies)
        
        # Find most common strategies
        strategy_counts = {}
        for strategy in all_strategies:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        top_strategies = sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        recommendations.extend([strategy for strategy, _ in top_strategies])
        
        # Add scenario-specific recommendations
        recommendations.append("Develop scenario-based contingency plans")
        recommendations.append("Implement early warning systems for key risk indicators")
        recommendations.append("Build strategic flexibility to adapt to changing conditions")
        
        return recommendations[:5]  # Limit to top 5
