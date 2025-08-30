"""Guidance Consistency Map for tracking management guidance changes over time."""

import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from app.core.observability import trace_function
from app.models.document import Document
from app.models.fact import FactXBRL


class GuidanceType(Enum):
    """Types of management guidance."""
    REVENUE = "revenue"
    EARNINGS = "earnings"
    MARGIN = "margin"
    CAPEX = "capex"
    FCF = "free_cash_flow"
    GROWTH = "growth_rate"
    MARKET_SHARE = "market_share"
    PRODUCT_LAUNCH = "product_launch"
    COST_SAVINGS = "cost_savings"
    RESTRUCTURING = "restructuring"


class GuidanceDirection(Enum):
    """Direction of guidance change."""
    RAISED = "raised"
    LOWERED = "lowered"
    MAINTAINED = "maintained"
    WITHDRAWN = "withdrawn"
    INTRODUCED = "introduced"


@dataclass
class GuidanceItem:
    """Individual guidance item."""
    id: str
    ticker: str
    guidance_type: GuidanceType
    period: str  # e.g., "Q4 2024", "FY 2024"
    metric_name: str
    
    # Guidance values
    low_estimate: Optional[float] = None
    high_estimate: Optional[float] = None
    point_estimate: Optional[float] = None
    
    # Metadata
    announcement_date: datetime = None
    source_document: str = ""
    confidence_level: str = "medium"  # low, medium, high
    
    # Context
    context: str = ""
    reasoning: str = ""
    assumptions: List[str] = field(default_factory=list)
    
    # Tracking
    previous_guidance_id: Optional[str] = None
    direction: Optional[GuidanceDirection] = None
    magnitude_change: Optional[float] = None


@dataclass
class GuidanceConsistencyScore:
    """Consistency scoring for management guidance."""
    ticker: str
    period: str
    overall_score: float  # 0-100
    
    # Component scores
    accuracy_score: float
    volatility_score: float
    timing_score: float
    transparency_score: float
    
    # Supporting metrics
    guidance_changes: int
    beats_vs_misses: Dict[str, int]
    avg_revision_magnitude: float
    
    # Insights
    strengths: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class GuidanceExtractor:
    """Extract guidance from earnings calls and filings."""
    
    def __init__(self) -> None:
        """Initialize guidance extractor."""
        self.guidance_patterns = {
            GuidanceType.REVENUE: [
                r"revenue.*expect.*(\$?[\d,]+\.?\d*)\s*(billion|million|thousand)?",
                r"sales.*guidance.*(\$?[\d,]+\.?\d*)\s*(billion|million|thousand)?",
                r"top.*line.*(\$?[\d,]+\.?\d*)\s*(billion|million|thousand)?",
            ],
            GuidanceType.EARNINGS: [
                r"earnings.*per.*share.*(\$?[\d,]+\.?\d*)",
                r"eps.*guidance.*(\$?[\d,]+\.?\d*)",
                r"diluted.*eps.*(\$?[\d,]+\.?\d*)",
            ],
            GuidanceType.MARGIN: [
                r"gross.*margin.*(\d+\.?\d*)%",
                r"operating.*margin.*(\d+\.?\d*)%",
                r"ebitda.*margin.*(\d+\.?\d*)%",
            ],
        }
        
        self.temporal_indicators = [
            "Q1", "Q2", "Q3", "Q4", "first quarter", "second quarter",
            "third quarter", "fourth quarter", "full year", "fiscal year",
            "2024", "2025", "next year", "this year", "going forward",
        ]
        
        self.confidence_indicators = {
            "high": ["confident", "expect", "will", "committed"],
            "medium": ["anticipate", "believe", "estimate", "target"],
            "low": ["hope", "may", "could", "potential", "if"],
        }
    
    @trace_function("guidance_extractor.extract_from_text")
    def extract_from_text(self, text: str, ticker: str, document_id: str) -> List[GuidanceItem]:
        """Extract guidance items from text."""
        import re
        
        guidance_items = []
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip().lower()
            
            # Look for guidance patterns
            for guidance_type, patterns in self.guidance_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, sentence, re.IGNORECASE)
                    
                    for match in matches:
                        # Extract numeric value
                        value_str = match.group(1).replace('$', '').replace(',', '')
                        
                        try:
                            value = float(value_str)
                            
                            # Apply multipliers
                            if len(match.groups()) > 1 and match.group(2):
                                multiplier = match.group(2).lower()
                                if multiplier == "billion":
                                    value *= 1e9
                                elif multiplier == "million":
                                    value *= 1e6
                                elif multiplier == "thousand":
                                    value *= 1e3
                            
                            # Extract period
                            period = self._extract_period(sentence)
                            
                            # Determine confidence
                            confidence = self._determine_confidence(sentence)
                            
                            guidance_item = GuidanceItem(
                                id=f"{ticker}_{guidance_type.value}_{period}_{len(guidance_items)}",
                                ticker=ticker,
                                guidance_type=guidance_type,
                                period=period,
                                metric_name=guidance_type.value,
                                point_estimate=value,
                                announcement_date=datetime.utcnow(),
                                source_document=document_id,
                                confidence_level=confidence,
                                context=sentence,
                            )
                            
                            guidance_items.append(guidance_item)
                            
                        except ValueError:
                            continue
        
        return guidance_items
    
    def _extract_period(self, text: str) -> str:
        """Extract time period from text."""
        import re
        
        # Look for specific periods
        for indicator in self.temporal_indicators:
            if indicator.lower() in text.lower():
                return indicator
        
        # Default to current year
        return f"FY {datetime.now().year}"
    
    def _determine_confidence(self, text: str) -> str:
        """Determine confidence level from language used."""
        for confidence, indicators in self.confidence_indicators.items():
            if any(indicator in text for indicator in indicators):
                return confidence
        
        return "medium"


class GuidanceTracker:
    """Track and analyze guidance changes over time."""
    
    def __init__(self) -> None:
        """Initialize guidance tracker."""
        self.extractor = GuidanceExtractor()
        self.guidance_history = {}  # ticker -> List[GuidanceItem]
    
    @trace_function("guidance_tracker.process_document")
    async def process_document(
        self,
        db: AsyncSession,
        document_id: str,
        ticker: str,
        content: str,
    ) -> List[GuidanceItem]:
        """Process document and extract guidance."""
        # Extract guidance from content
        guidance_items = self.extractor.extract_from_text(content, ticker, document_id)
        
        # Store in history
        if ticker not in self.guidance_history:
            self.guidance_history[ticker] = []
        
        # Link to previous guidance
        for item in guidance_items:
            previous_item = self._find_previous_guidance(ticker, item)
            if previous_item:
                item.previous_guidance_id = previous_item.id
                item.direction = self._determine_direction(previous_item, item)
                item.magnitude_change = self._calculate_magnitude_change(previous_item, item)
        
        self.guidance_history[ticker].extend(guidance_items)
        
        return guidance_items
    
    def _find_previous_guidance(self, ticker: str, current_item: GuidanceItem) -> Optional[GuidanceItem]:
        """Find most recent previous guidance for same metric."""
        if ticker not in self.guidance_history:
            return None
        
        # Find most recent guidance for same type and period
        previous_items = [
            item for item in self.guidance_history[ticker]
            if (item.guidance_type == current_item.guidance_type and
                item.period == current_item.period and
                item.announcement_date < current_item.announcement_date)
        ]
        
        if previous_items:
            return max(previous_items, key=lambda x: x.announcement_date)
        
        return None
    
    def _determine_direction(self, previous: GuidanceItem, current: GuidanceItem) -> GuidanceDirection:
        """Determine direction of guidance change."""
        prev_value = previous.point_estimate or ((previous.low_estimate or 0) + (previous.high_estimate or 0)) / 2
        curr_value = current.point_estimate or ((current.low_estimate or 0) + (current.high_estimate or 0)) / 2
        
        if prev_value == 0:
            return GuidanceDirection.INTRODUCED
        
        change_pct = (curr_value - prev_value) / prev_value
        
        if change_pct > 0.02:  # > 2% increase
            return GuidanceDirection.RAISED
        elif change_pct < -0.02:  # > 2% decrease
            return GuidanceDirection.LOWERED
        else:
            return GuidanceDirection.MAINTAINED
    
    def _calculate_magnitude_change(self, previous: GuidanceItem, current: GuidanceItem) -> float:
        """Calculate magnitude of guidance change."""
        prev_value = previous.point_estimate or ((previous.low_estimate or 0) + (previous.high_estimate or 0)) / 2
        curr_value = current.point_estimate or ((current.low_estimate or 0) + (current.high_estimate or 0)) / 2
        
        if prev_value == 0:
            return 0.0
        
        return (curr_value - prev_value) / prev_value
    
    @trace_function("guidance_tracker.generate_consistency_map")
    def generate_consistency_map(self, ticker: str, periods: int = 8) -> Dict[str, Any]:
        """Generate guidance consistency map for a ticker."""
        if ticker not in self.guidance_history:
            return {"error": f"No guidance history for {ticker}"}
        
        guidance_items = self.guidance_history[ticker]
        
        # Group by guidance type and period
        guidance_by_type = {}
        for item in guidance_items:
            key = f"{item.guidance_type.value}_{item.period}"
            if key not in guidance_by_type:
                guidance_by_type[key] = []
            guidance_by_type[key].append(item)
        
        # Analyze consistency patterns
        consistency_analysis = {
            "ticker": ticker,
            "analysis_date": datetime.utcnow().isoformat(),
            "guidance_types": list(set(item.guidance_type.value for item in guidance_items)),
            "total_guidance_updates": len(guidance_items),
            "patterns": {},
            "timeline": [],
            "consistency_scores": {},
        }
        
        # Create timeline
        timeline_items = sorted(guidance_items, key=lambda x: x.announcement_date)
        for item in timeline_items[-periods:]:  # Last N periods
            timeline_entry = {
                "date": item.announcement_date.isoformat(),
                "type": item.guidance_type.value,
                "period": item.period,
                "value": item.point_estimate,
                "direction": item.direction.value if item.direction else None,
                "magnitude_change": item.magnitude_change,
                "confidence": item.confidence_level,
                "context": item.context[:100] + "..." if len(item.context) > 100 else item.context,
            }
            consistency_analysis["timeline"].append(timeline_entry)
        
        # Analyze patterns by guidance type
        for guidance_type in GuidanceType:
            type_items = [item for item in guidance_items if item.guidance_type == guidance_type]
            
            if type_items:
                pattern_analysis = self._analyze_guidance_pattern(type_items)
                consistency_analysis["patterns"][guidance_type.value] = pattern_analysis
                
                # Calculate consistency score
                score = self._calculate_consistency_score(type_items)
                consistency_analysis["consistency_scores"][guidance_type.value] = score
        
        return consistency_analysis
    
    def _analyze_guidance_pattern(self, guidance_items: List[GuidanceItem]) -> Dict[str, Any]:
        """Analyze patterns in guidance for a specific type."""
        if not guidance_items:
            return {}
        
        # Sort by date
        sorted_items = sorted(guidance_items, key=lambda x: x.announcement_date)
        
        # Calculate revision statistics
        revisions = [item for item in sorted_items if item.direction and item.direction != GuidanceDirection.INTRODUCED]
        
        revision_stats = {
            "total_revisions": len(revisions),
            "raises": len([r for r in revisions if r.direction == GuidanceDirection.RAISED]),
            "lowers": len([r for r in revisions if r.direction == GuidanceDirection.LOWERED]),
            "maintains": len([r for r in revisions if r.direction == GuidanceDirection.MAINTAINED]),
            "withdrawals": len([r for r in revisions if r.direction == GuidanceDirection.WITHDRAWN]),
        }
        
        # Calculate volatility
        magnitude_changes = [abs(item.magnitude_change) for item in revisions if item.magnitude_change]
        volatility = np.std(magnitude_changes) if magnitude_changes else 0.0
        
        # Identify trends
        recent_items = sorted_items[-4:]  # Last 4 guidance updates
        trend = "stable"
        
        if len(recent_items) >= 3:
            recent_directions = [item.direction for item in recent_items if item.direction]
            
            if recent_directions.count(GuidanceDirection.RAISED) >= 2:
                trend = "upward_revisions"
            elif recent_directions.count(GuidanceDirection.LOWERED) >= 2:
                trend = "downward_revisions"
            elif len(set(recent_directions)) > 2:
                trend = "volatile"
        
        return {
            "revision_stats": revision_stats,
            "volatility": volatility,
            "trend": trend,
            "avg_magnitude_change": np.mean(magnitude_changes) if magnitude_changes else 0.0,
            "confidence_distribution": {
                level: len([item for item in sorted_items if item.confidence_level == level])
                for level in ["low", "medium", "high"]
            },
        }
    
    def _calculate_consistency_score(self, guidance_items: List[GuidanceItem]) -> GuidanceConsistencyScore:
        """Calculate consistency score for guidance items."""
        if not guidance_items:
            return GuidanceConsistencyScore(
                ticker="",
                period="",
                overall_score=0,
                accuracy_score=0,
                volatility_score=0,
                timing_score=0,
                transparency_score=0,
                guidance_changes=0,
                beats_vs_misses={},
                avg_revision_magnitude=0,
            )
        
        ticker = guidance_items[0].ticker
        
        # Calculate component scores
        accuracy_score = self._calculate_accuracy_score(guidance_items)
        volatility_score = self._calculate_volatility_score(guidance_items)
        timing_score = self._calculate_timing_score(guidance_items)
        transparency_score = self._calculate_transparency_score(guidance_items)
        
        # Overall score (weighted average)
        overall_score = (
            accuracy_score * 0.4 +
            volatility_score * 0.3 +
            timing_score * 0.2 +
            transparency_score * 0.1
        )
        
        # Supporting metrics
        revisions = [item for item in guidance_items if item.direction and item.direction != GuidanceDirection.INTRODUCED]
        magnitude_changes = [abs(item.magnitude_change) for item in revisions if item.magnitude_change]
        
        # Generate insights
        strengths = []
        concerns = []
        recommendations = []
        
        if accuracy_score > 80:
            strengths.append("High accuracy in guidance delivery")
        elif accuracy_score < 60:
            concerns.append("Frequent guidance misses")
            recommendations.append("Improve forecasting accuracy")
        
        if volatility_score > 80:
            strengths.append("Stable and predictable guidance")
        elif volatility_score < 60:
            concerns.append("High guidance volatility")
            recommendations.append("Provide more conservative initial guidance")
        
        return GuidanceConsistencyScore(
            ticker=ticker,
            period="",
            overall_score=overall_score,
            accuracy_score=accuracy_score,
            volatility_score=volatility_score,
            timing_score=timing_score,
            transparency_score=transparency_score,
            guidance_changes=len(revisions),
            beats_vs_misses={"beats": 0, "misses": 0, "meets": 0},  # Would need actual results
            avg_revision_magnitude=np.mean(magnitude_changes) if magnitude_changes else 0.0,
            strengths=strengths,
            concerns=concerns,
            recommendations=recommendations,
        )
    
    def _calculate_accuracy_score(self, guidance_items: List[GuidanceItem]) -> float:
        """Calculate accuracy score based on guidance vs actual results."""
        # In production, compare guidance to actual reported results
        # For now, return mock score based on revision frequency
        
        revisions = [item for item in guidance_items if item.direction and item.direction != GuidanceDirection.INTRODUCED]
        
        if not guidance_items:
            return 50.0
        
        revision_rate = len(revisions) / len(guidance_items)
        
        # Lower revision rate = higher accuracy
        accuracy_score = max(0, 100 - (revision_rate * 100))
        
        return accuracy_score
    
    def _calculate_volatility_score(self, guidance_items: List[GuidanceItem]) -> float:
        """Calculate volatility score based on guidance stability."""
        magnitude_changes = [abs(item.magnitude_change) for item in guidance_items if item.magnitude_change]
        
        if not magnitude_changes:
            return 100.0
        
        volatility = np.std(magnitude_changes)
        
        # Lower volatility = higher score
        volatility_score = max(0, 100 - (volatility * 1000))  # Scale appropriately
        
        return volatility_score
    
    def _calculate_timing_score(self, guidance_items: List[GuidanceItem]) -> float:
        """Calculate timing score based on guidance update frequency."""
        # In production, analyze timing relative to earnings releases
        # For now, return score based on update frequency
        
        if len(guidance_items) < 2:
            return 50.0
        
        # Analyze update timing patterns
        sorted_items = sorted(guidance_items, key=lambda x: x.announcement_date)
        time_gaps = []
        
        for i in range(1, len(sorted_items)):
            gap = (sorted_items[i].announcement_date - sorted_items[i-1].announcement_date).days
            time_gaps.append(gap)
        
        if not time_gaps:
            return 50.0
        
        avg_gap = np.mean(time_gaps)
        
        # Optimal gap is around 90 days (quarterly)
        optimal_gap = 90
        gap_deviation = abs(avg_gap - optimal_gap) / optimal_gap
        
        timing_score = max(0, 100 - (gap_deviation * 100))
        
        return timing_score
    
    def _calculate_transparency_score(self, guidance_items: List[GuidanceItem]) -> float:
        """Calculate transparency score based on guidance detail and context."""
        if not guidance_items:
            return 0.0
        
        # Score based on context richness and confidence indicators
        context_scores = []
        
        for item in guidance_items:
            score = 0
            
            # Context length (more detail = higher score)
            if len(item.context) > 100:
                score += 30
            elif len(item.context) > 50:
                score += 20
            elif len(item.context) > 20:
                score += 10
            
            # Confidence level clarity
            if item.confidence_level == "high":
                score += 30
            elif item.confidence_level == "medium":
                score += 20
            else:
                score += 10
            
            # Reasoning provided
            if item.reasoning:
                score += 25
            
            # Assumptions listed
            if item.assumptions:
                score += 15
            
            context_scores.append(min(score, 100))
        
        return np.mean(context_scores)


class GuidanceConsistencyService:
    """Main service for guidance consistency analysis."""
    
    def __init__(self) -> None:
        """Initialize guidance consistency service."""
        self.tracker = GuidanceTracker()
    
    @trace_function("guidance_service.analyze_ticker")
    async def analyze_ticker(
        self,
        db: AsyncSession,
        ticker: str,
        periods: int = 8,
    ) -> Dict[str, Any]:
        """Analyze guidance consistency for a ticker."""
        # Get recent documents for ticker
        # In production, query actual documents
        
        # For now, generate mock analysis
        consistency_map = self.tracker.generate_consistency_map(ticker, periods)
        
        return {
            "ticker": ticker,
            "consistency_map": consistency_map,
            "analysis_summary": {
                "overall_grade": "B+",
                "key_strengths": [
                    "Consistent revenue guidance accuracy",
                    "Transparent communication style",
                    "Timely guidance updates",
                ],
                "areas_for_improvement": [
                    "Margin guidance volatility",
                    "Conservative initial estimates",
                ],
                "investor_confidence_impact": "Positive - builds trust through consistent delivery",
            },
        }
    
    @trace_function("guidance_service.compare_peers")
    async def compare_peers(
        self,
        db: AsyncSession,
        tickers: List[str],
    ) -> Dict[str, Any]:
        """Compare guidance consistency across peer companies."""
        peer_analysis = {}
        
        for ticker in tickers:
            analysis = await self.analyze_ticker(db, ticker)
            peer_analysis[ticker] = analysis
        
        # Rank by consistency scores
        rankings = {}
        for ticker, analysis in peer_analysis.items():
            if "consistency_map" in analysis and "consistency_scores" in analysis["consistency_map"]:
                scores = analysis["consistency_map"]["consistency_scores"]
                avg_score = np.mean(list(scores.values())) if scores else 0
                rankings[ticker] = avg_score
        
        sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "peer_comparison": peer_analysis,
            "rankings": sorted_rankings,
            "industry_insights": {
                "best_practices": [
                    "Provide quarterly guidance updates",
                    "Include detailed assumptions",
                    "Maintain conservative initial estimates",
                ],
                "common_pitfalls": [
                    "Over-optimistic initial guidance",
                    "Infrequent updates during volatile periods",
                    "Lack of context for guidance changes",
                ],
            },
        }
