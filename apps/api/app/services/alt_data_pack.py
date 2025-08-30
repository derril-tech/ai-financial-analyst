"""Alternative Data Pack for web, jobs, app rankings, and satellite data integration."""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import re

import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.observability import trace_function
from app.core.config import settings


class AltDataSource(Enum):
    """Types of alternative data sources."""
    WEB_SCRAPING = "web_scraping"
    JOB_POSTINGS = "job_postings"
    APP_RANKINGS = "app_rankings"
    SATELLITE_IMAGERY = "satellite_imagery"
    SOCIAL_SENTIMENT = "social_sentiment"
    PATENT_FILINGS = "patent_filings"
    EXECUTIVE_MOVEMENTS = "executive_movements"
    SUPPLY_CHAIN = "supply_chain"
    ESG_METRICS = "esg_metrics"
    FOOT_TRAFFIC = "foot_traffic"


class DataFrequency(Enum):
    """Data collection frequency."""
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class AltDataPoint:
    """Individual alternative data point."""
    source: AltDataSource
    ticker: str
    timestamp: datetime
    
    # Data content
    metric_name: str
    value: float
    unit: str
    
    # Metadata
    confidence_score: float  # 0-1 scale
    data_quality: str  # "high", "medium", "low"
    collection_method: str
    
    # Context
    raw_data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Validation
    is_validated: bool = False
    validation_notes: str = ""


@dataclass
class AltDataInsight:
    """Insight derived from alternative data."""
    ticker: str
    insight_type: str
    title: str
    description: str
    
    # Supporting data
    supporting_metrics: List[AltDataPoint]
    confidence_level: str  # "high", "medium", "low"
    
    # Impact assessment
    potential_impact: str  # "positive", "negative", "neutral"
    magnitude: str  # "high", "medium", "low"
    time_horizon: str  # "immediate", "short_term", "medium_term", "long_term"
    
    # Investment relevance
    investment_thesis: str
    risk_factors: List[str] = field(default_factory=list)
    
    # Generated insights
    generated_at: datetime = field(default_factory=datetime.utcnow)


class WebScrapingService:
    """Web scraping service for company data."""
    
    def __init__(self) -> None:
        """Initialize web scraping service."""
        self.session = None
        self.rate_limits = {}  # Domain -> last request time
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    @trace_function("web_scraping.scrape_company_news")
    async def scrape_company_news(
        self,
        ticker: str,
        sources: List[str] = None,
        lookback_days: int = 7,
    ) -> List[AltDataPoint]:
        """Scrape company news and sentiment."""
        if sources is None:
            sources = ["reuters", "bloomberg", "yahoo_finance"]
        
        data_points = []
        
        for source in sources:
            try:
                news_data = await self._scrape_news_source(ticker, source, lookback_days)
                
                # Process news sentiment
                sentiment_score = self._analyze_news_sentiment(news_data)
                
                data_point = AltDataPoint(
                    source=AltDataSource.WEB_SCRAPING,
                    ticker=ticker,
                    timestamp=datetime.utcnow(),
                    metric_name=f"news_sentiment_{source}",
                    value=sentiment_score,
                    unit="sentiment_score",
                    confidence_score=0.7,
                    data_quality="medium",
                    collection_method="web_scraping",
                    raw_data={"articles": news_data, "source": source},
                    tags=["news", "sentiment", source],
                )
                
                data_points.append(data_point)
                
            except Exception as e:
                print(f"Error scraping {source} for {ticker}: {e}")
        
        return data_points
    
    async def _scrape_news_source(
        self,
        ticker: str,
        source: str,
        lookback_days: int,
    ) -> List[Dict[str, Any]]:
        """Scrape news from specific source."""
        # Mock implementation - in production, use actual scraping
        
        # Simulate rate limiting
        await self._respect_rate_limit(source)
        
        # Mock news articles
        mock_articles = [
            {
                "title": f"{ticker} Reports Strong Quarterly Results",
                "content": f"Company {ticker} exceeded expectations with strong revenue growth...",
                "published_date": datetime.utcnow() - timedelta(days=1),
                "url": f"https://{source}.com/article/{ticker}-results",
            },
            {
                "title": f"{ticker} Announces New Product Launch",
                "content": f"{ticker} unveiled its latest innovation targeting new markets...",
                "published_date": datetime.utcnow() - timedelta(days=3),
                "url": f"https://{source}.com/article/{ticker}-product",
            },
        ]
        
        return mock_articles
    
    async def _respect_rate_limit(self, domain: str, min_interval: float = 1.0) -> None:
        """Respect rate limits for domain."""
        now = datetime.utcnow().timestamp()
        
        if domain in self.rate_limits:
            time_since_last = now - self.rate_limits[domain]
            if time_since_last < min_interval:
                await asyncio.sleep(min_interval - time_since_last)
        
        self.rate_limits[domain] = now
    
    def _analyze_news_sentiment(self, articles: List[Dict[str, Any]]) -> float:
        """Analyze sentiment of news articles."""
        # Simplified sentiment analysis
        positive_keywords = ["strong", "growth", "exceeded", "positive", "beat", "outperformed"]
        negative_keywords = ["weak", "decline", "missed", "negative", "disappointed", "underperformed"]
        
        sentiment_scores = []
        
        for article in articles:
            text = f"{article['title']} {article['content']}".lower()
            
            positive_count = sum(1 for keyword in positive_keywords if keyword in text)
            negative_count = sum(1 for keyword in negative_keywords if keyword in text)
            
            # Calculate sentiment score (-1 to 1)
            if positive_count + negative_count > 0:
                sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            else:
                sentiment = 0.0
            
            sentiment_scores.append(sentiment)
        
        return np.mean(sentiment_scores) if sentiment_scores else 0.0


class JobPostingsAnalyzer:
    """Analyze job postings for hiring trends and company growth signals."""
    
    def __init__(self) -> None:
        """Initialize job postings analyzer."""
        self.job_boards = ["linkedin", "indeed", "glassdoor", "company_careers"]
    
    @trace_function("job_analyzer.analyze_hiring_trends")
    async def analyze_hiring_trends(
        self,
        ticker: str,
        company_name: str,
        lookback_months: int = 6,
    ) -> List[AltDataPoint]:
        """Analyze hiring trends for company."""
        data_points = []
        
        # Get job posting data
        job_data = await self._collect_job_postings(company_name, lookback_months)
        
        # Analyze hiring velocity
        hiring_velocity = self._calculate_hiring_velocity(job_data)
        
        data_points.append(AltDataPoint(
            source=AltDataSource.JOB_POSTINGS,
            ticker=ticker,
            timestamp=datetime.utcnow(),
            metric_name="hiring_velocity",
            value=hiring_velocity,
            unit="jobs_per_month",
            confidence_score=0.8,
            data_quality="high",
            collection_method="job_board_api",
            raw_data={"job_postings": len(job_data)},
            tags=["hiring", "growth", "employment"],
        ))
        
        # Analyze skill demand trends
        skill_trends = self._analyze_skill_trends(job_data)
        
        for skill, trend_score in skill_trends.items():
            data_points.append(AltDataPoint(
                source=AltDataSource.JOB_POSTINGS,
                ticker=ticker,
                timestamp=datetime.utcnow(),
                metric_name=f"skill_demand_{skill}",
                value=trend_score,
                unit="trend_score",
                confidence_score=0.7,
                data_quality="medium",
                collection_method="skill_extraction",
                raw_data={"skill": skill},
                tags=["skills", "technology", "trends"],
            ))
        
        # Analyze geographic expansion
        geo_expansion = self._analyze_geographic_expansion(job_data)
        
        data_points.append(AltDataPoint(
            source=AltDataSource.JOB_POSTINGS,
            ticker=ticker,
            timestamp=datetime.utcnow(),
            metric_name="geographic_expansion",
            value=geo_expansion,
            unit="expansion_score",
            confidence_score=0.6,
            data_quality="medium",
            collection_method="location_analysis",
            raw_data={"locations": len(set(job["location"] for job in job_data))},
            tags=["expansion", "geography", "growth"],
        ))
        
        return data_points
    
    async def _collect_job_postings(
        self,
        company_name: str,
        lookback_months: int,
    ) -> List[Dict[str, Any]]:
        """Collect job postings for company."""
        # Mock job postings data
        mock_jobs = []
        
        for i in range(50):  # Mock 50 job postings
            job = {
                "title": f"Software Engineer {i}",
                "department": np.random.choice(["Engineering", "Sales", "Marketing", "Operations"]),
                "location": np.random.choice(["San Francisco", "New York", "Austin", "Remote"]),
                "posted_date": datetime.utcnow() - timedelta(days=np.random.randint(0, lookback_months * 30)),
                "skills": np.random.choice([
                    ["Python", "AWS", "Machine Learning"],
                    ["React", "Node.js", "TypeScript"],
                    ["Java", "Spring", "Kubernetes"],
                    ["Data Science", "SQL", "Tableau"],
                ]),
                "seniority": np.random.choice(["Junior", "Mid", "Senior", "Staff"]),
            }
            mock_jobs.append(job)
        
        return mock_jobs
    
    def _calculate_hiring_velocity(self, job_data: List[Dict[str, Any]]) -> float:
        """Calculate hiring velocity (jobs per month)."""
        if not job_data:
            return 0.0
        
        # Group by month
        monthly_counts = {}
        for job in job_data:
            month_key = job["posted_date"].strftime("%Y-%m")
            monthly_counts[month_key] = monthly_counts.get(month_key, 0) + 1
        
        # Calculate average monthly hiring
        if monthly_counts:
            return np.mean(list(monthly_counts.values()))
        
        return 0.0
    
    def _analyze_skill_trends(self, job_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze trending skills in job postings."""
        skill_counts = {}
        
        for job in job_data:
            for skill in job.get("skills", []):
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        # Calculate trend scores (normalized by total jobs)
        total_jobs = len(job_data)
        skill_trends = {}
        
        for skill, count in skill_counts.items():
            # Trend score based on frequency and recency
            trend_score = count / total_jobs
            skill_trends[skill] = trend_score
        
        # Return top 5 trending skills
        sorted_skills = sorted(skill_trends.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_skills[:5])
    
    def _analyze_geographic_expansion(self, job_data: List[Dict[str, Any]]) -> float:
        """Analyze geographic expansion based on job locations."""
        locations = set(job["location"] for job in job_data)
        
        # Expansion score based on number of unique locations
        # and distribution of jobs across locations
        location_counts = {}
        for job in job_data:
            location = job["location"]
            location_counts[location] = location_counts.get(location, 0) + 1
        
        # Calculate distribution entropy (higher = more distributed)
        total_jobs = len(job_data)
        entropy = 0.0
        
        for count in location_counts.values():
            prob = count / total_jobs
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        # Normalize entropy by max possible entropy
        max_entropy = np.log2(len(locations)) if len(locations) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Expansion score combines location diversity and distribution
        expansion_score = len(locations) * normalized_entropy
        
        return expansion_score


class AppRankingsTracker:
    """Track mobile app rankings and performance metrics."""
    
    def __init__(self) -> None:
        """Initialize app rankings tracker."""
        self.app_stores = ["ios_app_store", "google_play_store"]
        self.categories = ["overall", "finance", "productivity", "social", "games"]
    
    @trace_function("app_tracker.track_app_performance")
    async def track_app_performance(
        self,
        ticker: str,
        app_names: List[str],
    ) -> List[AltDataPoint]:
        """Track app performance metrics."""
        data_points = []
        
        for app_name in app_names:
            # Get app ranking data
            ranking_data = await self._get_app_rankings(app_name)
            
            # Process rankings for each store and category
            for store in self.app_stores:
                for category in self.categories:
                    ranking = ranking_data.get(store, {}).get(category, {})
                    
                    if ranking:
                        # Current ranking
                        data_points.append(AltDataPoint(
                            source=AltDataSource.APP_RANKINGS,
                            ticker=ticker,
                            timestamp=datetime.utcnow(),
                            metric_name=f"app_ranking_{store}_{category}",
                            value=ranking.get("current_rank", 999),
                            unit="rank_position",
                            confidence_score=0.9,
                            data_quality="high",
                            collection_method="app_store_api",
                            raw_data=ranking,
                            tags=["app", "ranking", store, category],
                        ))
                        
                        # Rating score
                        if "rating" in ranking:
                            data_points.append(AltDataPoint(
                                source=AltDataSource.APP_RANKINGS,
                                ticker=ticker,
                                timestamp=datetime.utcnow(),
                                metric_name=f"app_rating_{store}",
                                value=ranking["rating"],
                                unit="rating_score",
                                confidence_score=0.9,
                                data_quality="high",
                                collection_method="app_store_api",
                                raw_data={"app_name": app_name},
                                tags=["app", "rating", store],
                            ))
                        
                        # Download velocity
                        if "download_velocity" in ranking:
                            data_points.append(AltDataPoint(
                                source=AltDataSource.APP_RANKINGS,
                                ticker=ticker,
                                timestamp=datetime.utcnow(),
                                metric_name=f"app_downloads_{store}",
                                value=ranking["download_velocity"],
                                unit="downloads_per_day",
                                confidence_score=0.7,
                                data_quality="medium",
                                collection_method="estimated_downloads",
                                raw_data={"app_name": app_name},
                                tags=["app", "downloads", store],
                            ))
        
        return data_points
    
    async def _get_app_rankings(self, app_name: str) -> Dict[str, Any]:
        """Get app rankings from app stores."""
        # Mock app ranking data
        mock_data = {
            "ios_app_store": {
                "overall": {
                    "current_rank": np.random.randint(1, 200),
                    "previous_rank": np.random.randint(1, 200),
                    "rating": np.random.uniform(3.5, 5.0),
                    "review_count": np.random.randint(1000, 50000),
                    "download_velocity": np.random.randint(100, 10000),
                },
                "finance": {
                    "current_rank": np.random.randint(1, 50),
                    "previous_rank": np.random.randint(1, 50),
                    "rating": np.random.uniform(4.0, 5.0),
                    "review_count": np.random.randint(500, 20000),
                    "download_velocity": np.random.randint(50, 5000),
                },
            },
            "google_play_store": {
                "overall": {
                    "current_rank": np.random.randint(1, 200),
                    "previous_rank": np.random.randint(1, 200),
                    "rating": np.random.uniform(3.5, 5.0),
                    "review_count": np.random.randint(2000, 100000),
                    "download_velocity": np.random.randint(200, 20000),
                },
                "finance": {
                    "current_rank": np.random.randint(1, 50),
                    "previous_rank": np.random.randint(1, 50),
                    "rating": np.random.uniform(4.0, 5.0),
                    "review_count": np.random.randint(1000, 40000),
                    "download_velocity": np.random.randint(100, 10000),
                },
            },
        }
        
        return mock_data


class SatelliteImageryAnalyzer:
    """Analyze satellite imagery for business intelligence."""
    
    def __init__(self) -> None:
        """Initialize satellite imagery analyzer."""
        self.imagery_providers = ["planet", "maxar", "sentinel"]
    
    @trace_function("satellite_analyzer.analyze_facility_activity")
    async def analyze_facility_activity(
        self,
        ticker: str,
        facility_locations: List[Tuple[float, float]],  # (lat, lon) pairs
        analysis_period_days: int = 30,
    ) -> List[AltDataPoint]:
        """Analyze facility activity from satellite imagery."""
        data_points = []
        
        for i, (lat, lon) in enumerate(facility_locations):
            # Get satellite imagery data
            imagery_data = await self._get_satellite_imagery(lat, lon, analysis_period_days)
            
            # Analyze parking lot occupancy
            parking_occupancy = self._analyze_parking_occupancy(imagery_data)
            
            data_points.append(AltDataPoint(
                source=AltDataSource.SATELLITE_IMAGERY,
                ticker=ticker,
                timestamp=datetime.utcnow(),
                metric_name=f"parking_occupancy_facility_{i}",
                value=parking_occupancy,
                unit="occupancy_percentage",
                confidence_score=0.8,
                data_quality="high",
                collection_method="satellite_image_analysis",
                raw_data={"lat": lat, "lon": lon, "facility_id": i},
                tags=["satellite", "parking", "activity"],
            ))
            
            # Analyze construction activity
            construction_activity = self._analyze_construction_activity(imagery_data)
            
            data_points.append(AltDataPoint(
                source=AltDataSource.SATELLITE_IMAGERY,
                ticker=ticker,
                timestamp=datetime.utcnow(),
                metric_name=f"construction_activity_facility_{i}",
                value=construction_activity,
                unit="activity_score",
                confidence_score=0.7,
                data_quality="medium",
                collection_method="change_detection",
                raw_data={"lat": lat, "lon": lon, "facility_id": i},
                tags=["satellite", "construction", "expansion"],
            ))
            
            # Analyze shipping/logistics activity
            logistics_activity = self._analyze_logistics_activity(imagery_data)
            
            data_points.append(AltDataPoint(
                source=AltDataSource.SATELLITE_IMAGERY,
                ticker=ticker,
                timestamp=datetime.utcnow(),
                metric_name=f"logistics_activity_facility_{i}",
                value=logistics_activity,
                unit="activity_score",
                confidence_score=0.6,
                data_quality="medium",
                collection_method="object_detection",
                raw_data={"lat": lat, "lon": lon, "facility_id": i},
                tags=["satellite", "logistics", "shipping"],
            ))
        
        return data_points
    
    async def _get_satellite_imagery(
        self,
        lat: float,
        lon: float,
        days: int,
    ) -> Dict[str, Any]:
        """Get satellite imagery for location."""
        # Mock satellite imagery data
        mock_data = {
            "location": {"lat": lat, "lon": lon},
            "images": [
                {
                    "date": datetime.utcnow() - timedelta(days=i),
                    "cloud_cover": np.random.uniform(0, 30),
                    "resolution": "3m",
                    "parking_spots_occupied": np.random.randint(50, 200),
                    "total_parking_spots": 250,
                    "construction_area": np.random.uniform(0, 5000),  # sq meters
                    "truck_count": np.random.randint(0, 20),
                }
                for i in range(0, days, 3)  # Every 3 days
            ],
        }
        
        return mock_data
    
    def _analyze_parking_occupancy(self, imagery_data: Dict[str, Any]) -> float:
        """Analyze parking lot occupancy trends."""
        images = imagery_data.get("images", [])
        
        if not images:
            return 0.0
        
        occupancy_rates = []
        for image in images:
            occupied = image.get("parking_spots_occupied", 0)
            total = image.get("total_parking_spots", 1)
            occupancy_rate = (occupied / total) * 100
            occupancy_rates.append(occupancy_rate)
        
        # Return average occupancy rate
        return np.mean(occupancy_rates)
    
    def _analyze_construction_activity(self, imagery_data: Dict[str, Any]) -> float:
        """Analyze construction activity levels."""
        images = imagery_data.get("images", [])
        
        if len(images) < 2:
            return 0.0
        
        # Calculate change in construction area over time
        construction_areas = [img.get("construction_area", 0) for img in images]
        
        # Calculate trend (positive = increasing construction)
        if len(construction_areas) > 1:
            # Simple linear trend
            x = np.arange(len(construction_areas))
            slope, _ = np.polyfit(x, construction_areas, 1)
            
            # Normalize to 0-100 scale
            activity_score = min(max(slope / 100, 0), 100)
            return activity_score
        
        return 0.0
    
    def _analyze_logistics_activity(self, imagery_data: Dict[str, Any]) -> float:
        """Analyze logistics and shipping activity."""
        images = imagery_data.get("images", [])
        
        if not images:
            return 0.0
        
        truck_counts = [img.get("truck_count", 0) for img in images]
        
        # Calculate average truck activity
        avg_trucks = np.mean(truck_counts)
        
        # Normalize to activity score (0-100)
        # Assume 10+ trucks = high activity (100)
        activity_score = min((avg_trucks / 10) * 100, 100)
        
        return activity_score


class AltDataInsightGenerator:
    """Generate actionable insights from alternative data."""
    
    def __init__(self) -> None:
        """Initialize insight generator."""
        self.insight_templates = self._load_insight_templates()
    
    def _load_insight_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load insight generation templates."""
        return {
            "hiring_acceleration": {
                "threshold": 20,  # jobs per month
                "impact": "positive",
                "magnitude": "medium",
                "time_horizon": "medium_term",
                "template": "Company showing strong hiring acceleration with {value} jobs per month, indicating business expansion and growth confidence.",
            },
            "app_ranking_improvement": {
                "threshold": -10,  # negative = improvement in rank
                "impact": "positive",
                "magnitude": "medium",
                "time_horizon": "short_term",
                "template": "Mobile app ranking improved significantly, moving up {value} positions, suggesting increased user adoption and engagement.",
            },
            "facility_expansion": {
                "threshold": 50,  # construction activity score
                "impact": "positive",
                "magnitude": "high",
                "time_horizon": "long_term",
                "template": "Satellite imagery shows significant construction activity (score: {value}), indicating facility expansion and capacity growth.",
            },
            "negative_sentiment": {
                "threshold": -0.3,  # sentiment score
                "impact": "negative",
                "magnitude": "medium",
                "time_horizon": "short_term",
                "template": "News sentiment analysis shows negative trend (score: {value}), potentially impacting near-term stock performance.",
            },
        }
    
    @trace_function("insight_generator.generate_insights")
    def generate_insights(
        self,
        ticker: str,
        data_points: List[AltDataPoint],
    ) -> List[AltDataInsight]:
        """Generate insights from alternative data points."""
        insights = []
        
        # Group data points by metric type
        metrics_by_type = {}
        for dp in data_points:
            metric_type = dp.metric_name.split('_')[0]  # First part of metric name
            if metric_type not in metrics_by_type:
                metrics_by_type[metric_type] = []
            metrics_by_type[metric_type].append(dp)
        
        # Generate insights for each metric type
        for metric_type, points in metrics_by_type.items():
            metric_insights = self._generate_metric_insights(ticker, metric_type, points)
            insights.extend(metric_insights)
        
        # Generate cross-metric insights
        cross_insights = self._generate_cross_metric_insights(ticker, data_points)
        insights.extend(cross_insights)
        
        return insights
    
    def _generate_metric_insights(
        self,
        ticker: str,
        metric_type: str,
        data_points: List[AltDataPoint],
    ) -> List[AltDataInsight]:
        """Generate insights for specific metric type."""
        insights = []
        
        if not data_points:
            return insights
        
        # Analyze trends
        values = [dp.value for dp in data_points]
        avg_value = np.mean(values)
        
        # Check against thresholds
        for insight_type, template in self.insight_templates.items():
            if metric_type in insight_type:
                threshold = template["threshold"]
                
                if (threshold > 0 and avg_value > threshold) or (threshold < 0 and avg_value < threshold):
                    insight = AltDataInsight(
                        ticker=ticker,
                        insight_type=insight_type,
                        title=f"{metric_type.replace('_', ' ').title()} Signal Detected",
                        description=template["template"].format(value=avg_value),
                        supporting_metrics=data_points,
                        confidence_level=self._calculate_confidence_level(data_points),
                        potential_impact=template["impact"],
                        magnitude=template["magnitude"],
                        time_horizon=template["time_horizon"],
                        investment_thesis=self._generate_investment_thesis(insight_type, avg_value),
                        risk_factors=self._identify_risk_factors(insight_type),
                    )
                    
                    insights.append(insight)
        
        return insights
    
    def _generate_cross_metric_insights(
        self,
        ticker: str,
        data_points: List[AltDataPoint],
    ) -> List[AltDataInsight]:
        """Generate insights from multiple data sources."""
        insights = []
        
        # Look for correlated signals
        hiring_data = [dp for dp in data_points if "hiring" in dp.metric_name]
        construction_data = [dp for dp in data_points if "construction" in dp.metric_name]
        
        if hiring_data and construction_data:
            hiring_avg = np.mean([dp.value for dp in hiring_data])
            construction_avg = np.mean([dp.value for dp in construction_data])
            
            if hiring_avg > 15 and construction_avg > 40:
                insight = AltDataInsight(
                    ticker=ticker,
                    insight_type="expansion_convergence",
                    title="Multiple Growth Signals Detected",
                    description=f"Both hiring acceleration ({hiring_avg:.1f} jobs/month) and facility expansion (construction score: {construction_avg:.1f}) suggest coordinated business growth.",
                    supporting_metrics=hiring_data + construction_data,
                    confidence_level="high",
                    potential_impact="positive",
                    magnitude="high",
                    time_horizon="medium_term",
                    investment_thesis="Multiple independent data sources confirm business expansion, reducing false signal risk",
                    risk_factors=["Economic downturn could halt expansion", "Regulatory changes affecting growth plans"],
                )
                
                insights.append(insight)
        
        return insights
    
    def _calculate_confidence_level(self, data_points: List[AltDataPoint]) -> str:
        """Calculate confidence level for insight."""
        if not data_points:
            return "low"
        
        avg_confidence = np.mean([dp.confidence_score for dp in data_points])
        data_quality_scores = {"high": 1.0, "medium": 0.7, "low": 0.4}
        avg_quality = np.mean([data_quality_scores.get(dp.data_quality, 0.5) for dp in data_points])
        
        combined_score = (avg_confidence + avg_quality) / 2
        
        if combined_score > 0.8:
            return "high"
        elif combined_score > 0.6:
            return "medium"
        else:
            return "low"
    
    def _generate_investment_thesis(self, insight_type: str, value: float) -> str:
        """Generate investment thesis for insight."""
        thesis_templates = {
            "hiring_acceleration": f"Increased hiring suggests revenue growth expectations and market share expansion opportunities.",
            "app_ranking_improvement": f"Improved app rankings indicate stronger user engagement and potential revenue acceleration.",
            "facility_expansion": f"Capital investment in facilities suggests management confidence in long-term growth prospects.",
            "negative_sentiment": f"Negative sentiment may create temporary valuation discount and potential buying opportunity.",
        }
        
        return thesis_templates.get(insight_type, "Alternative data signal suggests potential investment opportunity.")
    
    def _identify_risk_factors(self, insight_type: str) -> List[str]:
        """Identify risk factors for insight type."""
        risk_factors = {
            "hiring_acceleration": [
                "Hiring may not translate to immediate revenue growth",
                "Increased labor costs could impact margins",
                "Economic downturn could force hiring freezes",
            ],
            "app_ranking_improvement": [
                "App store rankings can be volatile",
                "Improved rankings may not sustain long-term",
                "Competition could quickly erode gains",
            ],
            "facility_expansion": [
                "Construction delays or cost overruns",
                "Demand may not materialize as expected",
                "Regulatory approval risks",
            ],
            "negative_sentiment": [
                "Sentiment could worsen further",
                "Negative news may have lasting impact",
                "Market overreaction risks",
            ],
        }
        
        return risk_factors.get(insight_type, ["General market risks", "Data quality limitations"])


class AltDataPackService:
    """Main service for alternative data integration and analysis."""
    
    def __init__(self) -> None:
        """Initialize alt data pack service."""
        self.web_scraper = WebScrapingService()
        self.job_analyzer = JobPostingsAnalyzer()
        self.app_tracker = AppRankingsTracker()
        self.satellite_analyzer = SatelliteImageryAnalyzer()
        self.insight_generator = AltDataInsightGenerator()
    
    @trace_function("alt_data_service.collect_comprehensive_data")
    async def collect_comprehensive_data(
        self,
        db: AsyncSession,
        ticker: str,
        company_name: str,
        data_sources: List[AltDataSource] = None,
        config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Collect comprehensive alternative data for company."""
        if data_sources is None:
            data_sources = [
                AltDataSource.WEB_SCRAPING,
                AltDataSource.JOB_POSTINGS,
                AltDataSource.APP_RANKINGS,
                AltDataSource.SATELLITE_IMAGERY,
            ]
        
        if config is None:
            config = {}
        
        all_data_points = []
        collection_results = {}
        
        # Collect data from each source
        for source in data_sources:
            try:
                if source == AltDataSource.WEB_SCRAPING:
                    async with self.web_scraper:
                        data_points = await self.web_scraper.scrape_company_news(ticker)
                        all_data_points.extend(data_points)
                        collection_results["web_scraping"] = {"status": "success", "data_points": len(data_points)}
                
                elif source == AltDataSource.JOB_POSTINGS:
                    data_points = await self.job_analyzer.analyze_hiring_trends(ticker, company_name)
                    all_data_points.extend(data_points)
                    collection_results["job_postings"] = {"status": "success", "data_points": len(data_points)}
                
                elif source == AltDataSource.APP_RANKINGS:
                    app_names = config.get("app_names", [company_name])
                    data_points = await self.app_tracker.track_app_performance(ticker, app_names)
                    all_data_points.extend(data_points)
                    collection_results["app_rankings"] = {"status": "success", "data_points": len(data_points)}
                
                elif source == AltDataSource.SATELLITE_IMAGERY:
                    facility_locations = config.get("facility_locations", [(37.7749, -122.4194)])  # Default SF
                    data_points = await self.satellite_analyzer.analyze_facility_activity(ticker, facility_locations)
                    all_data_points.extend(data_points)
                    collection_results["satellite_imagery"] = {"status": "success", "data_points": len(data_points)}
                
            except Exception as e:
                collection_results[source.value] = {"status": "error", "error": str(e)}
        
        # Generate insights
        insights = self.insight_generator.generate_insights(ticker, all_data_points)
        
        # Create summary
        summary = self._create_data_summary(all_data_points, insights)
        
        return {
            "ticker": ticker,
            "company_name": company_name,
            "collection_timestamp": datetime.utcnow().isoformat(),
            "data_points": [
                {
                    "source": dp.source.value,
                    "metric_name": dp.metric_name,
                    "value": dp.value,
                    "unit": dp.unit,
                    "confidence_score": dp.confidence_score,
                    "timestamp": dp.timestamp.isoformat(),
                    "tags": dp.tags,
                }
                for dp in all_data_points
            ],
            "insights": [
                {
                    "type": insight.insight_type,
                    "title": insight.title,
                    "description": insight.description,
                    "confidence": insight.confidence_level,
                    "impact": insight.potential_impact,
                    "magnitude": insight.magnitude,
                    "time_horizon": insight.time_horizon,
                    "investment_thesis": insight.investment_thesis,
                    "risk_factors": insight.risk_factors,
                }
                for insight in insights
            ],
            "collection_results": collection_results,
            "summary": summary,
        }
    
    def _create_data_summary(
        self,
        data_points: List[AltDataPoint],
        insights: List[AltDataInsight],
    ) -> Dict[str, Any]:
        """Create summary of collected data and insights."""
        # Data source breakdown
        source_counts = {}
        for dp in data_points:
            source = dp.source.value
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Insight breakdown
        positive_insights = len([i for i in insights if i.potential_impact == "positive"])
        negative_insights = len([i for i in insights if i.potential_impact == "negative"])
        
        # Confidence distribution
        high_confidence = len([i for i in insights if i.confidence_level == "high"])
        medium_confidence = len([i for i in insights if i.confidence_level == "medium"])
        low_confidence = len([i for i in insights if i.confidence_level == "low"])
        
        return {
            "total_data_points": len(data_points),
            "data_sources": source_counts,
            "total_insights": len(insights),
            "insight_sentiment": {
                "positive": positive_insights,
                "negative": negative_insights,
                "neutral": len(insights) - positive_insights - negative_insights,
            },
            "confidence_distribution": {
                "high": high_confidence,
                "medium": medium_confidence,
                "low": low_confidence,
            },
            "key_signals": [
                insight.title for insight in insights
                if insight.confidence_level == "high" and insight.magnitude in ["high", "medium"]
            ][:5],  # Top 5 key signals
        }
