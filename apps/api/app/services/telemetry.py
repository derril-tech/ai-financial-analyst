"""Telemetry and monitoring services for performance tracking."""

import time
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading

import psutil
import redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.observability import trace_function, get_request_id


@dataclass
class PerformanceMetric:
    """Performance metric data point."""
    timestamp: datetime
    metric_name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    request_id: Optional[str] = None


@dataclass
class CostMetric:
    """Cost tracking metric."""
    timestamp: datetime
    operation: str
    cost_usd: float
    tokens_used: int = 0
    api_calls: int = 0
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Centralized metrics collection system."""
    
    def __init__(self) -> None:
        """Initialize metrics collector."""
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.metrics_buffer = deque(maxlen=10000)
        self.cost_buffer = deque(maxlen=10000)
        self.lock = threading.Lock()
        
        # Performance counters
        self.request_counts = defaultdict(int)
        self.latency_buckets = defaultdict(list)
        self.error_counts = defaultdict(int)
        
        # Cost tracking
        self.daily_costs = defaultdict(float)
        self.operation_costs = defaultdict(float)
    
    @trace_function("metrics_collector.record_latency")
    def record_latency(
        self, 
        operation: str, 
        latency_ms: float, 
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record latency metric."""
        metric = PerformanceMetric(
            timestamp=datetime.utcnow(),
            metric_name="latency_ms",
            value=latency_ms,
            labels=labels or {},
            request_id=get_request_id(),
        )
        
        with self.lock:
            self.metrics_buffer.append(metric)
            self.latency_buckets[operation].append(latency_ms)
            
            # Keep only recent latencies for percentile calculation
            if len(self.latency_buckets[operation]) > 1000:
                self.latency_buckets[operation] = self.latency_buckets[operation][-1000:]
    
    @trace_function("metrics_collector.record_cost")
    def record_cost(
        self,
        operation: str,
        cost_usd: float,
        tokens_used: int = 0,
        api_calls: int = 1,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record cost metric."""
        cost_metric = CostMetric(
            timestamp=datetime.utcnow(),
            operation=operation,
            cost_usd=cost_usd,
            tokens_used=tokens_used,
            api_calls=api_calls,
            labels=labels or {},
        )
        
        with self.lock:
            self.cost_buffer.append(cost_metric)
            
            # Update daily totals
            today = datetime.utcnow().date()
            self.daily_costs[today] += cost_usd
            self.operation_costs[operation] += cost_usd
    
    def record_request(self, endpoint: str, status_code: int) -> None:
        """Record request count and status."""
        with self.lock:
            self.request_counts[f"{endpoint}:{status_code}"] += 1
            
            if status_code >= 400:
                self.error_counts[endpoint] += 1
    
    def get_latency_percentiles(self, operation: str) -> Dict[str, float]:
        """Get latency percentiles for an operation."""
        import numpy as np
        
        with self.lock:
            latencies = self.latency_buckets.get(operation, [])
        
        if not latencies:
            return {}
        
        return {
            "p50": np.percentile(latencies, 50),
            "p90": np.percentile(latencies, 90),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "avg": np.mean(latencies),
            "count": len(latencies),
        }
    
    def get_cost_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get cost summary for recent days."""
        cutoff_date = datetime.utcnow().date() - timedelta(days=days)
        
        with self.lock:
            recent_costs = {
                date: cost 
                for date, cost in self.daily_costs.items() 
                if date >= cutoff_date
            }
            
            operation_costs = dict(self.operation_costs)
        
        total_cost = sum(recent_costs.values())
        
        return {
            "total_cost_usd": total_cost,
            "daily_costs": recent_costs,
            "operation_costs": operation_costs,
            "avg_daily_cost": total_cost / max(len(recent_costs), 1),
        }
    
    def get_error_rates(self) -> Dict[str, float]:
        """Get error rates by endpoint."""
        with self.lock:
            error_rates = {}
            
            for endpoint_status, count in self.request_counts.items():
                endpoint, status = endpoint_status.rsplit(":", 1)
                
                if endpoint not in error_rates:
                    total_requests = sum(
                        c for es, c in self.request_counts.items() 
                        if es.startswith(f"{endpoint}:")
                    )
                    error_count = self.error_counts.get(endpoint, 0)
                    error_rates[endpoint] = error_count / max(total_requests, 1)
        
        return error_rates


class CacheMetrics:
    """Cache performance metrics."""
    
    def __init__(self) -> None:
        """Initialize cache metrics."""
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.hit_counts = defaultdict(int)
        self.miss_counts = defaultdict(int)
        self.lock = threading.Lock()
    
    def record_hit(self, cache_type: str) -> None:
        """Record cache hit."""
        with self.lock:
            self.hit_counts[cache_type] += 1
    
    def record_miss(self, cache_type: str) -> None:
        """Record cache miss."""
        with self.lock:
            self.miss_counts[cache_type] += 1
    
    def get_hit_rates(self) -> Dict[str, float]:
        """Get cache hit rates by type."""
        with self.lock:
            hit_rates = {}
            
            for cache_type in set(list(self.hit_counts.keys()) + list(self.miss_counts.keys())):
                hits = self.hit_counts.get(cache_type, 0)
                misses = self.miss_counts.get(cache_type, 0)
                total = hits + misses
                
                if total > 0:
                    hit_rates[cache_type] = hits / total
                else:
                    hit_rates[cache_type] = 0.0
        
        return hit_rates


class SystemMetrics:
    """System resource metrics."""
    
    @staticmethod
    def get_cpu_usage() -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=1)
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get memory usage statistics."""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent_used": memory.percent,
        }
    
    @staticmethod
    def get_disk_usage() -> Dict[str, float]:
        """Get disk usage statistics."""
        disk = psutil.disk_usage('/')
        return {
            "total_gb": disk.total / (1024**3),
            "used_gb": disk.used / (1024**3),
            "free_gb": disk.free / (1024**3),
            "percent_used": (disk.used / disk.total) * 100,
        }
    
    @staticmethod
    def get_network_stats() -> Dict[str, int]:
        """Get network I/O statistics."""
        net_io = psutil.net_io_counters()
        return {
            "bytes_sent": net_io.bytes_sent,
            "bytes_recv": net_io.bytes_recv,
            "packets_sent": net_io.packets_sent,
            "packets_recv": net_io.packets_recv,
        }


class FlameGraphProfiler:
    """Profiler for generating flame graphs of performance."""
    
    def __init__(self) -> None:
        """Initialize flame graph profiler."""
        self.call_stacks = []
        self.profiling_active = False
    
    def start_profiling(self) -> None:
        """Start profiling session."""
        self.profiling_active = True
        self.call_stacks = []
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop profiling and return flame graph data."""
        self.profiling_active = False
        
        # Process call stacks into flame graph format
        flame_data = self._process_call_stacks()
        return flame_data
    
    def record_call(self, function_name: str, duration_ms: float) -> None:
        """Record function call for flame graph."""
        if self.profiling_active:
            self.call_stacks.append({
                "function": function_name,
                "duration_ms": duration_ms,
                "timestamp": time.time(),
            })
    
    def _process_call_stacks(self) -> Dict[str, Any]:
        """Process call stacks into flame graph format."""
        # Simplified flame graph data structure
        flame_data = {
            "name": "root",
            "value": sum(call["duration_ms"] for call in self.call_stacks),
            "children": [],
        }
        
        # Group by function name
        function_totals = defaultdict(float)
        for call in self.call_stacks:
            function_totals[call["function"]] += call["duration_ms"]
        
        # Create children nodes
        for function, total_duration in function_totals.items():
            flame_data["children"].append({
                "name": function,
                "value": total_duration,
                "children": [],
            })
        
        return flame_data


class ABTestingFramework:
    """A/B testing framework for model experiments."""
    
    def __init__(self) -> None:
        """Initialize A/B testing framework."""
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.experiments = {}
        self.results = defaultdict(list)
    
    @trace_function("ab_testing.create_experiment")
    def create_experiment(
        self,
        experiment_id: str,
        variants: List[str],
        traffic_split: Dict[str, float],
        success_metric: str,
    ) -> None:
        """Create new A/B test experiment."""
        # Validate traffic split
        if abs(sum(traffic_split.values()) - 1.0) > 0.01:
            raise ValueError("Traffic split must sum to 1.0")
        
        experiment = {
            "id": experiment_id,
            "variants": variants,
            "traffic_split": traffic_split,
            "success_metric": success_metric,
            "created_at": datetime.utcnow(),
            "status": "active",
        }
        
        self.experiments[experiment_id] = experiment
        
        # Store in Redis for persistence
        self.redis_client.hset(
            "ab_experiments", 
            experiment_id, 
            str(experiment)
        )
    
    def assign_variant(self, experiment_id: str, user_id: str) -> str:
        """Assign user to experiment variant."""
        import hashlib
        
        if experiment_id not in self.experiments:
            return "control"
        
        experiment = self.experiments[experiment_id]
        
        # Use consistent hashing for assignment
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        normalized_hash = (hash_value % 10000) / 10000.0
        
        # Assign based on traffic split
        cumulative_prob = 0.0
        for variant, prob in experiment["traffic_split"].items():
            cumulative_prob += prob
            if normalized_hash <= cumulative_prob:
                return variant
        
        return list(experiment["variants"])[0]  # Fallback
    
    def record_result(
        self,
        experiment_id: str,
        user_id: str,
        variant: str,
        metric_value: float,
    ) -> None:
        """Record experiment result."""
        result = {
            "experiment_id": experiment_id,
            "user_id": user_id,
            "variant": variant,
            "metric_value": metric_value,
            "timestamp": datetime.utcnow(),
        }
        
        self.results[experiment_id].append(result)
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze experiment results."""
        if experiment_id not in self.results:
            return {"error": "No results found"}
        
        results = self.results[experiment_id]
        
        # Group by variant
        variant_results = defaultdict(list)
        for result in results:
            variant_results[result["variant"]].append(result["metric_value"])
        
        # Calculate statistics
        analysis = {}
        for variant, values in variant_results.items():
            if values:
                import numpy as np
                analysis[variant] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
        
        # Statistical significance test (simplified)
        if len(variant_results) == 2:
            variants = list(variant_results.keys())
            if len(variants) >= 2:
                from scipy import stats
                
                values_a = variant_results[variants[0]]
                values_b = variant_results[variants[1]]
                
                if len(values_a) > 1 and len(values_b) > 1:
                    t_stat, p_value = stats.ttest_ind(values_a, values_b)
                    analysis["statistical_test"] = {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                    }
        
        return analysis


class TelemetryService:
    """Main telemetry service coordinating all metrics."""
    
    def __init__(self) -> None:
        """Initialize telemetry service."""
        self.metrics_collector = MetricsCollector()
        self.cache_metrics = CacheMetrics()
        self.system_metrics = SystemMetrics()
        self.profiler = FlameGraphProfiler()
        self.ab_testing = ABTestingFramework()
    
    @trace_function("telemetry_service.get_dashboard_data")
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "performance": {
                "latency_percentiles": {
                    op: self.metrics_collector.get_latency_percentiles(op)
                    for op in ["query", "retrieval", "valuation", "export"]
                },
                "error_rates": self.metrics_collector.get_error_rates(),
                "cache_hit_rates": self.cache_metrics.get_hit_rates(),
            },
            "costs": self.metrics_collector.get_cost_summary(),
            "system": {
                "cpu_usage": self.system_metrics.get_cpu_usage(),
                "memory_usage": self.system_metrics.get_memory_usage(),
                "disk_usage": self.system_metrics.get_disk_usage(),
                "network_stats": self.system_metrics.get_network_stats(),
            },
        }
    
    def start_performance_profiling(self) -> None:
        """Start performance profiling session."""
        self.profiler.start_profiling()
    
    def stop_performance_profiling(self) -> Dict[str, Any]:
        """Stop profiling and get flame graph data."""
        return self.profiler.stop_profiling()
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {},
        }
        
        # Check system resources
        cpu_usage = self.system_metrics.get_cpu_usage()
        memory_usage = self.system_metrics.get_memory_usage()
        
        health_status["checks"]["cpu"] = {
            "status": "healthy" if cpu_usage < 80 else "warning",
            "usage_percent": cpu_usage,
        }
        
        health_status["checks"]["memory"] = {
            "status": "healthy" if memory_usage["percent_used"] < 85 else "warning",
            "usage_percent": memory_usage["percent_used"],
        }
        
        # Check error rates
        error_rates = self.metrics_collector.get_error_rates()
        avg_error_rate = sum(error_rates.values()) / max(len(error_rates), 1)
        
        health_status["checks"]["error_rate"] = {
            "status": "healthy" if avg_error_rate < 0.05 else "warning",
            "avg_error_rate": avg_error_rate,
        }
        
        # Overall status
        if any(check["status"] == "warning" for check in health_status["checks"].values()):
            health_status["status"] = "warning"
        
        return health_status


# Global telemetry instance
telemetry = TelemetryService()
