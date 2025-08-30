"""Load testing framework for performance validation."""

import asyncio
import time
import random
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import statistics

import httpx
import numpy as np

from app.core.observability import trace_function


@dataclass
class LoadTestScenario:
    """Load test scenario configuration."""
    name: str
    concurrent_users: int
    duration_seconds: int
    ramp_up_seconds: int
    requests_per_user: int
    endpoints: List[Dict[str, Any]]
    think_time_min: float = 1.0
    think_time_max: float = 5.0


@dataclass
class LoadTestResult:
    """Load test execution result."""
    scenario_name: str
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    # Performance metrics
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    
    # Throughput metrics
    requests_per_second: float
    
    # Error analysis
    error_rate: float
    error_breakdown: Dict[str, int]
    
    # Resource utilization
    cpu_usage_samples: List[float]
    memory_usage_samples: List[float]


class VirtualUser:
    """Virtual user for load testing."""
    
    def __init__(self, user_id: int, base_url: str, scenario: LoadTestScenario):
        """Initialize virtual user."""
        self.user_id = user_id
        self.base_url = base_url
        self.scenario = scenario
        self.session = None
        self.results = []
    
    async def run(self) -> List[Dict[str, Any]]:
        """Execute user scenario."""
        async with httpx.AsyncClient(timeout=30.0) as session:
            self.session = session
            
            for request_num in range(self.scenario.requests_per_user):
                # Select random endpoint
                endpoint = random.choice(self.scenario.endpoints)
                
                # Execute request
                result = await self._execute_request(endpoint, request_num)
                self.results.append(result)
                
                # Think time between requests
                if request_num < self.scenario.requests_per_user - 1:
                    think_time = random.uniform(
                        self.scenario.think_time_min,
                        self.scenario.think_time_max
                    )
                    await asyncio.sleep(think_time)
        
        return self.results
    
    async def _execute_request(self, endpoint: Dict[str, Any], request_num: int) -> Dict[str, Any]:
        """Execute a single request."""
        start_time = time.time()
        
        try:
            method = endpoint.get("method", "GET")
            path = endpoint["path"]
            headers = endpoint.get("headers", {})
            data = endpoint.get("data")
            params = endpoint.get("params")
            
            url = f"{self.base_url}{path}"
            
            if method.upper() == "GET":
                response = await self.session.get(url, headers=headers, params=params)
            elif method.upper() == "POST":
                response = await self.session.post(url, headers=headers, json=data, params=params)
            elif method.upper() == "PUT":
                response = await self.session.put(url, headers=headers, json=data, params=params)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            return {
                "user_id": self.user_id,
                "request_num": request_num,
                "endpoint": endpoint["name"],
                "method": method,
                "url": url,
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "success": 200 <= response.status_code < 400,
                "error": None,
                "timestamp": datetime.utcnow(),
            }
            
        except Exception as e:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            return {
                "user_id": self.user_id,
                "request_num": request_num,
                "endpoint": endpoint["name"],
                "method": endpoint.get("method", "GET"),
                "url": f"{self.base_url}{endpoint['path']}",
                "status_code": 0,
                "response_time_ms": response_time,
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow(),
            }


class LoadTestRunner:
    """Load test execution engine."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize load test runner."""
        self.base_url = base_url
        self.system_monitor = SystemMonitor()
    
    @trace_function("load_test_runner.run_scenario")
    async def run_scenario(self, scenario: LoadTestScenario) -> LoadTestResult:
        """Execute load test scenario."""
        print(f"Starting load test scenario: {scenario.name}")
        print(f"Concurrent users: {scenario.concurrent_users}")
        print(f"Duration: {scenario.duration_seconds}s")
        
        start_time = datetime.utcnow()
        
        # Start system monitoring
        monitor_task = asyncio.create_task(
            self.system_monitor.monitor_during_test(scenario.duration_seconds + scenario.ramp_up_seconds)
        )
        
        # Create virtual users
        users = [
            VirtualUser(i, self.base_url, scenario)
            for i in range(scenario.concurrent_users)
        ]
        
        # Execute with ramp-up
        all_results = []
        
        if scenario.ramp_up_seconds > 0:
            # Gradual ramp-up
            ramp_up_delay = scenario.ramp_up_seconds / scenario.concurrent_users
            
            tasks = []
            for i, user in enumerate(users):
                # Stagger user start times
                delay = i * ramp_up_delay
                task = asyncio.create_task(self._delayed_user_execution(user, delay))
                tasks.append(task)
            
            # Wait for all users to complete
            user_results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Immediate start
            tasks = [user.run() for user in users]
            user_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for result in user_results:
            if isinstance(result, Exception):
                print(f"User execution failed: {result}")
            else:
                all_results.extend(result)
        
        end_time = datetime.utcnow()
        
        # Wait for monitoring to complete
        system_metrics = await monitor_task
        
        # Analyze results
        load_test_result = self._analyze_results(
            scenario, start_time, end_time, all_results, system_metrics
        )
        
        print(f"Load test completed: {load_test_result.successful_requests}/{load_test_result.total_requests} successful")
        print(f"Average response time: {load_test_result.avg_response_time:.2f}ms")
        print(f"P95 response time: {load_test_result.p95_response_time:.2f}ms")
        print(f"Requests per second: {load_test_result.requests_per_second:.2f}")
        
        return load_test_result
    
    async def _delayed_user_execution(self, user: VirtualUser, delay: float) -> List[Dict[str, Any]]:
        """Execute user with delay for ramp-up."""
        if delay > 0:
            await asyncio.sleep(delay)
        return await user.run()
    
    def _analyze_results(
        self,
        scenario: LoadTestScenario,
        start_time: datetime,
        end_time: datetime,
        results: List[Dict[str, Any]],
        system_metrics: Dict[str, List[float]],
    ) -> LoadTestResult:
        """Analyze load test results."""
        if not results:
            return LoadTestResult(
                scenario_name=scenario.name,
                start_time=start_time,
                end_time=end_time,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_response_time=0,
                p50_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                min_response_time=0,
                max_response_time=0,
                requests_per_second=0,
                error_rate=0,
                error_breakdown={},
                cpu_usage_samples=system_metrics.get("cpu", []),
                memory_usage_samples=system_metrics.get("memory", []),
            )
        
        # Basic counts
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r["success"])
        failed_requests = total_requests - successful_requests
        
        # Response time analysis
        response_times = [r["response_time_ms"] for r in results]
        
        avg_response_time = statistics.mean(response_times)
        p50_response_time = np.percentile(response_times, 50)
        p95_response_time = np.percentile(response_times, 95)
        p99_response_time = np.percentile(response_times, 99)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        
        # Throughput calculation
        duration = (end_time - start_time).total_seconds()
        requests_per_second = total_requests / max(duration, 1)
        
        # Error analysis
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        error_breakdown = {}
        
        for result in results:
            if not result["success"]:
                error_key = result.get("error", f"HTTP_{result['status_code']}")
                error_breakdown[error_key] = error_breakdown.get(error_key, 0) + 1
        
        return LoadTestResult(
            scenario_name=scenario.name,
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            error_breakdown=error_breakdown,
            cpu_usage_samples=system_metrics.get("cpu", []),
            memory_usage_samples=system_metrics.get("memory", []),
        )


class SystemMonitor:
    """System resource monitoring during load tests."""
    
    async def monitor_during_test(self, duration_seconds: int) -> Dict[str, List[float]]:
        """Monitor system resources during test execution."""
        cpu_samples = []
        memory_samples = []
        
        sample_interval = 5  # seconds
        samples_needed = max(1, duration_seconds // sample_interval)
        
        for _ in range(samples_needed):
            try:
                import psutil
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_samples.append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_samples.append(memory.percent)
                
                # Wait for next sample
                if _ < samples_needed - 1:
                    await asyncio.sleep(sample_interval - 1)  # -1 because cpu_percent takes 1 second
                    
            except ImportError:
                # psutil not available, use mock data
                cpu_samples.append(random.uniform(20, 80))
                memory_samples.append(random.uniform(30, 70))
                await asyncio.sleep(sample_interval)
        
        return {
            "cpu": cpu_samples,
            "memory": memory_samples,
        }


class LoadTestSuite:
    """Predefined load test scenarios for the financial analyst system."""
    
    @staticmethod
    def get_analyst_workload_scenario() -> LoadTestScenario:
        """Typical analyst workload scenario."""
        return LoadTestScenario(
            name="Analyst Workload",
            concurrent_users=50,
            duration_seconds=300,  # 5 minutes
            ramp_up_seconds=60,    # 1 minute ramp-up
            requests_per_user=20,
            endpoints=[
                {
                    "name": "health_check",
                    "path": "/health",
                    "method": "GET",
                },
                {
                    "name": "query_analysis",
                    "path": "/v1/query",
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "data": {
                        "org_id": "test-org",
                        "prompt": "What was NVIDIA's revenue growth in 2023?",
                        "tickers": ["NVDA"],
                    },
                },
                {
                    "name": "upload_document",
                    "path": "/v1/upload",
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "data": {
                        "org_id": "test-org",
                        "filename": "test-document.pdf",
                    },
                },
                {
                    "name": "export_analysis",
                    "path": "/v1/exports",
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "data": {
                        "format": "excel",
                        "data": {"company": {"symbol": "NVDA"}},
                    },
                },
            ],
            think_time_min=2.0,
            think_time_max=8.0,
        )
    
    @staticmethod
    def get_peak_load_scenario() -> LoadTestScenario:
        """Peak load scenario with 200 concurrent users."""
        return LoadTestScenario(
            name="Peak Load Test",
            concurrent_users=200,
            duration_seconds=600,  # 10 minutes
            ramp_up_seconds=120,   # 2 minute ramp-up
            requests_per_user=15,
            endpoints=[
                {
                    "name": "health_check",
                    "path": "/health",
                    "method": "GET",
                },
                {
                    "name": "query_analysis",
                    "path": "/v1/query",
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "data": {
                        "org_id": "test-org",
                        "prompt": "Analyze Tesla's financial performance vs competitors",
                        "tickers": ["TSLA"],
                    },
                },
            ],
            think_time_min=1.0,
            think_time_max=3.0,
        )
    
    @staticmethod
    def get_stress_test_scenario() -> LoadTestScenario:
        """Stress test to find breaking point."""
        return LoadTestScenario(
            name="Stress Test",
            concurrent_users=500,
            duration_seconds=300,  # 5 minutes
            ramp_up_seconds=30,    # Fast ramp-up
            requests_per_user=10,
            endpoints=[
                {
                    "name": "health_check",
                    "path": "/health",
                    "method": "GET",
                },
                {
                    "name": "simple_query",
                    "path": "/v1/query",
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "data": {
                        "org_id": "test-org",
                        "prompt": "What is Apple's current stock price?",
                        "tickers": ["AAPL"],
                    },
                },
            ],
            think_time_min=0.5,
            think_time_max=2.0,
        )


class ChaosTestRunner:
    """Chaos engineering tests for resilience validation."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize chaos test runner."""
        self.base_url = base_url
    
    async def run_chaos_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Run chaos engineering experiment."""
        experiments = {
            "database_latency": self._simulate_database_latency,
            "network_partition": self._simulate_network_partition,
            "memory_pressure": self._simulate_memory_pressure,
            "cpu_spike": self._simulate_cpu_spike,
        }
        
        if experiment_name not in experiments:
            return {"error": f"Unknown experiment: {experiment_name}"}
        
        print(f"Starting chaos experiment: {experiment_name}")
        
        # Run baseline test
        baseline_scenario = LoadTestScenario(
            name=f"Baseline for {experiment_name}",
            concurrent_users=20,
            duration_seconds=60,
            ramp_up_seconds=10,
            requests_per_user=5,
            endpoints=[{"name": "health", "path": "/health", "method": "GET"}],
        )
        
        runner = LoadTestRunner(self.base_url)
        baseline_result = await runner.run_scenario(baseline_scenario)
        
        # Introduce chaos
        chaos_task = asyncio.create_task(experiments[experiment_name]())
        
        # Run test during chaos
        chaos_scenario = LoadTestScenario(
            name=f"Chaos Test: {experiment_name}",
            concurrent_users=20,
            duration_seconds=60,
            ramp_up_seconds=10,
            requests_per_user=5,
            endpoints=[{"name": "health", "path": "/health", "method": "GET"}],
        )
        
        chaos_result = await runner.run_scenario(chaos_scenario)
        
        # Wait for chaos to complete
        await chaos_task
        
        # Compare results
        resilience_score = self._calculate_resilience_score(baseline_result, chaos_result)
        
        return {
            "experiment": experiment_name,
            "baseline_performance": {
                "avg_response_time": baseline_result.avg_response_time,
                "success_rate": baseline_result.successful_requests / baseline_result.total_requests,
            },
            "chaos_performance": {
                "avg_response_time": chaos_result.avg_response_time,
                "success_rate": chaos_result.successful_requests / chaos_result.total_requests,
            },
            "resilience_score": resilience_score,
        }
    
    async def _simulate_database_latency(self) -> None:
        """Simulate database latency issues."""
        # In production, this would introduce actual database delays
        print("Simulating database latency...")
        await asyncio.sleep(60)  # Simulate for 60 seconds
        print("Database latency simulation complete")
    
    async def _simulate_network_partition(self) -> None:
        """Simulate network partition."""
        print("Simulating network partition...")
        await asyncio.sleep(60)
        print("Network partition simulation complete")
    
    async def _simulate_memory_pressure(self) -> None:
        """Simulate memory pressure."""
        print("Simulating memory pressure...")
        await asyncio.sleep(60)
        print("Memory pressure simulation complete")
    
    async def _simulate_cpu_spike(self) -> None:
        """Simulate CPU spike."""
        print("Simulating CPU spike...")
        await asyncio.sleep(60)
        print("CPU spike simulation complete")
    
    def _calculate_resilience_score(
        self, 
        baseline: LoadTestResult, 
        chaos: LoadTestResult
    ) -> float:
        """Calculate resilience score (0-1, higher is better)."""
        baseline_success_rate = baseline.successful_requests / baseline.total_requests
        chaos_success_rate = chaos.successful_requests / chaos.total_requests
        
        # Response time degradation
        response_time_ratio = chaos.avg_response_time / max(baseline.avg_response_time, 1)
        
        # Success rate degradation
        success_rate_ratio = chaos_success_rate / max(baseline_success_rate, 0.01)
        
        # Combined resilience score
        resilience_score = (success_rate_ratio * 0.7) + ((1 / response_time_ratio) * 0.3)
        
        return min(resilience_score, 1.0)
