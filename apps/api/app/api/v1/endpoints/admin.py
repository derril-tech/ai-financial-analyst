"""Admin endpoints for system management."""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List

from app.core.database import get_db
from app.core.security import SecurityManager, UserRole
from app.core.observability import trace_function
from app.services.evaluation import EvaluationFramework
from app.services.telemetry import telemetry
from app.services.scaling import ScalingService
from app.services.load_testing import LoadTestRunner, LoadTestSuite, ChaosTestRunner

router = APIRouter()
security_manager = SecurityManager()
evaluation_framework = EvaluationFramework()
scaling_service = ScalingService()


@router.get("/health")
@trace_function("admin_endpoint.system_health")
async def system_health(
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Get comprehensive system health status."""
    try:
        # Get telemetry health
        telemetry_health = await telemetry.health_check()
        
        # Get scaling health
        scaling_health = await scaling_service.health_check()
        
        # Combine health checks
        overall_status = "healthy"
        if (telemetry_health.get("status") != "healthy" or 
            scaling_health.get("status") != "healthy"):
            overall_status = "warning"
        
        return {
            "status": overall_status,
            "telemetry": telemetry_health,
            "scaling": scaling_health,
            "timestamp": telemetry_health["timestamp"],
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/metrics")
@trace_function("admin_endpoint.get_metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get system performance metrics."""
    try:
        dashboard_data = telemetry.get_dashboard_data()
        return dashboard_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )


@router.post("/evaluation/run")
@trace_function("admin_endpoint.run_evaluation")
async def run_evaluation(
    model_version: str = "current",
    dataset_name: str = "financial_qa",
    sample_size: int = None,
    background_tasks: BackgroundTasks = None,
) -> Dict[str, Any]:
    """Run model evaluation."""
    try:
        # Run evaluation in background if requested
        if background_tasks:
            background_tasks.add_task(
                evaluation_framework.run_evaluation,
                model_version,
                dataset_name,
                sample_size
            )
            return {
                "status": "started",
                "message": "Evaluation running in background",
                "model_version": model_version,
                "dataset": dataset_name,
            }
        else:
            # Run synchronously
            result = await evaluation_framework.run_evaluation(
                model_version, dataset_name, sample_size
            )
            return {
                "status": "completed",
                "result": result,
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.get("/evaluation/results")
@trace_function("admin_endpoint.get_evaluation_results")
async def get_evaluation_results(limit: int = 10) -> Dict[str, Any]:
    """Get recent evaluation results."""
    try:
        recent_results = evaluation_framework.results_history[-limit:]
        
        return {
            "results": recent_results,
            "total_evaluations": len(evaluation_framework.results_history),
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve evaluation results: {str(e)}"
        )


@router.post("/load-test/run")
@trace_function("admin_endpoint.run_load_test")
async def run_load_test(
    scenario_name: str = "analyst_workload",
    background_tasks: BackgroundTasks = None,
) -> Dict[str, Any]:
    """Run load test scenario."""
    try:
        # Get predefined scenario
        scenarios = {
            "analyst_workload": LoadTestSuite.get_analyst_workload_scenario(),
            "peak_load": LoadTestSuite.get_peak_load_scenario(),
            "stress_test": LoadTestSuite.get_stress_test_scenario(),
        }
        
        if scenario_name not in scenarios:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown scenario: {scenario_name}"
            )
        
        scenario = scenarios[scenario_name]
        runner = LoadTestRunner()
        
        if background_tasks:
            background_tasks.add_task(runner.run_scenario, scenario)
            return {
                "status": "started",
                "message": f"Load test '{scenario_name}' running in background",
                "scenario": scenario_name,
            }
        else:
            result = await runner.run_scenario(scenario)
            return {
                "status": "completed",
                "result": {
                    "scenario_name": result.scenario_name,
                    "total_requests": result.total_requests,
                    "successful_requests": result.successful_requests,
                    "error_rate": result.error_rate,
                    "avg_response_time": result.avg_response_time,
                    "p95_response_time": result.p95_response_time,
                    "requests_per_second": result.requests_per_second,
                },
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Load test failed: {str(e)}"
        )


@router.post("/chaos-test/run")
@trace_function("admin_endpoint.run_chaos_test")
async def run_chaos_test(
    experiment_name: str,
    background_tasks: BackgroundTasks = None,
) -> Dict[str, Any]:
    """Run chaos engineering test."""
    try:
        chaos_runner = ChaosTestRunner()
        
        if background_tasks:
            background_tasks.add_task(
                chaos_runner.run_chaos_experiment, 
                experiment_name
            )
            return {
                "status": "started",
                "message": f"Chaos experiment '{experiment_name}' running in background",
                "experiment": experiment_name,
            }
        else:
            result = await chaos_runner.run_chaos_experiment(experiment_name)
            return {
                "status": "completed",
                "result": result,
            }
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chaos test failed: {str(e)}"
        )


@router.post("/scaling/evaluate")
@trace_function("admin_endpoint.evaluate_scaling")
async def evaluate_scaling() -> Dict[str, Any]:
    """Evaluate current scaling needs."""
    try:
        scaling_evaluation = await scaling_service.autoscaler.evaluate_scaling()
        return scaling_evaluation
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Scaling evaluation failed: {str(e)}"
        )


@router.post("/scaling/execute")
@trace_function("admin_endpoint.execute_scaling")
async def execute_scaling(
    action: str,  # scale_up, scale_down
    target_count: int,
) -> Dict[str, Any]:
    """Execute scaling action."""
    try:
        action_plan = {
            "action": action,
            "target_count": target_count,
        }
        
        success = await scaling_service.autoscaler.execute_scaling_action(action_plan)
        
        return {
            "success": success,
            "action": action,
            "target_count": target_count,
            "message": f"Scaling action {'completed' if success else 'failed'}",
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Scaling execution failed: {str(e)}"
        )


@router.post("/maintenance")
@trace_function("admin_endpoint.run_maintenance")
async def run_maintenance(
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """Run system maintenance tasks."""
    try:
        background_tasks.add_task(scaling_service.execute_maintenance_tasks)
        
        return {
            "status": "started",
            "message": "Maintenance tasks running in background",
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Maintenance failed: {str(e)}"
        )


@router.get("/security/audit-log")
@trace_function("admin_endpoint.get_audit_log")
async def get_audit_log(
    org_id: str,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Get security audit log."""
    try:
        # In production, query actual audit log table
        return {
            "audit_entries": [],
            "total_entries": 0,
            "org_id": org_id,
            "limit": limit,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve audit log: {str(e)}"
        )


@router.post("/security/scan-content")
@trace_function("admin_endpoint.scan_content")
async def scan_content(
    content: str,
    org_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Scan content for security risks."""
    try:
        from app.core.security import ContentFilter, PromptInjectionGuard
        
        content_filter = ContentFilter()
        injection_guard = PromptInjectionGuard()
        
        # Scan for MNPI/PII
        content_risks = content_filter.scan_content(content)
        
        # Scan for prompt injection
        injection_risks = injection_guard.scan_prompt(content)
        
        return {
            "content_risks": content_risks,
            "injection_risks": injection_risks,
            "overall_risk_score": max(
                content_risks["risk_score"],
                injection_risks["risk_score"]
            ),
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Content scanning failed: {str(e)}"
        )


@router.get("/ab-tests")
@trace_function("admin_endpoint.list_ab_tests")
async def list_ab_tests() -> Dict[str, Any]:
    """List active A/B tests."""
    try:
        experiments = telemetry.ab_testing.experiments
        
        return {
            "experiments": list(experiments.keys()),
            "total_experiments": len(experiments),
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list A/B tests: {str(e)}"
        )


@router.post("/ab-tests/{experiment_id}/analyze")
@trace_function("admin_endpoint.analyze_ab_test")
async def analyze_ab_test(experiment_id: str) -> Dict[str, Any]:
    """Analyze A/B test results."""
    try:
        analysis = telemetry.ab_testing.analyze_experiment(experiment_id)
        
        return {
            "experiment_id": experiment_id,
            "analysis": analysis,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"A/B test analysis failed: {str(e)}"
        )
