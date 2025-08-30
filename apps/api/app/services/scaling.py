"""Scaling and operations services for production deployment."""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
import queue

import redis
from celery import Celery
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.observability import trace_function


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class WorkerMetrics:
    """Worker performance metrics."""
    worker_id: str
    cpu_usage: float
    memory_usage: float
    active_tasks: int
    completed_tasks: int
    failed_tasks: int
    last_heartbeat: datetime


class AutoScaler:
    """Auto-scaling service for worker management."""
    
    def __init__(self) -> None:
        """Initialize auto-scaler."""
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.min_workers = 2
        self.max_workers = 20
        self.target_cpu_usage = 70.0
        self.scale_up_threshold = 80.0
        self.scale_down_threshold = 30.0
        self.worker_metrics = {}
        self.scaling_cooldown = timedelta(minutes=5)
        self.last_scaling_action = None
    
    @trace_function("autoscaler.evaluate_scaling")
    async def evaluate_scaling(self) -> Dict[str, Any]:
        """Evaluate if scaling action is needed."""
        current_time = datetime.utcnow()
        
        # Check cooldown period
        if (self.last_scaling_action and 
            current_time - self.last_scaling_action < self.scaling_cooldown):
            return {"action": "none", "reason": "cooldown_period"}
        
        # Get current worker metrics
        active_workers = await self._get_active_workers()
        
        if not active_workers:
            return {"action": "scale_up", "reason": "no_active_workers", "target_count": self.min_workers}
        
        # Calculate average metrics
        avg_cpu = sum(w.cpu_usage for w in active_workers) / len(active_workers)
        total_active_tasks = sum(w.active_tasks for w in active_workers)
        
        # Get queue depth
        queue_depth = await self._get_queue_depth()
        
        # Scaling decision logic
        current_worker_count = len(active_workers)
        
        # Scale up conditions
        if (avg_cpu > self.scale_up_threshold or 
            queue_depth > current_worker_count * 10 or
            total_active_tasks > current_worker_count * 5):
            
            if current_worker_count < self.max_workers:
                target_count = min(current_worker_count + 2, self.max_workers)
                return {
                    "action": "scale_up",
                    "reason": f"high_load (cpu: {avg_cpu:.1f}%, queue: {queue_depth})",
                    "current_count": current_worker_count,
                    "target_count": target_count,
                }
        
        # Scale down conditions
        elif (avg_cpu < self.scale_down_threshold and 
              queue_depth < current_worker_count * 2 and
              total_active_tasks < current_worker_count * 2):
            
            if current_worker_count > self.min_workers:
                target_count = max(current_worker_count - 1, self.min_workers)
                return {
                    "action": "scale_down",
                    "reason": f"low_load (cpu: {avg_cpu:.1f}%, queue: {queue_depth})",
                    "current_count": current_worker_count,
                    "target_count": target_count,
                }
        
        return {
            "action": "none",
            "reason": "within_thresholds",
            "current_count": current_worker_count,
            "metrics": {
                "avg_cpu": avg_cpu,
                "queue_depth": queue_depth,
                "active_tasks": total_active_tasks,
            }
        }
    
    async def _get_active_workers(self) -> List[WorkerMetrics]:
        """Get list of active workers with metrics."""
        # In production, this would query actual worker metrics
        # For now, return mock data
        return [
            WorkerMetrics(
                worker_id=f"worker-{i}",
                cpu_usage=60.0 + i * 5,
                memory_usage=50.0 + i * 3,
                active_tasks=2 + i,
                completed_tasks=100 + i * 10,
                failed_tasks=i,
                last_heartbeat=datetime.utcnow(),
            )
            for i in range(3)  # Mock 3 workers
        ]
    
    async def _get_queue_depth(self) -> int:
        """Get current task queue depth."""
        try:
            # Check Celery queue depth
            queue_length = self.redis_client.llen("celery")
            return queue_length
        except Exception:
            return 0
    
    async def execute_scaling_action(self, action_plan: Dict[str, Any]) -> bool:
        """Execute scaling action."""
        if action_plan["action"] == "none":
            return True
        
        try:
            if action_plan["action"] == "scale_up":
                success = await self._scale_up_workers(action_plan["target_count"])
            elif action_plan["action"] == "scale_down":
                success = await self._scale_down_workers(action_plan["target_count"])
            else:
                return False
            
            if success:
                self.last_scaling_action = datetime.utcnow()
            
            return success
            
        except Exception as e:
            print(f"Scaling action failed: {e}")
            return False
    
    async def _scale_up_workers(self, target_count: int) -> bool:
        """Scale up worker instances."""
        # In production, this would:
        # 1. Launch new container instances
        # 2. Update load balancer configuration
        # 3. Wait for health checks to pass
        
        print(f"Scaling up to {target_count} workers")
        return True
    
    async def _scale_down_workers(self, target_count: int) -> bool:
        """Scale down worker instances."""
        # In production, this would:
        # 1. Gracefully drain tasks from selected workers
        # 2. Terminate worker instances
        # 3. Update load balancer configuration
        
        print(f"Scaling down to {target_count} workers")
        return True


class PriorityQueueManager:
    """Priority queue management for task scheduling."""
    
    def __init__(self) -> None:
        """Initialize priority queue manager."""
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.queues = {
            TaskPriority.CRITICAL: "celery:critical",
            TaskPriority.HIGH: "celery:high", 
            TaskPriority.NORMAL: "celery:normal",
            TaskPriority.LOW: "celery:low",
        }
    
    @trace_function("priority_queue.enqueue_task")
    async def enqueue_task(
        self,
        task_name: str,
        task_args: List[Any],
        task_kwargs: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> str:
        """Enqueue task with priority."""
        task_id = str(uuid.uuid4())
        
        task_payload = {
            "id": task_id,
            "task": task_name,
            "args": task_args,
            "kwargs": task_kwargs,
            "priority": priority.value,
            "enqueued_at": datetime.utcnow().isoformat(),
        }
        
        queue_name = self.queues[priority]
        
        # Add to appropriate priority queue
        self.redis_client.lpush(queue_name, json.dumps(task_payload))
        
        return task_id
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics for all priority queues."""
        stats = {}
        
        for priority, queue_name in self.queues.items():
            queue_length = self.redis_client.llen(queue_name)
            stats[priority.name.lower()] = {
                "queue_length": queue_length,
                "queue_name": queue_name,
            }
        
        return stats
    
    async def rebalance_queues(self) -> Dict[str, Any]:
        """Rebalance tasks across queues based on age and priority."""
        rebalanced = 0
        
        # Move old low-priority tasks to normal priority
        low_queue = self.queues[TaskPriority.LOW]
        normal_queue = self.queues[TaskPriority.NORMAL]
        
        # Get tasks from low priority queue
        tasks = []
        while True:
            task_data = self.redis_client.rpop(low_queue)
            if not task_data:
                break
            
            try:
                task = json.loads(task_data)
                enqueued_at = datetime.fromisoformat(task["enqueued_at"])
                
                # If task is older than 1 hour, promote to normal priority
                if datetime.utcnow() - enqueued_at > timedelta(hours=1):
                    task["priority"] = TaskPriority.NORMAL.value
                    self.redis_client.lpush(normal_queue, json.dumps(task))
                    rebalanced += 1
                else:
                    tasks.append(task_data)
                    
            except (json.JSONDecodeError, ValueError):
                continue
        
        # Put remaining tasks back
        for task_data in reversed(tasks):
            self.redis_client.rpush(low_queue, task_data)
        
        return {"rebalanced_tasks": rebalanced}


class BackpressureManager:
    """Backpressure management to prevent system overload."""
    
    def __init__(self) -> None:
        """Initialize backpressure manager."""
        self.redis_client = redis.from_url(settings.REDIS_URL)
        self.max_queue_size = 10000
        self.max_concurrent_requests = 1000
        self.current_load = 0
        self.lock = threading.Lock()
    
    @trace_function("backpressure.check_capacity")
    async def check_capacity(self, operation: str) -> Dict[str, Any]:
        """Check if system has capacity for new requests."""
        # Check queue sizes
        total_queue_size = 0
        for queue_name in ["celery:critical", "celery:high", "celery:normal", "celery:low"]:
            total_queue_size += self.redis_client.llen(queue_name)
        
        # Check concurrent requests
        with self.lock:
            current_requests = self.current_load
        
        capacity_check = {
            "has_capacity": True,
            "queue_utilization": total_queue_size / self.max_queue_size,
            "request_utilization": current_requests / self.max_concurrent_requests,
            "total_queue_size": total_queue_size,
            "current_requests": current_requests,
        }
        
        # Apply backpressure if limits exceeded
        if total_queue_size > self.max_queue_size:
            capacity_check["has_capacity"] = False
            capacity_check["reason"] = "queue_full"
        
        elif current_requests > self.max_concurrent_requests:
            capacity_check["has_capacity"] = False
            capacity_check["reason"] = "too_many_requests"
        
        return capacity_check
    
    def acquire_slot(self) -> bool:
        """Acquire a request slot."""
        with self.lock:
            if self.current_load < self.max_concurrent_requests:
                self.current_load += 1
                return True
            return False
    
    def release_slot(self) -> None:
        """Release a request slot."""
        with self.lock:
            if self.current_load > 0:
                self.current_load -= 1


class SnapshotManager:
    """Enhanced snapshot management for reproducibility."""
    
    def __init__(self) -> None:
        """Initialize snapshot manager."""
        self.redis_client = redis.from_url(settings.REDIS_URL)
    
    @trace_function("snapshot_manager.create_snapshot")
    async def create_snapshot(
        self,
        org_id: str,
        snapshot_type: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new data snapshot."""
        snapshot_id = f"{snapshot_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        snapshot_data = {
            "id": snapshot_id,
            "org_id": org_id,
            "type": snapshot_type,
            "data": data,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "version": "1.0",
        }
        
        # Store snapshot
        snapshot_key = f"snapshot:{org_id}:{snapshot_id}"
        self.redis_client.setex(
            snapshot_key,
            timedelta(days=90).total_seconds(),  # 90 day retention
            json.dumps(snapshot_data, default=str)
        )
        
        # Add to snapshot index
        index_key = f"snapshots:{org_id}:{snapshot_type}"
        self.redis_client.zadd(
            index_key,
            {snapshot_id: datetime.utcnow().timestamp()}
        )
        
        return snapshot_id
    
    @trace_function("snapshot_manager.get_snapshot")
    async def get_snapshot(self, org_id: str, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve snapshot by ID."""
        snapshot_key = f"snapshot:{org_id}:{snapshot_id}"
        snapshot_data = self.redis_client.get(snapshot_key)
        
        if snapshot_data:
            return json.loads(snapshot_data)
        
        return None
    
    async def list_snapshots(
        self,
        org_id: str,
        snapshot_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List snapshots for organization."""
        if snapshot_type:
            index_key = f"snapshots:{org_id}:{snapshot_type}"
        else:
            # Get all snapshot types
            pattern = f"snapshots:{org_id}:*"
            keys = self.redis_client.keys(pattern)
            
            all_snapshots = []
            for key in keys:
                snapshot_ids = self.redis_client.zrevrange(key, 0, limit - 1)
                for snapshot_id in snapshot_ids:
                    snapshot = await self.get_snapshot(org_id, snapshot_id.decode())
                    if snapshot:
                        all_snapshots.append(snapshot)
            
            return sorted(all_snapshots, key=lambda x: x["created_at"], reverse=True)[:limit]
        
        # Get snapshots for specific type
        snapshot_ids = self.redis_client.zrevrange(index_key, 0, limit - 1)
        snapshots = []
        
        for snapshot_id in snapshot_ids:
            snapshot = await self.get_snapshot(org_id, snapshot_id.decode())
            if snapshot:
                snapshots.append(snapshot)
        
        return snapshots


class DisasterRecoveryManager:
    """Disaster recovery and backup management."""
    
    def __init__(self) -> None:
        """Initialize DR manager."""
        self.backup_schedule = {
            "database": "0 2 * * *",  # Daily at 2 AM
            "redis": "0 */6 * * *",   # Every 6 hours
            "files": "0 3 * * 0",     # Weekly on Sunday at 3 AM
        }
    
    @trace_function("dr_manager.create_backup")
    async def create_backup(self, backup_type: str) -> Dict[str, Any]:
        """Create system backup."""
        backup_id = f"{backup_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        backup_result = {
            "backup_id": backup_id,
            "type": backup_type,
            "status": "completed",
            "created_at": datetime.utcnow().isoformat(),
            "size_bytes": 0,
            "location": "",
        }
        
        if backup_type == "database":
            backup_result.update(await self._backup_database(backup_id))
        elif backup_type == "redis":
            backup_result.update(await self._backup_redis(backup_id))
        elif backup_type == "files":
            backup_result.update(await self._backup_files(backup_id))
        else:
            backup_result["status"] = "failed"
            backup_result["error"] = f"Unknown backup type: {backup_type}"
        
        return backup_result
    
    async def _backup_database(self, backup_id: str) -> Dict[str, Any]:
        """Backup PostgreSQL database."""
        # In production, this would:
        # 1. Create pg_dump with proper credentials
        # 2. Compress the dump
        # 3. Upload to S3 or backup storage
        # 4. Verify backup integrity
        
        return {
            "size_bytes": 1024 * 1024 * 100,  # Mock 100MB
            "location": f"s3://backups/database/{backup_id}.sql.gz",
            "tables_backed_up": 15,
        }
    
    async def _backup_redis(self, backup_id: str) -> Dict[str, Any]:
        """Backup Redis data."""
        # In production, this would:
        # 1. Create Redis RDB snapshot
        # 2. Upload to backup storage
        
        return {
            "size_bytes": 1024 * 1024 * 50,  # Mock 50MB
            "location": f"s3://backups/redis/{backup_id}.rdb",
            "keys_backed_up": 10000,
        }
    
    async def _backup_files(self, backup_id: str) -> Dict[str, Any]:
        """Backup application files."""
        # In production, this would:
        # 1. Create tar archive of application files
        # 2. Upload to backup storage
        
        return {
            "size_bytes": 1024 * 1024 * 500,  # Mock 500MB
            "location": f"s3://backups/files/{backup_id}.tar.gz",
            "files_backed_up": 5000,
        }
    
    async def verify_backup(self, backup_id: str) -> Dict[str, Any]:
        """Verify backup integrity."""
        # In production, this would:
        # 1. Download backup file
        # 2. Verify checksums
        # 3. Test restore process
        
        return {
            "backup_id": backup_id,
            "verification_status": "passed",
            "checksum_valid": True,
            "restore_test_passed": True,
            "verified_at": datetime.utcnow().isoformat(),
        }
    
    async def get_recovery_plan(self, disaster_type: str) -> Dict[str, Any]:
        """Get disaster recovery plan."""
        recovery_plans = {
            "database_failure": {
                "steps": [
                    "1. Identify latest valid database backup",
                    "2. Provision new database instance",
                    "3. Restore from backup",
                    "4. Update application configuration",
                    "5. Verify data integrity",
                    "6. Resume operations",
                ],
                "estimated_rto": "2 hours",  # Recovery Time Objective
                "estimated_rpo": "24 hours",  # Recovery Point Objective
            },
            "complete_outage": {
                "steps": [
                    "1. Assess scope of outage",
                    "2. Activate backup infrastructure",
                    "3. Restore database from latest backup",
                    "4. Restore Redis cache",
                    "5. Deploy application to backup environment",
                    "6. Update DNS to point to backup",
                    "7. Verify all services operational",
                ],
                "estimated_rto": "4 hours",
                "estimated_rpo": "24 hours",
            },
        }
        
        return recovery_plans.get(disaster_type, {
            "error": f"No recovery plan found for disaster type: {disaster_type}"
        })


class ScalingService:
    """Main scaling service coordinating all scaling operations."""
    
    def __init__(self) -> None:
        """Initialize scaling service."""
        self.autoscaler = AutoScaler()
        self.priority_queue = PriorityQueueManager()
        self.backpressure = BackpressureManager()
        self.snapshot_manager = SnapshotManager()
        self.dr_manager = DisasterRecoveryManager()
    
    @trace_function("scaling_service.health_check")
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive scaling health check."""
        # Check autoscaler status
        scaling_evaluation = await self.autoscaler.evaluate_scaling()
        
        # Check queue stats
        queue_stats = await self.priority_queue.get_queue_stats()
        
        # Check backpressure
        capacity_check = await self.backpressure.check_capacity("health_check")
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "scaling": scaling_evaluation,
            "queues": queue_stats,
            "capacity": capacity_check,
            "status": "healthy" if capacity_check["has_capacity"] else "degraded",
        }
    
    async def execute_maintenance_tasks(self) -> Dict[str, Any]:
        """Execute routine maintenance tasks."""
        results = {}
        
        # Rebalance queues
        rebalance_result = await self.priority_queue.rebalance_queues()
        results["queue_rebalancing"] = rebalance_result
        
        # Create backups
        backup_result = await self.dr_manager.create_backup("database")
        results["backup"] = backup_result
        
        # Cleanup old snapshots
        # In production, implement snapshot cleanup
        results["snapshot_cleanup"] = {"cleaned_snapshots": 0}
        
        return results
