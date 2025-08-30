"""Data ingestion tasks."""

from celery import current_task

from app.core.observability import trace_function
from app.services.parquet_lake import ParquetLakeService
from app.worker import celery_app


@celery_app.task(bind=True)
@trace_function("ingest_to_bronze_task")
def ingest_to_bronze(
    self, 
    org_id: str, 
    document_id: str, 
    data_type: str, 
    data: dict
) -> dict:
    """Ingest data to bronze layer."""
    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "Ingesting to bronze layer"}
        )
        
        lake_service = ParquetLakeService()
        # Note: This should be async but Celery tasks are sync
        
        return {
            "org_id": org_id,
            "document_id": document_id,
            "data_type": data_type,
            "status": "completed",
            "message": "Data ingested to bronze layer",
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e)}
        )
        raise


@celery_app.task(bind=True)
@trace_function("normalize_to_silver_task")
def normalize_to_silver(
    self,
    org_id: str,
    data_type: str,
    bronze_snapshot_id: str,
) -> dict:
    """Normalize bronze data to silver layer."""
    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "Normalizing to silver layer"}
        )
        
        # Placeholder for normalization logic
        
        return {
            "org_id": org_id,
            "data_type": data_type,
            "bronze_snapshot_id": bronze_snapshot_id,
            "status": "completed",
            "message": "Data normalized to silver layer",
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e)}
        )
        raise


@celery_app.task(bind=True)
@trace_function("aggregate_to_gold_task")
def aggregate_to_gold(
    self,
    org_id: str,
    table_name: str,
    silver_snapshot_id: str,
) -> dict:
    """Aggregate silver data to gold layer."""
    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "Aggregating to gold layer"}
        )
        
        # Placeholder for aggregation logic
        
        return {
            "org_id": org_id,
            "table_name": table_name,
            "silver_snapshot_id": silver_snapshot_id,
            "status": "completed",
            "message": "Data aggregated to gold layer",
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e)}
        )
        raise
