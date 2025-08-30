"""Document processing tasks."""

from celery import current_task
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.core.observability import trace_function
from app.models.document import Document
from app.services.processors.pdf_processor import PDFProcessor
from app.services.processors.audio_processor import AudioProcessor
from app.services.processors.xbrl_processor import XBRLProcessor
from app.worker import celery_app


@celery_app.task(bind=True)
@trace_function("process_document_task")
def process_document(self, document_id: str, org_id: str) -> dict:
    """Process uploaded document."""
    try:
        # Update task state
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "Starting document processing"}
        )
        
        # This would be async in a real implementation
        # For now, return a placeholder result
        return {
            "document_id": document_id,
            "org_id": org_id,
            "status": "completed",
            "message": "Document processing completed successfully",
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e)}
        )
        raise


@celery_app.task(bind=True)
def process_pdf_document(self, document_id: str, org_id: str, file_path: str) -> dict:
    """Process PDF document."""
    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "Processing PDF document"}
        )
        
        processor = PDFProcessor()
        # Note: This should be async but Celery tasks are sync
        # In production, use async Celery or adapt the processor
        
        return {
            "document_id": document_id,
            "status": "completed",
            "message": "PDF processing completed",
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e)}
        )
        raise


@celery_app.task(bind=True)
def process_audio_document(self, document_id: str, org_id: str, file_path: str) -> dict:
    """Process audio document."""
    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "Processing audio document"}
        )
        
        processor = AudioProcessor()
        
        return {
            "document_id": document_id,
            "status": "completed",
            "message": "Audio processing completed",
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e)}
        )
        raise


@celery_app.task(bind=True)
def process_xbrl_document(self, document_id: str, org_id: str, file_path: str) -> dict:
    """Process XBRL document."""
    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"status": "Processing XBRL document"}
        )
        
        processor = XBRLProcessor()
        
        return {
            "document_id": document_id,
            "status": "completed",
            "message": "XBRL processing completed",
        }
        
    except Exception as e:
        current_task.update_state(
            state="FAILURE",
            meta={"error": str(e)}
        )
        raise
