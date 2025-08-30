"""Export endpoints for generating reports."""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import io
from typing import Dict, Any

from app.core.database import get_db
from app.core.observability import trace_function
from app.services.exports import ExportService
from app.schemas.exports import ExportRequest, ExportResponse

router = APIRouter()


@router.post("/", response_model=ExportResponse)
@trace_function("export_endpoint.create_export")
async def create_export(
    request: ExportRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> ExportResponse:
    """Create export file from analysis data."""
    try:
        export_service = ExportService()
        
        # Generate export file
        file_bytes = await export_service.export_analysis(
            format_type=request.format,
            data=request.data,
        )
        
        # Generate filename
        filename = export_service.get_filename(
            format_type=request.format,
            company_symbol=request.data.get('company', {}).get('symbol', 'analysis')
        )
        
        # In production, would upload to S3 and return URL
        # For now, return mock response
        return ExportResponse(
            export_id=f"export_{request.format}_{filename}",
            filename=filename,
            format=request.format,
            status="completed",
            download_url=f"/api/v1/exports/download/{filename}",
            size_bytes=len(file_bytes),
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Export generation failed: {str(e)}"
        )


@router.get("/download/{filename}")
@trace_function("export_endpoint.download_export")
async def download_export(
    filename: str,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Download export file."""
    try:
        # In production, would retrieve from S3
        # For now, generate mock file
        export_service = ExportService()
        
        # Determine format from filename
        format_type = filename.split('.')[-1]
        
        # Mock data for demo
        mock_data = {
            'company': {
                'name': 'NVIDIA Corporation',
                'symbol': 'NVDA',
                'sector': 'Technology',
                'market_cap': 1800000000000,
                'enterprise_value': 1750000000000,
            },
            'valuation': {
                'share_price': 485.20,
                'current_price': 452.30,
                'upside': 7.3,
                'enterprise_value': 1750000000000,
                'equity_value': 1720000000000,
                'model_type': 'Three-Stage DCF',
                'terminal_value': 1200000000000,
                'pv_explicit_period': 550000000000,
                'pv_terminal_value': 1200000000000,
                'assumptions': {
                    'wacc': 0.095,
                    'terminal_growth_rate': 0.025,
                    'tax_rate': 0.21,
                },
            },
            'financial_data': [
                {'year': 2021, 'revenue': 26914000000, 'net_income': 4368000000},
                {'year': 2022, 'revenue': 60922000000, 'net_income': 9752000000},
                {'year': 2023, 'revenue': 60922000000, 'net_income': 29760000000},
            ],
            'analysis_text': 'NVIDIA demonstrates strong growth in AI and data center markets with significant competitive advantages in GPU technology.',
            'citations': [
                {'source': 'NVDA 10-K 2023', 'locator': 'Page 45'},
                {'source': 'Q3 2023 Earnings Call', 'locator': '15:30'},
            ],
        }
        
        file_bytes = await export_service.export_analysis(format_type, mock_data)
        
        # Determine content type
        content_types = {
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'pdf': 'application/pdf',
        }
        
        content_type = content_types.get(format_type, 'application/octet-stream')
        
        # Return file as streaming response
        return StreamingResponse(
            io.BytesIO(file_bytes),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Download failed: {str(e)}"
        )


@router.get("/{export_id}/status")
@trace_function("export_endpoint.get_export_status")
async def get_export_status(
    export_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Get export status."""
    # In production, would check database/queue for actual status
    return {
        "export_id": export_id,
        "status": "completed",
        "progress": 100,
        "message": "Export completed successfully",
    }
