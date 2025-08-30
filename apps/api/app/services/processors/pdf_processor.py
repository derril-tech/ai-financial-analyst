"""PDF processing service."""

import io
import uuid
from pathlib import Path
from typing import Any, List, Dict

import fitz  # PyMuPDF
import pandas as pd
from camelot import read_pdf
from unstructured.partition.pdf import partition_pdf
from PIL import Image

from app.core.config import settings
from app.core.feature_flags import is_enabled
from app.core.observability import trace_function
from app.models.artifact import Artifact
from app.schemas.processing import ProcessingResult, ExtractedTable, ExtractedImage


class PDFProcessor:
    """Service for processing PDF documents."""
    
    def __init__(self) -> None:
        """Initialize PDF processor."""
        self.enabled = is_enabled("enable_pdf_processing")
    
    @trace_function("pdf_processor.process_document")
    async def process_document(
        self,
        document_id: str,
        org_id: str,
        file_path: str,
    ) -> ProcessingResult:
        """Process PDF document and extract content."""
        if not self.enabled:
            return ProcessingResult(
                document_id=document_id,
                status="skipped",
                message="PDF processing disabled",
                artifacts=[],
            )
        
        try:
            # Extract text and structure with Unstructured
            elements = partition_pdf(file_path, strategy="hi_res")
            
            # Extract tables with Camelot
            tables = self._extract_tables_camelot(file_path)
            
            # Extract images with PyMuPDF
            images = self._extract_images_pymupdf(file_path)
            
            # Process elements into structured data
            sections = self._process_elements(elements)
            
            # Create artifacts
            artifacts = []
            
            # Text sections artifact
            if sections:
                text_artifact = await self._create_text_artifact(
                    document_id, org_id, sections
                )
                artifacts.append(text_artifact)
            
            # Table artifacts
            for i, table in enumerate(tables):
                table_artifact = await self._create_table_artifact(
                    document_id, org_id, table, i
                )
                artifacts.append(table_artifact)
            
            # Image artifacts
            for i, image in enumerate(images):
                image_artifact = await self._create_image_artifact(
                    document_id, org_id, image, i
                )
                artifacts.append(image_artifact)
            
            return ProcessingResult(
                document_id=document_id,
                status="completed",
                message=f"Extracted {len(sections)} sections, {len(tables)} tables, {len(images)} images",
                artifacts=artifacts,
            )
            
        except Exception as e:
            return ProcessingResult(
                document_id=document_id,
                status="failed",
                message=f"PDF processing failed: {str(e)}",
                artifacts=[],
            )
    
    def _extract_tables_camelot(self, file_path: str) -> List[ExtractedTable]:
        """Extract tables using Camelot."""
        try:
            tables = read_pdf(file_path, pages="all", flavor="lattice")
            extracted_tables = []
            
            for i, table in enumerate(tables):
                # Convert to DataFrame
                df = table.df
                
                # Clean up the table
                df = self._clean_table(df)
                
                extracted_table = ExtractedTable(
                    page_number=table.page,
                    table_index=i,
                    data=df.to_dict("records"),
                    headers=df.columns.tolist(),
                    confidence=table.accuracy,
                    bbox=table._bbox,
                )
                extracted_tables.append(extracted_table)
            
            return extracted_tables
            
        except Exception as e:
            print(f"Camelot table extraction failed: {e}")
            return []
    
    def _extract_images_pymupdf(self, file_path: str) -> List[ExtractedImage]:
        """Extract images using PyMuPDF."""
        try:
            doc = fitz.open(file_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        # Convert to PIL Image
                        img_data = pix.tobytes("png")
                        pil_image = Image.open(io.BytesIO(img_data))
                        
                        extracted_image = ExtractedImage(
                            page_number=page_num + 1,
                            image_index=img_index,
                            image_data=img_data,
                            width=pix.width,
                            height=pix.height,
                            format="png",
                        )
                        images.append(extracted_image)
                    
                    pix = None
            
            doc.close()
            return images
            
        except Exception as e:
            print(f"PyMuPDF image extraction failed: {e}")
            return []
    
    def _process_elements(self, elements: List[Any]) -> List[Dict[str, Any]]:
        """Process Unstructured elements into sections."""
        sections = []
        current_section = None
        
        for element in elements:
            element_type = element.category
            text = element.text.strip()
            
            if not text:
                continue
            
            # Handle different element types
            if element_type == "Title":
                # Start new section
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    "type": "section",
                    "title": text,
                    "content": [],
                    "page_number": getattr(element.metadata, "page_number", None),
                }
            
            elif element_type in ["NarrativeText", "Text"]:
                if current_section is None:
                    current_section = {
                        "type": "section",
                        "title": "Introduction",
                        "content": [],
                        "page_number": getattr(element.metadata, "page_number", None),
                    }
                
                current_section["content"].append({
                    "type": "text",
                    "text": text,
                    "page_number": getattr(element.metadata, "page_number", None),
                })
            
            elif element_type == "Table":
                if current_section is None:
                    current_section = {
                        "type": "section",
                        "title": "Tables",
                        "content": [],
                        "page_number": getattr(element.metadata, "page_number", None),
                    }
                
                current_section["content"].append({
                    "type": "table",
                    "text": text,
                    "page_number": getattr(element.metadata, "page_number", None),
                })
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean extracted table data."""
        # Remove empty rows and columns
        df = df.dropna(how="all").dropna(axis=1, how="all")
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Clean cell values
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str).str.strip()
        
        return df
    
    async def _create_text_artifact(
        self,
        document_id: str,
        org_id: str,
        sections: List[Dict[str, Any]],
    ) -> Artifact:
        """Create text artifact from sections."""
        artifact_id = str(uuid.uuid4())
        
        # TODO: Upload to S3 and create artifact record
        # For now, return a placeholder
        return Artifact(
            id=artifact_id,
            org_id=org_id,
            document_id=document_id,
            type="text_sections",
            path_s3=f"{org_id}/processed/{document_id}/text_sections.json",
            meta={
                "section_count": len(sections),
                "total_characters": sum(
                    len(str(section)) for section in sections
                ),
            },
        )
    
    async def _create_table_artifact(
        self,
        document_id: str,
        org_id: str,
        table: ExtractedTable,
        index: int,
    ) -> Artifact:
        """Create table artifact."""
        artifact_id = str(uuid.uuid4())
        
        # TODO: Upload to S3 and create artifact record
        return Artifact(
            id=artifact_id,
            org_id=org_id,
            document_id=document_id,
            type="table",
            path_s3=f"{org_id}/processed/{document_id}/table_{index}.json",
            meta={
                "page_number": table.page_number,
                "table_index": table.table_index,
                "row_count": len(table.data),
                "column_count": len(table.headers),
                "confidence": table.confidence,
            },
        )
    
    async def _create_image_artifact(
        self,
        document_id: str,
        org_id: str,
        image: ExtractedImage,
        index: int,
    ) -> Artifact:
        """Create image artifact."""
        artifact_id = str(uuid.uuid4())
        
        # TODO: Upload to S3 and create artifact record
        return Artifact(
            id=artifact_id,
            org_id=org_id,
            document_id=document_id,
            type="image",
            path_s3=f"{org_id}/processed/{document_id}/image_{index}.png",
            meta={
                "page_number": image.page_number,
                "image_index": image.image_index,
                "width": image.width,
                "height": image.height,
                "format": image.format,
            },
        )
