"""Text chunking service for different document types."""

import re
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.core.observability import trace_function


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    id: str
    text: str
    chunk_type: str  # passage, table_row, slide, transcript_segment
    metadata: Dict[str, Any]
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class ChunkingService:
    """Service for chunking different types of content."""
    
    def __init__(self) -> None:
        """Initialize chunking service."""
        self.max_chunk_size = 1000  # characters
        self.overlap_size = 100     # characters
        self.min_chunk_size = 50    # characters
    
    @trace_function("chunking_service.chunk_text_passages")
    def chunk_text_passages(
        self, 
        text: str, 
        document_id: str,
        page_number: Optional[int] = None
    ) -> List[Chunk]:
        """Chunk text into overlapping passages."""
        if not text or len(text) < self.min_chunk_size:
            return []
        
        chunks = []
        
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text.strip())
        
        current_chunk = ""
        current_start = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed max size, finalize current chunk
            if (len(current_chunk) + len(paragraph) + 1 > self.max_chunk_size and 
                len(current_chunk) >= self.min_chunk_size):
                
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    text=current_chunk.strip(),
                    chunk_type="passage",
                    metadata={
                        "document_id": document_id,
                        "page_number": page_number,
                        "chunk_index": len(chunks),
                    },
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + paragraph
                current_start += len(current_chunk) - len(overlap_text) - len(paragraph) - 1
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip() and len(current_chunk) >= self.min_chunk_size:
            chunk = Chunk(
                id=str(uuid.uuid4()),
                text=current_chunk.strip(),
                chunk_type="passage",
                metadata={
                    "document_id": document_id,
                    "page_number": page_number,
                    "chunk_index": len(chunks),
                },
                start_char=current_start,
                end_char=current_start + len(current_chunk),
            )
            chunks.append(chunk)
        
        return chunks
    
    @trace_function("chunking_service.chunk_table")
    def chunk_table(
        self, 
        table_data: List[Dict[str, Any]], 
        headers: List[str],
        document_id: str,
        table_index: int,
        page_number: Optional[int] = None
    ) -> List[Chunk]:
        """Chunk table data by rows with header context."""
        if not table_data or not headers:
            return []
        
        chunks = []
        
        # Create header context
        header_text = " | ".join(headers)
        
        # Chunk by rows or groups of rows
        for i, row in enumerate(table_data):
            # Create row text
            row_values = [str(row.get(header, "")) for header in headers]
            row_text = " | ".join(row_values)
            
            # Combine header and row
            chunk_text = f"Headers: {header_text}\nRow {i+1}: {row_text}"
            
            chunk = Chunk(
                id=str(uuid.uuid4()),
                text=chunk_text,
                chunk_type="table_row",
                metadata={
                    "document_id": document_id,
                    "table_index": table_index,
                    "page_number": page_number,
                    "row_index": i,
                    "headers": headers,
                    "row_data": row,
                },
            )
            chunks.append(chunk)
        
        # Also create a summary chunk for the entire table
        if len(table_data) > 1:
            summary_rows = table_data[:3]  # First 3 rows as sample
            summary_text = f"Table with {len(table_data)} rows and columns: {', '.join(headers)}\n"
            summary_text += f"Headers: {header_text}\n"
            
            for i, row in enumerate(summary_rows):
                row_values = [str(row.get(header, "")) for header in headers]
                summary_text += f"Row {i+1}: {' | '.join(row_values)}\n"
            
            if len(table_data) > 3:
                summary_text += f"... and {len(table_data) - 3} more rows"
            
            summary_chunk = Chunk(
                id=str(uuid.uuid4()),
                text=summary_text,
                chunk_type="table_summary",
                metadata={
                    "document_id": document_id,
                    "table_index": table_index,
                    "page_number": page_number,
                    "row_count": len(table_data),
                    "headers": headers,
                },
            )
            chunks.append(summary_chunk)
        
        return chunks
    
    @trace_function("chunking_service.chunk_transcript")
    def chunk_transcript(
        self, 
        segments: List[Dict[str, Any]], 
        document_id: str
    ) -> List[Chunk]:
        """Chunk transcript by speaker segments and time windows."""
        if not segments:
            return []
        
        chunks = []
        
        # Group segments by speaker and time windows
        current_speaker = None
        current_segments = []
        current_duration = 0
        max_duration = 300  # 5 minutes
        
        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "")
            start_time = segment.get("start_time", 0)
            end_time = segment.get("end_time", 0)
            duration = end_time - start_time
            
            # Start new chunk if speaker changes or duration exceeds limit
            if (current_speaker != speaker or 
                current_duration + duration > max_duration) and current_segments:
                
                chunk = self._create_transcript_chunk(
                    current_segments, document_id, len(chunks)
                )
                chunks.append(chunk)
                
                current_segments = []
                current_duration = 0
            
            current_segments.append(segment)
            current_duration += duration
            current_speaker = speaker
        
        # Add final chunk
        if current_segments:
            chunk = self._create_transcript_chunk(
                current_segments, document_id, len(chunks)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_transcript_chunk(
        self, 
        segments: List[Dict[str, Any]], 
        document_id: str, 
        chunk_index: int
    ) -> Chunk:
        """Create a transcript chunk from segments."""
        if not segments:
            raise ValueError("Cannot create chunk from empty segments")
        
        # Combine segment texts
        texts = []
        speakers = set()
        start_time = min(seg.get("start_time", 0) for seg in segments)
        end_time = max(seg.get("end_time", 0) for seg in segments)
        
        for segment in segments:
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "")
            speakers.add(speaker)
            
            if text.strip():
                texts.append(f"{speaker}: {text}")
        
        chunk_text = "\n".join(texts)
        
        return Chunk(
            id=str(uuid.uuid4()),
            text=chunk_text,
            chunk_type="transcript_segment",
            metadata={
                "document_id": document_id,
                "chunk_index": chunk_index,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "speakers": list(speakers),
                "segment_count": len(segments),
            },
        )
    
    @trace_function("chunking_service.chunk_slide")
    def chunk_slide(
        self, 
        slide_content: Dict[str, Any], 
        document_id: str,
        slide_index: int
    ) -> List[Chunk]:
        """Chunk presentation slide content."""
        chunks = []
        
        title = slide_content.get("title", "")
        content = slide_content.get("content", "")
        notes = slide_content.get("notes", "")
        
        # Combine slide elements
        slide_text_parts = []
        
        if title:
            slide_text_parts.append(f"Title: {title}")
        
        if content:
            slide_text_parts.append(f"Content: {content}")
        
        if notes:
            slide_text_parts.append(f"Notes: {notes}")
        
        if slide_text_parts:
            slide_text = "\n".join(slide_text_parts)
            
            chunk = Chunk(
                id=str(uuid.uuid4()),
                text=slide_text,
                chunk_type="slide",
                metadata={
                    "document_id": document_id,
                    "slide_index": slide_index,
                    "has_title": bool(title),
                    "has_content": bool(content),
                    "has_notes": bool(notes),
                },
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of a chunk."""
        if len(text) <= self.overlap_size:
            return text
        
        # Try to break at sentence boundary
        overlap_start = len(text) - self.overlap_size
        sentence_end = text.rfind('.', overlap_start)
        
        if sentence_end > overlap_start:
            return text[sentence_end + 1:].strip()
        else:
            return text[-self.overlap_size:].strip()
    
    @trace_function("chunking_service.chunk_xbrl_facts")
    def chunk_xbrl_facts(
        self, 
        facts: List[Dict[str, Any]], 
        document_id: str
    ) -> List[Chunk]:
        """Chunk XBRL facts by taxonomy and context."""
        if not facts:
            return []
        
        chunks = []
        
        # Group facts by taxonomy and period
        fact_groups = {}
        
        for fact in facts:
            taxonomy = fact.get("taxonomy", "unknown")
            period_key = f"{fact.get('period_start', '')}-{fact.get('period_end', '')}"
            group_key = f"{taxonomy}:{period_key}"
            
            if group_key not in fact_groups:
                fact_groups[group_key] = []
            fact_groups[group_key].append(fact)
        
        # Create chunks for each group
        for group_key, group_facts in fact_groups.items():
            taxonomy, period = group_key.split(":", 1)
            
            # Create descriptive text for the facts
            fact_texts = []
            for fact in group_facts:
                tag = fact.get("tag", "")
                value = fact.get("value", "")
                unit = fact.get("unit", "")
                
                if value is not None:
                    fact_text = f"{tag}: {value}"
                    if unit:
                        fact_text += f" {unit}"
                    fact_texts.append(fact_text)
            
            if fact_texts:
                chunk_text = f"XBRL Facts - {taxonomy} ({period}):\n" + "\n".join(fact_texts)
                
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    chunk_type="xbrl_facts",
                    metadata={
                        "document_id": document_id,
                        "taxonomy": taxonomy,
                        "period": period,
                        "fact_count": len(group_facts),
                        "facts": group_facts,
                    },
                )
                chunks.append(chunk)
        
        return chunks
