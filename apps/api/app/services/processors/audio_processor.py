"""Audio processing service using WhisperX and pyannote."""

import uuid
from pathlib import Path
from typing import List, Dict, Any

import whisperx
import torch
from pyannote.audio import Pipeline

from app.core.config import settings
from app.core.feature_flags import is_enabled
from app.core.observability import trace_function
from app.models.artifact import Artifact
from app.schemas.processing import ProcessingResult, TranscriptSegment


class AudioProcessor:
    """Service for processing audio files."""
    
    def __init__(self) -> None:
        """Initialize audio processor."""
        self.enabled = is_enabled("enable_audio_processing")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"
        
        # Initialize models lazily
        self._whisper_model = None
        self._diarization_pipeline = None
        self._alignment_model = None
    
    def _load_whisper_model(self) -> Any:
        """Load WhisperX model."""
        if self._whisper_model is None:
            self._whisper_model = whisperx.load_model(
                "large-v2", 
                self.device, 
                compute_type=self.compute_type
            )
        return self._whisper_model
    
    def _load_diarization_pipeline(self) -> Any:
        """Load pyannote diarization pipeline."""
        if self._diarization_pipeline is None:
            # Note: Requires HuggingFace token for pyannote models
            try:
                self._diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=True  # Set HF_TOKEN environment variable
                )
            except Exception as e:
                print(f"Failed to load diarization pipeline: {e}")
                self._diarization_pipeline = None
        return self._diarization_pipeline
    
    def _load_alignment_model(self) -> Any:
        """Load alignment model."""
        if self._alignment_model is None:
            self._alignment_model, self._alignment_metadata = whisperx.load_align_model(
                language_code="en", 
                device=self.device
            )
        return self._alignment_model, self._alignment_metadata
    
    @trace_function("audio_processor.process_document")
    async def process_document(
        self,
        document_id: str,
        org_id: str,
        file_path: str,
    ) -> ProcessingResult:
        """Process audio document and extract transcript."""
        if not self.enabled:
            return ProcessingResult(
                document_id=document_id,
                status="skipped",
                message="Audio processing disabled",
                artifacts=[],
            )
        
        try:
            # Load audio
            audio = whisperx.load_audio(file_path)
            
            # Transcribe with WhisperX
            model = self._load_whisper_model()
            result = model.transcribe(audio, batch_size=16)
            
            # Align transcript
            align_model, align_metadata = self._load_alignment_model()
            aligned_result = whisperx.align(
                result["segments"], 
                align_model, 
                align_metadata, 
                audio, 
                self.device, 
                return_char_alignments=False
            )
            
            # Perform speaker diarization
            diarization_pipeline = self._load_diarization_pipeline()
            segments_with_speakers = []
            
            if diarization_pipeline is not None:
                # Run diarization
                diarization = diarization_pipeline(file_path)
                
                # Assign speakers to segments
                segments_with_speakers = self._assign_speakers(
                    aligned_result["segments"], 
                    diarization
                )
            else:
                # No diarization available
                segments_with_speakers = [
                    TranscriptSegment(
                        speaker=None,
                        text=seg["text"],
                        start_time=seg.get("start", 0.0),
                        end_time=seg.get("end", 0.0),
                        confidence=seg.get("confidence"),
                    )
                    for seg in aligned_result["segments"]
                ]
            
            # Classify speakers (CEO, CFO, Analyst, etc.)
            classified_segments = self._classify_speakers(segments_with_speakers)
            
            # Create transcript artifact
            transcript_artifact = await self._create_transcript_artifact(
                document_id, org_id, classified_segments
            )
            
            return ProcessingResult(
                document_id=document_id,
                status="completed",
                message=f"Processed {len(classified_segments)} transcript segments",
                artifacts=[transcript_artifact],
            )
            
        except Exception as e:
            return ProcessingResult(
                document_id=document_id,
                status="failed",
                message=f"Audio processing failed: {str(e)}",
                artifacts=[],
            )
    
    def _assign_speakers(
        self, 
        segments: List[Dict[str, Any]], 
        diarization: Any
    ) -> List[TranscriptSegment]:
        """Assign speakers to transcript segments."""
        segments_with_speakers = []
        
        for segment in segments:
            start_time = segment.get("start", 0.0)
            end_time = segment.get("end", 0.0)
            
            # Find the most overlapping speaker
            best_speaker = None
            max_overlap = 0.0
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                overlap_start = max(start_time, turn.start)
                overlap_end = min(end_time, turn.end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_speaker = speaker
            
            transcript_segment = TranscriptSegment(
                speaker=best_speaker,
                text=segment["text"],
                start_time=start_time,
                end_time=end_time,
                confidence=segment.get("confidence"),
            )
            segments_with_speakers.append(transcript_segment)
        
        return segments_with_speakers
    
    def _classify_speakers(
        self, 
        segments: List[TranscriptSegment]
    ) -> List[TranscriptSegment]:
        """Classify speakers into roles (CEO, CFO, Analyst, etc.)."""
        # Simple heuristic-based classification
        # In production, this would use a more sophisticated model
        
        speaker_roles = {}
        
        for segment in segments:
            if segment.speaker and segment.speaker not in speaker_roles:
                # Analyze text content for role indicators
                text_lower = segment.text.lower()
                
                if any(phrase in text_lower for phrase in [
                    "chief executive", "ceo", "president"
                ]):
                    speaker_roles[segment.speaker] = "CEO"
                elif any(phrase in text_lower for phrase in [
                    "chief financial", "cfo", "finance"
                ]):
                    speaker_roles[segment.speaker] = "CFO"
                elif any(phrase in text_lower for phrase in [
                    "analyst", "question", "thank you for taking"
                ]):
                    speaker_roles[segment.speaker] = "Analyst"
                else:
                    speaker_roles[segment.speaker] = "Unknown"
        
        # Update segments with classified roles
        for segment in segments:
            if segment.speaker in speaker_roles:
                segment.speaker = f"{speaker_roles[segment.speaker]}_{segment.speaker}"
        
        return segments
    
    async def _create_transcript_artifact(
        self,
        document_id: str,
        org_id: str,
        segments: List[TranscriptSegment],
    ) -> Artifact:
        """Create transcript artifact."""
        artifact_id = str(uuid.uuid4())
        
        # Calculate statistics
        total_duration = max(seg.end_time for seg in segments) if segments else 0
        speaker_count = len(set(seg.speaker for seg in segments if seg.speaker))
        
        # TODO: Upload to S3 and create artifact record
        return Artifact(
            id=artifact_id,
            org_id=org_id,
            document_id=document_id,
            type="transcript",
            path_s3=f"{org_id}/processed/{document_id}/transcript.json",
            meta={
                "segment_count": len(segments),
                "total_duration": total_duration,
                "speaker_count": speaker_count,
                "language": "en",  # TODO: Detect language
            },
        )
