"""XBRL processing service."""

import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.core.config import settings
from app.core.feature_flags import is_enabled
from app.core.observability import trace_function
from app.models.artifact import Artifact
from app.schemas.processing import ProcessingResult, XBRLFact


class XBRLProcessor:
    """Service for processing XBRL documents."""
    
    def __init__(self) -> None:
        """Initialize XBRL processor."""
        self.enabled = is_enabled("enable_xbrl_processing")
        
        # Common XBRL namespaces
        self.namespaces = {
            "xbrli": "http://www.xbrl.org/2003/instance",
            "link": "http://www.xbrl.org/2003/linkbase",
            "xlink": "http://www.w3.org/1999/xlink",
            "us-gaap": "http://fasb.org/us-gaap/2023",
            "dei": "http://xbrl.sec.gov/dei/2023",
            "ifrs": "http://xbrl.ifrs.org/taxonomy/2021-03-24/ifrs-full",
        }
    
    @trace_function("xbrl_processor.process_document")
    async def process_document(
        self,
        document_id: str,
        org_id: str,
        file_path: str,
    ) -> ProcessingResult:
        """Process XBRL document and extract facts."""
        if not self.enabled:
            return ProcessingResult(
                document_id=document_id,
                status="skipped",
                message="XBRL processing disabled",
                artifacts=[],
            )
        
        try:
            # Parse XBRL document
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract contexts (time periods and entities)
            contexts = self._extract_contexts(root)
            
            # Extract units
            units = self._extract_units(root)
            
            # Extract facts
            facts = self._extract_facts(root, contexts, units)
            
            # Create facts artifact
            facts_artifact = await self._create_facts_artifact(
                document_id, org_id, facts
            )
            
            # Create taxonomy artifact
            taxonomy_artifact = await self._create_taxonomy_artifact(
                document_id, org_id, root
            )
            
            return ProcessingResult(
                document_id=document_id,
                status="completed",
                message=f"Extracted {len(facts)} XBRL facts",
                artifacts=[facts_artifact, taxonomy_artifact],
            )
            
        except Exception as e:
            return ProcessingResult(
                document_id=document_id,
                status="failed",
                message=f"XBRL processing failed: {str(e)}",
                artifacts=[],
            )
    
    def _extract_contexts(self, root: ET.Element) -> Dict[str, Dict[str, Any]]:
        """Extract context information from XBRL."""
        contexts = {}
        
        for context in root.findall(".//xbrli:context", self.namespaces):
            context_id = context.get("id")
            if not context_id:
                continue
            
            # Extract entity information
            entity_elem = context.find(".//xbrli:entity", self.namespaces)
            entity_id = None
            if entity_elem is not None:
                identifier = entity_elem.find(".//xbrli:identifier", self.namespaces)
                if identifier is not None:
                    entity_id = identifier.text
            
            # Extract period information
            period_elem = context.find(".//xbrli:period", self.namespaces)
            period_info = {}
            
            if period_elem is not None:
                # Instant period
                instant = period_elem.find(".//xbrli:instant", self.namespaces)
                if instant is not None:
                    period_info = {
                        "type": "instant",
                        "instant": instant.text,
                    }
                else:
                    # Duration period
                    start_date = period_elem.find(".//xbrli:startDate", self.namespaces)
                    end_date = period_elem.find(".//xbrli:endDate", self.namespaces)
                    if start_date is not None and end_date is not None:
                        period_info = {
                            "type": "duration",
                            "start_date": start_date.text,
                            "end_date": end_date.text,
                        }
            
            contexts[context_id] = {
                "entity_id": entity_id,
                "period": period_info,
            }
        
        return contexts
    
    def _extract_units(self, root: ET.Element) -> Dict[str, str]:
        """Extract unit information from XBRL."""
        units = {}
        
        for unit in root.findall(".//xbrli:unit", self.namespaces):
            unit_id = unit.get("id")
            if not unit_id:
                continue
            
            # Extract measure
            measure = unit.find(".//xbrli:measure", self.namespaces)
            if measure is not None:
                units[unit_id] = measure.text
        
        return units
    
    def _extract_facts(
        self, 
        root: ET.Element, 
        contexts: Dict[str, Dict[str, Any]], 
        units: Dict[str, str]
    ) -> List[XBRLFact]:
        """Extract facts from XBRL document."""
        facts = []
        
        # Find all elements that are not in the xbrli namespace (these are facts)
        for elem in root:
            if elem.tag.startswith("{http://www.xbrl.org/2003/instance}"):
                continue  # Skip XBRL infrastructure elements
            
            # Extract fact information
            context_ref = elem.get("contextRef")
            unit_ref = elem.get("unitRef")
            decimals = elem.get("decimals")
            
            # Get context information
            context_info = contexts.get(context_ref, {})
            period_info = context_info.get("period", {})
            
            # Get unit information
            unit = units.get(unit_ref) if unit_ref else None
            
            # Parse value
            value = self._parse_fact_value(elem.text, decimals)
            
            # Extract taxonomy and tag from element name
            namespace_uri, tag = self._parse_element_name(elem.tag)
            taxonomy = self._get_taxonomy_from_namespace(namespace_uri)
            
            fact = XBRLFact(
                taxonomy=taxonomy,
                tag=tag,
                value=value,
                unit=unit,
                period_start=period_info.get("start_date"),
                period_end=period_info.get("end_date") or period_info.get("instant"),
                decimals=int(decimals) if decimals and decimals != "INF" else None,
                context=context_ref,
            )
            facts.append(fact)
        
        return facts
    
    def _parse_fact_value(self, text: Optional[str], decimals: Optional[str]) -> Any:
        """Parse fact value with proper type conversion."""
        if not text:
            return None
        
        text = text.strip()
        
        # Try to parse as number
        try:
            if "." in text or decimals:
                return float(text)
            else:
                return int(text)
        except ValueError:
            # Return as string if not a number
            return text
    
    def _parse_element_name(self, tag: str) -> tuple[str, str]:
        """Parse element tag to extract namespace and local name."""
        if tag.startswith("{"):
            # Namespace format: {namespace}localname
            end_brace = tag.find("}")
            namespace = tag[1:end_brace]
            local_name = tag[end_brace + 1:]
            return namespace, local_name
        else:
            # No namespace
            return "", tag
    
    def _get_taxonomy_from_namespace(self, namespace_uri: str) -> str:
        """Get taxonomy name from namespace URI."""
        if "us-gaap" in namespace_uri:
            return "us-gaap"
        elif "ifrs" in namespace_uri:
            return "ifrs"
        elif "dei" in namespace_uri:
            return "dei"
        else:
            return "unknown"
    
    async def _create_facts_artifact(
        self,
        document_id: str,
        org_id: str,
        facts: List[XBRLFact],
    ) -> Artifact:
        """Create facts artifact."""
        artifact_id = str(uuid.uuid4())
        
        # Calculate statistics
        taxonomies = set(fact.taxonomy for fact in facts)
        numeric_facts = sum(1 for fact in facts if isinstance(fact.value, (int, float)))
        
        # TODO: Upload to S3 and create artifact record
        return Artifact(
            id=artifact_id,
            org_id=org_id,
            document_id=document_id,
            type="xbrl_facts",
            path_s3=f"{org_id}/processed/{document_id}/xbrl_facts.json",
            meta={
                "fact_count": len(facts),
                "numeric_fact_count": numeric_facts,
                "taxonomies": list(taxonomies),
            },
        )
    
    async def _create_taxonomy_artifact(
        self,
        document_id: str,
        org_id: str,
        root: ET.Element,
    ) -> Artifact:
        """Create taxonomy artifact."""
        artifact_id = str(uuid.uuid4())
        
        # Extract schema references
        schema_refs = []
        for schema_ref in root.findall(".//link:schemaRef", self.namespaces):
            href = schema_ref.get("{http://www.w3.org/1999/xlink}href")
            if href:
                schema_refs.append(href)
        
        # TODO: Upload to S3 and create artifact record
        return Artifact(
            id=artifact_id,
            org_id=org_id,
            document_id=document_id,
            type="xbrl_taxonomy",
            path_s3=f"{org_id}/processed/{document_id}/xbrl_taxonomy.json",
            meta={
                "schema_references": schema_refs,
                "namespaces": list(self.namespaces.keys()),
            },
        )
