"""Hybrid retrieval service combining dense and sparse search."""

import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text, func

from app.core.config import settings
from app.core.observability import trace_function
from app.services.embedding import EmbeddingService
from app.models.vector_index import VectorIndex


@dataclass
class RetrievalResult:
    """Result from retrieval search."""
    id: str
    document_id: str
    collection: str
    text: str
    metadata: Dict[str, Any]
    score: float
    retrieval_method: str  # dense, sparse, hybrid


class HybridRetriever:
    """Hybrid retrieval combining dense vector search and sparse BM25."""
    
    def __init__(self) -> None:
        """Initialize hybrid retriever."""
        self.embedding_service = EmbeddingService()
        self.redis_client = redis.from_url(settings.REDIS_URL)
        
        # Retrieval parameters
        self.dense_weight = 0.7
        self.sparse_weight = 0.3
        self.min_score_threshold = 0.5
    
    @trace_function("hybrid_retriever.search")
    async def search(
        self,
        db: AsyncSession,
        query: str,
        org_id: str,
        collections: Optional[List[str]] = None,
        limit: int = 20,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Perform hybrid search combining dense and sparse retrieval."""
        dense_weight = dense_weight or self.dense_weight
        sparse_weight = sparse_weight or self.sparse_weight
        
        # Perform dense and sparse searches in parallel
        dense_task = self._dense_search(db, query, org_id, collections, limit * 2)
        sparse_task = self._sparse_search(db, query, org_id, collections, limit * 2)
        
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
        
        # Combine results using Reciprocal Rank Fusion (RRF)
        fused_results = self._reciprocal_rank_fusion(
            dense_results, sparse_results, dense_weight, sparse_weight
        )
        
        # Filter by minimum score and limit
        filtered_results = [
            result for result in fused_results 
            if result.score >= self.min_score_threshold
        ]
        
        return filtered_results[:limit]
    
    @trace_function("hybrid_retriever.dense_search")
    async def _dense_search(
        self,
        db: AsyncSession,
        query: str,
        org_id: str,
        collections: Optional[List[str]] = None,
        limit: int = 40,
    ) -> List[RetrievalResult]:
        """Perform dense vector search."""
        # Generate query embedding
        query_embedding = await self.embedding_service.embed_text(query)
        
        # Search for each collection or all collections
        all_results = []
        
        if collections:
            for collection in collections:
                results = await self.embedding_service.similarity_search(
                    db, org_id, query_embedding, collection, limit
                )
                all_results.extend(results)
        else:
            results = await self.embedding_service.similarity_search(
                db, org_id, query_embedding, None, limit
            )
            all_results.extend(results)
        
        # Convert to RetrievalResult objects
        retrieval_results = []
        for result in all_results:
            retrieval_result = RetrievalResult(
                id=result["id"],
                document_id=result["document_id"],
                collection=result["collection"],
                text=result["text"],
                metadata=result["metadata"],
                score=result["similarity"],
                retrieval_method="dense",
            )
            retrieval_results.append(retrieval_result)
        
        # Sort by score descending
        retrieval_results.sort(key=lambda x: x.score, reverse=True)
        return retrieval_results[:limit]
    
    @trace_function("hybrid_retriever.sparse_search")
    async def _sparse_search(
        self,
        db: AsyncSession,
        query: str,
        org_id: str,
        collections: Optional[List[str]] = None,
        limit: int = 40,
    ) -> List[RetrievalResult]:
        """Perform sparse BM25-style search using PostgreSQL full-text search."""
        # Prepare query for full-text search
        query_terms = self._prepare_search_query(query)
        
        if not query_terms:
            return []
        
        # Build SQL query
        query_parts = [
            "SELECT vi.id, vi.document_id, vi.collection, vi.payload,",
            "ts_rank_cd(to_tsvector('english', vi.payload->>'text'), query) as rank",
            "FROM vector_index vi,",
            "to_tsquery('english', :query_terms) query",
            "WHERE vi.org_id = :org_id",
            "AND to_tsvector('english', vi.payload->>'text') @@ query",
        ]
        
        params = {
            "query_terms": query_terms,
            "org_id": org_id,
        }
        
        if collections:
            placeholders = ", ".join([f":collection_{i}" for i in range(len(collections))])
            query_parts.append(f"AND vi.collection IN ({placeholders})")
            for i, collection in enumerate(collections):
                params[f"collection_{i}"] = collection
        
        query_parts.extend([
            "ORDER BY rank DESC",
            "LIMIT :limit"
        ])
        
        params["limit"] = limit
        
        query_sql = " ".join(query_parts)
        
        try:
            result = await db.execute(text(query_sql), params)
            rows = result.fetchall()
            
            retrieval_results = []
            for row in rows:
                retrieval_result = RetrievalResult(
                    id=row.id,
                    document_id=row.document_id,
                    collection=row.collection,
                    text=row.payload.get("text", ""),
                    metadata=row.payload.get("metadata", {}),
                    score=float(row.rank) if row.rank else 0.0,
                    retrieval_method="sparse",
                )
                retrieval_results.append(retrieval_result)
            
            return retrieval_results
            
        except Exception as e:
            print(f"Sparse search failed: {e}")
            return []
    
    def _prepare_search_query(self, query: str) -> str:
        """Prepare query for PostgreSQL full-text search."""
        # Clean and tokenize query
        query = re.sub(r'[^\w\s]', ' ', query.lower())
        terms = [term.strip() for term in query.split() if len(term.strip()) > 2]
        
        if not terms:
            return ""
        
        # Create tsquery format (AND operation between terms)
        return " & ".join(terms)
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        dense_weight: float,
        sparse_weight: float,
        k: int = 60,
    ) -> List[RetrievalResult]:
        """Combine results using Reciprocal Rank Fusion."""
        # Create lookup maps
        dense_lookup = {result.id: (i + 1, result) for i, result in enumerate(dense_results)}
        sparse_lookup = {result.id: (i + 1, result) for i, result in enumerate(sparse_results)}
        
        # Get all unique result IDs
        all_ids = set(dense_lookup.keys()) | set(sparse_lookup.keys())
        
        # Calculate RRF scores
        fused_results = []
        for result_id in all_ids:
            dense_rank, dense_result = dense_lookup.get(result_id, (float('inf'), None))
            sparse_rank, sparse_result = sparse_lookup.get(result_id, (float('inf'), None))
            
            # Use the result object from whichever method found it (prefer dense)
            result_obj = dense_result or sparse_result
            if not result_obj:
                continue
            
            # Calculate RRF score
            dense_rrf = dense_weight / (k + dense_rank) if dense_rank != float('inf') else 0
            sparse_rrf = sparse_weight / (k + sparse_rank) if sparse_rank != float('inf') else 0
            
            total_score = dense_rrf + sparse_rrf
            
            # Create new result with fused score
            fused_result = RetrievalResult(
                id=result_obj.id,
                document_id=result_obj.document_id,
                collection=result_obj.collection,
                text=result_obj.text,
                metadata=result_obj.metadata,
                score=total_score,
                retrieval_method="hybrid",
            )
            fused_results.append(fused_result)
        
        # Sort by fused score descending
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results
    
    @trace_function("hybrid_retriever.search_tables")
    async def search_tables(
        self,
        db: AsyncSession,
        query: str,
        org_id: str,
        limit: int = 10,
    ) -> List[RetrievalResult]:
        """Search specifically in table content."""
        return await self.search(
            db, query, org_id, collections=["table_row", "table_summary"], limit=limit
        )
    
    @trace_function("hybrid_retriever.search_transcripts")
    async def search_transcripts(
        self,
        db: AsyncSession,
        query: str,
        org_id: str,
        limit: int = 10,
    ) -> List[RetrievalResult]:
        """Search specifically in transcript content."""
        return await self.search(
            db, query, org_id, collections=["transcript_segment"], limit=limit
        )
    
    @trace_function("hybrid_retriever.search_documents")
    async def search_documents(
        self,
        db: AsyncSession,
        query: str,
        org_id: str,
        document_ids: List[str],
        limit: int = 10,
    ) -> List[RetrievalResult]:
        """Search within specific documents."""
        # Generate query embedding
        query_embedding = await self.embedding_service.embed_text(query)
        
        # Build query with document filter
        query_sql = """
            SELECT id, org_id, document_id, collection, payload,
            1 - (embedding <=> :query_embedding) as similarity
            FROM vector_index
            WHERE org_id = :org_id
            AND document_id = ANY(:document_ids)
            AND 1 - (embedding <=> :query_embedding) >= :threshold
            ORDER BY embedding <=> :query_embedding
            LIMIT :limit
        """
        
        params = {
            "query_embedding": query_embedding,
            "org_id": org_id,
            "document_ids": document_ids,
            "threshold": self.min_score_threshold,
            "limit": limit,
        }
        
        try:
            result = await db.execute(text(query_sql), params)
            rows = result.fetchall()
            
            return [
                RetrievalResult(
                    id=row.id,
                    document_id=row.document_id,
                    collection=row.collection,
                    text=row.payload.get("text", ""),
                    metadata=row.payload.get("metadata", {}),
                    score=float(row.similarity),
                    retrieval_method="dense",
                )
                for row in rows
            ]
            
        except Exception as e:
            print(f"Document search failed: {e}")
            return []


class TableAwareRetriever:
    """Specialized retriever for table content with cell coordinate awareness."""
    
    def __init__(self, hybrid_retriever: HybridRetriever) -> None:
        """Initialize table-aware retriever."""
        self.hybrid_retriever = hybrid_retriever
    
    @trace_function("table_aware_retriever.search_table_cells")
    async def search_table_cells(
        self,
        db: AsyncSession,
        query: str,
        org_id: str,
        table_schema: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search table content with cell coordinate information."""
        # First get table results
        table_results = await self.hybrid_retriever.search_tables(
            db, query, org_id, limit
        )
        
        # Enhance results with cell coordinates and context
        enhanced_results = []
        for result in table_results:
            metadata = result.metadata
            
            if result.collection == "table_row":
                # Extract cell coordinates and values
                row_data = metadata.get("row_data", {})
                headers = metadata.get("headers", [])
                row_index = metadata.get("row_index", 0)
                
                # Find relevant cells based on query
                relevant_cells = self._find_relevant_cells(query, row_data, headers)
                
                enhanced_result = {
                    "id": result.id,
                    "document_id": result.document_id,
                    "text": result.text,
                    "score": result.score,
                    "table_info": {
                        "table_index": metadata.get("table_index", 0),
                        "row_index": row_index,
                        "headers": headers,
                        "relevant_cells": relevant_cells,
                        "full_row": row_data,
                    },
                }
                enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _find_relevant_cells(
        self, 
        query: str, 
        row_data: Dict[str, Any], 
        headers: List[str]
    ) -> List[Dict[str, Any]]:
        """Find cells in a row that are most relevant to the query."""
        query_lower = query.lower()
        relevant_cells = []
        
        for i, header in enumerate(headers):
            cell_value = str(row_data.get(header, ""))
            
            # Calculate relevance score
            relevance = 0.0
            
            # Exact match in cell value
            if query_lower in cell_value.lower():
                relevance += 1.0
            
            # Header relevance
            if any(term in header.lower() for term in query_lower.split()):
                relevance += 0.5
            
            # Numeric value relevance (for financial queries)
            if self._is_numeric(cell_value) and any(
                term in query_lower for term in ["revenue", "income", "profit", "loss", "cash"]
            ):
                relevance += 0.3
            
            if relevance > 0:
                relevant_cells.append({
                    "column_index": i,
                    "header": header,
                    "value": cell_value,
                    "relevance": relevance,
                })
        
        # Sort by relevance
        relevant_cells.sort(key=lambda x: x["relevance"], reverse=True)
        return relevant_cells[:5]  # Top 5 most relevant cells
    
    def _is_numeric(self, value: str) -> bool:
        """Check if a value is numeric."""
        try:
            float(str(value).replace(",", "").replace("$", "").replace("%", ""))
            return True
        except (ValueError, TypeError):
            return False
