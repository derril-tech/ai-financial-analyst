"""Embedding service for vector indexing."""

import hashlib
import uuid
from typing import List, Optional, Dict, Any

import openai
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.core.config import settings
from app.core.observability import trace_function
from app.models.vector_index import VectorIndex


class EmbeddingService:
    """Service for generating and managing embeddings."""
    
    def __init__(self) -> None:
        """Initialize embedding service."""
        self.client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "text-embedding-3-large"
        self.dimension = 3072  # text-embedding-3-large dimension
        self.batch_size = 100
    
    @trace_function("embedding_service.embed_text")
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            return [0.0] * self.dimension
        
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            return [0.0] * self.dimension
    
    @trace_function("embedding_service.embed_batch")
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []
        
        # Filter empty texts
        non_empty_texts = [text for text in texts if text.strip()]
        if not non_empty_texts:
            return [[0.0] * self.dimension] * len(texts)
        
        try:
            # Process in batches
            all_embeddings = []
            for i in range(0, len(non_empty_texts), self.batch_size):
                batch = non_empty_texts[i:i + self.batch_size]
                
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    encoding_format="float"
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
            
        except Exception as e:
            print(f"Batch embedding generation failed: {e}")
            return [[0.0] * self.dimension] * len(texts)
    
    @trace_function("embedding_service.store_embedding")
    async def store_embedding(
        self,
        db: AsyncSession,
        org_id: str,
        document_id: str,
        collection: str,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> str:
        """Store embedding in vector index."""
        vector_id = str(uuid.uuid4())
        
        # Create payload with text and metadata
        payload = {
            "text": text,
            "metadata": metadata,
            "text_hash": hashlib.sha256(text.encode()).hexdigest()[:16],
        }
        
        vector_entry = VectorIndex(
            id=vector_id,
            org_id=org_id,
            document_id=document_id,
            collection=collection,
            embedding=embedding,
            payload=payload,
        )
        
        db.add(vector_entry)
        await db.commit()
        
        return vector_id
    
    @trace_function("embedding_service.similarity_search")
    async def similarity_search(
        self,
        db: AsyncSession,
        org_id: str,
        query_embedding: List[float],
        collection: Optional[str] = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Perform similarity search using cosine similarity."""
        # Build query
        query_parts = [
            "SELECT id, org_id, document_id, collection, payload,",
            "1 - (embedding <=> :query_embedding) as similarity",
            "FROM vector_index",
            "WHERE org_id = :org_id",
        ]
        
        params = {
            "query_embedding": query_embedding,
            "org_id": org_id,
        }
        
        if collection:
            query_parts.append("AND collection = :collection")
            params["collection"] = collection
        
        query_parts.extend([
            "AND 1 - (embedding <=> :query_embedding) >= :threshold",
            "ORDER BY embedding <=> :query_embedding",
            "LIMIT :limit"
        ])
        
        params.update({
            "threshold": threshold,
            "limit": limit,
        })
        
        query_sql = " ".join(query_parts)
        
        try:
            result = await db.execute(text(query_sql), params)
            rows = result.fetchall()
            
            return [
                {
                    "id": row.id,
                    "document_id": row.document_id,
                    "collection": row.collection,
                    "similarity": float(row.similarity),
                    "text": row.payload.get("text", ""),
                    "metadata": row.payload.get("metadata", {}),
                }
                for row in rows
            ]
            
        except Exception as e:
            print(f"Similarity search failed: {e}")
            return []
    
    @trace_function("embedding_service.create_index")
    async def create_index(self, db: AsyncSession) -> None:
        """Create vector index for faster similarity search."""
        try:
            # Create IVFFlat index for approximate nearest neighbor search
            await db.execute(text("""
                CREATE INDEX IF NOT EXISTS vector_index_embedding_idx 
                ON vector_index 
                USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = 100)
            """))
            
            # Create additional indexes for filtering
            await db.execute(text("""
                CREATE INDEX IF NOT EXISTS vector_index_org_collection_idx 
                ON vector_index (org_id, collection)
            """))
            
            await db.commit()
            
        except Exception as e:
            print(f"Index creation failed: {e}")
    
    def get_semantic_cache_key(self, query: str, filters: Dict[str, Any]) -> str:
        """Generate cache key for semantic queries."""
        cache_input = f"{query}:{sorted(filters.items())}"
        return hashlib.sha256(cache_input.encode()).hexdigest()[:16]
