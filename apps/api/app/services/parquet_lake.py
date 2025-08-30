"""Parquet data lake service."""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from minio import Minio

from app.core.config import settings
from app.core.observability import trace_function


class ParquetLakeService:
    """Service for managing parquet data lake."""
    
    def __init__(self) -> None:
        """Initialize parquet lake service."""
        self.minio_client = Minio(
            settings.S3_ENDPOINT.replace("http://", "").replace("https://", ""),
            access_key=settings.S3_ACCESS_KEY,
            secret_key=settings.S3_SECRET_KEY,
            secure=settings.S3_ENDPOINT.startswith("https://"),
        )
        
        # Data lake structure
        self.layers = {
            "bronze": "raw ingested data",
            "silver": "cleaned and normalized data", 
            "gold": "aggregated and business-ready data",
        }
    
    @trace_function("parquet_lake.write_bronze")
    async def write_bronze(
        self,
        org_id: str,
        document_id: str,
        data_type: str,
        data: Dict[str, Any] | List[Dict[str, Any]],
        snapshot_id: Optional[str] = None,
    ) -> str:
        """Write raw data to bronze layer."""
        snapshot_id = snapshot_id or str(uuid.uuid4())
        
        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        # Add metadata columns
        df["_org_id"] = org_id
        df["_document_id"] = document_id
        df["_snapshot_id"] = snapshot_id
        df["_ingested_at"] = datetime.utcnow()
        
        # Write to parquet
        path = f"datasets/bronze/{org_id}/{data_type}/{document_id}/{snapshot_id}.parquet"
        await self._write_parquet(df, path)
        
        return snapshot_id
    
    @trace_function("parquet_lake.write_silver")
    async def write_silver(
        self,
        org_id: str,
        data_type: str,
        data: pd.DataFrame,
        partition_cols: Optional[List[str]] = None,
        snapshot_id: Optional[str] = None,
    ) -> str:
        """Write cleaned data to silver layer."""
        snapshot_id = snapshot_id or str(uuid.uuid4())
        
        # Add metadata columns
        data = data.copy()
        data["_org_id"] = org_id
        data["_snapshot_id"] = snapshot_id
        data["_processed_at"] = datetime.utcnow()
        
        # Write to parquet with partitioning
        if partition_cols:
            path = f"datasets/silver/{org_id}/{data_type}"
            await self._write_partitioned_parquet(data, path, partition_cols)
        else:
            path = f"datasets/silver/{org_id}/{data_type}/{snapshot_id}.parquet"
            await self._write_parquet(data, path)
        
        return snapshot_id
    
    @trace_function("parquet_lake.write_gold")
    async def write_gold(
        self,
        org_id: str,
        table_name: str,
        data: pd.DataFrame,
        partition_cols: Optional[List[str]] = None,
        snapshot_id: Optional[str] = None,
    ) -> str:
        """Write aggregated data to gold layer."""
        snapshot_id = snapshot_id or str(uuid.uuid4())
        
        # Add metadata columns
        data = data.copy()
        data["_org_id"] = org_id
        data["_snapshot_id"] = snapshot_id
        data["_created_at"] = datetime.utcnow()
        
        # Write to parquet
        if partition_cols:
            path = f"datasets/gold/{org_id}/{table_name}"
            await self._write_partitioned_parquet(data, path, partition_cols)
        else:
            path = f"datasets/gold/{org_id}/{table_name}/{snapshot_id}.parquet"
            await self._write_parquet(data, path)
        
        return snapshot_id
    
    @trace_function("parquet_lake.read_bronze")
    async def read_bronze(
        self,
        org_id: str,
        document_id: str,
        data_type: str,
        snapshot_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Read data from bronze layer."""
        if snapshot_id:
            path = f"datasets/bronze/{org_id}/{data_type}/{document_id}/{snapshot_id}.parquet"
            return await self._read_parquet(path)
        else:
            # Read all snapshots for document
            path_prefix = f"datasets/bronze/{org_id}/{data_type}/{document_id}/"
            return await self._read_parquet_prefix(path_prefix)
    
    @trace_function("parquet_lake.read_silver")
    async def read_silver(
        self,
        org_id: str,
        data_type: str,
        filters: Optional[List[tuple]] = None,
        snapshot_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Read data from silver layer."""
        if snapshot_id:
            path = f"datasets/silver/{org_id}/{data_type}/{snapshot_id}.parquet"
            return await self._read_parquet(path)
        else:
            path_prefix = f"datasets/silver/{org_id}/{data_type}/"
            return await self._read_parquet_prefix(path_prefix, filters)
    
    @trace_function("parquet_lake.read_gold")
    async def read_gold(
        self,
        org_id: str,
        table_name: str,
        filters: Optional[List[tuple]] = None,
        snapshot_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Read data from gold layer."""
        if snapshot_id:
            path = f"datasets/gold/{org_id}/{table_name}/{snapshot_id}.parquet"
            return await self._read_parquet(path)
        else:
            path_prefix = f"datasets/gold/{org_id}/{table_name}/"
            return await self._read_parquet_prefix(path_prefix, filters)
    
    async def _write_parquet(self, df: pd.DataFrame, s3_path: str) -> None:
        """Write DataFrame to parquet file in S3."""
        # Convert to Arrow table
        table = pa.Table.from_pandas(df)
        
        # Write to bytes buffer
        buffer = pa.BufferOutputStream()
        pq.write_table(table, buffer)
        
        # Upload to S3
        buffer_bytes = buffer.getvalue().to_pybytes()
        self.minio_client.put_object(
            bucket_name=settings.S3_BUCKET,
            object_name=s3_path,
            data=buffer_bytes,
            length=len(buffer_bytes),
            content_type="application/octet-stream",
        )
    
    async def _write_partitioned_parquet(
        self, 
        df: pd.DataFrame, 
        s3_path: str, 
        partition_cols: List[str]
    ) -> None:
        """Write partitioned parquet dataset to S3."""
        # For simplicity, write as single file for now
        # In production, implement proper partitioning
        await self._write_parquet(df, f"{s3_path}/data.parquet")
    
    async def _read_parquet(self, s3_path: str) -> pd.DataFrame:
        """Read parquet file from S3."""
        try:
            # Get object from S3
            response = self.minio_client.get_object(
                bucket_name=settings.S3_BUCKET,
                object_name=s3_path,
            )
            
            # Read parquet
            table = pq.read_table(response)
            df = table.to_pandas()
            
            return df
            
        except Exception as e:
            print(f"Failed to read parquet from {s3_path}: {e}")
            return pd.DataFrame()
    
    async def _read_parquet_prefix(
        self, 
        s3_prefix: str, 
        filters: Optional[List[tuple]] = None
    ) -> pd.DataFrame:
        """Read all parquet files with given prefix."""
        try:
            # List objects with prefix
            objects = self.minio_client.list_objects(
                bucket_name=settings.S3_BUCKET,
                prefix=s3_prefix,
            )
            
            # Read all parquet files
            dfs = []
            for obj in objects:
                if obj.object_name.endswith(".parquet"):
                    df = await self._read_parquet(obj.object_name)
                    if not df.empty:
                        dfs.append(df)
            
            # Combine DataFrames
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # Apply filters if provided
                if filters:
                    for column, operator, value in filters:
                        if operator == "==":
                            combined_df = combined_df[combined_df[column] == value]
                        elif operator == "!=":
                            combined_df = combined_df[combined_df[column] != value]
                        elif operator == ">":
                            combined_df = combined_df[combined_df[column] > value]
                        elif operator == ">=":
                            combined_df = combined_df[combined_df[column] >= value]
                        elif operator == "<":
                            combined_df = combined_df[combined_df[column] < value]
                        elif operator == "<=":
                            combined_df = combined_df[combined_df[column] <= value]
                
                return combined_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Failed to read parquet with prefix {s3_prefix}: {e}")
            return pd.DataFrame()
    
    def get_snapshot_metadata(self, snapshot_id: str) -> Dict[str, Any]:
        """Get metadata for a snapshot."""
        return {
            "snapshot_id": snapshot_id,
            "created_at": datetime.utcnow().isoformat(),
            "reproducible": True,
        }
