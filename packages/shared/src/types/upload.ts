/**
 * Upload-related types
 */

export interface UploadMetadata {
  title?: string;
  ticker?: string;
  fiscal_year?: number;
  fiscal_period?: 'Q1' | 'Q2' | 'Q3' | 'Q4' | 'FY';
  language?: string;
  uploader?: string;
}

export interface UploadResponse {
  document_id: string;
  org_id: string;
  kind: string;
  title: string;
  ticker?: string;
  fiscal_year?: number;
  fiscal_period?: string;
  language: string;
  checksum: string;
  path_s3: string;
  meta: Record<string, any>;
}
