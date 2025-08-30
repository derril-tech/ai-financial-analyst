/**
 * Document-related types
 */

export interface Document {
  id: string;
  org_id: string;
  kind: 'pdf' | 'audio' | 'video' | 'xbrl' | 'spreadsheet' | 'presentation' | 'unknown';
  title: string;
  ticker?: string;
  fiscal_year?: number;
  fiscal_period?: 'Q1' | 'Q2' | 'Q3' | 'Q4' | 'FY';
  language: string;
  checksum: string;
  path_s3: string;
  meta: Record<string, any>;
  created_at: string;
}

export interface Artifact {
  id: string;
  org_id: string;
  document_id: string;
  type: 'table' | 'image' | 'text' | 'transcript' | 'xbrl_facts' | 'xbrl_taxonomy';
  path_s3: string;
  meta: Record<string, any>;
  created_at: string;
}
