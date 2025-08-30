/**
 * Query and answer types
 */

export interface Citation {
  source: string;
  kind: 'pdf' | 'xbrl' | 'audio' | 'slide' | 'api';
  locator: string;
  confidence: number;
}

export interface QueryRequest {
  org_id: string;
  prompt: string;
  tickers?: string[];
  options?: Record<string, any>;
}

export interface QueryResponse {
  query_id: string;
  text: string;
  confidence?: number;
  citations: Citation[];
  exports?: Record<string, any>;
}
