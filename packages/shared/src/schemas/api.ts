/**
 * API validation schemas using Zod
 */

import { z } from 'zod';

export const DocumentMetadataSchema = z.object({
  title: z.string().optional(),
  ticker: z.string().max(10).optional(),
  fiscal_year: z.number().int().min(1900).max(2100).optional(),
  fiscal_period: z.enum(['Q1', 'Q2', 'Q3', 'Q4', 'FY']).optional(),
  language: z.string().max(10).default('en'),
  uploader: z.string().optional(),
});

export const QueryRequestSchema = z.object({
  org_id: z.string(),
  prompt: z.string().min(1),
  tickers: z.array(z.string()).optional(),
  options: z.record(z.any()).optional(),
});

export const CitationSchema = z.object({
  source: z.string(),
  kind: z.enum(['pdf', 'xbrl', 'audio', 'slide', 'api']),
  locator: z.string(),
  confidence: z.number().min(0).max(1),
});

export type DocumentMetadata = z.infer<typeof DocumentMetadataSchema>;
export type QueryRequest = z.infer<typeof QueryRequestSchema>;
export type Citation = z.infer<typeof CitationSchema>;
