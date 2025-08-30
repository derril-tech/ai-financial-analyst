-- Initialize database with pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create initial schema
CREATE SCHEMA IF NOT EXISTS public;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA public TO postgres;
