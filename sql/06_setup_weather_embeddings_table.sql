-- Setup script for weather_documents_embeddings table
-- Run this manually in your Lakebase Postgres database before running the notebook
-- Replace {{EMBEDDING_DIM}} with your model's dimension (e.g., 384 for all-MiniLM-L6-v2)

-- Enable pgvector extension
--CREATE EXTENSION IF NOT EXISTS vector;

-- Create the embeddings table
-- IMPORTANT: Replace {{EMBEDDING_DIM}} below with the correct dimension for your model:
--   - sentence-transformers/all-MiniLM-L6-v2: 384
--   - sentence-transformers/all-mpnet-base-v2: 768
--   - BAAI/bge-small-en-v1.5: 384
--   - BAAI/bge-base-en-v1.5: 768
--   - BAAI/bge-large-en-v1.5: 1024
CREATE TABLE IF NOT EXISTS weather_documents_embeddings (
    id TEXT PRIMARY KEY,
    location TEXT NOT NULL,
    source_type TEXT NOT NULL,
    headline TEXT NOT NULL,
    effective_at TIMESTAMPTZ,
    embedding VECTOR({{EMBEDDING_DIM}}) NOT NULL,
    model_name TEXT NOT NULL,
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create HNSW index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_weather_documents_embeddings_embedding
ON weather_documents_embeddings
USING hnsw (embedding vector_cosine_ops);

-- Verify the table was created
SELECT 
    table_name,
    column_name,
    data_type,
    udt_name
FROM information_schema.columns
WHERE table_name = 'weather_documents_embeddings'
ORDER BY ordinal_position;