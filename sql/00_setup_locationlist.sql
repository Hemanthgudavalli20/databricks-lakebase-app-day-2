-- Setup script for locationlist table
-- Run this manually in your Lakebase Postgres database before running the app

-- Create the locationlist table
CREATE TABLE IF NOT EXISTS locationlist (
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    city TEXT,
    state TEXT,
    email TEXT NOT NULL,
    searched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (latitude, longitude, email)
);

-- Create index for email lookups
CREATE INDEX IF NOT EXISTS idx_locationlist_email 
ON locationlist (email);

-- Verify the table was created
SELECT 
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name = 'locationlist'
ORDER BY ordinal_position;