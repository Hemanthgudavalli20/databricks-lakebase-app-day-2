# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Ticker News -> Vector Embeddings (Lakebase)
# MAGIC
# MAGIC This notebook is part of the **Context Engineering on Databricks** course.
# MAGIC
# MAGIC It:
# MAGIC 1. Reads the `watchlist` table in Lakebase to find out which ticker
# MAGIC    symbols are currently being tracked.
# MAGIC 2. Fetches recent news for those tickers directly from the Massive
# MAGIC    `/v2/reference/news` endpoint (see `massive_client.py` for the same
# MAGIC    call shape used by the Flask app's `POST /news/sync` route), rate
# MAGIC    limited to stay within the free Massive API tier's strict quota, and
# MAGIC    upserts the results into the `ticker_news_documents` table.
# MAGIC 3. Computes a sentence embedding for each article (title + description)
# MAGIC    using Spark, distributed across the cluster via a pandas UDF, and
# MAGIC    writes them into a `ticker_news_embeddings` table using the
# MAGIC    `pgvector` Postgres extension so downstream RAG / context-engineering
# MAGIC    exercises can run similarity search directly in Postgres.
# MAGIC 4. Fetches the full article body for each `article_url` (via
# MAGIC    `trafilatura`, which strips nav/ads/boilerplate from the raw HTML),
# MAGIC    splits it into overlapping text chunks, embeds each chunk, and writes
# MAGIC    them into a `ticker_news_chunk_embeddings` table - so RAG exercises can
# MAGIC    retrieve fine-grained passages from article bodies, not just
# MAGIC    title/description.
# MAGIC
# MAGIC It re-uses the SAME Lakebase secret (scope `database`, key `lakebase-url`)
# MAGIC that `lakebase.py` uses in the Flask app, so no extra secrets need to be
# MAGIC created for this notebook.

# COMMAND ----------

# DBTITLE 1,Install all required packages
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary
# MAGIC %pip install -q 'databricks-sdk>=0.118.0' sentence-transformers trafilatura requests pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC Widgets let you override the source/destination table names and the
# MAGIC embedding model without editing the notebook - useful when running this
# MAGIC as a scheduled Databricks Job.

# COMMAND ----------

dbutils.widgets.text("watchlist_table_name", "watchlist", "Source table (watchlist symbols)")
dbutils.widgets.text("news_table_name", "ticker_news_documents", "Destination table (raw news)")
dbutils.widgets.text("embeddings_table_name", "ticker_news_embeddings", "Destination table (vectors)")
dbutils.widgets.text("chunk_embeddings_table_name", "ticker_news_chunk_embeddings", "Destination table (chunk vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("massive_secret_scope", "massive", "Massive API secret scope")
dbutils.widgets.text("massive_secret_key", "api-key", "Massive API secret key")
dbutils.widgets.text("massive_api_base_url", "https://api.massive.com", "Massive API base URL")
dbutils.widgets.text("news_fetch_limit", "50", "Max articles to fetch per ticker")
dbutils.widgets.text("max_requests_per_minute", "5", "Massive API rate limit (free tier is strict)")
dbutils.widgets.text("chunk_size", "800", "Article content chunk size (chars)")
dbutils.widgets.text("chunk_overlap", "100", "Article content chunk overlap (chars)")

WATCHLIST_TABLE_NAME = dbutils.widgets.get("watchlist_table_name")
NEWS_TABLE_NAME = dbutils.widgets.get("news_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")
CHUNK_EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("chunk_embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
MASSIVE_SECRET_SCOPE = dbutils.widgets.get("massive_secret_scope")
MASSIVE_SECRET_KEY = dbutils.widgets.get("massive_secret_key")
MASSIVE_API_BASE_URL = dbutils.widgets.get("massive_api_base_url")
NEWS_FETCH_LIMIT = int(dbutils.widgets.get("news_fetch_limit"))
MAX_REQUESTS_PER_MINUTE = int(dbutils.widgets.get("max_requests_per_minute"))
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))

# Different sentence-transformers models emit different vector sizes, and the
# pgvector column type (VECTOR(N)) must match exactly. Rather than hardcoding
# one dimension, switch on the model name so swapping EMBEDDING_MODEL_NAME via
# the widget above automatically resizes the destination table's vector column.
match EMBEDDING_MODEL_NAME:
    case "sentence-transformers/all-MiniLM-L6-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-MiniLM-L12-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "sentence-transformers/paraphrase-multilingual-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "BAAI/bge-small-en-v1.5":
        EMBEDDING_DIM = 384
    case "BAAI/bge-base-en-v1.5":
        EMBEDDING_DIM = 768
    case "BAAI/bge-large-en-v1.5":
        EMBEDDING_DIM = 1024
    case "text-embedding-3-small":
        EMBEDDING_DIM = 1536
    case "text-embedding-3-large":
        EMBEDDING_DIM = 3072
    case _:
        raise ValueError(
            f"Unknown embedding model {EMBEDDING_MODEL_NAME!r} - add its output "
            "dimension to the match/case block above before running this notebook."
        )

print(f"Using model {EMBEDDING_MODEL_NAME!r} -> {EMBEDDING_DIM}-dim vectors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the Lakebase connection URL
# MAGIC
# MAGIC Same secret, same decoding scheme as `lakebase.py`: a single base64-encoded
# MAGIC Postgres URL (`postgresql://role:password@host:5432/db?sslmode=require`)
# MAGIC stored in a Databricks secret scope. We parse it into the pieces psycopg3
# MAGIC needs for connection (host/port/dbname/user/password).

# COMMAND ----------

# DBTITLE 1,Parse Lakebase Connection Info
import base64
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()


def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope="database", key="lakebase-url")
    return base64.b64decode(secret.value).decode("utf-8")


lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

# Extract connection details directly from the secret URL
db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip('/')
db_user = parsed.username
db_password = parsed.password

print(f"Connection details:")
print(f"  Host: {db_host}:{db_port}")
print(f"  Database: {db_name}")
print(f"  User: {db_user}")
print(f"  Using raw credentials from secret (no OAuth)")

# COMMAND ----------

# DBTITLE 1,Test psycopg2 connection (weather-db)
import psycopg2

# The secret URL in cell 7 points to the old "new-database" project whose
# branch is archived and endpoint disabled. Override the connection variables
# to use the active "weather-db" project with short-lived OAuth credentials.
db_host = "ep-shiny-rice-d8wqgz3x.database.us-east-2.cloud.databricks.com"
db_port = 5432
db_name = "databricks_postgres"

from databricks.sdk import WorkspaceClient as _W
_w = _W()
_cred = _w.postgres.generate_database_credential(
    endpoint="projects/weather-db/branches/production/endpoints/primary"
)
db_user = "hemanthbigdataengineer@gmail.com"
db_password = _cred.token

print(f"Testing connection to {db_host}:{db_port}/{db_name}")
print(f"Using OAuth token authentication as user: {db_user}\n")

try:
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require',
        connect_timeout=15
    )
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {WATCHLIST_TABLE_NAME}")
    count = cursor.fetchone()[0]
    print(f"✅ Connection successful! Found {count} rows in {WATCHLIST_TABLE_NAME}")
    
    cursor.execute(f"SELECT * FROM {WATCHLIST_TABLE_NAME} LIMIT 5")
    rows = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    print(f"\nColumns: {colnames}")
    for row in rows:
        print(row)
    
    cursor.close()
    conn.close()
    print("\n✅ psycopg2 with OAuth authentication working correctly!")
except Exception as e:
    import traceback
    print(f"❌ Connection failed: {e}")
    print(f"\nFull traceback:")
    traceback.print_exc()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Database Setup Instructions
# MAGIC
# MAGIC Before running this notebook, you must manually create the required tables
# MAGIC in your Lakebase Postgres database:
# MAGIC
# MAGIC 1. Run `sql/01_setup_news_table.sql` to create `ticker_news_documents`
# MAGIC 2. Run `sql/02_setup_embeddings_table.sql` to create `ticker_news_embeddings`
# MAGIC    - Replace `{{EMBEDDING_DIM}}` with your model's dimension (e.g., 384)
# MAGIC 3. Run `sql/03_setup_chunk_embeddings_table.sql` to create `ticker_news_chunk_embeddings`
# MAGIC    - Replace `{{EMBEDDING_DIM}}` with your model's dimension (e.g., 384)
# MAGIC
# MAGIC This notebook uses psycopg2 with OAuth token authentication for all database operations.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch news from Massive for watchlisted tickers
# MAGIC
# MAGIC This ETL is now self-contained: instead of relying on the Flask app's
# MAGIC `POST /news/sync` route to have populated `ticker_news_documents` ahead of
# MAGIC time, the notebook queries the `watchlist` table in Lakebase directly to
# MAGIC find out which tickers are being tracked, then pulls news for exactly
# MAGIC those tickers from Massive itself.
# MAGIC
# MAGIC The free Massive API tier is rate-limited very aggressively, so requests
# MAGIC are made **serially** (not distributed across Spark workers) with a sleep
# MAGIC between calls that enforces `MAX_REQUESTS_PER_MINUTE` (default 5/min).

# COMMAND ----------

# DBTITLE 1,Fetch news and sync using Lakebase SDK
import hashlib
import json as _json
import time

import psycopg2
import requests

NWS_BASE_URL = "https://api.weather.gov"
_NWS_SESSION = requests.Session()
_NWS_SESSION.headers.update({
    "User-Agent": "databricks-lakebase-app/1.0 (contact@example.com)",
    "Accept": "application/geo+json",
})


def get_watchlist_locations() -> list[dict]:
    """Distinct locations currently tracked across all users in the
    locationlist table — these are the coordinates we fetch weather for."""
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require'
    )
    try:
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT DISTINCT latitude, longitude, city, state
            FROM {WATCHLIST_TABLE_NAME}
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """)
        rows = cursor.fetchall()
        return [
            {"latitude": float(r[0]), "longitude": float(r[1]), "city": r[2], "state": r[3]}
            for r in rows
        ]
    finally:
        cursor.close()
        conn.close()


def fetch_weather_for_location(lat: float, lon: float) -> list[dict]:
    """Fetch active alerts and forecast for a location from the NWS API
    (api.weather.gov) and normalize into the weather_documents schema."""
    docs = []

    # --- Alerts ---
    try:
        resp = _NWS_SESSION.get(
            f"{NWS_BASE_URL}/alerts/active",
            params={"point": f"{lat},{lon}"},
            timeout=30,
        )
        resp.raise_for_status()
        alerts = resp.json().get("features", [])
    except Exception as exc:
        print(f"    Could not fetch alerts: {exc}")
        alerts = []

    for alert in alerts:
        props = alert.get("properties", {})
        alert_id = alert.get("id") or props.get("id")
        if not alert_id:
            continue
        description = props.get("description") or ""
        instruction = props.get("instruction") or ""
        narrative = (description + "\n\n" + instruction).strip() if instruction else description
        docs.append({
            "id": alert_id,
            "source_type": "alert",
            "headline": props.get("event"),
            "narrative_text": narrative or None,
            "issued_at": props.get("sent"),
            "effective_at": props.get("effective"),
            "payload": _json.dumps(alert),
        })

    # --- Forecast ---
    try:
        point_resp = _NWS_SESSION.get(f"{NWS_BASE_URL}/points/{lat},{lon}", timeout=30)
        point_resp.raise_for_status()
        forecast_url = point_resp.json().get("properties", {}).get("forecast", "")

        forecast = None
        if forecast_url:
            fc_resp = _NWS_SESSION.get(forecast_url, timeout=30)
            fc_resp.raise_for_status()
            forecast = fc_resp.json()
    except Exception as exc:
        print(f"    Could not fetch forecast: {exc}")
        forecast = None

    if forecast:
        generated_at = forecast.get("properties", {}).get("generatedAt")
        for period in forecast.get("properties", {}).get("periods", []):
            start_time = period.get("startTime")
            if not start_time:
                continue
            raw = f"{lat},{lon}_{start_time}"
            doc_id = "fc_" + hashlib.md5(raw.encode()).hexdigest()
            docs.append({
                "id": doc_id,
                "source_type": "forecast",
                "headline": period.get("name"),
                "narrative_text": period.get("detailedForecast"),
                "issued_at": generated_at,
                "effective_at": start_time,
                "payload": _json.dumps(period),
            })

    return docs


print("NOTE: Before running this cell, ensure you've run sql/05_setup_weather_documents.sql")
print("      to create the weather_documents table in your Lakebase database.\n")

locations = get_watchlist_locations()
print(f"Found {len(locations)} distinct watchlisted locations")

# NWS API is free and open — no rate limit, but be respectful with 1s between calls.
_seconds_between_requests = 1.0

all_news_rows = []  # Reused variable name for downstream cell compatibility
for i, loc in enumerate(locations):
    if i > 0:
        time.sleep(_seconds_between_requests)

    lat, lon = loc["latitude"], loc["longitude"]
    city, state = loc["city"], loc["state"]
    location_str = f"{city}, {state}" if city and state else f"{lat},{lon}"

    try:
        docs = fetch_weather_for_location(lat, lon)
        for doc in docs:
            doc["location"] = location_str
        all_news_rows.extend(docs)
        print(f"  ({location_str}) → {len(docs)} weather documents")
    except Exception as exc:
        print(f"  Skipping ({location_str}): {exc}")
        continue

print(f"\nCollected {len(all_news_rows)} weather documents to insert.")
print(f"Run the next cell to insert them into {NEWS_TABLE_NAME}.")

# COMMAND ----------

# DBTITLE 1,Insert weather documents using psycopg2
import psycopg2

# Build connection using psycopg2
conn = psycopg2.connect(
    host=db_host,
    port=db_port,
    dbname=db_name,
    user=db_user,
    password=db_password,
    sslmode='require'
)

try:
    cursor = conn.cursor()
    
    # Prepare data tuples for batch insert into weather_documents
    insert_data = [
        (
            row['id'],
            row['location'],
            row['source_type'],
            row['headline'],
            row['narrative_text'],
            row['issued_at'],
            row['effective_at'],
            row['payload']
        )
        for row in all_news_rows
    ]
    
    if not insert_data:
        print("No weather documents to insert (locationlist may be empty).")
    else:
        # Batch insert with ON CONFLICT DO UPDATE to refresh data on re-fetch
        insert_sql = f"""
            INSERT INTO {NEWS_TABLE_NAME} (
                id, location, source_type, headline, narrative_text,
                issued_at, effective_at, payload, synced_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (id) DO UPDATE
                SET headline = EXCLUDED.headline,
                    narrative_text = EXCLUDED.narrative_text,
                    issued_at = EXCLUDED.issued_at,
                    effective_at = EXCLUDED.effective_at,
                    payload = EXCLUDED.payload,
                    synced_at = EXCLUDED.synced_at
        """
        
        # executemany is much faster than individual INSERTs
        cursor.executemany(insert_sql, insert_data)
        
        conn.commit()
        inserted_count = cursor.rowcount
        print(f"✅ Successfully upserted {len(insert_data)} weather documents into {NEWS_TABLE_NAME}")
        print(f"   (Existing documents were updated via ON CONFLICT DO UPDATE)")
    
finally:
    cursor.close()
    conn.close()

print(f"\nReady to compute embeddings! Run the cells below to continue.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load raw news documents
# MAGIC
# MAGIC Reads the whole `ticker_news_documents` table (just synced from Massive
# MAGIC above) using psycopg3 into a pandas DataFrame for embedding computation.

# COMMAND ----------

# DBTITLE 1,Load weather documents from Lakebase
import pandas as pd
import psycopg2

# Load weather documents using psycopg2
conn = psycopg2.connect(
    host=db_host,
    port=db_port,
    dbname=db_name,
    user=db_user,
    password=db_password,
    sslmode='require'
)

try:
    # Query with embedding_text computed from headline + narrative_text
    query = f"""
        SELECT 
            id,
            location,
            source_type,
            headline,
            narrative_text,
            effective_at,
            TRIM(CONCAT(COALESCE(headline, ''), '. ', COALESCE(narrative_text, ''))) AS embedding_text
        FROM {NEWS_TABLE_NAME}
        WHERE TRIM(CONCAT(COALESCE(headline, ''), '. ', COALESCE(narrative_text, ''))) IS NOT NULL
          AND TRIM(CONCAT(COALESCE(headline, ''), '. ', COALESCE(narrative_text, ''))) != ''
    """
    
    news_df = pd.read_sql_query(query, conn)
    print(f"Loaded {len(news_df)} weather documents from {NEWS_TABLE_NAME}")
    if len(news_df) > 0:
        display(news_df.head(5))
    else:
        print("(Table is empty — add locations via the app's POST /locationlist endpoint first.)")
finally:
    conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute embeddings
# MAGIC
# MAGIC Loads the sentence-transformers model once and applies it in batches
# MAGIC to the news documents.

# COMMAND ----------

# DBTITLE 1,Compute embeddings (distributed pandas UDF)
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# Set up HuggingFace cache
os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

# Compute embeddings in batches for memory efficiency
print("Computing embeddings...")
batch_size = 32
all_embeddings = []

for i in range(0, len(news_df), batch_size):
    batch = news_df.iloc[i:i+batch_size]
    vectors = model.encode(batch["embedding_text"].tolist(), show_progress_bar=False)
    all_embeddings.extend(vectors.tolist())
    if (i + batch_size) % 128 == 0:
        print(f"  Processed {min(i + batch_size, len(news_df))}/{len(news_df)} documents")

# Compute embeddings DataFrame for weather documents
embeddings_df = pd.DataFrame({
    "id": news_df["id"],
    "location": news_df["location"],
    "source_type": news_df["source_type"],
    "headline": news_df["headline"],
    "effective_at": news_df["effective_at"].astype(str),
    "embedding": all_embeddings,
})

print(f"Computed {len(embeddings_df)} embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure the pgvector destination table exists
# MAGIC
# MAGIC The `pgvector` extension must be enabled and the destination table
# MAGIC created with the correct vector dimension before inserting embeddings.

# COMMAND ----------

# Before running the cells below, ensure you've manually run:
#   sql/02_setup_embeddings_table.sql
# Replace {{EMBEDDING_DIM}} in that file with the value below:
print(f"Required EMBEDDING_DIM for SQL setup: {EMBEDDING_DIM}")
print(f"Table name: {EMBEDDINGS_TABLE_NAME}")
print("\nRun sql/02_setup_embeddings_table.sql in your Lakebase database before continuing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert embeddings into Lakebase
# MAGIC
# MAGIC Written in batches via psycopg2's `executemany` for throughput.
# MAGIC Each embedding is cast to Postgres' `vector` type via `::vector`.

# COMMAND ----------

# DBTITLE 1,Insert embeddings using psycopg2
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime

# Add model_name and embedded_at columns
embeddings_df['model_name'] = EMBEDDING_MODEL_NAME
embeddings_df['embedded_at'] = datetime.now()

embeddings_rows = embeddings_df.to_dict('records')

if len(embeddings_rows) > 0:
    print(f"Inserting {len(embeddings_rows)} embeddings into {EMBEDDINGS_TABLE_NAME}...")
    
    # Build connection from parsed URL
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require'
    )
    
    try:
        cursor = conn.cursor()
        
        # Prepare data tuples for batch insert
        # Format embedding as PostgreSQL array literal: '{val1,val2,...}'
        insert_data = [
            (
                row['id'],
                row['ticker'],
                row['title'],
                str(row['published_utc']) if row['published_utc'] else None,
                '{' + ','.join(str(float(x)) for x in row['embedding']) + '}',
                row['model_name'],
                row['embedded_at']
            )
            for row in embeddings_rows
        ]
        
        # Batch insert with ON CONFLICT DO NOTHING for deduplication
        insert_sql = f"""
            INSERT INTO {EMBEDDINGS_TABLE_NAME} (
                id, ticker, title, published_utc, embedding, model_name, embedded_at
            ) VALUES %s
            ON CONFLICT (id) DO NOTHING
        """
        
        # execute_values is much faster than individual INSERTs
        template = "(%s, %s, %s, %s, %s::double precision[], %s, %s)"
        execute_values(cursor, insert_sql, insert_data, template=template, page_size=100)
        
        conn.commit()
        inserted_count = cursor.rowcount
        print(f"✅ Successfully inserted {inserted_count} new embeddings")
        print(f"   (Duplicates were skipped via ON CONFLICT DO NOTHING)")
        print("\nIMPORTANT: Run this SQL in your Lakebase database to cast arrays to vectors:")
        print(f"  UPDATE {EMBEDDINGS_TABLE_NAME} SET embedding = embedding::vector WHERE embedding IS NOT NULL;")
        
    finally:
        cursor.close()
        conn.close()
else:
    print("No embeddings to write.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch and chunk article content
# MAGIC
# MAGIC Title/description only gets you so far - the actual article body lives at
# MAGIC `article_url` on the publisher's site. This step fetches each URL, uses
# MAGIC `trafilatura` to extract just the article text (stripping nav/ads/related
# MAGIC links/etc.), and splits it into overlapping chunks so each chunk can be
# MAGIC embedded and retrieved independently. Any URL that fails to fetch/extract
# MAGIC (paywall, timeout, dead link) is skipped rather than failing the whole job.

# COMMAND ----------

import pandas as pd
import requests
import trafilatura

# Filter news_df for articles with URLs
content_df = news_df[news_df['article_url'].notna() & (news_df['article_url'] != '')].copy()

print(f"Fetching and chunking content from {len(content_df)} article URLs...")

# Fetch and chunk article content
out_article_ids, out_tickers, out_chunk_indexes, out_chunk_texts = [], [], [], []

for idx, row in content_df.iterrows():
    article_id = row['id']
    ticker = row['ticker']
    article_url = row['article_url']
    
    try:
        resp = requests.get(article_url, timeout=15)
        resp.raise_for_status()
        text = trafilatura.extract(resp.text)
    except Exception as e:
        # Dead link, paywall, timeout, etc. - skip this article's
        # content chunks rather than failing the whole job.
        continue

    if not text:
        continue

    # Split into overlapping chunks
    for chunk_index, start in enumerate(range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)):
        chunk_text = text[start : start + CHUNK_SIZE].strip()
        if not chunk_text:
            continue
        out_article_ids.append(article_id)
        out_tickers.append(ticker)
        out_chunk_indexes.append(str(chunk_index))
        out_chunk_texts.append(chunk_text)
        if start + CHUNK_SIZE >= len(text):
            break
    
    # Progress update every 10 articles
    if (idx + 1) % 10 == 0:
        print(f"  Processed {idx + 1}/{len(content_df)} articles")

chunks_df = pd.DataFrame({
    "article_id": out_article_ids,
    "ticker": out_tickers,
    "chunk_index": out_chunk_indexes,
    "chunk_text": out_chunk_texts,
})

print(f"Extracted {len(chunks_df)} content chunks from {len(content_df)} article URLs")
display(chunks_df.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute chunk embeddings
# MAGIC
# MAGIC Same approach as the title/description embeddings above - load the model
# MAGIC once and process in batches, but one vector per content chunk instead of
# MAGIC per article.

# COMMAND ----------

import os
import pandas as pd
from sentence_transformers import SentenceTransformer

# Model should already be loaded from earlier, but ensure cache is set
os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

print(f"Computing chunk embeddings using {EMBEDDING_MODEL_NAME}...")
# Reuse the model if already loaded, otherwise load it
if 'model' not in locals():
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

# Compute chunk embeddings in batches
batch_size = 32
all_chunk_embeddings = []

for i in range(0, len(chunks_df), batch_size):
    batch = chunks_df.iloc[i:i+batch_size]
    vectors = model.encode(batch["chunk_text"].tolist(), show_progress_bar=False)
    all_chunk_embeddings.extend(vectors.tolist())
    if (i + batch_size) % 128 == 0:
        print(f"  Processed {min(i + batch_size, len(chunks_df))}/{len(chunks_df)} chunks")

# Create chunk embeddings DataFrame
chunk_embeddings_df = pd.DataFrame({
    "article_id": chunks_df["article_id"],
    "ticker": chunks_df["ticker"],
    "chunk_index": chunks_df["chunk_index"],
    "chunk_text": chunks_df["chunk_text"],
    "embedding": all_chunk_embeddings,
})

print(f"Computed {len(chunk_embeddings_df)} chunk embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ensure the chunk embeddings destination table exists

# COMMAND ----------

# Before running the cells below, ensure you've manually run:
#   sql/03_setup_chunk_embeddings_table.sql
# Replace {{EMBEDDING_DIM}} in that file with the value below:
print(f"Required EMBEDDING_DIM for SQL setup: {EMBEDDING_DIM}")
print(f"Table name: {CHUNK_EMBEDDINGS_TABLE_NAME}")
print("\nRun sql/03_setup_chunk_embeddings_table.sql in your Lakebase database before continuing.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert chunk embeddings into Lakebase

# COMMAND ----------

# DBTITLE 1,Insert chunk embeddings using psycopg2
import psycopg2
from datetime import datetime

# Add id (article_id_chunk_index), model_name, and embedded_at columns
chunk_embeddings_df['id'] = chunk_embeddings_df['article_id'] + '_' + chunk_embeddings_df['chunk_index']
chunk_embeddings_df['model_name'] = EMBEDDING_MODEL_NAME
chunk_embeddings_df['embedded_at'] = datetime.now()
chunk_embeddings_df['chunk_index'] = chunk_embeddings_df['chunk_index'].astype(int)

chunk_embeddings_rows = chunk_embeddings_df.to_dict('records')

if len(chunk_embeddings_rows) > 0:
    print(f"Inserting {len(chunk_embeddings_rows)} chunk embeddings into {CHUNK_EMBEDDINGS_TABLE_NAME}...")
    
    # Build connection using psycopg2
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        sslmode='require'
    )
    
    try:
        cursor = conn.cursor()
        
        # Prepare data tuples for batch insert
        # Format embedding as PostgreSQL array literal: '{val1,val2,...}'
        insert_data = [
            (
                row['id'],
                row['article_id'],
                row['ticker'],
                int(row['chunk_index']),
                row['chunk_text'],
                '{' + ','.join(str(float(x)) for x in row['embedding']) + '}',
                row['model_name'],
                row['embedded_at']
            )
            for row in chunk_embeddings_rows
        ]
        
        # Batch insert with ON CONFLICT DO NOTHING for deduplication
        insert_sql = f"""
            INSERT INTO {CHUNK_EMBEDDINGS_TABLE_NAME} (
                id, article_id, ticker, chunk_index, chunk_text, embedding, model_name, embedded_at
            ) VALUES (%s, %s, %s, %s, %s, %s::double precision[], %s, %s)
            ON CONFLICT (id) DO NOTHING
        """
        
        # executemany in psycopg2 is much faster than individual INSERTs
        cursor.executemany(insert_sql, insert_data)
        
        conn.commit()
        inserted_count = cursor.rowcount
        print(f"✅ Successfully inserted {inserted_count} new chunk embeddings")
        print(f"   (Duplicates were skipped via ON CONFLICT DO NOTHING)")
        print("\nIMPORTANT: Run this SQL in your Lakebase database to cast arrays to vectors:")
        print(f"  UPDATE {CHUNK_EMBEDDINGS_TABLE_NAME} SET embedding = embedding::vector WHERE embedding IS NOT NULL;")
        
    finally:
        cursor.close()
        conn.close()
else:
    print("No chunk embeddings to write.")