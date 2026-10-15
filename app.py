"""
Databricks App boilerplate:
- Serves a small Flask API
- Reads/writes to Lakebase (Databricks-managed Postgres) via lakebase.py
- Pulls data from the Massive API via massive_client.py and syncs it into Lakebase

Run locally:
    python app.py
Deploy as a Databricks App using app.yaml.
"""

import logging
import os
import re

import requests
from databricks.sdk import WorkspaceClient
from flask import Flask, jsonify, render_template, request

import lakebase
from massive_client import MassiveClient
from weather_client import WeatherClient

# Lazy-load the embedding model on first search request
_embedding_model = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("massive-app")

app = Flask(__name__)
_w = WorkspaceClient()

TABLE_NAME = os.environ.get("MASSIVE_TABLE_NAME", "massive_records")
WATCHLIST_TABLE_NAME = os.environ.get("WATCHLIST_TABLE_NAME", "watchlist")
NEWS_TABLE_NAME = os.environ.get("NEWS_TABLE_NAME", "ticker_news_documents")
LOCATIONLIST_TABLE_NAME = os.environ.get("LOCATIONLIST_TABLE_NAME", "locationlist")
WEATHER_DOCUMENTS_TABLE_NAME = os.environ.get("WEATHER_DOCUMENTS_TABLE_NAME", "weather_documents")

# Tickers to fetch news for by default (comma-separated), e.g. "AAPL,MSFT,GOOGL"
DEFAULT_NEWS_TICKERS = [
    t.strip().upper()
    for t in os.environ.get("NEWS_TICKERS", "AAPL,MSFT,GOOGL,AMZN,TSLA").split(",")
    if t.strip()
]

# Basic stock ticker shape check: 1-10 uppercase letters, with an optional
# ".X" or ".XX" share-class suffix (e.g. "BRK.B"). This rejects obviously
# malformed input before we even call the Massive API.
_TICKER_RE = re.compile(r"^[A-Z]{1,10}(\.[A-Z]{1,2})?$")


def ensure_table():
    """Create the destination table in Lakebase if it doesn't exist yet."""
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id TEXT PRIMARY KEY,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )


def ensure_watchlist_table():
    """Create the watchlist table in Lakebase if it doesn't exist yet."""
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {WATCHLIST_TABLE_NAME} (
            symbol TEXT NOT NULL,
            email TEXT NOT NULL,
            latest_price NUMERIC,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (symbol, email)
        )
        """
    )


def ensure_locationlist_table():
    """Create the locationlist table in Lakebase if it doesn't exist yet."""
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {LOCATIONLIST_TABLE_NAME} (
            latitude DOUBLE PRECISION NOT NULL,
            longitude DOUBLE PRECISION NOT NULL,
            city TEXT,
            state TEXT,
            email TEXT NOT NULL,
            searched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (latitude, longitude, email)
        )
        """
    )


def ensure_weather_documents_table():
    """Create the weather_documents table in Lakebase if it doesn't exist yet."""
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {WEATHER_DOCUMENTS_TABLE_NAME} (
            id TEXT PRIMARY KEY,
            location TEXT NOT NULL,
            source_type TEXT NOT NULL,
            headline TEXT,
            narrative_text TEXT,
            issued_at TIMESTAMPTZ,
            effective_at TIMESTAMPTZ,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{WEATHER_DOCUMENTS_TABLE_NAME}_location "
        f"ON {WEATHER_DOCUMENTS_TABLE_NAME} (location)"
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{WEATHER_DOCUMENTS_TABLE_NAME}_source_type "
        f"ON {WEATHER_DOCUMENTS_TABLE_NAME} (source_type)"
    )


def ensure_news_table():
    """
    Create the raw ticker-news documents table in Lakebase if it doesn't
    exist yet. This is the RAW document store the Spark notebook
    (notebooks/ingest_ticker_news_embeddings.py) reads from to compute
    vector embeddings into a separate `<NEWS_TABLE_NAME>_embeddings` table.
    """
    lakebase.run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {NEWS_TABLE_NAME} (
            id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            author TEXT,
            article_url TEXT,
            publisher_name TEXT,
            keywords JSONB,
            sentiment TEXT,
            sentiment_reasoning TEXT,
            published_utc TIMESTAMPTZ,
            payload JSONB NOT NULL,
            synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    lakebase.run_write(
        f"CREATE INDEX IF NOT EXISTS idx_{NEWS_TABLE_NAME}_ticker "
        f"ON {NEWS_TABLE_NAME} (ticker)"
    )


def _current_user_email() -> str:
    """
    Resolve the current user's email so the watchlist can be personalized.

    Databricks Apps inject the logged-in user's identity via the
    X-Forwarded-Email header on every request. Fall back to the Databricks
    SDK's current_user API for local development where that header isn't set.
    """
    header_email = request.headers.get("X-Forwarded-Email")
    if header_email:
        return header_email
    return _w.current_user.me().user_name


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.errorhandler(Exception)
def handle_exception(err):
    """Ensure all unhandled errors return JSON (not an HTML error page),
    so the frontend's resp.json() call never chokes on HTML."""
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/")
def index():
    """Simple UI to submit a list of stock symbols to sync from Massive."""
    return render_template("index.html")


@app.route("/records")
def list_records():
    """Read records already synced into Lakebase."""
    limit = int(request.args.get("limit", 100))
    rows = lakebase.run_query(
        f"SELECT id, payload, synced_at FROM {TABLE_NAME} ORDER BY synced_at DESC LIMIT %s",
        (limit,),
    )
    return jsonify(rows)


@app.route("/sync", methods=["POST"])
def sync_from_massive():
    """
    Pull data from the Massive API (paginated, potentially huge dataset) and
    upsert it into Lakebase in batches.
    """
    ensure_table()
    client = MassiveClient()

    path = request.json.get("path", "/records") if request.is_json else "/records"
    batch_size = int(request.args.get("batch_size", 500))

    batch = []
    total = 0
    for item in client.paginated_get(path):
        batch.append(item)
        if len(batch) >= batch_size:
            total += _upsert_batch(batch)
            batch = []

    if batch:
        total += _upsert_batch(batch)

    return jsonify({"synced": total})


@app.route("/news/sync", methods=["POST"])
def sync_news_from_massive():
    """
    Pull recent news articles for a set of tickers from Massive (ONE API
    call per ticker, via MassiveClient.get_news) and upsert them into the
    ticker_news_documents table in Lakebase.

    Body (optional JSON): {"tickers": ["AAPL", "MSFT"], "limit": 50}
    Defaults to DEFAULT_NEWS_TICKERS when no tickers are supplied.
    """
    ensure_news_table()
    client = MassiveClient()

    body = request.json if request.is_json else {}
    tickers = body.get("tickers") or DEFAULT_NEWS_TICKERS
    tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]
    limit = int(body.get("limit", 50))

    total = 0
    for ticker in tickers:
        if not _TICKER_RE.match(ticker):
            continue
        articles = client.get_news(ticker, limit=limit)
        total += _upsert_news_batch(ticker, articles)

    return jsonify({"synced": total, "tickers": tickers})


@app.route("/watchlist", methods=["GET"])
def get_watchlist():
    """Return the current user's watchlist symbols, with their last known price."""
    ensure_watchlist_table()
    email = _current_user_email()
    rows = lakebase.run_query(
        f"SELECT symbol, email, latest_price, updated_at FROM {WATCHLIST_TABLE_NAME} "
        f"WHERE email = %s ORDER BY symbol ASC",
        (email,),
    )
    return jsonify(rows)


@app.route("/watchlist", methods=["POST"])
def add_to_watchlist():
    """
    Fetch the latest price for a single stock symbol from Massive using
    exactly ONE API call (see MassiveClient.get_latest_price), then add/
    update that symbol on the watchlist in Lakebase.
    """
    ensure_watchlist_table()

    if request.is_json:
        symbol = request.json.get("symbol", "")
    else:
        symbol = request.form.get("symbol", "")

    symbol = symbol.strip().upper() if isinstance(symbol, str) else ""

    if not symbol or not _TICKER_RE.match(symbol):
        return jsonify({"error": f"Invalid ticker symbol: {symbol!r}"}), 400

    client = MassiveClient()
    try:
        data = client.get_latest_price(symbol)  # <-- single API call, latest price only
    except requests.HTTPError:
        # Massive returns a 404/4xx for tickers it doesn't recognize.
        return jsonify({"error": f"Unknown ticker symbol: {symbol}"}), 400

    price = _extract_latest_price(data)
    if price is None:
        # No usable price in the response (e.g. delisted/invalid ticker
        # that still 200s with an empty result set) - don't add it.
        return jsonify({"error": f"No price data available for ticker: {symbol}"}), 400

    email = _current_user_email()

    lakebase.run_write(
        f"""
        INSERT INTO {WATCHLIST_TABLE_NAME} (symbol, email, latest_price, updated_at)
        VALUES (%s, %s, %s, now())
        ON CONFLICT (symbol, email) DO UPDATE
            SET latest_price = EXCLUDED.latest_price,
                updated_at = EXCLUDED.updated_at
        """,
        (symbol, email, price),
    )

    return jsonify({"symbol": symbol, "email": email, "latest_price": price})


@app.route("/watchlist/<symbol>", methods=["DELETE"])
def delete_from_watchlist(symbol: str):
    """Remove a single symbol from the current user's watchlist."""
    ensure_watchlist_table()

    symbol = symbol.strip().upper() if isinstance(symbol, str) else ""
    if not symbol or not _TICKER_RE.match(symbol):
        return jsonify({"error": f"Invalid ticker symbol: {symbol!r}"}), 400

    email = _current_user_email()
    deleted = lakebase.run_write(
        f"DELETE FROM {WATCHLIST_TABLE_NAME} WHERE symbol = %s AND email = %s",
        (symbol, email),
    )

    if not deleted:
        return jsonify({"error": f"{symbol} is not on your watchlist"}), 404

    return jsonify({"symbol": symbol, "email": email, "deleted": True})


@app.route("/locationlist", methods=["GET"])
def get_locationlist():
    """Return the current user's searched locations, with search timestamps."""
    ensure_locationlist_table()
    email = _current_user_email()
    rows = lakebase.run_query(
        f"SELECT latitude, longitude, city, state, email, searched_at FROM {LOCATIONLIST_TABLE_NAME} "
        f"WHERE email = %s ORDER BY searched_at DESC",
        (email,),
    )
    return jsonify(rows)


@app.route("/locationlist", methods=["POST"])
def add_to_locationlist():
    """
    Accept a latitude/longitude pair submitted by the user (for lookup via
    api.weather.gov / weather_client.py), then add/update that location on
    the locationlist in Lakebase with the searcher's email and timestamp.
    """
    ensure_locationlist_table()

    if request.is_json:
        latitude = request.json.get("latitude")
        longitude = request.json.get("longitude")
    else:
        latitude = request.form.get("latitude")
        longitude = request.form.get("longitude")

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError):
        return jsonify({"error": "latitude and longitude must be valid numbers"}), 400

    if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
        return jsonify({"error": "latitude must be in [-90, 90], longitude in [-180, 180]"}), 400

    # Resolve city/state from the NWS /points endpoint (same call used by
    # weather_client.py for forecast lookups). If the API is unavailable or
    # returns no relativeLocation, we still store the coordinates with null
    # city/state so the user's search is never lost.
    city = None
    state = None
    try:
        wc = WeatherClient()
        loc = wc.get_location_name(latitude, longitude)
        city = loc.get("city")
        state = loc.get("state")
    except Exception as exc:
        logger.warning("Could not resolve city/state for (%s, %s): %s", latitude, longitude, exc)

    email = _current_user_email()

    lakebase.run_write(
        f"""
        INSERT INTO {LOCATIONLIST_TABLE_NAME} (latitude, longitude, city, state, email, searched_at)
        VALUES (%s, %s, %s, %s, %s, now())
        ON CONFLICT (latitude, longitude, email) DO UPDATE
            SET city = EXCLUDED.city,
                state = EXCLUDED.state,
                searched_at = EXCLUDED.searched_at
        """,
        (latitude, longitude, city, state, email),
    )

    # Fetch alerts + forecast for this location and upsert into weather_documents
    weather_doc_count = _upsert_weather_documents(latitude, longitude, city, state)
    logger.info("Upserted %d weather documents for (%s, %s)", weather_doc_count, latitude, longitude)

    return jsonify({"latitude": latitude, "longitude": longitude, "city": city, "state": state, "email": email, "weather_documents": weather_doc_count})


@app.route("/locationlist", methods=["DELETE"])
def delete_from_locationlist():
    """Remove a single location from the current user's locationlist."""
    ensure_locationlist_table()

    try:
        latitude = float(request.args.get("latitude"))
        longitude = float(request.args.get("longitude"))
    except (TypeError, ValueError):
        return jsonify({"error": "latitude and longitude query params are required"}), 400

    email = _current_user_email()
    deleted = lakebase.run_write(
        f"DELETE FROM {LOCATIONLIST_TABLE_NAME} WHERE latitude = %s AND longitude = %s AND email = %s",
        (latitude, longitude, email),
    )

    if not deleted:
        return jsonify({"error": f"Location ({latitude}, {longitude}) is not on your locationlist"}), 404

    return jsonify({"latitude": latitude, "longitude": longitude, "email": email, "deleted": True})


def _upsert_weather_documents(latitude: float, longitude: float, city: str | None, state: str | None) -> int:
    """
    Fetch active alerts and the forecast for a location from the NWS API,
    normalize them into a common schema, and upsert into the
    weather_documents table in Lakebase.

    Returns the number of documents upserted.
    """
    import hashlib
    import json as _json

    ensure_weather_documents_table()
    location_str = f"{city}, {state}" if city and state else f"{latitude},{longitude}"
    wc = WeatherClient()

    docs: list[dict] = []

    # --- Alerts ---
    try:
        alerts = wc.get_active_alerts(latitude, longitude)
    except Exception as exc:
        logger.warning("Could not fetch alerts for (%s, %s): %s", latitude, longitude, exc)
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
            "location": location_str,
            "source_type": "alert",
            "headline": props.get("event"),
            "narrative_text": narrative or None,
            "issued_at": props.get("sent"),
            "effective_at": props.get("effective"),
            "payload": _json.dumps(alert),
        })

    # --- Forecast periods ---
    try:
        point_info = wc.get_point_info(latitude, longitude, include_forecast=True)
        forecast = point_info.get("properties", {}).get("_forecast")
    except Exception as exc:
        logger.warning("Could not fetch forecast for (%s, %s): %s", latitude, longitude, exc)
        forecast = None

    if forecast:
        generated_at = forecast.get("properties", {}).get("generatedAt")
        for period in forecast.get("properties", {}).get("periods", []):
            start_time = period.get("startTime")
            if not start_time:
                continue
            raw = f"{latitude},{longitude}_{start_time}"
            doc_id = "fc_" + hashlib.md5(raw.encode()).hexdigest()
            docs.append({
                "id": doc_id,
                "location": location_str,
                "source_type": "forecast",
                "headline": period.get("name"),
                "narrative_text": period.get("detailedForecast"),
                "issued_at": generated_at,
                "effective_at": start_time,
                "payload": _json.dumps(period),
            })

    # --- Upsert ---
    count = 0
    if docs:
        with lakebase.get_connection() as conn:
            with conn.cursor() as cur:
                for doc in docs:
                    cur.execute(
                        f"""
                        INSERT INTO {WEATHER_DOCUMENTS_TABLE_NAME} (
                            id, location, source_type, headline, narrative_text,
                            issued_at, effective_at, payload, synced_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                        ON CONFLICT (id) DO UPDATE
                            SET headline = EXCLUDED.headline,
                                narrative_text = EXCLUDED.narrative_text,
                                issued_at = EXCLUDED.issued_at,
                                effective_at = EXCLUDED.effective_at,
                                payload = EXCLUDED.payload,
                                synced_at = EXCLUDED.synced_at
                        """,
                        (
                            doc["id"],
                            doc["location"],
                            doc["source_type"],
                            doc["headline"],
                            doc["narrative_text"],
                            doc["issued_at"],
                            doc["effective_at"],
                            doc["payload"],
                        ),
                    )
                    count += 1
            conn.commit()

    return count


def _extract_latest_price(data: dict) -> float | None:
    """Pull the trade price out of the Massive 'previous close' response shape.

    The /v2/aggs/ticker/{symbol}/prev endpoint returns "results" as a LIST
    containing a single aggregate bar (not a dict), e.g.:
        {"status": "OK", "resultsCount": 1, "results": [{"c": 148.845, ...}]}
    Previously this code treated "results" as a dict, so isinstance(results, dict)
    was always False for this endpoint's real shape and the price silently
    resolved to None. Unwrap the list here, and check "status"/"resultsCount"
    so invalid tickers (empty results) are detected instead of "succeeding"
    with a null price.

    Adjust the key lookup here if the real Massive API returns a different
    field name for the traded/close price.
    """
    if not isinstance(data, dict):
        return None
    if data.get("status") not in (None, "OK") or data.get("resultsCount") == 0:
        return None
    results = data.get("results", data)
    if isinstance(results, list):
        results = results[0] if results else None
    if isinstance(results, dict):
        for key in ("c", "p", "price", "last_price", "vw"):
            if key in results:
                return results[key]
    return None


def _upsert_batch(items: list[dict]) -> int:
    """Upsert a batch of Massive API items into Lakebase, one statement per row.

    For very large batches, consider psycopg2.extras.execute_values for
    higher throughput instead of per-row execute calls.
    """
    import json as _json

    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for item in items:
                cur.execute(
                    f"""
                    INSERT INTO {TABLE_NAME} (id, payload, synced_at)
                    VALUES (%s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (str(item.get("id")), _json.dumps(item)),
                )
                count += 1
            conn.commit()
    return count


def _upsert_news_batch(ticker: str, articles: list[dict]) -> int:
    """Upsert news articles for a single ticker into the news documents table.

    Flattens the top-level "insights" sentiment entry that matches this
    ticker (if present) into its own columns so the Spark notebook can read
    plain text columns instead of parsing JSONB for the common case.
    """
    import json as _json

    count = 0
    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            for article in articles:
                sentiment = None
                sentiment_reasoning = None
                for insight in article.get("insights", []) or []:
                    if insight.get("ticker") == ticker:
                        sentiment = insight.get("sentiment")
                        sentiment_reasoning = insight.get("sentiment_reasoning")
                        break

                publisher = article.get("publisher") or {}
                cur.execute(
                    f"""
                    INSERT INTO {NEWS_TABLE_NAME} (
                        id, ticker, title, description, author, article_url,
                        publisher_name, keywords, sentiment, sentiment_reasoning,
                        published_utc, payload, synced_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                    ON CONFLICT (id) DO UPDATE
                        SET ticker = EXCLUDED.ticker,
                            title = EXCLUDED.title,
                            description = EXCLUDED.description,
                            author = EXCLUDED.author,
                            article_url = EXCLUDED.article_url,
                            publisher_name = EXCLUDED.publisher_name,
                            keywords = EXCLUDED.keywords,
                            sentiment = EXCLUDED.sentiment,
                            sentiment_reasoning = EXCLUDED.sentiment_reasoning,
                            published_utc = EXCLUDED.published_utc,
                            payload = EXCLUDED.payload,
                            synced_at = EXCLUDED.synced_at
                    """,
                    (
                        str(article.get("id")),
                        ticker,
                        article.get("title", ""),
                        article.get("description"),
                        article.get("author"),
                        article.get("article_url"),
                        publisher.get("name"),
                        _json.dumps(article.get("keywords", [])),
                        sentiment,
                        sentiment_reasoning,
                        article.get("published_utc"),
                        _json.dumps(article),
                    ),
                )
                count += 1
            conn.commit()
    return count


if __name__ == '__main__':
    host = os.getenv('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_RUN_PORT', 8000))
    app.run(debug=True, host=host, port=port)
    print(f"Flask app running on http://{host}:{port}")