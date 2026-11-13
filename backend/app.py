
import os
import threading
import logging
import random
import time

from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH, override=True)

from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_socketio import SocketIO

from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed


# ==================================================
# CONFIGURATION
# ==================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATUSES = (
    "Not Started",
    "In Progress",
    "Completed",
    "Blocked"
)

MARKET_SYMBOLS = [
    symbol.strip().upper()
    for symbol in os.getenv(
        "ALPACA_SYMBOLS",
        "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,AMD"
    ).split(",")
    if symbol.strip()
]

MAX_RECENT_TRADES = 500

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "tasks.db"
    )
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

CORS(app)

db = SQLAlchemy(app)

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading"
)


# ==================================================
# TASK MANAGER DATABASE
# ==================================================

class Task(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    title = db.Column(
        db.String(200),
        nullable=False
    )

    description = db.Column(
        db.Text,
        default=""
    )

    status = db.Column(
        db.String(30),
        default="Not Started"
    )

    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self):

        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status,
            "created_at": (
                self.created_at.isoformat()
                if self.created_at else None
            )
        }


# ==================================================
# TASK MANAGER ROUTES
# ==================================================

@app.route("/api/items", methods=["GET"])
def get_items():

    tasks = Task.query.order_by(
        Task.id.desc()
    ).all()

    return jsonify([
        task.to_dict()
        for task in tasks
    ])


@app.route("/api/items", methods=["POST"])
def create_item():

    data = request.get_json(silent=True) or {}

    title = data.get("title", "").strip()

    if not title:
        return jsonify({
            "error": "Title is required"
        }), 400

    status = data.get(
        "status",
        "Not Started"
    )

    if status not in STATUSES:
        return jsonify({
            "error": "Invalid status"
        }), 400

    task = Task(
        title=title,
        description=data.get("description", ""),
        status=status
    )

    db.session.add(task)
    db.session.commit()

    return jsonify(task.to_dict()), 201


@app.route("/api/items/<int:item_id>", methods=["PUT"])
def update_item(item_id):

    task = db.session.get(Task, item_id)

    if not task:
        return jsonify({
            "error": "Task not found"
        }), 404

    data = request.get_json(silent=True) or {}

    if "title" in data:

        title = data["title"].strip()

        if not title:
            return jsonify({
                "error": "Title cannot be empty"
            }), 400

        task.title = title

    if "description" in data:
        task.description = data["description"]

    if "status" in data:

        if data["status"] not in STATUSES:
            return jsonify({
                "error": "Invalid status"
            }), 400

        task.status = data["status"]

    db.session.commit()

    return jsonify(task.to_dict())


@app.route("/api/items/<int:item_id>", methods=["DELETE"])
def delete_item(item_id):

    task = db.session.get(Task, item_id)

    if not task:
        return jsonify({
            "error": "Task not found"
        }), 404

    db.session.delete(task)
    db.session.commit()

    return jsonify({
        "message": "Task deleted"
    })


# ==================================================
# ALPACA MARKET DATA
# ==================================================

ALPACA_API_KEY = os.getenv(
    "ALPACA_API_KEY",
    ""
).strip()

ALPACA_API_SECRET = os.getenv(
    "ALPACA_API_SECRET",
    ""
).strip()

ALPACA_DATA_FEED = os.getenv(
    "ALPACA_DATA_FEED",
    "iex"
).strip().lower()

logger.info(
    "Alpaca API key loaded: %s",
    bool(ALPACA_API_KEY)
)

logger.info(
    "Alpaca secret loaded: %s",
    bool(ALPACA_API_SECRET)
)

logger.info(
    "Alpaca feed: %s",
    ALPACA_DATA_FEED
)

# Real trades and simulated trades are kept separate.
_recent_trades = deque(maxlen=MAX_RECENT_TRADES)
_simulated_trades = deque(maxlen=MAX_RECENT_TRADES)

_last_trade_by_symbol = {}

_market_lock = threading.Lock()
_stream_lock = threading.Lock()
_simulation_lock = threading.Lock()

_alpaca_stream = None
_stream_started = False
_market_error = None

_simulation_enabled = False
_simulation_thread = None
_simulation_stop = threading.Event()


# ==================================================
# COMPANY HEADQUARTERS
# ==================================================

SIMULATION_COMPANIES = {

    "AAPL": {
        "company": "Apple",
        "sector": "Technology",
        "price": 270,
        "city": "Cupertino",
        "country": "USA",
        "latitude": 37.3349,
        "longitude": -122.0090
    },

    "MSFT": {
        "company": "Microsoft",
        "sector": "Technology",
        "price": 510,
        "city": "Redmond",
        "country": "USA",
        "latitude": 47.6740,
        "longitude": -122.1215
    },

    "NVDA": {
        "company": "NVIDIA",
        "sector": "Technology",
        "price": 190,
        "city": "Santa Clara",
        "country": "USA",
        "latitude": 37.3708,
        "longitude": -121.9630
    },

    "AMZN": {
        "company": "Amazon",
        "sector": "Consumer",
        "price": 245,
        "city": "Seattle",
        "country": "USA",
        "latitude": 47.6225,
        "longitude": -122.3363
    },

    "GOOGL": {
        "company": "Alphabet",
        "sector": "Technology",
        "price": 300,
        "city": "Mountain View",
        "country": "USA",
        "latitude": 37.4220,
        "longitude": -122.0841
    },

    "META": {
        "company": "Meta",
        "sector": "Technology",
        "price": 650,
        "city": "Menlo Park",
        "country": "USA",
        "latitude": 37.4848,
        "longitude": -122.1484
    },

    "TSLA": {
        "company": "Tesla",
        "sector": "Automotive",
        "price": 345,
        "city": "Austin",
        "country": "USA",
        "latitude": 30.2672,
        "longitude": -97.7431
    },

    "AMD": {
        "company": "AMD",
        "sector": "Technology",
        "price": 220,
        "city": "Santa Clara",
        "country": "USA",
        "latitude": 37.3875,
        "longitude": -121.9636
    }
}


# ==================================================
# SIMULATED BUYER LOCATIONS
# ==================================================

SIMULATION_CITIES = [

    {
        "city": "London",
        "country": "UK",
        "latitude": 51.5072,
        "longitude": -0.1276
    },

    {
        "city": "New York",
        "country": "USA",
        "latitude": 40.7128,
        "longitude": -74.0060
    },

    {
        "city": "Tokyo",
        "country": "Japan",
        "latitude": 35.6762,
        "longitude": 139.6503
    },

    {
        "city": "Singapore",
        "country": "Singapore",
        "latitude": 1.3521,
        "longitude": 103.8198
    },

    {
        "city": "Hong Kong",
        "country": "China",
        "latitude": 22.3193,
        "longitude": 114.1694
    },

    {
        "city": "Paris",
        "country": "France",
        "latitude": 48.8566,
        "longitude": 2.3522
    },

    {
        "city": "Frankfurt",
        "country": "Germany",
        "latitude": 50.1109,
        "longitude": 8.6821
    },

    {
        "city": "Sydney",
        "country": "Australia",
        "latitude": -33.8688,
        "longitude": 151.2093
    },

    {
        "city": "Dubai",
        "country": "UAE",
        "latitude": 25.2048,
        "longitude": 55.2708
    },

    {
        "city": "Toronto",
        "country": "Canada",
        "latitude": 43.6532,
        "longitude": -79.3832
    },

    {
        "city": "Seoul",
        "country": "South Korea",
        "latitude": 37.5665,
        "longitude": 126.9780
    },

    {
        "city": "Mumbai",
        "country": "India",
        "latitude": 19.0760,
        "longitude": 72.8777
    },

    {
        "city": "Zurich",
        "country": "Switzerland",
        "latitude": 47.3769,
        "longitude": 8.5417
    },

    {
        "city": "Amsterdam",
        "country": "Netherlands",
        "latitude": 52.3676,
        "longitude": 4.9041
    },

    {
        "city": "San Francisco",
        "country": "USA",
        "latitude": 37.7749,
        "longitude": -122.4194
    }
]


# ==================================================
# SIMULATION ENGINE
# ==================================================

def generate_simulated_trade():

    symbol = random.choice(
        list(SIMULATION_COMPANIES.keys())
    )

    company = SIMULATION_COMPANIES[symbol]

    buyer = random.choice(SIMULATION_CITIES)

    # Simulated share price variation.
    price = round(
        company["price"] * random.uniform(0.985, 1.015),
        2
    )

    shares = random.randint(1, 1500)

    notional = round(
        price * shares,
        2
    )

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    trade_id = (
        f"fake-{int(time.time() * 1000)}-"
        f"{random.randint(1000, 9999)}"
    )

    return {

        "id": trade_id,

        "symbol": symbol,
        "company": company["company"],
        "sector": company["sector"],

        "price": price,
        "size": shares,
        "shares": shares,
        "notional": notional,

        "exchange": "FAKEPACA",
        "timestamp": timestamp,

        "simulated": True,
        "source": "FAKEPACA",

        "side": random.choice([
            "buy",
            "sell"
        ]),

        # Company headquarters: route origin.
        "origin": {
            "city": company["city"],
            "country": company["country"],
            "latitude": company["latitude"],
            "longitude": company["longitude"]
        },

        # Simulated buyer/seller: route destination.
        "destination": {
            "city": buyer["city"],
            "country": buyer["country"],
            "latitude": buyer["latitude"],
            "longitude": buyer["longitude"]
        },

        "buyer_location": (
            f"{buyer['city']}, {buyer['country']}"
        ),

        "seller_location": (
            f"{company['city']}, {company['country']}"
        )
    }


def simulation_worker():

    logger.info("FAKEPACA simulation started")

    while not _simulation_stop.is_set():

        with _simulation_lock:
            enabled = _simulation_enabled

        if not enabled:
            break

        try:

            trade = generate_simulated_trade()

            with _market_lock:

                _simulated_trades.appendleft(trade)

            socketio.emit(
                "market_trade",
                trade,
                namespace="/market"
            )

            logger.info(
                "SIMULATED TRADE: %s | %s shares | $%s | %s -> %s",
                trade["symbol"],
                trade["shares"],
                trade["notional"],
                trade["seller_location"],
                trade["buyer_location"]
            )

        except Exception:

            logger.exception(
                "Simulation trade generation failed"
            )

        # Generate a new transaction every 1–3 seconds.
        _simulation_stop.wait(
            random.uniform(1.0, 3.0)
        )

    logger.info("FAKEPACA simulation stopped")


def set_simulation_enabled(enabled):

    global _simulation_enabled
    global _simulation_thread

    with _simulation_lock:

        if enabled == _simulation_enabled:
            return

        _simulation_enabled = enabled

        if enabled:

            _simulation_stop.clear()

            _simulation_thread = threading.Thread(
                target=simulation_worker,
                daemon=True
            )

            _simulation_thread.start()

        else:

            _simulation_stop.set()

    socketio.emit(
        "simulation_status",
        {
            "enabled": enabled,
            "mode": "FAKEPACA" if enabled else "REAL"
        },
        namespace="/market"
    )


# ==================================================
# MARKET STATUS
# ==================================================

def market_status():

    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        return "credentials_missing"

    if _market_error:
        return "error"

    if _stream_started:
        return "connecting"

    return "offline"


# ==================================================
# REAL TRADE SERIALIZATION
# ==================================================

def serialize_trade(trade):

    symbol = trade.symbol

    price = float(trade.price)
    size = float(trade.size)

    return {

        "id": (
            f"real-{symbol}-"
            f"{trade.timestamp.isoformat()}"
        ),

        "symbol": symbol,
        "price": price,
        "size": size,
        "shares": size,
        "notional": price * size,

        "exchange": str(trade.exchange),
        "timestamp": trade.timestamp.isoformat(),

        "company": symbol,

        "simulated": False,
        "source": "ALPACA",

        # Real Alpaca trade prints do not provide buyer
        # or seller identities or their locations.
        "origin": None,
        "destination": None
    }


async def on_trade(trade):

    global _market_error

    try:

        formatted = serialize_trade(trade)

        with _market_lock:

            _recent_trades.appendleft(formatted)

            _last_trade_by_symbol[
                formatted["symbol"]
            ] = formatted

        socketio.emit(
            "market_trade",
            formatted,
            namespace="/market"
        )

        logger.info(
            "Real trade received: %s $%s x %s",
            formatted["symbol"],
            formatted["price"],
            formatted["size"]
        )

    except Exception:

        logger.exception(
            "Could not process Alpaca trade"
        )


# ==================================================
# ALPACA STREAM
# ==================================================

def start_alpaca_stream():

    global _alpaca_stream
    global _stream_started
    global _market_error

    with _stream_lock:

        if _stream_started:
            return

        if not ALPACA_API_KEY or not ALPACA_API_SECRET:

            _market_error = (
                "Alpaca credentials are missing"
            )

            logger.warning(_market_error)
            return

        try:

            feed = (
                DataFeed.IEX
                if ALPACA_DATA_FEED == "iex"
                else DataFeed.SIP
            )

            _alpaca_stream = StockDataStream(
                ALPACA_API_KEY,
                ALPACA_API_SECRET,
                feed=feed
            )

            _alpaca_stream.subscribe_trades(
                on_trade,
                *MARKET_SYMBOLS
            )

            _stream_started = True
            _market_error = None

            def run_stream():

                global _market_error
                global _stream_started

                try:

                    logger.info(
                        "Starting Alpaca stream for %s",
                        ", ".join(MARKET_SYMBOLS)
                    )

                    _alpaca_stream.run()

                except Exception as exc:

                    _market_error = str(exc)
                    _stream_started = False

                    logger.exception(
                        "Alpaca stream stopped"
                    )

            thread = threading.Thread(
                target=run_stream,
                daemon=True
            )

            thread.start()

        except Exception as exc:

            _market_error = str(exc)
            _stream_started = False

            logger.exception(
                "Could not start Alpaca stream"
            )


# ==================================================
# MARKET API ROUTES
# ==================================================

@app.route("/api/market/config", methods=["GET"])
def market_config():

    with _simulation_lock:
        simulation_enabled = _simulation_enabled

    return jsonify({

        "symbols": MARKET_SYMBOLS,

        "feed": ALPACA_DATA_FEED.upper(),

        "stream_url": "/market",

        "status": market_status(),

        "credentials_configured": bool(
            ALPACA_API_KEY and ALPACA_API_SECRET
        ),

        "simulation_enabled": simulation_enabled,

        "simulation_available": True,

        "location_note": (
            "Real map pins show company reference locations. "
            "Simulated routes show synthetic transaction locations."
        )
    })


@app.route("/api/market/snapshot", methods=["GET"])
def market_snapshot():

    # Do not start or reconnect the real Alpaca stream
    # just because simulation is active.
    with _simulation_lock:
        simulation_enabled = _simulation_enabled

    if not simulation_enabled:
        start_alpaca_stream()

    with _market_lock:

        latest = dict(_last_trade_by_symbol)

        recent = list(
            _simulated_trades
            if simulation_enabled
            else _recent_trades
        )

    summary = {}

    for symbol in MARKET_SYMBOLS:

        symbol_trades = [
            trade
            for trade in recent
            if trade["symbol"] == symbol
        ]

        summary[symbol] = {

            "latest": next(
                (
                    trade
                    for trade in symbol_trades
                ),
                latest.get(symbol)
            ),

            "trade_count": len(symbol_trades),

            "volume": sum(
                trade["size"]
                for trade in symbol_trades
            ),

            "notional": sum(
                trade["notional"]
                for trade in symbol_trades
            )
        }

    return jsonify({

        "status": (
            "simulation"
            if simulation_enabled
            else market_status()
        ),

        "simulation_enabled": simulation_enabled,

        "symbols": summary,

        "recent_trades": recent[:50]
    })


@app.route("/api/market/trades", methods=["GET"])
def market_trades():

    with _simulation_lock:
        simulation_enabled = _simulation_enabled

    if not simulation_enabled:
        start_alpaca_stream()

    symbol = request.args.get(
        "symbol",
        ""
    ).upper().strip()

    source = request.args.get(
        "source",
        "active"
    ).lower().strip()

    try:

        limit = max(
            1,
            min(
                int(request.args.get("limit", 100)),
                MAX_RECENT_TRADES
            )
        )

    except ValueError:

        return jsonify({
            "error": "limit must be an integer"
        }), 400

    with _market_lock:

        if source == "real":
            trades = list(_recent_trades)

        elif source == "simulated":
            trades = list(_simulated_trades)

        elif simulation_enabled:
            trades = list(_simulated_trades)

        else:
            trades = list(_recent_trades)

    if symbol:

        trades = [
            trade
            for trade in trades
            if trade["symbol"] == symbol
        ]

    return jsonify(trades[:limit])


# ==================================================
# SIMULATION CONTROL ROUTES
# ==================================================

@app.route("/api/market/simulation", methods=["GET"])
def get_simulation_status():

    with _simulation_lock:
        enabled = _simulation_enabled

    return jsonify({

        "enabled": enabled,

        "mode": (
            "FAKEPACA"
            if enabled
            else "REAL"
        ),

        "description": (
            "Synthetic transactions for globe visualisation"
        )
    })


@app.route("/api/market/simulation", methods=["POST"])
def toggle_simulation():

    data = request.get_json(silent=True) or {}

    enabled = data.get("enabled")

    if not isinstance(enabled, bool):

        return jsonify({
            "error": "enabled must be true or false"
        }), 400

    set_simulation_enabled(enabled)

    return jsonify({

        "enabled": enabled,

        "mode": (
            "FAKEPACA"
            if enabled
            else "REAL"
        ),

        "message": (
            "FAKEPACA simulation enabled"
            if enabled
            else "FAKEPACA simulation disabled"
        )
    })


# ==================================================
# COMPANY LOCATIONS
# ==================================================

@app.route("/api/market/locations", methods=["GET"])
def market_locations():

    locations = {

        "AAPL": (
            "Apple", "Cupertino", "USA",
            37.3349, -122.0090, "Technology"
        ),

        "MSFT": (
            "Microsoft", "Redmond", "USA",
            47.6740, -122.1215, "Technology"
        ),

        "NVDA": (
            "NVIDIA", "Santa Clara", "USA",
            37.3708, -121.9630, "Technology"
        ),

        "AMZN": (
            "Amazon", "Seattle", "USA",
            47.6225, -122.3363, "Consumer"
        ),

        "GOOGL": (
            "Alphabet", "Mountain View", "USA",
            37.4220, -122.0841, "Technology"
        ),

        "META": (
            "Meta", "Menlo Park", "USA",
            37.4848, -122.1484, "Technology"
        ),

        "TSLA": (
            "Tesla", "Austin", "USA",
            30.2672, -97.7431, "Automotive"
        ),

        "AMD": (
            "AMD", "Santa Clara", "USA",
            37.3875, -121.9636, "Technology"
        )
    }

    symbols = []

    for symbol in MARKET_SYMBOLS:

        location = locations.get(symbol)

        if not location:
            continue

        company, city, country, lat, lon, sector = location

        symbols.append({

            "symbol": symbol,
            "company": company,

            "city": city,
            "country": country,

            "latitude": lat,
            "longitude": lon,

            "sector": sector,

            "location_type": (
                "company_reference_location"
            )
        })

    return jsonify(symbols)


# ==================================================
# SOCKET.IO
# ==================================================

@socketio.on("connect", namespace="/market")
def market_client_connected():

    logger.info(
        "Market dashboard connected"
    )

    with _simulation_lock:
        enabled = _simulation_enabled

    socketio.emit(
        "simulation_status",
        {
            "enabled": enabled,
            "mode": (
                "FAKEPACA"
                if enabled
                else "REAL"
            )
        },
        namespace="/market",
        to=request.sid
    )


@socketio.on("disconnect", namespace="/market")
def market_client_disconnected():

    logger.info(
        "Market dashboard disconnected"
    )


# ==================================================
# START APPLICATION
# ==================================================

with app.app_context():
    db.create_all()


# Research API (/api/research/*): see backend/research/ and docs/MARKETMIND_RESEARCH_API.md
from research.api import init_research
init_research(app, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'research.db'))


if __name__ == "__main__":

    start_alpaca_stream()

    socketio.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True
    )
