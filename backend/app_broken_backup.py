        symbols.append({
            'symbol': symbol,
            'company': location.get('company', symbol),
            'city': location.get('city'),
            'country': location.get('country'),
            'latitude': location.get('latitude'),
            'longitude': location.get('longitude'),
            'sector': location.get('sector', 'Other'),
            'location_type': 'company_reference_location',
        })
    return jsonify(symbols)


@app.route('/api/market/trades', methods=['GET'])
def market_trades():
    """Return recent trades. Optional query parameters: symbol and limit."""
    _start_alpaca_stream()
    symbol = request.args.get('symbol', '').upper().strip()
    try:
        limit = max(1, min(int(request.args.get('limit', 100)), MAX_RECENT_TRADES))
    except ValueError:
        return jsonify({'error': 'limit must be an integer'}), 400
    with _market_lock:
        trades = list(_recent_trades)
    if symbol:
        trades = [trade for trade in trades if trade['symbol'] == symbol]
    return jsonify(trades[:limit])


@app.route('/api/market/snapshot', methods=['GET'])
def market_snapshot():
    """Return latest known trade per symbol plus aggregate volume/notional."""
    _start_alpaca_stream()
    with _market_lock:
        latest = dict(_last_trade_by_symbol)
        recent = list(_recent_trades)
    summary = {}
    for symbol in MARKET_SYMBOLS:
        symbol_trades = [trade for trade in recent if trade['symbol'] == symbol]
        summary[symbol] = {
            'latest': latest.get(symbol),
            'trade_count': len(symbol_trades),
            'volume': sum(trade['size'] for trade in symbol_trades),
            'notional': sum(trade['notional'] for trade in symbol_trades),
        }
    return jsonify({
        'status': _market_status(),
        'symbols': summary,
        'recent_trades': recent[:50],
    })


@app.route('/api/market/stream', methods=['GET'])
def market_stream_events():
    """Server-Sent Events stream for browser EventSource clients.

    Frontend example: const events = new EventSource('/api/market/stream')
    events.onmessage = (event) => handleTrade(JSON.parse(event.data))
    """
    _start_alpaca_stream()
    subscriber = queue.Queue(maxsize=200)
    with _market_lock:
        _subscriber_queues.add(subscriber)
        initial_trades = list(_recent_trades)[:50]

    @stream_with_context
    def generate():
        try:
            yield 'retry: 3000\n\n'
            # Send a snapshot first so the map can populate immediately.
            for trade in reversed(initial_trades):
                yield f"data: {json.dumps(trade, separators=(',', ':'))}\n\n"
            while True:
                try:
                    trade = subscriber.get(timeout=20)
                    yield f"data: {json.dumps(trade, separators=(',', ':'))}\n\n"
                except queue.Empty:
                    yield ': keep-alive\n\n'
        finally:
            with _market_lock:
                _subscriber_queues.discard(subscriber)

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        },
    )


@app.route('/api/market/config', methods=['GET'])
def market_config():
    """Public, non-secret frontend configuration."""
    return jsonify({
        'symbols': MARKET_SYMBOLS,
        'feed': os.getenv('ALPACA_DATA_FEED', 'iex').upper(),
        'stream_url': '/api/market/stream',
        'location_note': 'Pins show company reference locations, not individual trade origins.',
    })


if __name__ == '__main__':
    # Start the market stream before serving requests in local development.
    _start_alpaca_stream()
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', '5000')), debug=True, threaded=True, use_reloader=False)
