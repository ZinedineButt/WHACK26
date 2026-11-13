# WHACK26: Global Market Intelligence + Autonomous Research

A React + Flask app with two tabs:

- **Global intelligence**: an interactive globe of companies with two modes.
  - *Correlation*: next-day lead-lag links between companies, "Ask the globe" (plain-English questions answered with real data), a replay of the last month, and the top research papers for each company.
  - *Daily movement*: one bar per company, height = size of the day's % change, green = gain, red = loss, with slow looping playback.
- **Research**: an AI-assisted pipeline that proposes hypotheses, then tests them with correlation, causal checks, out-of-sample validation, a backtest, peer tests and a literature-grounded self-critique.

All statistics are computed in Python from real market data. The AI (Gemini) only proposes ideas and writes explanations; it never produces numbers.

---

## 1. What you need installed

| Tool | Version | Check with |
|---|---|---|
| Git | any recent | `git --version` |
| Python | 3.10 or newer (3.11 or 3.12 recommended) | `python --version` (Windows: `py --version`) |
| Node.js (includes npm) | 18 or newer (20 LTS recommended) | `node --version` |

Download links: <https://git-scm.com>, <https://www.python.org/downloads> (tick **Add python.exe to PATH** on Windows), <https://nodejs.org>.

## 2. Get the code

```bash
git clone https://github.com/ZinedineButt/WHACK26.git
cd WHACK26
```

## 3. API keys

The app needs keys from free services. Keys are read **only by the backend** and must never be put in the frontend or committed to Git.

| Key | Used for | Where to get it | Required? |
|---|---|---|---|
| `ALPACA_API_KEY`, `ALPACA_API_SECRET` | Live trade stream on the globe, fallback price data | <https://alpaca.markets> (free paper account) | Needed for the live stream; data still works without it if you have Tiingo |
| `TIINGO_API_KEY` | Daily prices for correlations, replay and research | <https://www.tiingo.com> (free) | Strongly recommended |
| `GEMINI_API_KEY` | Hypothesis generation, "Ask the globe", critiques | <https://aistudio.google.com/apikey> (free tier) | Needed for the AI features. Without it a scripted demo mode runs |
| `FRED_API_KEY` | Macro series (yields, oil, VIX) | <https://fred.stlouisfed.org/docs/api/api_key.html> | Optional (a public CSV fallback is used) |
| `OPENALEX_API_KEY` | Research-paper search | <https://openalex.org/settings/api> | Optional (arXiv is the fallback) |

Create two files inside the `backend` folder:

**`backend/.env`** (market map and Alpaca):

```
ALPACA_API_KEY=your_alpaca_key
ALPACA_API_SECRET=your_alpaca_secret
ALPACA_DATA_FEED=iex
ALPACA_SYMBOLS=AAPL,TSLA,NVDA,MSFT,AMZN,GOOGL,META,AMD
```

**`backend/.env.local`** (research keys; start from `backend/.env.example`):

```
GEMINI_API_KEY=your_gemini_key
GEMINI_MODEL=gemini-3.5-flash-lite
TIINGO_API_KEY=your_tiingo_key
FRED_API_KEY=
OPENALEX_API_KEY=
```

Windows quick copy: `copy backend\.env.example backend\.env.local`, then edit it. macOS/Linux: `cp backend/.env.example backend/.env.local`.

`.env.local` is git-ignored. **Do not commit real keys.** If a key has ever been pushed to GitHub, treat it as leaked and regenerate it.

## 4. Install and run the backend (port 5000)

### Windows (PowerShell)

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

(Calling the venv's `python.exe` directly means you never need to activate it, which avoids PowerShell script-permission errors.)

### macOS / Linux

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

You should see log lines such as `Alpaca API key loaded: True`. Leave this terminal open.

Check it works by opening <http://localhost:5000/api/research/health>. You want `"gemini_configured": true` and `"equity_data_usable": true`.

## 5. Install and run the frontend (port 5173)

Open a **second** terminal:

### Windows (PowerShell)

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

### macOS / Linux

```bash
cd frontend
npm install
npm run dev
```

Open the address Vite prints, normally <http://localhost:5173>.

If the backend runs somewhere other than `http://localhost:5000`, create `frontend/.env.local` containing `VITE_API_URL=http://your-host:5000` and restart `npm run dev`.

## 6. Using the app

**Global intelligence tab**
- Drag to rotate, scroll to zoom, click a company to see its next-day links and top papers. Press `Esc`, click "✕ Clear", or click the company again to unselect.
- **Correlation | Daily movement** switch sits at the top centre of the globe.
- **Ask the globe**: type a question like "does Walmart lead Coca-Cola the next day?". Needs a Gemini key for best results.
- **Replay the last month**: steps through real trading days (correlation mode). Daily movement mode loops through them automatically; use ❚❚ / ▶ and the slider to control it.
- The first load of correlations can take about 20 to 30 seconds while prices download; later loads are cached.

**Research tab**
- Pick a one-click example or leave the default theme ("What drives moves in the stock market?") and let the AI explore. Runs take a few minutes.
- Each step shows its key result and chart; open the folds for the detailed checks.

## 7. Run the tests (optional)

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q
```

(macOS/Linux: `python -m pytest -q` with the venv active.) The full suite takes a few minutes. To check the globe API only: `pytest -q tests/test_globe_api.py`.

To build the frontend for production: `cd frontend && npm run build` (output in `frontend/dist`).

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `python` is not recognised (Windows) | Use `py` instead, or reinstall Python with "Add to PATH" ticked |
| `npm : File ...npm.ps1 cannot be loaded because running scripts is disabled` | Use `npm.cmd install` / `npm.cmd run dev`, or run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Globe page shows "Market data: Failed to fetch" | The backend is not running or not on port 5000. Start it (section 4) |
| Correlation lines or replay are empty | No price data: check `/api/research/health` shows `equity_data_usable: true`; add `TIINGO_API_KEY` |
| Some companies are missing links | Free price APIs rate-limit (HTTP 429). The backend falls back to Alpaca and retries on the next request; wait a minute and reload |
| "Ask the globe" gives keyword-only answers | `GEMINI_API_KEY` missing or invalid in `backend/.env.local` |
| No paper results | Free search APIs can be rate-limited; try again shortly (add `OPENALEX_API_KEY` to improve this) |
| Port 5000 already in use | Set `PORT=5001` in `backend/.env` and `VITE_API_URL=http://localhost:5001` in `frontend/.env.local` |
| Live stream shows no trades | Normal when the US market is closed; use the replay and daily movement modes instead |
| `pip install` fails building a package | Upgrade pip (`python -m pip install --upgrade pip`) and use Python 3.10 to 3.12 |

## 9. Project layout

```
backend/
  app.py                 Flask app, live market stream, task API
  requirements.txt
  research/              hypothesis engine, statistics, validation, globe API (globe_api.py)
  scripts/download_papers.py   optional: downloads open-access papers per stock into /papers
  tests/                 pytest suite
frontend/
  src/App.jsx            globe, modes, hero, panels
  src/components/        Research launcher and report
papers/                  downloaded open-access papers (INDEX.md lists them)
docs/
```

## 10. Notes on the data

- Prices are split/dividend-adjusted daily closes from Tiingo (or Alpaca as a fallback). Nothing in the correlation, replay or movement views is simulated.
- Globe links are **next-day lead-lag** correlations (the leader's move today vs the follower's move tomorrow), so there is no look-ahead. They are small and many are noise; faded lines are not statistically significant after correction for the number of pairs tested.
- Correlation is not causation. The Research tab exists to test whether a pattern survives held-out data.
