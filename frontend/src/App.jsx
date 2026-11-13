import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
  useCallback,
  memo
} from 'react'
import Globe from 'react-globe.gl'
import { io } from 'socket.io-client'
import StatusSelect from './StatusSelect'
import ResearchLauncher from './components/ResearchLauncher'

const API = import.meta.env.VITE_API_URL || 'http://localhost:5000'

const STATUSES = [
  'Not Started',
  'In Progress',
  'Completed',
  'Blocked'
]

const MAX_FEED = 150
const MAX_ARCS = 20
const MAX_RINGS = 4
const SOCKET_FLUSH_INTERVAL = 500
const ARC_LIFETIME = 1600
const SIMULATION_INTERVAL = 150

const money = value =>
  new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2
  }).format(Number(value) || 0)

const compact = value =>
  new Intl.NumberFormat('en-US', {
    notation: 'compact',
    maximumFractionDigits: 1
  }).format(Number(value) || 0)

const time = value =>
  value
    ? new Date(value).toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      })
    : '—'

const dateTime = value =>
  value ? new Date(value).toLocaleString() : '—'

const ZOOM_MODES = {
  GLOBAL: 'global',
  REGIONAL: 'regional',
  COMPANY: 'company',
  CLOSEST: 'closest'
}

const ZOOM_LABELS = {
  global: 'GLOBAL NETWORK',
  regional: 'REGIONAL MARKETS',
  company: 'COMPANY INTELLIGENCE',
  closest: 'TRADING TERMINAL'
}

const INITIAL_ALTITUDE = 2.5
const COMPANY_ALTITUDE = 0.38
const ZOOM_LOCK_ALTITUDE = 0.23

const BUY_COLOR = '#36d99c'
const SELL_COLOR = '#ff596d'

const BUYER_LOCATIONS = [
  { city: 'London', country: 'United Kingdom', lat: 51.5072, lng: -0.1276 },
  { city: 'New York', country: 'United States', lat: 40.7128, lng: -74.0060 },
  { city: 'Tokyo', country: 'Japan', lat: 35.6762, lng: 139.6503 },
  { city: 'Singapore', country: 'Singapore', lat: 1.3521, lng: 103.8198 },
  { city: 'Hong Kong', country: 'Hong Kong', lat: 22.3193, lng: 114.1694 },
  { city: 'Dubai', country: 'UAE', lat: 25.2048, lng: 55.2708 },
  { city: 'Paris', country: 'France', lat: 48.8566, lng: 2.3522 },
  { city: 'Frankfurt', country: 'Germany', lat: 50.1109, lng: 8.6821 },
  { city: 'Sydney', country: 'Australia', lat: -33.8688, lng: 151.2093 },
  { city: 'Toronto', country: 'Canada', lat: 43.6532, lng: -79.3832 },
  { city: 'Mumbai', country: 'India', lat: 19.0760, lng: 72.8777 },
  { city: 'Seoul', country: 'South Korea', lat: 37.5665, lng: 126.9780 },
  { city: 'Amsterdam', country: 'Netherlands', lat: 52.3676, lng: 4.9041 },
  { city: 'San Francisco', country: 'United States', lat: 37.7749, lng: -122.4194 },
  { city: 'Zurich', country: 'Switzerland', lat: 47.3769, lng: 8.5417 },
  { city: 'Sao Paulo', country: 'Brazil', lat: -23.5505, lng: -46.6333 },
  { city: 'Osaka', country: 'Japan', lat: 34.6937, lng: 135.5023 },
  { city: 'Beijing', country: 'China', lat: 39.9042, lng: 116.4074 },
  { city: 'Shanghai', country: 'China', lat: 31.2304, lng: 121.4737 },
  { city: 'Taipei', country: 'Taiwan', lat: 25.0330, lng: 121.5654 },
  { city: 'Bangkok', country: 'Thailand', lat: 13.7563, lng: 100.5018 },
  { city: 'Mexico City', country: 'Mexico', lat: 19.4326, lng: -99.1332 },
  { city: 'Johannesburg', country: 'South Africa', lat: -26.2041, lng: 28.0473 },
  { city: 'Stockholm', country: 'Sweden', lat: 59.3293, lng: 18.0686 }
]

const COMPANY_NAMES = {
  INTC: 'Intel', ORCL: 'Oracle', CRM: 'Salesforce', NFLX: 'Netflix', DIS: 'Walt Disney', JPM: 'JPMorgan Chase',
  GS: 'Goldman Sachs', BAC: 'Bank of America', V: 'Visa', MA: 'Mastercard', WMT: 'Walmart', KO: 'Coca-Cola',
  BA: 'Boeing', UBER: 'Uber', PLTR: 'Palantir Technologies', COIN: 'Coinbase', SONY: 'Sony', TM: 'Toyota Motor',
  HMC: 'Honda Motor', NTDOY: 'Nintendo', MUFG: 'Mitsubishi UFJ Financial', SMFG: 'Sumitomo Mitsui Financial',
  MFG: 'Mizuho Financial', BABA: 'Alibaba', BIDU: 'Baidu', JD: 'JD.com', PDD: 'PDD Holdings', TCEHY: 'Tencent',
  TSM: 'Taiwan Semiconductor', KB: 'KB Financial', INFY: 'Infosys', WIT: 'Wipro', SAP: 'SAP', ASML: 'ASML',
  NVO: 'Novo Nordisk', AZN: 'AstraZeneca', GSK: 'GSK', BP: 'BP', SHEL: 'Shell', UBS: 'UBS', NVS: 'Novartis',
  SHOP: 'Shopify', RY: 'Royal Bank of Canada', BHP: 'BHP Group'
}

const COMPANY_LOCATIONS = {
  AAPL: { city: 'Cupertino', country: 'USA', lat: 37.3349, lng: -122.0090 },
  MSFT: { city: 'Redmond', country: 'USA', lat: 47.6740, lng: -122.1215 },
  NVDA: { city: 'Santa Clara', country: 'USA', lat: 37.3708, lng: -121.9630 },
  AMZN: { city: 'Seattle', country: 'USA', lat: 47.6225, lng: -122.3363 },
  GOOGL: { city: 'Mountain View', country: 'USA', lat: 37.4220, lng: -122.0841 },
  META: { city: 'Menlo Park', country: 'USA', lat: 37.4848, lng: -122.1484 },
  TSLA: { city: 'Austin', country: 'USA', lat: 30.2672, lng: -97.7431 },
  AMD: { city: 'Santa Clara', country: 'USA', lat: 37.3875, lng: -121.9636 },
  INTC: { city: 'Santa Clara', country: 'USA', lat: 37.3875, lng: -121.9636 },
  ORCL: { city: 'Austin', country: 'USA', lat: 30.2672, lng: -97.7431 },
  CRM: { city: 'San Francisco', country: 'USA', lat: 37.7749, lng: -122.4194 },
  NFLX: { city: 'Los Gatos', country: 'USA', lat: 37.2358, lng: -121.9624 },
  DIS: { city: 'Burbank', country: 'USA', lat: 34.1808, lng: -118.3090 },
  JPM: { city: 'New York', country: 'USA', lat: 40.7555, lng: -73.9755 },
  GS: { city: 'New York', country: 'USA', lat: 40.7150, lng: -74.0140 },
  BAC: { city: 'Charlotte', country: 'USA', lat: 35.2271, lng: -80.8431 },
  V: { city: 'San Francisco', country: 'USA', lat: 37.7749, lng: -122.4194 },
  MA: { city: 'Purchase', country: 'USA', lat: 41.0409, lng: -73.7140 },
  WMT: { city: 'Bentonville', country: 'USA', lat: 36.3729, lng: -94.2088 },
  KO: { city: 'Atlanta', country: 'USA', lat: 33.7490, lng: -84.3880 },
  BA: { city: 'Arlington', country: 'USA', lat: 38.8816, lng: -77.0910 },
  UBER: { city: 'San Francisco', country: 'USA', lat: 37.7749, lng: -122.4194 },
  PLTR: { city: 'Denver', country: 'USA', lat: 39.7392, lng: -104.9903 },
  COIN: { city: 'New York', country: 'USA', lat: 40.7128, lng: -74.0060 },

  SONY: { city: 'Tokyo', country: 'Japan', lat: 35.6654, lng: 139.7707 },
  TM: { city: 'Toyota City', country: 'Japan', lat: 35.0826, lng: 137.1560 },
  HMC: { city: 'Tokyo', country: 'Japan', lat: 35.6762, lng: 139.6503 },
  NTDOY: { city: 'Kyoto', country: 'Japan', lat: 35.0116, lng: 135.7681 },
  MUFG: { city: 'Tokyo', country: 'Japan', lat: 35.6762, lng: 139.6503 },
  SMFG: { city: 'Tokyo', country: 'Japan', lat: 35.6762, lng: 139.6503 },
  MFG: { city: 'Tokyo', country: 'Japan', lat: 35.6762, lng: 139.6503 },

  BABA: { city: 'Hangzhou', country: 'China', lat: 30.2741, lng: 120.1551 },
  BIDU: { city: 'Beijing', country: 'China', lat: 39.9042, lng: 116.4074 },
  JD: { city: 'Beijing', country: 'China', lat: 39.9042, lng: 116.4074 },
  PDD: { city: 'Shanghai', country: 'China', lat: 31.2304, lng: 121.4737 },
  TCEHY: { city: 'Shenzhen', country: 'China', lat: 22.5431, lng: 114.0579 },

  TSM: { city: 'Hsinchu', country: 'Taiwan', lat: 24.8138, lng: 120.9675 },
  KB: { city: 'Seoul', country: 'South Korea', lat: 37.5665, lng: 126.9780 },

  INFY: { city: 'Bengaluru', country: 'India', lat: 12.9716, lng: 77.5946 },
  WIT: { city: 'Bengaluru', country: 'India', lat: 12.9716, lng: 77.5946 },

  SAP: { city: 'Walldorf', country: 'Germany', lat: 49.2940, lng: 8.6420 },
  ASML: { city: 'Veldhoven', country: 'Netherlands', lat: 51.4180, lng: 5.4060 },
  NVO: { city: 'Bagsvaerd', country: 'Denmark', lat: 55.7667, lng: 12.4500 },
  AZN: { city: 'Cambridge', country: 'United Kingdom', lat: 52.2053, lng: 0.1218 },
  GSK: { city: 'London', country: 'United Kingdom', lat: 51.5072, lng: -0.1276 },
  BP: { city: 'London', country: 'United Kingdom', lat: 51.5072, lng: -0.1276 },
  SHEL: { city: 'London', country: 'United Kingdom', lat: 51.5072, lng: -0.1276 },
  UBS: { city: 'Zurich', country: 'Switzerland', lat: 47.3769, lng: 8.5417 },
  NVS: { city: 'Basel', country: 'Switzerland', lat: 47.5596, lng: 7.5886 },

  SHOP: { city: 'Ottawa', country: 'Canada', lat: 45.4215, lng: -75.6972 },
  RY: { city: 'Toronto', country: 'Canada', lat: 43.6532, lng: -79.3832 },

  BHP: { city: 'Melbourne', country: 'Australia', lat: -37.8136, lng: 144.9631 }
}

// --------------------------------------------------
// STYLES
// --------------------------------------------------

const styles = `
* { box-sizing: border-box; }

body {
  margin: 0;
  background: #05080d;
  color: #e8f0f8;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
}

button, input { font: inherit; }
button { cursor: pointer; }

.app-shell {
  min-height: 100vh;
  background: #05080d;
}

.app-nav {
  position: fixed;
  top: 16px;
  left: 18px;
  z-index: 100;
  display: flex;
  gap: 8px;
  padding: 5px;
  background: rgba(7,13,22,.85);
  border: 1px solid rgba(120,160,190,.15);
  border-radius: 10px;
  backdrop-filter: blur(16px);
}

.nav-tab {
  padding: 9px 13px;
  border: 0;
  border-radius: 7px;
  color: #8295a9;
  background: transparent;
  font-size: 11px;
}

.nav-tab.selected {
  color: #69e8bd;
  background: rgba(54,211,153,.12);
}

.market-experience {
  position: relative;
  height: 100vh;
  min-height: 650px;
  overflow: hidden;
  background: radial-gradient(ellipse at 50% 45%,#142638 0%,#09121e 45%,#05080d 100%);
}

.globe-container {
  position: absolute;
  inset: 0;
  z-index: 1;
}

.globe-container canvas { outline: none; }

.globe-header {
  position: absolute;
  z-index: 10;
  top: 22px;
  left: 24px;
  right: 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  pointer-events: none;
}

.globe-header > * { pointer-events: auto; }

.brand { display: flex; align-items: center; gap: 11px; }

.brand-mark {
  width: 36px;
  height: 36px;
  display: grid;
  place-items: center;
  border-radius: 9px;
  background: linear-gradient(135deg,#39d8a6,#137e80);
  color: #03120e;
  font-weight: 900;
}

.brand-name { font-size: 15px; font-weight: 850; letter-spacing: 1px; }
.brand-name span { color: #64e6bd; }
.brand-sub { color: #70879d; font-size: 9px; letter-spacing: 2px; margin-top: 3px; }

.header-right { display: flex; align-items: center; gap: 15px; }

.connection-pill {
  display: flex;
  align-items: center;
  gap: 7px;
  color: #8fa5b8;
  font-size: 10px;
  letter-spacing: 1px;
}

.connection-pill i, .live-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #e4a83c;
}

.connection-pill.live i, .live-dot {
  background: #36d99c;
  box-shadow: 0 0 12px #36d99c;
}

.connection-pill.offline i { background: #ff657a; }

.feed-source {
  padding: 7px 10px;
  color: #8da3b8;
  border: 1px solid #243448;
  border-radius: 6px;
  font-size: 9px;
  letter-spacing: 1px;
}

.demo-control {
  position: absolute;
  z-index: 25;
  top: 82px;
  right: 24px;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 11px 13px;
  background: rgba(7,16,27,.94);
  border: 1px solid rgba(100,150,180,.25);
  border-radius: 9px;
  backdrop-filter: blur(15px);
}

.demo-copy strong {
  display: block;
  font-size: 10px;
  letter-spacing: 1px;
}

.demo-copy small {
  display: block;
  color: #8295a9;
  font-size: 9px;
  margin-top: 4px;
}

.demo-toggle {
  width: 42px;
  height: 23px;
  padding: 3px;
  border: 1px solid #405064;
  border-radius: 20px;
  background: #1b2938;
  transition: .2s;
}

.demo-toggle.active {
  background: #168b6c;
  border-color: #36d99c;
}

.demo-toggle span {
  display: block;
  width: 15px;
  height: 15px;
  border-radius: 50%;
  background: white;
  transition: .2s;
}

.demo-toggle.active span { transform: translateX(18px); }

.demo-badge {
  position: absolute;
  z-index: 20;
  top: 145px;
  right: 24px;
  color: #70efc0;
  background: rgba(12,55,45,.9);
  border: 1px solid rgba(54,217,156,.45);
  padding: 6px 10px;
  border-radius: 5px;
  font-size: 9px;
  letter-spacing: 1px;
}

.globe-search {
  position: absolute;
  z-index: 15;
  top: 83px;
  left: 24px;
  width: 260px;
}

.globe-search input {
  width: 100%;
  padding: 12px 14px;
  background: rgba(8,16,27,.88);
  color: white;
  border: 1px solid rgba(105,160,190,.22);
  border-radius: 8px;
  outline: none;
  backdrop-filter: blur(15px);
}

.globe-search input:focus { border-color: #38d7a2; }

.search-results {
  margin-top: 5px;
  background: rgba(8,16,27,.97);
  border: 1px solid #263b4c;
  border-radius: 8px;
  overflow: hidden;
}

.search-result {
  display: flex;
  width: 100%;
  justify-content: space-between;
  padding: 11px;
  background: transparent;
  border: 0;
  color: white;
  text-align: left;
}

.search-result:hover { background: #142536; }
.search-result small { color: #71889b; }

.zoom-indicator {
  position: absolute;
  z-index: 10;
  top: 84px;
  right: 25px;
  text-align: right;
  pointer-events: none;
}

.zoom-indicator .eyebrow { color: #5de3b5; font-size: 9px; letter-spacing: 2px; }
.zoom-indicator strong { display: block; margin-top: 7px; font-size: 12px; letter-spacing: 2px; }

.globe-stats {
  position: absolute;
  z-index: 10;
  left: 24px;
  bottom: 25px;
  display: flex;
  gap: 8px;
  pointer-events: none;
}

.globe-stat {
  min-width: 120px;
  padding: 11px 13px;
  background: rgba(8,17,28,.82);
  border: 1px solid rgba(100,150,180,.16);
  border-radius: 8px;
  backdrop-filter: blur(12px);
}

.globe-stat span { display: block; color: #7890a4; font-size: 9px; letter-spacing: 1px; }
.globe-stat strong { display: block; margin-top: 6px; font-size: 17px; }

.immersive-feed {
  position: absolute;
  z-index: 12;
  top: 185px;
  right: 24px;
  width: 310px;
  max-height: calc(100vh - 280px);
  display: flex;
  flex-direction: column;
  background: rgba(7,14,24,.88);
  border: 1px solid rgba(100,150,180,.2);
  border-radius: 10px;
  backdrop-filter: blur(18px);
  overflow: hidden;
}

.immersive-feed.company-mode { width: 340px; }
.immersive-feed.closest-mode { width: 370px; }

.overlay-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 15px;
  border-bottom: 1px solid rgba(100,150,180,.15);
}

.overlay-heading h3 { margin: 5px 0 0; font-size: 13px; }
.panel-kicker { color: #60dcb3; font-size: 9px; letter-spacing: 1.6px; }
.feed-status { color: #36d99c; font-size: 9px; }

.feed-item {
  width: 100%;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 5px 10px;
  padding: 11px 14px;
  background: transparent;
  border: 0;
  border-bottom: 1px solid rgba(100,150,180,.09);
  color: #e6eff7;
  text-align: left;
}

.feed-item:hover { background: rgba(50,130,130,.12); }
.feed-item strong { font-size: 11px; }
.feed-item .feed-price { color: #65e6bd; text-align: right; font-size: 11px; }
.feed-item .feed-price.sell { color: #ff657a; }
.feed-item .feed-side {
  font-size: 9px;
  font-weight: 800;
  letter-spacing: .8px;
}
.feed-item .feed-side.buy { color: #36d99c; }
.feed-item .feed-side.sell { color: #ff596d; }
.feed-item small { color: #71899e; font-size: 9px; }
.feed-item .feed-notional { text-align: right; }
.feed-list { overflow-y: auto; min-height: 100px; }

.feed-empty {
  padding: 20px;
  color: #8295a9;
  font-size: 11px;
  line-height: 1.7;
}

.feed-footer {
  padding: 10px 14px;
  color: #74899b;
  font-size: 9px;
  border-top: 1px solid rgba(100,150,180,.15);
}

.company-focus {
  position: absolute;
  z-index: 11;
  left: 50%;
  top: 105px;
  transform: translateX(-50%);
  width: min(400px,calc(100% - 40px));
  text-align: center;
  pointer-events: none;
}

.company-focus h2 { margin: 5px 0; font-size: 30px; letter-spacing: 2px; }
.company-focus p { margin: 0; color: #94a9bc; font-size: 11px; }
.company-focus .focus-price { margin-top: 12px; font-size: 25px; font-weight: 750; }
.company-focus .focus-caption { color: #7c93a8; font-size: 9px; }
.company-focus .focus-change { margin-top: 4px; font-size: 11px; font-variant-numeric: tabular-nums; }
.company-focus .focus-change.up, .terminal-card .up, .terminal-table td.up { color: #36d99c; }
.company-focus .focus-change.down, .terminal-card .down, .terminal-table td.down { color: #ff596d; }
.terminal-replay { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin: 4px 0 14px; padding: 10px 14px; border-radius: 8px; background: #0b1420; border: 1px solid #1d3041; font-size: 11px; color: #aebfce; }
.terminal-replay button { padding: 5px 11px; color: #dbe7f1; background: #142436; border: 1px solid #30475c; border-radius: 6px; font-size: 11px; cursor: pointer; }
.terminal-replay input[type=range] { flex: 1; min-width: 140px; accent-color: #36d5a1; }
.terminal-replay b { color: #60dcb3; }
.terminal-spark { width: 100%; height: 56px; margin: 0 0 12px; }

.company-details {
  position: absolute;
  z-index: 12;
  bottom: 25px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 20px;
  padding: 12px 18px;
  background: rgba(7,15,25,.88);
  border: 1px solid #263c4c;
  border-radius: 9px;
  backdrop-filter: blur(14px);
}

.company-details div span { display: block; color: #8198ab; font-size: 8px; letter-spacing: 1px; }
.company-details div strong { display: block; margin-top: 5px; font-size: 12px; }

.link-panel {
  position: absolute;
  z-index: 12;
  top: 140px;
  left: 24px;
  width: 290px;
  max-height: calc(100vh - 250px);
  overflow-y: auto;
  padding: 14px;
  background: rgba(7,14,24,.9);
  border: 1px solid rgba(100,150,180,.2);
  border-radius: 10px;
  backdrop-filter: blur(18px);
}
.link-legend.inline { margin: 0 0 0 auto; gap: 8px; }
.link-more { width: 100%; margin: 4px 0 2px; padding: 5px; font-size: 10px; color: #9fb3c4; background: transparent; border: 1px dashed #2b4156; border-radius: 6px; cursor: pointer; }
.link-more:hover { color: #60dcb3; border-color: #60dcb3; }
.paper-row span { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.ask-wrap { position: absolute; z-index: 16; left: 50%; bottom: 24px; transform: translateX(-50%); width: min(600px, calc(100% - 48px)); display: flex; flex-direction: column; gap: 8px; align-items: stretch; }
.ask-bar { display: flex; align-items: center; gap: 8px; padding: 6px 6px 6px 14px; background: rgba(7,16,27,.94); border: 1px solid rgba(96,220,179,.45); border-radius: 12px; backdrop-filter: blur(16px); box-shadow: 0 0 30px rgba(54,217,156,.12); }
.ask-bar input { flex: 1; min-width: 0; padding: 9px 4px; color: #eaf3fa; background: transparent; border: 0; outline: 0; font-size: 13px; }
.ask-bar button { padding: 8px 16px; border: 0; border-radius: 8px; background: #36d5a1; color: #04120e; font-weight: 700; font-size: 12px; }
.ask-bar button:disabled { opacity: .5; cursor: default; }
.ask-chips { display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; }
.ask-chips button { padding: 5px 11px; font-size: 10px; color: #9fb3c4; background: rgba(7,16,27,.85); border: 1px solid #243749; border-radius: 20px; }
.ask-chips button:hover { color: #60dcb3; border-color: #60dcb3; }
.ask-error { padding: 8px 12px; font-size: 11px; color: #ffb3bd; background: rgba(42,17,25,.92); border: 1px solid #5a2431; border-radius: 8px; }
.ask-card { padding: 14px 16px; background: rgba(7,14,24,.95); border: 1px solid rgba(100,150,180,.25); border-radius: 12px; backdrop-filter: blur(18px); max-height: 46vh; overflow-y: auto; }
.ask-card-head { display: flex; justify-content: space-between; gap: 10px; align-items: flex-start; margin-bottom: 8px; }
.ask-q { margin-top: 4px; font-size: 13px; color: #dbe7f1; }
.ask-pair { padding: 9px 0; border-top: 1px solid rgba(100,150,180,.12); }
.ask-pair-top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 5px; font-size: 13px; }
.ask-pair-top span { color: #6f879c; }
.ask-pair-top b.pos { color: #36d99c; } .ask-pair-top b.neg { color: #ff596d; }
.ask-pair-sub { margin-top: 5px; font-size: 10px; color: #8fa4b6; }
.ask-pair-sub em { font-style: normal; } .ask-pair-sub em.ok { color: #36d99c; } .ask-pair-sub em.no { color: #ffb36b; }
.ask-pair-why { margin-top: 4px; font-size: 11px; color: #b6c6d4; }
.ask-actions { display: flex; align-items: center; gap: 12px; margin-top: 8px; flex-wrap: wrap; }
.ask-actions small { flex: 1; min-width: 180px; font-size: 9px; color: #7d92a5; }
.replay-control { position: absolute; z-index: 25; top: 82px; right: 24px; width: 310px; box-sizing: border-box; display: flex; flex-direction: column; gap: 9px; padding: 11px 13px; background: rgba(7,16,27,.94); border: 1px solid rgba(100,150,180,.25); border-radius: 10px; backdrop-filter: blur(16px); }
.replay-top { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.replay-date { font-size: 18px; font-weight: 750; display: flex; align-items: baseline; gap: 8px; }
.replay-date small { font-size: 10px; font-weight: 400; color: #7d92a5; }
.replay-scrub { display: flex; align-items: center; gap: 8px; }
.replay-scrub button { width: 26px; height: 22px; font-size: 9px; color: #dbe7f1; background: #142436; border: 1px solid #30475c; border-radius: 5px; }
.replay-scrub input { flex: 1; min-width: 0; height: 4px; margin: 0; -webkit-appearance: none; appearance: none; background: #1c2e40; border-radius: 2px; outline: none; cursor: pointer; }
.replay-scrub input::-webkit-slider-thumb { -webkit-appearance: none; width: 12px; height: 12px; border-radius: 50%; background: #36d5a1; border: 0; }
.replay-scrub input::-moz-range-thumb { width: 12px; height: 12px; border-radius: 50%; background: #36d5a1; border: 0; }
@media (max-width:850px) { .replay-control { width: 255px; right: 12px; } .immersive-feed.replay-shift { top: 365px; } }
.replay-movers { display: flex; flex-wrap: wrap; gap: 6px; font-size: 11px; font-variant-numeric: tabular-nums; }
.replay-movers span { padding: 2px 7px; border-radius: 5px; background: #0e1a28; }
.replay-movers .up { color: #36d99c; } .replay-movers .down { color: #ff596d; }
.replay-stat { font-size: 10px; line-height: 1.4; color: #aebfce; } .replay-stat b { color: #60dcb3; font-size: 12px; }
.replay-legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 9px; color: #7d92a5; }
.replay-legend i { display: inline-block; width: 12px; height: 4px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }
.replay-legend i.f { background: #36d99c; } .replay-legend i.d { background: #ffb36b; }
.immersive-feed.replay-shift { top: 345px; max-height: calc(100vh - 430px); }
.mode-switch { position: absolute; z-index: 26; top: 66px; left: 50%; transform: translateX(-50%); display: flex; gap: 2px; padding: 3px; background: rgba(7,16,27,.92); border: 1px solid rgba(100,150,180,.25); border-radius: 9px; backdrop-filter: blur(14px); }
.mode-switch button { padding: 6px 14px; font-size: 11px; color: #8fa4b6; background: transparent; border: 0; border-radius: 6px; }
.mode-switch button.on { color: #04120e; background: #36d5a1; font-weight: 700; }
.move-panel { width: 290px; }
.move-summary { display: flex; gap: 10px; font-size: 11px; color: #8fa4b6; margin: 8px 0 4px; font-variant-numeric: tabular-nums; }
.move-summary .up, .move-row em.up { color: #36d99c; } .move-summary .down, .move-row em.down { color: #ff596d; }
.move-head { margin-top: 10px; }
.move-row { grid-template-columns: 56px 1fr 64px; }
.replay-scrub button:disabled { opacity: .35; cursor: default; }
.replay-legend i.l { background: #ff596d; }
.link-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.link-clear { padding: 4px 9px; font-size: 10px; color: #ffb3bd; background: #2a1119; border: 1px solid #5a2431; border-radius: 5px; cursor: pointer; }
.link-clear:hover { background: #3a1621; }
.link-row.static { grid-template-columns: 78px 1fr 44px; cursor: default; }
.link-row.static:hover { background: transparent; }
.network-panel { width: 280px; }
.link-controls { display: flex; align-items: center; gap: 6px; margin: 10px 0 8px; flex-wrap: wrap; }
.link-controls button { padding: 4px 9px; font-size: 10px; color: #9fb3c4; background: #0e1a28; border: 1px solid #243749; border-radius: 5px; cursor: pointer; }
.link-controls button.on { color: #04120e; background: #36d5a1; border-color: #36d5a1; }
.link-controls label { margin-left: auto; font-size: 9px; color: #8fa4b6; display: flex; gap: 4px; align-items: center; }
.link-legend { display: flex; gap: 10px; flex-wrap: wrap; font-size: 9px; color: #8fa4b6; margin-bottom: 8px; }
.link-legend i { display: inline-block; width: 14px; height: 4px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }
.link-legend i.pos, .link-bar i.pos { background: #36d99c; }
.link-legend i.neg, .link-bar i.neg { background: #ff596d; }
.link-row { width: 100%; display: grid; grid-template-columns: 48px 1fr 44px; gap: 8px; align-items: center; padding: 6px 2px; background: transparent; border: 0; color: #e6eef5; font-size: 11px; cursor: pointer; text-align: left; }
.link-row:hover { background: rgba(100,150,180,.08); }
.link-row em { font-style: normal; text-align: right; font-variant-numeric: tabular-nums; }
.link-bar { height: 6px; background: #14212f; border-radius: 3px; overflow: hidden; }
.link-bar i { display: block; height: 100%; border-radius: 3px; transition: width .5s ease; }
.link-note { color: #8fa4b6; font-size: 10px; padding: 6px 0; }
.papers-head { margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(100,150,180,.15); }
.paper-row { display: block; padding: 8px 0; text-decoration: none; color: #dbe7f1; border-bottom: 1px solid rgba(100,150,180,.09); }
.paper-row span { display: block; font-size: 11px; line-height: 1.35; }
.paper-row small { display: block; margin-top: 3px; font-size: 9px; color: #7f95a8; }
.paper-row:hover span { color: #60dcb3; }
@media (max-width:850px) { .link-panel { width: 235px; top: 170px; left: 12px; max-height: 45vh; } }

.closest-terminal {
  position: absolute;
  z-index: 13;
  inset: 0;
  display: flex;
  flex-direction: column;
  padding: 90px 25px 25px;
  background: linear-gradient(180deg,rgba(5,9,15,.94),#05080d 25%);
  overflow-y: auto;
}

.terminal-top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 20px;
  max-width: 1200px;
  width: 100%;
  margin: 0 auto 25px;
}

.terminal-top h1 { margin: 7px 0; font-size: clamp(26px,4vw,44px); }
.terminal-top p { color: #8095a9; font-size: 12px; }

.back-button {
  background: #142436;
  border: 1px solid #30475c;
  border-radius: 7px;
  padding: 10px 14px;
  color: #dce8f2;
}

.back-button:hover { background: #1c3449; border-color: #42d9ae; }

.terminal-metrics {
  width: 100%;
  max-width: 1200px;
  margin: 0 auto 20px;
  display: grid;
  grid-template-columns: repeat(4,1fr);
  gap: 10px;
}

.terminal-card { padding: 17px; border-radius: 8px; background: #0b1420; border: 1px solid #1d3041; }
.terminal-card span { display: block; color: #8095a9; font-size: 9px; letter-spacing: 1px; }
.terminal-card strong { display: block; margin-top: 9px; font-size: 21px; }

.terminal-content {
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  border: 1px solid #1d3041;
  background: #09111b;
  border-radius: 9px;
  overflow: hidden;
}

.terminal-content h3 { margin: 0; padding: 17px; font-size: 12px; letter-spacing: 1px; }
.terminal-table-wrap { overflow-x: auto; max-height: 42vh; overflow-y: auto; }
.terminal-table { width: 100%; border-collapse: collapse; font-size: 11px; }
.terminal-table th { position: sticky; top: 0; background: #101d2a; color: #8196a9; text-align: left; font-size: 9px; letter-spacing: 1px; }
.terminal-table th, .terminal-table td { padding: 12px; border-bottom: 1px solid #182a39; white-space: nowrap; }
.terminal-table td { color: #d5e0eb; }
.terminal-table .price { color: #65e6bd; }
.terminal-table .buy-type { color: #36d99c; font-weight: 800; }
.terminal-table .sell-type { color: #ff596d; font-weight: 800; }

.zoom-hint {
  position: absolute;
  z-index: 10;
  bottom: 105px;
  left: 50%;
  transform: translateX(-50%);
  padding: 8px 13px;
  border-radius: 20px;
  color: #9ab0c3;
  background: rgba(7,15,25,.8);
  font-size: 9px;
  pointer-events: none;
}

.notice {
  position: absolute;
  z-index: 30;
  top: 125px;
  left: 50%;
  transform: translateX(-50%);
  padding: 12px 17px;
  background: #392b17;
  border: 1px solid #8b6427;
  border-radius: 7px;
  font-size: 11px;
}

.tasks-page { min-height: 100vh; max-width: 1150px; margin: auto; padding: 100px 24px 40px; }
.tasks-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 25px; }
.tasks-header h1 { margin: 8px 0; }
.task-add { display: flex; gap: 10px; margin-bottom: 20px; }
.task-add input { flex: 1; padding: 13px; color: white; background: #101b28; border: 1px solid #26394b; border-radius: 7px; }
.primary-button { padding: 10px 18px; border: 0; border-radius: 7px; background: #36d5a1; color: #04120e; font-weight: 700; }
.task-panel { background: #0b1420; border: 1px solid #1d3041; border-radius: 9px; overflow-x: auto; }
.feed-table { width: 100%; border-collapse: collapse; font-size: 11px; }
.feed-table th, .feed-table td { padding: 14px; border-bottom: 1px solid #1b2a39; text-align: left; }
.feed-table th { color: #8196a9; font-size: 9px; }
.delete-button { color: #ff7587; background: transparent; border: 0; }
.notice.warning { position: relative; transform: none; top: auto; left: auto; margin-bottom: 15px; }
.feed-count { color: #7c92a6; font-size: 10px; }
.empty-feed { color: #8295a9; padding: 25px !important; text-align: center !important; }
.footer-note { max-width: 1200px; margin: 15px auto 0; color: #657b8e; font-size: 9px; }

@media (max-width:850px) {
  .immersive-feed { width: 255px; right: 12px; top: 190px; max-height: 45vh; }
  .immersive-feed.company-mode,.immersive-feed.closest-mode { width: 275px; }
  .globe-search { left: 12px; width: 210px; }
  .globe-stats { left: 12px; bottom: 12px; }
  .globe-stat { min-width: 90px; padding: 8px; }
  .company-details { bottom: 12px; gap: 10px; }
}

@media (max-width:580px) {
  .globe-header { top: 65px; left: 12px; right: 12px; }
  .header-right { gap: 6px; }
  .feed-source { display: none; }
  .demo-control { top: 115px; right: 12px; padding: 8px; }
  .demo-badge { top: 170px; right: 12px; }
  .globe-search { top: 120px; }
  .zoom-indicator { top: 175px; right: 12px; }
  .immersive-feed { top: auto; bottom: 80px; left: 10px; right: 10px; width: auto; max-height: 32vh; }
  .immersive-feed.company-mode,.immersive-feed.closest-mode { width: auto; }
  .globe-stats { bottom: 10px; gap: 4px; }
  .globe-stat { min-width: 0; }
  .globe-stat strong { font-size: 13px; }
  .company-focus { top: 155px; }
  .company-details { left: 10px; right: 10px; transform: none; justify-content: space-around; gap: 5px; }
  .company-details div strong { font-size: 10px; }
  .terminal-metrics { grid-template-columns: repeat(2,1fr); }
  .terminal-top { flex-direction: column; }
  .closest-terminal { padding: 115px 12px 15px; }
}
`

// --------------------------------------------------
// HELPERS
// --------------------------------------------------

function getZoomMode(altitude, hasSelectedCompany) {
  if (altitude > 1.8) return ZOOM_MODES.GLOBAL
  if (altitude > 0.8) return ZOOM_MODES.REGIONAL
  if (altitude > ZOOM_LOCK_ALTITUDE) return ZOOM_MODES.COMPANY
  return hasSelectedCompany ? ZOOM_MODES.CLOSEST : ZOOM_MODES.COMPANY
}

function distanceKm(lat1, lon1, lat2, lon2) {
  const radians = value => value * Math.PI / 180
  const dLat = radians(lat2 - lat1)
  const dLon = radians(lon2 - lon1)

  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(radians(lat1)) *
    Math.cos(radians(lat2)) *
    Math.sin(dLon / 2) ** 2

  return 6371 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
}

function randomBetween(min, max) {
  return Math.random() * (max - min) + min
}

function makeUniqueId() {
  return `demo-${Date.now()}-${Math.random().toString(36).slice(2, 12)}`
}

// |r| 0 -> hairline, |r| 1 -> thick. Colour encodes the sign; links that are not statistically solid fade.
const POS_RGB = '54,217,156'
const NEG_RGB = '255,89,109'
const corrColor = (r, significant) =>
  `rgba(${r >= 0 ? POS_RGB : NEG_RGB},${significant ? 1 : 0.75})`
const corrStroke = r => 0.35 + Math.min(1, Math.abs(r)) * 2.6

// Bar height (globe radii) from the absolute daily % change: linear, capped so one outlier cannot hide the rest.
const barHeight = change => (change == null ? 0.005 : 0.012 + Math.min(Math.abs(change), 10) * 0.034)
const MOVE_STEP_MS = 3500        // time on each day in daily movement mode
const MOVE_TRANSITION_MS = 1500  // bars grow/shrink over this long, so each change is easy to follow
const REPLAY_STEP_MS = 5000
const REPLAY_MOVERS = 5
const fmtPct = value => `${value > 0 ? '+' : value < 0 ? '−' : ''}${Math.abs(value).toFixed(2)}%`
const TOP_LINKS_SHOWN = 5
const ASK_SUGGESTIONS = [
  'Which chip makers lead each other?',
  'Do big banks move with each other?',
  'Does Tesla lead anything the next day?',
  'If Coca-Cola and PepsiCo diverge, do they come back together?'
]
const RIPPLE_PEERS = 4
const RIPPLE_MAX = 40
const RIPPLE_LIFETIME = 4500
const RIPPLE_MIN_GAP = 1500
const NETWORK_MAX_LINES = 30
const NETWORK_MAX_PER_COMPANY = 3
const SELECTED_MAX_ARCS = 15

function getArcAltitude(id) {
  const text = String(id || '')
  let hash = 0

  for (let i = 0; i < text.length; i++) {
    hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0
  }

  return 0.16 + (Math.abs(hash) % 9) * 0.012
}

function makeSimulatedTrade(location, companyLocation, existingTrade) {
  const priceBase = Number(existingTrade?.price) ||
    ({
      AAPL: 270,
      MSFT: 500,
      NVDA: 190,
      AMZN: 250,
      GOOGL: 280,
      META: 650,
      TSLA: 450,
      AMD: 220,
      SONY: 25,
      TM: 190,
      HMC: 30,
      NTDOY: 15,
      TSM: 300,
      BABA: 150,
      SAP: 250
    }[location.symbol] || 100)

  const price = Number(
    (priceBase * randomBetween(0.997, 1.003)).toFixed(2)
  )

  const size = Math.floor(randomBetween(1, 1500))
  const notional = Number((price * size).toFixed(2))
  const side = Math.random() < 0.5 ? 'BUY' : 'SELL'

  return {
    id: makeUniqueId(),
    createdAt: Date.now(),
    symbol: location.symbol,
    company: location.company || location.name || location.symbol,
    price,
    size,
    notional,
    side,
    type: side,
    exchange: 'FAKEPACA',
    timestamp: new Date().toISOString(),
    simulated: true,
    buyer_city: null,
    buyer_country: null,
    buyer_lat: null,
    buyer_lng: null,
    origin_city: companyLocation.city,
    origin_country: companyLocation.country,
    origin_lat: companyLocation.lat,
    origin_lng: companyLocation.lng
  }
}

// Stable callbacks: these are not recreated every time React renders.
const polygonGeometry = feature => feature.geometry
const polygonCapColor = () => 'rgba(35,220,180,0.02)'
const polygonSideColor = () => 'rgba(35,220,180,0.02)'
const polygonStrokeColor = () => 'rgba(65,245,195,0.25)'
const polygonLabel = feature => feature.properties?.ADMIN || 'Country'
const arcColor = arc => arc.color
const arcAltitude = arc => arc.arcAltitude
const ringColor = ring =>
  ring.side === 'SELL' ? SELL_COLOR : BUY_COLOR

// --------------------------------------------------
// IMMERSIVE GLOBE
// --------------------------------------------------

function MarketGlobe({
  locations,
  trades,
  feed,
  arcEvents,
  selected,
  onSelect,
  connection,
  onModeChange,
  fakeMode,
  onToggleFakeMode,
  onRunResearch
}) {
  const globeRef = useRef(null)

  const [mode, setMode] = useState(ZOOM_MODES.GLOBAL)
  const [camera, setCamera] = useState({ lat: 20, lng: 0 })
  const [viewport, setViewport] = useState({
    width: window.innerWidth,
    height: window.innerHeight
  })

  const [search, setSearch] = useState('')
  const [searchOpen, setSearchOpen] = useState(false)
  const [historicalTrades, setHistoricalTrades] = useState([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [countryPolygons, setCountryPolygons] = useState([])

  // Every company pin on the map takes part in the research overlay.
  const mapSymbols = useMemo(
    () => Object.keys(locations).sort().join(","),
    [locations]
  )
  const selectedCompanyName = selected ? locations[selected]?.company || "" : ""

  // Research overlay: daily-return correlation links from the selected company, plus its top papers.
  const [corrWindow, setCorrWindow] = useState('1y')
  const onlySignificant = false
  const [showAllLinks, setShowAllLinks] = useState(false)

  // "Ask the globe": a plain-English question -> AI picks company pairs -> real correlations are computed and drawn.
  const [askText, setAskText] = useState('')
  const [askBusy, setAskBusy] = useState(false)
  const [askError, setAskError] = useState('')
  const [askResult, setAskResult] = useState(null)
  // Two modes: correlation (links between companies) and daily movement (one bar per company).
  const [viewMode, setViewMode] = useState('correlation')
  const [moveDay, setMoveDay] = useState(null)
  const [movePlaying, setMovePlaying] = useState(true)

  // "Replay last month": real daily moves, one trading day at a time.
  const [replayOn, setReplayOn] = useState(false)
  const [replayData, setReplayData] = useState(null)
  const [replayLoading, setReplayLoading] = useState(false)
  const [replayDay, setReplayDay] = useState(0)
  const [replayPlaying, setReplayPlaying] = useState(true)

  useEffect(() => {
    if ((!replayOn && viewMode !== 'movement') || replayData || !mapSymbols) return undefined

    const controller = new AbortController()
    setReplayLoading(true)

    fetch(`${API}/api/research/globe/replay?symbols=${mapSymbols}`, { signal: controller.signal })
      .then(response => response.json())
      .then(data => {
        setReplayData(data)
        setReplayDay(0)
      })
      .catch(error => {
        if (error.name !== 'AbortError') {
          setReplayData({ status: 'unavailable', days: [], moves: {}, peers: {}, note: 'Replay service unreachable.' })
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setReplayLoading(false)
      })

    return () => controller.abort()
  }, [replayOn, replayData, mapSymbols, viewMode])

  // The last date is only ever an outcome (peers are judged the day AFTER the leaders move), so it is not a step.
  const replayLength = Math.max(0, (replayData?.days?.length || 0) - 1)

  useEffect(() => {
    if (!replayOn || !replayPlaying || replayLength < 2) return undefined

    const timer = window.setInterval(() => {
      setReplayDay(day => (day + 1) % replayLength)
    }, REPLAY_STEP_MS)

    return () => window.clearInterval(timer)
  }, [replayOn, replayPlaying, replayLength])

  const askSymbols = useMemo(
    () => new Set((askResult?.pairs || []).flatMap(pair => [pair.a, pair.b])),
    [askResult]
  )
  const [corrData, setCorrData] = useState(null)
  const [corrLoading, setCorrLoading] = useState(false)
  const [paperData, setPaperData] = useState(null)
  const [paperLoading, setPaperLoading] = useState(false)

  useEffect(() => {
    if (!selected) {
      setCorrData(null)
      return undefined
    }

    const controller = new AbortController()
    setCorrLoading(true)
    setCorrData(null)

    fetch(
      `${API}/api/research/globe/correlations?symbol=${encodeURIComponent(selected)}&window=${corrWindow}&symbols=${mapSymbols}`,
      { signal: controller.signal }
    )
      .then(response => response.json())
      .then(data => setCorrData(data))
      .catch(error => {
        if (error.name !== 'AbortError') {
          setCorrData({ status: 'unavailable', links: [], note: 'Correlation service unreachable.' })
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setCorrLoading(false)
      })

    return () => controller.abort()
  }, [selected, corrWindow, mapSymbols])

  // Market-wide link network (every pair of tracked companies), shown when no company is selected.
  const [netData, setNetData] = useState(null)
  const [netLoading, setNetLoading] = useState(false)

  useEffect(() => {
    if (selected) return undefined

    const controller = new AbortController()
    setNetLoading(true)

    fetch(`${API}/api/research/globe/network?window=${corrWindow}&symbols=${mapSymbols}`, { signal: controller.signal })
      .then(response => response.json())
      .then(data => setNetData(data))
      .catch(error => {
        if (error.name !== 'AbortError') {
          setNetData({ status: 'unavailable', links: [], note: 'Correlation service unreachable.' })
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setNetLoading(false)
      })

    return () => controller.abort()
  }, [selected, corrWindow, mapSymbols])

  useEffect(() => {
    if (!selected) {
      setPaperData(null)
      return undefined
    }

    const controller = new AbortController()
    setPaperLoading(true)
    setPaperData(null)

    fetch(
      `${API}/api/research/globe/papers?symbol=${encodeURIComponent(selected)}&name=${encodeURIComponent(selectedCompanyName)}`,
      { signal: controller.signal }
    )
      .then(response => response.json())
      .then(data => setPaperData(data))
      .catch(error => {
        if (error.name !== 'AbortError') {
          setPaperData({ status: 'unavailable', papers: [], note: 'Paper search unreachable.' })
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setPaperLoading(false)
      })

    return () => controller.abort()
  }, [selected, selectedCompanyName])

  useEffect(() => {
    let resizeTimer = null

    const resize = () => {
      if (resizeTimer) window.clearTimeout(resizeTimer)

      resizeTimer = window.setTimeout(() => {
        setViewport({
          width: window.innerWidth,
          height: window.innerHeight
        })
      }, 150)
    }

    window.addEventListener('resize', resize)

    return () => {
      window.removeEventListener('resize', resize)
      if (resizeTimer) window.clearTimeout(resizeTimer)
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    async function loadCountryPolygons() {
      try {
        const response = await fetch(
          'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson'
        )

        if (!response.ok) throw new Error('Could not load country outlines')

        const geojson = await response.json()

        if (!cancelled) {
          setCountryPolygons(
            Array.isArray(geojson.features) ? geojson.features : []
          )
        }
      } catch (error) {
        console.error('Country polygon loading error:', error)
      }
    }

    loadCountryPolygons()

    return () => {
      cancelled = true
    }
  }, [])

  const selectedLocation = selected ? locations[selected] : null
  const selectedTrade = selected ? trades[selected] : null

  const points = useMemo(() => {
    return Object.values(locations)
      .filter(location => {
        const lat = Number(location.lat ?? location.latitude)
        const lng = Number(location.lon ?? location.longitude)
        return Number.isFinite(lat) && Number.isFinite(lng)
      })
      .map(location => {
        const symbol = location.symbol
        const trade = trades[symbol] || {}

        return {
          ...location,
          symbol,
          lat: Number(location.lat ?? location.latitude),
          lng: Number(location.lon ?? location.longitude),
          price: trade.price,
          size: trade.size || 0,
          hasTrade: trade.price != null,
          active: symbol === selected,
          asked: askSymbols.has(symbol)
        }
      })
  }, [locations, trades, selected, askSymbols])

  const rings = useMemo(() => {
    return feed.slice(0, MAX_RINGS).map(trade => {
      const location = locations[trade.symbol]
      if (!location) return null

      const lat = Number(location.lat ?? location.latitude)
      const lng = Number(location.lon ?? location.longitude)

      if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null

      return {
        lat,
        lng,
        maxR: trade.simulated ? 2.5 : 1.8,
        propagationSpeed: 0.8,
        repeatPeriod: 2000,
        side: trade.side
      }
    }).filter(Boolean)
  }, [feed, locations])

  const visibleLinks = useMemo(() => {
    const links = corrData?.links || []
    return onlySignificant ? links.filter(link => link.significant) : links
  }, [corrData, onlySignificant])

  const linkBySymbol = useMemo(() => {
    const map = {}
    for (const link of visibleLinks) map[link.symbol] = link
    return map
  }, [visibleLinks])

  // One line per linked company: thickness = |r|, colour = sign of r.
  const correlationArcs = useMemo(() => {
    const origin = selected ? locations[selected] : null
    if (!origin) return []

    const startLat = Number(origin.lat ?? origin.latitude)
    const startLng = Number(origin.lon ?? origin.longitude)
    if (!Number.isFinite(startLat) || !Number.isFinite(startLng)) return []

    return visibleLinks.slice(0, SELECTED_MAX_ARCS).map(link => {
      const target = locations[link.symbol]
      if (!target) return null

      const endLat = Number(target.lat ?? target.latitude)
      const endLng = Number(target.lon ?? target.longitude)
      if (!Number.isFinite(endLat) || !Number.isFinite(endLng)) return null

      return {
        id: `corr-${selected}-${link.symbol}`,
        link,
        from: selected,
        symbol: link.symbol,
        startLat,
        startLng,
        endLat,
        endLng,
        color: corrColor(link.r, link.significant),
        stroke: corrStroke(link.r),
        arcAltitude: 0.18 + Math.abs(link.r) * 0.5
      }
    }).filter(Boolean)
  }, [selected, locations, visibleLinks])

  // No company selected: the strongest real return correlations between all tracked companies.
  const networkArcs = useMemo(() => {
    // Strongest first, but at most a few lines per company so one hub (e.g. big US tech) cannot hide the rest of the map.
    const degree = {}
    const links = []
    for (const link of netData?.links || []) {
      if ((degree[link.a] || 0) >= NETWORK_MAX_PER_COMPANY || (degree[link.b] || 0) >= NETWORK_MAX_PER_COMPANY) continue
      degree[link.a] = (degree[link.a] || 0) + 1
      degree[link.b] = (degree[link.b] || 0) + 1
      links.push(link)
      if (links.length >= NETWORK_MAX_LINES) break
    }

    return links.map(link => {
      const from = locations[link.a]
      const to = locations[link.b]
      if (!from || !to) return null

      const startLat = Number(from.lat ?? from.latitude)
      const startLng = Number(from.lon ?? from.longitude)
      const endLat = Number(to.lat ?? to.latitude)
      const endLng = Number(to.lon ?? to.longitude)
      if (![startLat, startLng, endLat, endLng].every(Number.isFinite)) return null

      return {
        id: `net-${link.a}-${link.b}`,
        link,
        from: link.a,
        symbol: link.b,
        startLat,
        startLng,
        endLat,
        endLng,
        color: `rgba(${link.r >= 0 ? POS_RGB : NEG_RGB},${link.significant ? 0.5 : 0.28})`,
        stroke: corrStroke(link.r),
        arcAltitude: 0.18 + Math.abs(link.r) * 0.5
      }
    }).filter(Boolean)
  }, [netData, locations])

  // Ripples: every trade tick (real stream, or the FAKEPACA demo ticks) travels along that company's real correlation
  // links to its most-linked peers. Direction of travel = where the tick happened; colour = sign of the real r.
  // No buyer or seller locations are invented: the only geography is company headquarters.
  const [ripples, setRipples] = useState([])
  const rippleSeen = useRef(new Set())
  const rippleLast = useRef({})
  const rippleTimers = useRef(new Set())

  useEffect(() => {
    if (selected || !netData?.links?.length) return

    const fresh = []
    for (const trade of feed.slice(0, 12)) {
      const key = trade.id || `${trade.symbol}-${trade.timestamp}`
      if (rippleSeen.current.has(key)) continue
      rippleSeen.current.add(key)
      if (rippleSeen.current.size > 500) rippleSeen.current = new Set([...rippleSeen.current].slice(-200))

      const now = Date.now()
      if (now - (rippleLast.current[trade.symbol] || 0) < RIPPLE_MIN_GAP) continue
      rippleLast.current[trade.symbol] = now

      const origin = locations[trade.symbol]
      if (!origin) continue

      const peers = netData.links
        .filter(link => link.a === trade.symbol)
        .slice(0, RIPPLE_PEERS)

      for (const link of peers) {
        const peer = link.b
        const target = locations[peer]
        if (!target) continue

        const startLat = Number(origin.lat ?? origin.latitude)
        const startLng = Number(origin.lon ?? origin.longitude)
        const endLat = Number(target.lat ?? target.latitude)
        const endLng = Number(target.lon ?? target.longitude)
        if (![startLat, startLng, endLat, endLng].every(Number.isFinite)) continue

        fresh.push({
          id: `rip-${key}-${peer}`,
          link: { ...link, ripple: true },
          from: trade.symbol,
          symbol: peer,
          startLat,
          startLng,
          endLat,
          endLng,
          color: corrColor(link.r, true),
          stroke: corrStroke(link.r) * 0.8,
          arcAltitude: 0.2 + Math.abs(link.r) * 0.45
        })
      }
    }

    if (!fresh.length) return

    setRipples(current => [...fresh, ...current].slice(0, RIPPLE_MAX))

    const ids = new Set(fresh.map(arc => arc.id))
    const timer = window.setTimeout(() => {
      rippleTimers.current.delete(timer)
      setRipples(current => current.filter(arc => !ids.has(arc.id)))
    }, RIPPLE_LIFETIME)
    rippleTimers.current.add(timer)
  }, [feed, selected, netData, locations])

  useEffect(() => () => {
    for (const timer of rippleTimers.current) window.clearTimeout(timer)
    rippleTimers.current.clear()
  }, [])

  // For one day: the biggest movers, and for each its most-correlated peers: did the peer move the way the link predicts?
  const buildReplayDay = useCallback((data, day) => {
    const moves = {}
    const nextMoves = {}
    for (const [symbol, series] of Object.entries(data.moves || {})) {
      if (series[day] != null) moves[symbol] = series[day]
      if (series[day + 1] != null) nextMoves[symbol] = series[day + 1]
    }

    const movers = Object.entries(moves)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, REPLAY_MOVERS)
      .map(([symbol, ret]) => ({ symbol, ret }))

    const links = []
    const seen = new Set()
    for (const mover of movers) {
      for (const peer of data.peers?.[mover.symbol] || []) {
        const retB = nextMoves[peer.symbol]
        const key = [mover.symbol, peer.symbol].sort().join('-')
        if (retB == null || seen.has(key)) continue
        seen.add(key)

        links.push({
          a: mover.symbol,
          b: peer.symbol,
          r: peer.r,
          retA: mover.ret,
          retB,
          // no look-ahead: the leader moves on day t, the peer is judged on day t+1
          followed: Math.sign(mover.ret) * Math.sign(retB) === Math.sign(peer.r)
        })
      }
    }

    return { moves, movers, links }
  }, [])

  const replayFrame = useMemo(() => {
    if (!replayOn || !replayLength) return { moves: {}, movers: [], arcs: [] }

    const day = buildReplayDay(replayData, Math.min(replayDay, replayLength - 1))

    const arcs = day.links.map(link => {
      const from = locations[link.a]
      const to = locations[link.b]
      if (!from || !to) return null

      const startLat = Number(from.lat ?? from.latitude)
      const startLng = Number(from.lon ?? from.longitude)
      const endLat = Number(to.lat ?? to.latitude)
      const endLng = Number(to.lon ?? to.longitude)
      if (![startLat, startLng, endLat, endLng].every(Number.isFinite)) return null

      return {
        id: `replay-${link.a}-${link.b}`,
        link: { ...link, replay: true },
        from: link.a,
        symbol: link.b,
        startLat,
        startLng,
        endLat,
        endLng,
        color: link.followed ? 'rgba(54,217,156,0.95)' : 'rgba(255,179,107,0.9)',
        stroke: 0.4 + Math.min(Math.abs(link.retA), 6) / 6 * 2.2,
        arcAltitude: 0.18 + Math.abs(link.r) * 0.4
      }
    }).filter(Boolean)

    return { moves: day.moves, movers: day.movers, arcs }
  }, [replayOn, replayData, replayDay, replayLength, locations, buildReplayDay])

  // Over the whole month: how often did a leader's most-correlated peers actually move the predicted way?
  const replayStats = useMemo(() => {
    if (!replayLength) return null

    let followed = 0
    let total = 0
    for (let day = 0; day < replayLength; day++) {
      for (const link of buildReplayDay(replayData, day).links) {
        total += 1
        if (link.followed) followed += 1
      }
    }

    return total ? { followed, total, share: Math.round(followed / total * 100) } : null
  }, [replayData, replayLength, buildReplayDay])

  // Daily movement mode: simple % change, close to close, for the chosen trading day (default: the latest).
  // The zoomed-in company, as it stood on the replayed day: real adjusted closes and daily moves up to that day only.
  const companyReplay = useMemo(() => {
    if (!replayOn || !selected || !replayLength) return null
    const closes = replayData.close?.[selected]
    const pcts = replayData.pct?.[selected]
    if (!closes && !pcts) return null
    const day = Math.min(replayDay, replayLength - 1)
    const first = closes?.[0]
    const rows = []
    for (let i = 0; i <= day; i++) {
      const close = closes?.[i] ?? null
      rows.push({
        date: replayData.days[i],
        close,
        pct: pcts?.[i] ?? null,
        mtd: close != null && first ? (close / first - 1) * 100 : null
      })
    }
    const seen = rows.map(r => r.close).filter(v => v != null)
    const last = rows[rows.length - 1]
    return {
      date: last.date,
      close: last.close,
      dayPct: last.pct,
      mtd: last.mtd,
      low: seen.length ? Math.min(...seen) : null,
      high: seen.length ? Math.max(...seen) : null,
      spark: seen,
      rows: [...rows].reverse(),
      hasClose: seen.length > 0
    }
  }, [replayOn, selected, replayData, replayDay, replayLength])

  const replayDateLabel = iso =>
    iso ? new Date(`${iso}T00:00:00`).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' }) : '—'

  const moveDays = replayData?.days || []
  const moveIdx = moveDays.length ? Math.min(moveDay ?? moveDays.length - 1, moveDays.length - 1) : 0

  // Autoplay: step through every stored day slowly and wrap round, so the animation never stops (until paused).
  useEffect(() => {
    if (viewMode !== 'movement' || !movePlaying || moveDays.length < 2) return undefined

    const length = moveDays.length
    const timer = window.setInterval(() => {
      setMoveDay(day => (((day ?? length - 1) + 1) % length))
    }, MOVE_STEP_MS)

    return () => window.clearInterval(timer)
  }, [viewMode, movePlaying, moveDays.length])

  const movement = useMemo(() => {
    if (viewMode !== 'movement' || !moveDays.length) return null

    const changes = {}
    for (const [symbol, series] of Object.entries(replayData.pct || {})) {
      if (series[moveIdx] != null) changes[symbol] = series[moveIdx]
    }

    const ranked = Object.entries(changes).sort((a, b) => b[1] - a[1]).map(([symbol, pct]) => ({ symbol, pct }))
    const ups = ranked.filter(item => item.pct > 0)
    const downs = ranked.filter(item => item.pct < 0)

    return {
      changes,
      gainers: ranked.slice(0, 3).filter(item => item.pct > 0),
      losers: ranked.slice(-3).reverse().filter(item => item.pct < 0),
      up: ups.length,
      down: downs.length,
      average: ranked.length ? ranked.reduce((sum, item) => sum + item.pct, 0) / ranked.length : 0
    }
  }, [viewMode, replayData, moveIdx, moveDays.length])

  const askArcs = useMemo(() => (askResult?.pairs || []).map(pair => {
    const from = locations[pair.a]
    const to = locations[pair.b]
    if (!from || !to) return null

    const startLat = Number(from.lat ?? from.latitude)
    const startLng = Number(from.lon ?? from.longitude)
    const endLat = Number(to.lat ?? to.latitude)
    const endLng = Number(to.lon ?? to.longitude)
    if (![startLat, startLng, endLat, endLng].every(Number.isFinite)) return null

    return {
      id: `ask-${pair.a}-${pair.b}`,
      link: { ...pair, r: pair.r ?? 0, n: pair.n ?? 0, significant: true, ripple: false },
      from: pair.a,
      symbol: pair.b,
      startLat,
      startLng,
      endLat,
      endLng,
      color: pair.kind === 'pair_divergence'
        ? (pair.verdict === 'converges' ? 'rgba(54,217,156,0.95)' : pair.verdict === 'mixed' ? 'rgba(255,179,107,0.9)' : 'rgba(255,89,109,0.9)')
        : corrColor(pair.r, true),
      stroke: pair.kind === 'pair_divergence' ? 1.6 : corrStroke(pair.r) * 1.3,
      arcAltitude: pair.kind === 'pair_divergence' ? 0.3 : 0.22 + Math.abs(pair.r) * 0.5
    }
  }).filter(Boolean), [askResult, locations])

  const globeArcs = viewMode === 'movement'
    ? []
    : selected
    ? correlationArcs
    : askResult ? askArcs : replayOn ? replayFrame.arcs : [...networkArcs, ...ripples]

  const askGlobe = useCallback(async text => {
    const question = (text ?? askText).trim()
    if (question.length < 4 || askBusy) return

    setAskBusy(true)
    setAskError('')

    try {
      const response = await fetch(`${API}/api/research/globe/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question,
          window: corrWindow,
          companies: Object.values(locations).map(l => ({ symbol: l.symbol, company: l.company || l.symbol }))
        })
      })
      const data = await response.json()

      if (!response.ok || !data.pairs?.length) {
        setAskResult(null)
        setAskError(data.note || data.error || 'No answer found.')
        return
      }

      setAskResult(data)

      // Fly to the middle of the companies involved.
      const pins = data.pairs.flatMap(pair => [locations[pair.a], locations[pair.b]]).filter(Boolean)
      const lats = pins.map(p => Number(p.lat ?? p.latitude))
      const lngs = pins.map(p => Number(p.lon ?? p.longitude))
      const lat = (Math.min(...lats) + Math.max(...lats)) / 2
      const lng = (Math.min(...lngs) + Math.max(...lngs)) / 2
      const spread = Math.max(Math.max(...lats) - Math.min(...lats), (Math.max(...lngs) - Math.min(...lngs)) / 1.6)
      if ([lat, lng].every(Number.isFinite)) {
        globeRef.current?.pointOfView({ lat, lng, altitude: Math.min(2.6, Math.max(0.9, 0.5 + spread / 40)) }, 1600)
      }
    } catch (error) {
      setAskError('Could not reach the research service.')
    } finally {
      setAskBusy(false)
    }
  }, [askText, askBusy, corrWindow, locations])

  const clearAsk = useCallback(() => {
    setAskResult(null)
    setAskError('')
    setAskText('')
    globeRef.current?.pointOfView({ lat: 20, lng: 0, altitude: INITIAL_ALTITUDE }, 1400)
  }, [])

  const handleZoom = useCallback(({ lat, lng, altitude }) => {
    const nextAltitude = Number(altitude)
    if (!Number.isFinite(nextAltitude)) return

    if (nextAltitude <= ZOOM_LOCK_ALTITUDE && !selected) {
      globeRef.current?.pointOfView({
        lat,
        lng,
        altitude: ZOOM_LOCK_ALTITUDE + 0.01
      }, 350)

      setCamera({ lat, lng })
      setMode(ZOOM_MODES.COMPANY)
      onModeChange(ZOOM_MODES.COMPANY)
      return
    }

    const nextMode = getZoomMode(nextAltitude, Boolean(selected))
    setCamera({ lat, lng })
    setMode(nextMode)
    onModeChange(nextMode)
  }, [selected, onModeChange])

  const flyToCompany = useCallback(symbol => {
    const location = locations[symbol]
    if (!location || !globeRef.current) return

    const lat = Number(location.lat ?? location.latitude)
    const lng = Number(location.lon ?? location.longitude)

    if (!Number.isFinite(lat) || !Number.isFinite(lng)) return

    onSelect(symbol)
    setCamera({ lat, lng })
    setMode(ZOOM_MODES.COMPANY)
    onModeChange(ZOOM_MODES.COMPANY)

    globeRef.current.pointOfView({
      lat,
      lng,
      altitude: COMPANY_ALTITUDE
    }, 1800)
  }, [locations, onSelect, onModeChange])

  const returnToGlobal = useCallback(() => {
    onSelect(null)
    setMode(ZOOM_MODES.GLOBAL)
    setCamera({ lat: 20, lng: 0 })
    onModeChange(ZOOM_MODES.GLOBAL)

    globeRef.current?.pointOfView({
      lat: 20,
      lng: 0,
      altitude: INITIAL_ALTITUDE
    }, 1800)
  }, [onSelect, onModeChange])

  // Escape clears the selection.
  useEffect(() => {
    if (!selected) return undefined

    const onKey = event => { if (event.key === 'Escape') returnToGlobal() }
    window.addEventListener('keydown', onKey)

    return () => window.removeEventListener('keydown', onKey)
  }, [selected, returnToGlobal])

  useEffect(() => {
    if (mode !== ZOOM_MODES.CLOSEST || !selected) {
      setHistoricalTrades([])
      return
    }

    let cancelled = false

    async function loadHistory() {
      setHistoryLoading(true)

      try {
        const response = await fetch(
          `${API}/api/market/trades?symbol=${encodeURIComponent(selected)}`
        )

        if (!response.ok) throw new Error('Could not load trade history')

        const data = await response.json()

        if (!cancelled) {
          const records = Array.isArray(data) ? data : data.trades || []
          setHistoricalTrades(
            records.filter(trade => trade.symbol === selected)
          )
        }
      } catch (error) {
        console.error('Trade history error:', error)
        if (!cancelled) setHistoricalTrades([])
      } finally {
        if (!cancelled) setHistoryLoading(false)
      }
    }

    loadHistory()

    return () => {
      cancelled = true
    }
  }, [mode, selected])

  const contextualFeed = useMemo(() => {
    if (mode === ZOOM_MODES.GLOBAL) return feed

    if (mode === ZOOM_MODES.REGIONAL) {
      return feed.filter(trade => {
        const location = locations[trade.symbol]
        if (!location) return false

        const lat = Number(location.lat ?? location.latitude)
        const lng = Number(location.lon ?? location.longitude)

        if (!Number.isFinite(lat) || !Number.isFinite(lng)) return false

        return distanceKm(camera.lat, camera.lng, lat, lng) < 4500
      })
    }

    return feed.filter(trade => trade.symbol === selected)
  }, [feed, mode, locations, camera, selected])

  const searchResults = useMemo(() => {
    if (!search.trim()) return []

    const term = search.toLowerCase()

    return Object.values(locations)
      .filter(location =>
        `${location.symbol} ${location.company || location.name || ''}`
          .toLowerCase()
          .includes(term)
      )
      .slice(0, 7)
  }, [search, locations])

  const activeCompany = selectedLocation || {
    symbol: selected,
    company: selected
  }

  const companyTrades = useMemo(() => [
    ...feed.filter(trade => trade.symbol === selected),
    ...historicalTrades.filter(trade => !trade.simulated)
  ], [feed, historicalTrades, selected])

  const companyTotalVolume = useMemo(
    () => companyTrades.reduce(
      (sum, trade) => sum + (Number(trade.size) || 0),
      0
    ),
    [companyTrades]
  )

  const companyTotalNotional = useMemo(
    () => companyTrades.reduce(
      (sum, trade) => sum + (
        Number(trade.notional) ||
        (Number(trade.price) || 0) * (Number(trade.size) || 0)
      ),
      0
    ),
    [companyTrades]
  )

  const totalShares = useMemo(
    () => Object.values(trades).reduce(
      (sum, trade) => sum + (Number(trade.size) || 0),
      0
    ),
    [trades]
  )

  const feedTitle =
    mode === ZOOM_MODES.GLOBAL
      ? 'GLOBAL LIVE TRADES'
      : mode === ZOOM_MODES.REGIONAL
        ? 'REGIONAL LIVE TRADES'
        : 'COMPANY LIVE TRADES'

  return (
    <div className="market-experience">
      <div className="globe-container">
        <Globe
          ref={globeRef}
          width={viewport.width}
          height={viewport.height}
          backgroundColor="rgba(0,0,0,0)"
          globeImageUrl="https://unpkg.com/three-globe/example/img/earth-dark.jpg"
          bumpImageUrl="https://unpkg.com/three-globe/example/img/earth-topology.png"

          showAtmosphere
          atmosphereColor="#36e6bd"
          atmosphereAltitude={0.15}

          polygonsData={countryPolygons}
          polygonGeoJsonGeometry={polygonGeometry}
          polygonAltitude={0.004}
          polygonCapColor={polygonCapColor}
          polygonSideColor={polygonSideColor}
          polygonStrokeColor={polygonStrokeColor}
          polygonStrokeResolution={1}
          polygonLabel={polygonLabel}

          pointsData={points}
          pointLat="lat"
          pointLng="lng"
          pointAltitude={point => movement
            ? barHeight(movement.changes[point.symbol])
            : point.active ? 0.13 : point.asked ? 0.11 : 0.065}
          pointRadius={point =>
            movement ? (movement.changes[point.symbol] == null ? 0.2 : point.active ? 0.7 : 0.5) :
            point.active ? 0.65 : point.asked ? 0.6
              : replayOn && !selected && replayLength ? 0.2 + Math.min(Math.abs(replayFrame.moves[point.symbol] || 0), 6) * 0.07
              : point.hasTrade ? 0.38 : 0.23
          }
          pointColor={point => {
            if (movement) {
              const change = movement.changes[point.symbol]
              if (change == null) return '#3a4a5c'
              return change >= 0 ? '#36d99c' : '#ff596d'
            }
            if (point.active) return '#ffffff'
            if (askResult && !selected) return point.asked ? '#ffffff' : '#3a4a5c'
            if (replayOn && !selected && replayLength) {
              const ret = replayFrame.moves[point.symbol]
              if (ret == null) return '#3a4a5c'
              const alpha = 0.45 + Math.min(Math.abs(ret) / 4, 0.55)
              return `rgba(${ret >= 0 ? POS_RGB : NEG_RGB},${alpha})`
            }
            const link = linkBySymbol[point.symbol]
            if (link) return `rgb(${link.r >= 0 ? POS_RGB : NEG_RGB})`
            if (!point.hasTrade) return '#64748b'
            return '#45e3b1'
          }}
          pointResolution={movement ? 12 : 6}
          pointsTransitionDuration={movement ? MOVE_TRANSITION_MS : 900}
          pointLabel={point => {
            if (movement) {
              const change = movement.changes[point.symbol]
              return `${point.symbol} · ${point.company || point.name || ''} · ${
                change == null ? 'no data' : `${fmtPct(change)} on ${moveDays[moveIdx]}`
              }`
            }
            const link = linkBySymbol[point.symbol]
            return `${point.symbol} · ${point.company || point.name || ''} · ${
              point.hasTrade ? money(point.price) : 'Awaiting trade'
            }${link ? ` · r = ${link.r.toFixed(2)} with ${selected}` : ''}`
          }}
          onPointClick={point =>
            point.symbol === selected ? returnToGlobal() : flyToCompany(point.symbol)
          }
          onGlobeClick={() => { if (selected) returnToGlobal() }}

          ringsData={rings}
          ringLat="lat"
          ringLng="lng"
          ringMaxRadius="maxR"
          ringPropagationSpeed="propagationSpeed"
          ringRepeatPeriod="repeatPeriod"
          ringColor={ringColor}

          arcsData={globeArcs}
          arcStartLat="startLat"
          arcStartLng="startLng"
          arcEndLat="endLat"
          arcEndLng="endLng"
          arcColor={arcColor}
          arcAltitude={arcAltitude}
          arcStroke={arc => arc.stroke ?? 0.4}
          arcDashLength={arc => arc.link ? 0.8 : 0.4}
          arcDashGap={arc => arc.link ? 0.06 : 0.05}
          arcDashInitialGap={0}
          arcDashAnimateTime={arc => arc.link?.replay ? 3800 : arc.link?.ripple ? 1400 : arc.id?.startsWith('ask-') ? 2600 : arc.link ? 4000 : 1000}

          arcLabel={arc => arc.link?.replay
            ? `${arc.from} ${fmtPct(arc.link.retA)} today → ${arc.symbol} ${fmtPct(arc.link.retB)} next day · ${arc.link.followed ? 'peer followed' : 'peer diverged'} (prior-year next-day r = ${arc.link.r.toFixed(2)})`
            : arc.link
            ? `${arc.from} today → ${arc.symbol} next day · r = ${arc.link.r.toFixed(2)} · ${arc.link.n} days${arc.link.significant ? '' : ' · not significant'}`
            : `${arc.symbol} · ${arc.side} · ${arc.buyer} · ${money(arc.notional)}`}

          onZoom={handleZoom}
          animateIn={false}
        />
      </div>

      <header className="globe-header">
        <div className="brand">
          <div className="brand-mark">W</div>
          <div>
            <div className="brand-name">WHACK<span>26</span></div>
            <div className="brand-sub">GLOBAL FINANCIAL INTELLIGENCE</div>
          </div>
        </div>

        <div className="header-right">
          <div className={`connection-pill ${fakeMode ? 'live' : connection}`}>
            <i />
            {fakeMode
              ? 'SIMULATED STREAM'
              : connection === 'live'
                ? 'LIVE STREAM'
                : connection.toUpperCase()}
          </div>
          <span className="feed-source">
            {fakeMode ? 'FAKEPACA DEMO' : 'ALPACA · IEX'}
          </span>
        </div>
      </header>

      {viewMode === 'correlation' && !selected && mode !== ZOOM_MODES.CLOSEST && (
        <div className="replay-control">
          <div className="replay-top">
            <div className="demo-copy">
              <strong>REPLAY THE LAST MONTH</strong>
              <small>
                {!replayOn
                  ? 'Real daily moves, one day at a time'
                  : replayLoading
                    ? 'Loading real prices…'
                    : 'Real closing prices, adjusted'}
              </small>
            </div>

            <button
              className={`demo-toggle ${replayOn ? 'active' : ''}`}
              onClick={() => setReplayOn(on => !on)}
              aria-label="Toggle last-month replay"
              aria-pressed={replayOn}
            >
              <span />
            </button>
          </div>

          {replayOn && replayLength > 0 && (
            <>
              <div className="replay-date">
                {new Date(`${replayData.days[replayDay]}T00:00:00`).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' })}
                <small>day {replayDay + 1} of {replayLength} · peers judged {new Date(`${replayData.days[replayDay + 1]}T00:00:00`).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })}</small>
              </div>

              <div className="replay-scrub">
                <button onClick={() => setReplayPlaying(p => !p)} aria-label={replayPlaying ? 'Pause' : 'Play'}>
                  {replayPlaying ? '❚❚' : '▶'}
                </button>
                <input
                  type="range"
                  min={0}
                  max={replayLength - 1}
                  value={replayDay}
                  onChange={event => {
                    setReplayPlaying(false)
                    setReplayDay(Number(event.target.value))
                  }}
                />
              </div>

              <div className="replay-movers">
                {replayFrame.movers.slice(0, 4).map(m => (
                  <span key={m.symbol} className={m.ret >= 0 ? 'up' : 'down'}>
                    {m.symbol} {fmtPct(m.ret)}
                  </span>
                ))}
              </div>

              {replayStats && (
                <div
                  className="replay-stat"
                  title="For each day's biggest movers, we take the 3 companies whose NEXT-day move they best predicted over the year before this month. 'Followed' means that peer, the day after, moved the way that earlier pattern predicts. No same-day data is used."
                >
                  The next day, predicted peers moved as expected on <b>{replayStats.share}%</b> of {replayStats.total} links this month
                </div>
              )}

              <div className="replay-legend">
                <span><i className="f" /> peer followed</span>
                <span><i className="d" /> peer diverged</span>
                <span>line width = leader move</span>
              </div>
            </>
          )}

          {replayOn && !replayLoading && replayData && !replayLength && (
            <div className="link-note">{replayData.note || 'No replay data available.'}</div>
          )}
        </div>
      )}


      {mode !== ZOOM_MODES.CLOSEST && (
        <div className="globe-search">
          <input
            value={search}
            onChange={event => {
              setSearch(event.target.value)
              setSearchOpen(true)
            }}
            onFocus={() => setSearchOpen(true)}
            onKeyDown={event => {
              if (event.key === 'Escape') {
                setSearchOpen(false)
                setSearch('')
              }
            }}
            placeholder="Search companies or tickers..."
          />

          {searchOpen && searchResults.length > 0 && (
            <div className="search-results">
              {searchResults.map(location => (
                <button
                  className="search-result"
                  key={location.symbol}
                  onClick={() => {
                    flyToCompany(location.symbol)
                    setSearch('')
                    setSearchOpen(false)
                  }}
                >
                  <strong>{location.symbol}</strong>
                  <small>{location.company || location.name || ''}</small>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="zoom-indicator">
        <div className="eyebrow">EXPLORATION MODE</div>
        <strong>{ZOOM_LABELS[mode]}</strong>
      </div>

      {mode !== ZOOM_MODES.CLOSEST && (
        <aside className={`immersive-feed ${mode === ZOOM_MODES.COMPANY ? 'company-mode' : ''} ${replayOn && !selected ? 'replay-shift' : ''}`}>
          <div className="overlay-heading">
            <div>
              <div className="panel-kicker">MARKET EVENT STREAM</div>
              <h3>{feedTitle}</h3>
            </div>
            <span className="feed-status">
              ● {fakeMode ? 'DEMO' : 'LIVE'}
            </span>
          </div>

          <div className="feed-list">
            {contextualFeed.length ? (
              contextualFeed.slice(0, 35).map((trade, index) => (
                <button
                  className="feed-item"
                  key={trade.id || `${trade.symbol}-${trade.timestamp}-${index}`}
                  onClick={() => flyToCompany(trade.symbol)}
                >
                  <strong>
                    {trade.symbol}
                    {trade.simulated ? ' · SIM' : ''}
                  </strong>

                  <span className={`feed-price ${trade.side === 'SELL' ? 'sell' : ''}`}>
                    {money(trade.price)}
                  </span>

                  <small>
                    {Number(trade.size || 0).toLocaleString()} shares
                    {' · '}
                    {time(trade.timestamp)}
                  </small>

                  <small className="feed-notional">
                    {money(trade.notional)}
                  </small>

                  {trade.simulated && (
                    <small className={`feed-side ${trade.side === 'SELL' ? 'sell' : 'buy'}`}>
                      {trade.side} · {trade.origin_city} → {trade.buyer_city}
                    </small>
                  )}
                </button>
              ))
            ) : (
              <div className="feed-empty">
                Waiting for live trades. The market may be closed.
                <br /><br />
                Use Replay the last month to explore real recent moves.
              </div>
            )}
          </div>

          <div className="feed-footer">
            {contextualFeed.length} EVENTS ·
            {mode === ZOOM_MODES.GLOBAL
              ? ' ALL TRACKED SYMBOLS'
              : mode === ZOOM_MODES.REGIONAL
                ? ' VISIBLE REGION'
                : ` ${selected}`}
          </div>
        </aside>
      )}

      {selected && (
        <>
          {(mode === ZOOM_MODES.COMPANY || mode === ZOOM_MODES.CLOSEST) && (
            <div className="company-focus">
              <div className="panel-kicker">COMPANY INTELLIGENCE</div>
              <h2>{selected}</h2>
              <p>{activeCompany.company || activeCompany.name || selected}</p>
              {companyReplay?.hasClose ? (
                <>
                  <div className="focus-price">{money(companyReplay.close)}</div>
                  <div className="focus-caption">
                    REPLAY · {replayDateLabel(companyReplay.date).toUpperCase()} · ADJUSTED CLOSE · USD
                  </div>
                  <div className={`focus-change ${companyReplay.dayPct >= 0 ? 'up' : 'down'}`}>
                    {companyReplay.dayPct != null ? `${fmtPct(companyReplay.dayPct)} on the day` : ''}
                    {companyReplay.mtd != null ? ` · ${fmtPct(companyReplay.mtd)} since month start` : ''}
                  </div>
                </>
              ) : (
                <>
                  <div className="focus-price">
                    {selectedTrade?.price != null ? money(selectedTrade.price) : '—'}
                  </div>
                  <div className="focus-caption">LATEST OBSERVED TRADE · USD</div>
                </>
              )}
            </div>
          )}

          {mode === ZOOM_MODES.COMPANY && (
            <div className="company-details">
              <div><span>SECTOR</span><strong>{activeCompany.sector || '—'}</strong></div>
              <div><span>LOCATION</span><strong>{activeCompany.city || '—'}</strong></div>
              <div><span>SHARES</span><strong>{compact(selectedTrade?.size || 0)}</strong></div>
              <div><span>NOTIONAL</span><strong>{money(selectedTrade?.notional || 0)}</strong></div>
            </div>
          )}
        </>
      )}

      {mode === ZOOM_MODES.CLOSEST && selected && (
        <section className="closest-terminal">
          <div className="terminal-top">
            <div>
              <div className="panel-kicker">INDIVIDUAL COMPANY TRADING TERMINAL</div>
              <h1>{selected} · {activeCompany.company || selected}</h1>
              <p>{activeCompany.sector || 'Company'} · {activeCompany.city || 'Location unavailable'}</p>
            </div>

            <button className="back-button" onClick={returnToGlobal}>
              ← Return to Globe
            </button>
          </div>

          <div className="terminal-replay">
            {!replayOn ? (
              <>
                <span>View this company's real daily data from the last month, day by day.</span>
                <button onClick={() => { setReplayOn(true); setReplayPlaying(false) }}>Replay last month</button>
              </>
            ) : replayLoading ? (
              <span>Loading real prices…</span>
            ) : !replayLength ? (
              <>
                <span>{replayData?.note || 'Replay data is unavailable.'}</span>
                <button onClick={() => setReplayOn(false)}>Back to live</button>
              </>
            ) : (
              <>
                <button onClick={() => setReplayPlaying(p => !p)} aria-label={replayPlaying ? 'Pause' : 'Play'}>
                  {replayPlaying ? '❚❚' : '▶'}
                </button>
                <input
                  type="range"
                  min={0}
                  max={replayLength - 1}
                  value={replayDay}
                  onChange={event => {
                    setReplayPlaying(false)
                    setReplayDay(Number(event.target.value))
                  }}
                />
                <span>
                  <b>{replayDateLabel(replayData.days[replayDay])}</b> · day {replayDay + 1} of {replayLength}
                </span>
                <button onClick={() => setReplayOn(false)}>Back to live</button>
              </>
            )}
          </div>

          {companyReplay ? (
            <div className="terminal-metrics">
              <div className="terminal-card">
                <span>ADJUSTED CLOSE · {replayDateLabel(companyReplay.date).toUpperCase()}</span>
                <strong>{companyReplay.close != null ? money(companyReplay.close) : '—'}</strong>
              </div>
              <div className="terminal-card">
                <span>DAY CHANGE</span>
                <strong className={companyReplay.dayPct >= 0 ? 'up' : 'down'}>
                  {companyReplay.dayPct != null ? fmtPct(companyReplay.dayPct) : '—'}
                </strong>
              </div>
              <div className="terminal-card">
                <span>SINCE MONTH START</span>
                <strong className={companyReplay.mtd >= 0 ? 'up' : 'down'}>
                  {companyReplay.mtd != null ? fmtPct(companyReplay.mtd) : '—'}
                </strong>
              </div>
              <div className="terminal-card">
                <span>RANGE SO FAR</span>
                <strong>
                  {companyReplay.low != null ? `${money(companyReplay.low)} – ${money(companyReplay.high)}` : '—'}
                </strong>
              </div>
            </div>
          ) : (
            <div className="terminal-metrics">
              <div className="terminal-card">
                <span>AVAILABLE TRADES</span>
                <strong>{compact(companyTrades.length)}</strong>
              </div>
              <div className="terminal-card">
                <span>TOTAL SHARES</span>
                <strong>{compact(companyTotalVolume)}</strong>
              </div>
              <div className="terminal-card">
                <span>TOTAL NOTIONAL</span>
                <strong>{money(companyTotalNotional)}</strong>
              </div>
              <div className="terminal-card">
                <span>LATEST PRICE</span>
                <strong>{selectedTrade?.price != null ? money(selectedTrade.price) : '—'}</strong>
              </div>
            </div>
          )}

          <div className="terminal-content">
            {companyReplay && (
              <>
                <h3>DAILY HISTORY · UP TO {replayDateLabel(companyReplay.date).toUpperCase()}</h3>

                {companyReplay.spark.length > 1 && (() => {
                  const lo = Math.min(...companyReplay.spark)
                  const hi = Math.max(...companyReplay.spark)
                  const span = hi - lo || 1
                  const pts = companyReplay.spark
                    .map((v, i) => `${(i / (replayLength - 1)) * 100},${54 - ((v - lo) / span) * 50}`)
                    .join(' ')
                  return (
                    <svg className="terminal-spark" viewBox="0 0 100 56" preserveAspectRatio="none" aria-label="Closing price so far">
                      <polyline points={pts} fill="none" stroke="#36d5a1" strokeWidth="1.2" vectorEffect="non-scaling-stroke" />
                    </svg>
                  )
                })()}

                <div className="terminal-table-wrap">
                  <table className="terminal-table">
                    <thead>
                      <tr>
                        <th>DATE</th>
                        <th>ADJ. CLOSE</th>
                        <th>DAY CHANGE</th>
                        <th>SINCE MONTH START</th>
                      </tr>
                    </thead>
                    <tbody>
                      {companyReplay.rows.map(row => (
                        <tr key={row.date}>
                          <td>{replayDateLabel(row.date)}</td>
                          <td className="price">{row.close != null ? money(row.close) : '—'}</td>
                          <td className={row.pct == null ? '' : row.pct >= 0 ? 'up' : 'down'}>{row.pct != null ? fmtPct(row.pct) : '—'}</td>
                          <td className={row.mtd == null ? '' : row.mtd >= 0 ? 'up' : 'down'}>{row.mtd != null ? fmtPct(row.mtd) : '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}

            {!companyReplay && (
            <>
            <h3>AVAILABLE TRADE HISTORY {historyLoading ? '· LOADING' : ''}</h3>

            <div className="terminal-table-wrap">
              <table className="terminal-table">
                <thead>
                  <tr>
                    <th>TIME</th>
                    <th>PRICE</th>
                    <th>SHARES</th>
                    <th>NOTIONAL</th>
                    <th>EXCHANGE</th>
                    <th>TYPE</th>
                  </tr>
                </thead>

                <tbody>
                  {companyTrades.length ? companyTrades.map((trade, index) => (
                    <tr key={trade.id || `${trade.timestamp}-${index}`}>
                      <td>{dateTime(trade.timestamp)}</td>
                      <td className="price">{money(trade.price)}</td>
                      <td>{Number(trade.size || 0).toLocaleString()}</td>
                      <td>{money(trade.notional ?? trade.price * trade.size)}</td>
                      <td>{trade.exchange || '—'}</td>
                      <td className={trade.simulated
                        ? trade.side === 'SELL' ? 'sell-type' : 'buy-type'
                        : ''}>
                        {trade.simulated ? trade.side : 'REAL'}
                      </td>
                    </tr>
                  )) : (
                    <tr>
                      <td colSpan="6">
                        {historyLoading
                          ? 'Loading available trade records...'
                          : 'No trade records currently available.'}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            </>
            )}
          </div>

          <div className="footer-note">
            {companyReplay
              ? 'Replay shows real split- and dividend-adjusted daily closes, only up to the selected day. Switch back to live for streaming trade records.'
              : 'Real records come from your backend. Simulated records are generated locally and are not actual market transactions.'}
          </div>
        </section>
      )}

      {viewMode === 'correlation' && !selected && (
        <aside className="link-panel network-panel">
          <div className="panel-kicker">NEXT-DAY LEAD–LAG NETWORK</div>

          <div className="link-controls">
            {['3m', '1y', '3y'].map(w => (
              <button
                key={w}
                className={corrWindow === w ? 'on' : ''}
                onClick={() => setCorrWindow(w)}
              >
                {w.toUpperCase()}
              </button>
            ))}
          </div>

          <div className="link-legend">
            <span><i className="pos" /> follows same way</span>
            <span><i className="neg" /> follows opposite</span>
            <span>thicker = stronger</span>
          </div>

          {netLoading && <div className="link-note">Calculating correlations…</div>}
          {!netLoading && !networkArcs.length && (
            <div className="link-note">{netData?.note || 'No links to show.'}</div>
          )}

          {networkArcs.slice(0, 5).map(arc => (
            <div key={arc.id} className="link-row static">
              <strong>{arc.from}→{arc.symbol}</strong>
              <span className="link-bar">
                <i
                  className={arc.link.r >= 0 ? 'pos' : 'neg'}
                  style={{ width: `${Math.min(100, Math.abs(arc.link.r) * 100)}%`, opacity: arc.link.significant ? 1 : 0.45 }}
                />
              </span>
              <em>{arc.link.r > 0 ? '+' : ''}{arc.link.r.toFixed(2)}</em>
            </div>
          ))}

          <div
            className="link-note"
            title="Arrows run from the leader (its move today) to the follower (its move the next trading day). Faint lines are real next-day return correlations. Bright flashes are live trade ticks travelling from a leader to its followers (expected co-movement, not money flow)."
          >
            Leader today → follower next day · click a company
          </div>
        </aside>
      )}

      {viewMode === 'correlation' && selected && mode !== ZOOM_MODES.CLOSEST && (
        <aside className="link-panel">
          <div className="link-head">
            <div className="panel-kicker">{selected} TODAY → OTHERS NEXT DAY</div>
            <button className="link-clear" onClick={returnToGlobal} title="Clear selection (Esc)">
              ✕ Clear
            </button>
          </div>

          <div className="link-controls">
            {['3m', '1y', '3y'].map(w => (
              <button
                key={w}
                className={corrWindow === w ? 'on' : ''}
                onClick={() => setCorrWindow(w)}
              >
                {w.toUpperCase()}
              </button>
            ))}
            <span className="link-legend inline">
              <span><i className="pos" /> same way</span>
              <span><i className="neg" /> opposite</span>
            </span>
          </div>

          {corrLoading && <div className="link-note">Calculating correlations…</div>}

          {!corrLoading && corrData && !visibleLinks.length && (
            <div className="link-note">{corrData.note || 'No links to show.'}</div>
          )}

          {(showAllLinks ? visibleLinks : visibleLinks.slice(0, TOP_LINKS_SHOWN)).map(link => (
            <button
              key={link.symbol}
              className="link-row"
              onClick={() => flyToCompany(link.symbol)}
              title={`${link.n} days · p = ${link.p_value < 0.001 ? '<0.001' : link.p_value.toFixed(3)}${link.significant ? '' : ' · not significant'}`}
            >
              <strong>{link.symbol}</strong>
              <span className="link-bar">
                <i
                  className={link.r >= 0 ? 'pos' : 'neg'}
                  style={{ width: `${Math.min(100, Math.abs(link.r) * 100)}%`, opacity: link.significant ? 1 : 0.45 }}
                />
              </span>
              <em>{link.r > 0 ? '+' : ''}{link.r.toFixed(2)}</em>
            </button>
          ))}

          {visibleLinks.length > TOP_LINKS_SHOWN && (
            <button className="link-more" onClick={() => setShowAllLinks(v => !v)}>
              {showAllLinks ? 'Show fewer' : `Show ${visibleLinks.length - TOP_LINKS_SHOWN} more`}
            </button>
          )}

          <div className="panel-kicker papers-head">TOP RESEARCH PAPERS</div>

          {paperLoading && <div className="link-note">Searching literature…</div>}

          {!paperLoading && paperData && !paperData.papers?.length && (
            <div className="link-note">{paperData.note || 'No papers found.'}</div>
          )}

          {(paperData?.papers || []).map((paper, index) => (
            <a
              key={`${paper.title}-${index}`}
              className="paper-row"
              href={paper.url || undefined}
              target="_blank"
              rel="noreferrer"
              title={paper.title}
            >
              <span>{paper.title}</span>
              <small>
                {[paper.year, paper.venue, paper.cited_by != null ? `${paper.cited_by} cited` : null]
                  .filter(Boolean)
                  .join(' · ')}
              </small>
            </a>
          ))}
        </aside>
      )}

      {mode !== ZOOM_MODES.CLOSEST && (
        <div className="mode-switch" role="tablist" aria-label="Globe mode">
          <button
            role="tab"
            aria-selected={viewMode === 'correlation'}
            className={viewMode === 'correlation' ? 'on' : ''}
            onClick={() => setViewMode('correlation')}
          >
            Correlation
          </button>
          <button
            role="tab"
            aria-selected={viewMode === 'movement'}
            className={viewMode === 'movement' ? 'on' : ''}
            onClick={() => setViewMode('movement')}
          >
            Daily movement
          </button>
        </div>
      )}

      {viewMode === 'movement' && mode !== ZOOM_MODES.CLOSEST && (
        <aside className="link-panel move-panel">
          <div className="panel-kicker">DAILY MOVEMENT</div>

          {!movement && (
            <div className="link-note">
              {replayData && !moveDays.length ? replayData.note || 'No data available.' : 'Loading real prices…'}
            </div>
          )}

          {movement && (
            <>
              <div className="replay-date">
                {new Date(`${moveDays[moveIdx]}T00:00:00`).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' })}
                <small>day {moveIdx + 1} of {moveDays.length}{movePlaying ? ' · looping' : ' · paused'}</small>
              </div>

              <div className="replay-scrub">
                <button onClick={() => setMovePlaying(p => !p)} aria-label={movePlaying ? 'Pause' : 'Play'}>
                  {movePlaying ? '❚❚' : '▶'}
                </button>
                <button
                  onClick={() => setMoveDay((moveIdx - 1 + moveDays.length) % moveDays.length)}
                  aria-label="Previous day"
                >
                  ◀
                </button>
                <input
                  type="range"
                  min={0}
                  max={moveDays.length - 1}
                  value={moveIdx}
                  onChange={event => setMoveDay(Number(event.target.value))}
                />
                <button
                  onClick={() => setMoveDay((moveIdx + 1) % moveDays.length)}
                  aria-label="Next day"
                >
                  ▶
                </button>
              </div>

              <div className="move-summary">
                <span className="up">{movement.up} up</span>
                <span className="down">{movement.down} down</span>
                <span>avg {fmtPct(movement.average)}</span>
              </div>

              {movement.gainers.length > 0 && (
                <>
                  <div className="panel-kicker move-head">BIGGEST GAINS</div>
                  {movement.gainers.map(item => (
                    <button key={item.symbol} className="link-row move-row" onClick={() => flyToCompany(item.symbol)}>
                      <strong>{item.symbol}</strong>
                      <span />
                      <em className="up">{fmtPct(item.pct)}</em>
                    </button>
                  ))}
                </>
              )}

              {movement.losers.length > 0 && (
                <>
                  <div className="panel-kicker move-head">BIGGEST LOSSES</div>
                  {movement.losers.map(item => (
                    <button key={item.symbol} className="link-row move-row" onClick={() => flyToCompany(item.symbol)}>
                      <strong>{item.symbol}</strong>
                      <span />
                      <em className="down">{fmtPct(item.pct)}</em>
                    </button>
                  ))}
                </>
              )}

              <div className="replay-legend">
                <span><i className="f" /> gain</span>
                <span><i className="l" /> loss</span>
                <span>bar height = size of the day's move</span>
              </div>
            </>
          )}
        </aside>
      )}

      {viewMode === 'correlation' && !selected && mode !== ZOOM_MODES.CLOSEST && (
        <div className="ask-wrap">
          {askResult && (
            <div className="ask-card">
              <div className="ask-card-head">
                <div>
                  <div className="panel-kicker">
                    {askResult.kind === 'pair_divergence'
                      ? 'ANSWER · DO DIVERGENCES CLOSE? (3 YEARS, TESTED ON HELD-OUT DATA)'
                      : `ANSWER · NEXT-DAY RETURNS, ${askResult.window.toUpperCase()}`}
                  </div>
                  <div className="ask-q">“{askResult.question}”</div>
                </div>
                <button className="link-clear" onClick={clearAsk}>✕ Clear</button>
              </div>

              {askResult.kind === 'pair_divergence' && askResult.pairs.map(pair => (
                <div key={`${pair.a}-${pair.b}`} className="ask-pair">
                  <div className="ask-pair-top">
                    <strong>{pair.a} <span>vs</span> {pair.b}</strong>
                    <b className={pair.verdict === 'converges' ? 'pos' : pair.verdict === 'mixed' ? '' : 'neg'}>
                      {pair.verdict === 'converges' ? 'Gaps tend to close'
                        : pair.verdict === 'mixed' ? 'Some sign of closing'
                        : pair.verdict === 'did_not_hold' ? 'Did not hold up'
                        : pair.verdict === 'inconclusive' ? 'Too few divergences' : 'No reliable closing'}
                    </b>
                  </div>
                  <div className="ask-pair-sub">
                    {pair.events} divergences found
                    {pair.oos_n ? ` · ${Math.round((pair.oos_converged_share || 0) * 100)}% had narrowed after ${pair.best_horizon_days} day(s) on held-out data (${pair.oos_n} cases)` : ''}
                    {pair.half_life_days ? ` · half-life ${pair.half_life_days} days` : ' · no steady pull-back found'}
                  </div>
                  {pair.reason && <div className="ask-pair-why">{pair.reason}</div>}
                </div>
              ))}

              {askResult.kind !== 'pair_divergence' && askResult.pairs.map(pair => (
                <div key={`${pair.a}-${pair.b}`} className="ask-pair">
                  <div className="ask-pair-top">
                    <strong>{pair.a} <span>today →</span> {pair.b} <span>next day</span></strong>
                    <b className={pair.r >= 0 ? 'pos' : 'neg'}>r = {pair.r > 0 ? '+' : ''}{pair.r.toFixed(2)}</b>
                  </div>
                  <span className="link-bar">
                    <i
                      className={pair.r >= 0 ? 'pos' : 'neg'}
                      style={{ width: `${Math.min(100, Math.abs(pair.r) * 100)}%`, opacity: pair.significant ? 1 : 0.5 }}
                    />
                  </span>
                  <div className="ask-pair-sub">
                    {pair.strength} · {pair.b} tends to {pair.direction === 'positive' ? 'move the same way' : 'move the opposite way'} the day after {pair.a} moves · {pair.n} days
                    {pair.significant ? '' : ' · could be chance'}
                    {pair.matches_expectation === true && <em className="ok"> · matches the AI expectation</em>}
                    {pair.matches_expectation === false && <em className="no"> · opposite to the AI expectation</em>}
                  </div>
                  {pair.reason && <div className="ask-pair-why">{pair.reason}</div>}
                </div>
              ))}

              <div className="ask-actions">
                <button
                  className="primary-button"
                  onClick={() => onRunResearch?.(
                    askResult.research_question ||
                      `Does ${askResult.pairs[0].a} lead ${askResult.pairs[0].b}?`,
                    [...new Set(askResult.pairs.flatMap(pair => [pair.a, pair.b]))].slice(0, 4),
                    askResult.kind === 'pair_divergence' ? { a: askResult.pairs[0].a, b: askResult.pairs[0].b } : null
                  )}
                >
                  {askResult.kind === 'pair_divergence' ? 'Run the full pairs investigation →' : 'Test this properly in Research →'}
                </button>
                <small>Correlation is not causation. The AI picked the pairs; the numbers are computed from market data.</small>
              </div>
            </div>
          )}

          {askError && <div className="ask-error">{askError}</div>}

          <form
            className="ask-bar"
            onSubmit={event => {
              event.preventDefault()
              askGlobe()
            }}
          >
            <input
              value={askText}
              maxLength={200}
              disabled={askBusy}
              placeholder="Ask the globe: does Walmart lead Coca-Cola the next day?"
              onChange={event => setAskText(event.target.value)}
            />
            <button type="submit" disabled={askBusy || askText.trim().length < 4}>
              {askBusy ? 'Thinking…' : 'Ask'}
            </button>
          </form>

          {!askResult && !askBusy && (
            <div className="ask-chips">
              {ASK_SUGGESTIONS.map(text => (
                <button
                  key={text}
                  onClick={() => {
                    setAskText(text)
                    askGlobe(text)
                  }}
                >
                  {text}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {mode !== ZOOM_MODES.CLOSEST && (
        <div className="globe-stats">
          <div className="globe-stat">
            <span>COMPANIES</span>
            <strong>{compact(points.length)}</strong>
          </div>
          <div className="globe-stat">
            <span>RECENT EVENTS</span>
            <strong>{compact(feed.length)}</strong>
          </div>
          <div className="globe-stat">
            <span>SHARES TRADED</span>
            <strong>{compact(totalShares)}</strong>
          </div>
        </div>
      )}

      {mode !== ZOOM_MODES.CLOSEST && !askResult && (
        <div className="zoom-hint">
          DRAG TO ROTATE · SCROLL TO EXPLORE · SELECT A COMPANY
        </div>
      )}
    </div>
  )
}

// --------------------------------------------------
// ANIMATED RESEARCH NOTEBOOK HERO
// --------------------------------------------------


// --------------------------------------------------
// ANIMATED MARKET SCALE HERO
// --------------------------------------------------

function ResearchHero() {
  const heroRef = useRef(null)
  const changeTimer = useRef(null)
  const fadeTimer = useRef(null)

  // Pairs of assets that really do tend to move in opposite directions on daily returns.
  // `r` is the approximate long-run daily correlation; `change` is an illustrative divergence day.
  const pairs = [
    { r: -0.75, left: { symbol: 'VIX', change: -6.8 }, right: { symbol: 'SPY', change: 1.1 } },
    { r: -0.45, left: { symbol: 'USO', change: 2.3 }, right: { symbol: 'DAL', change: -2.9 } },
    { r: -0.40, left: { symbol: 'UUP', change: -0.6 }, right: { symbol: 'GLD', change: 1.2 } },
    { r: -0.40, left: { symbol: 'XLU', change: 1.4 }, right: { symbol: 'TNX', change: -3.1 } }
  ]

  const [pairIndex, setPairIndex] = useState(0)
  const [tickerFading, setTickerFading] = useState(false)

  // Swap to the next pair: fade text out, change data (scale tilts via CSS transition), fade text in.
  useEffect(() => {
    changeTimer.current = window.setInterval(() => {
      setTickerFading(true)

      fadeTimer.current = window.setTimeout(() => {
        setPairIndex(current => (current + 1) % pairs.length)
        fadeTimer.current = window.setTimeout(() => setTickerFading(false), 120)
      }, 350)
    }, 3000)

    return () => {
      window.clearInterval(changeTimer.current)
      window.clearTimeout(fadeTimer.current)
    }
  }, [])

  // Preserve the existing scroll fade into the globe.
  useEffect(() => {
    const updateProgress = () => {
      if (!heroRef.current) return

      const rect = heroRef.current.getBoundingClientRect()

      const progress = Math.max(
        0,
        Math.min(
          1,
          1 - Math.max(0, -rect.top) / (rect.height * 0.8)
        )
      )

      heroRef.current.style.setProperty(
        '--hero-progress',
        progress
      )
    }

    window.addEventListener('scroll', updateProgress, {
      passive: true
    })

    updateProgress()

    return () => {
      window.removeEventListener('scroll', updateProgress)
    }
  }, [])

  const pair = pairs[pairIndex]

  // The asset with the higher change always sits higher on the scale.
  const leftUp = pair.left.change >= pair.right.change
  const spread = Math.abs(pair.left.change - pair.right.change)
  const tilt = (leftUp ? 1 : -1) * Math.min(11, 4 + spread * 1.2)
  const fmt = v => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(2)}%`
  const sideUp = side => (side === 'left' ? leftUp : !leftUp)

  const tickerContentStyle = {
    opacity: tickerFading ? 0 : 1,
    transition: 'opacity 300ms ease'
  }

  return (
    <section
      ref={heroRef}
      className="research-hero market-scale-hero"
      style={{ '--scale-tilt': `${tilt}deg`, '--scale-untilt': `${-tilt}deg` }}
    >
      <div className="scale-grid" />
      <div className="scale-glow scale-glow-green" />
      <div className="scale-glow scale-glow-red" />

      <div className="scale-hero-content">

        {/* INTRODUCTION */}

        <div className="scale-intro">
          <div className="scale-eyebrow">
            <span className="scale-live-dot" />
            GLOBAL MARKET INTELLIGENCE
          </div>

          <h1>
            Discover Patterns
            <br />
            <span>Keep What Matters.</span>
          </h1>

          <p>
            Utilising AI to find trends 
            and test ideas against data
          </p>
        </div>

        {/* BALANCE SCALE */}

        <div className="scale-stage">
          <div className="scale-machine">

            <div className="scale-top">
              <div className="scale-cap" />

              <div className="scale-beam">
                <div className="scale-beam-light" />

                {/* LEFT CUP */}

                <div
                  className="scale-arm scale-arm-left"
                >
                  <div className="scale-cup">

                    <div className="scale-cup-hanger" />

                    <div
                      className={`scale-ticker-card ${
                        sideUp('left') ? 'ticker-green' : 'ticker-red'
                      }`}
                    >
                      <div style={tickerContentStyle}>

                        <strong>
                          {pair.left.symbol}
                        </strong>

                        <span className="scale-ticker-change">
                          {fmt(pair.left.change)}
                        </span>
                      </div>
                    </div>

                    <div className="scale-cup-bowl" />
                  </div>
                </div>

                {/* RIGHT CUP */}

                <div
                  className="scale-arm scale-arm-right"
                >
                  <div className="scale-cup">

                    <div className="scale-cup-hanger" />

                    <div
                      className={`scale-ticker-card ${
                        sideUp('right') ? 'ticker-green' : 'ticker-red'
                      }`}
                    >
                      <div style={tickerContentStyle}>

                        <strong>
                          {pair.right.symbol}
                        </strong>

                        <span className="scale-ticker-change">
                          {fmt(pair.right.change)}
                        </span>
                      </div>
                    </div>

                    <div className="scale-cup-bowl" />
                  </div>
                </div>

              </div>
            </div>

            {/* KEEP THE SCALE BODY UNCHANGED */}

            <div className="scale-pivot">
              <div className="scale-pivot-shine" />
            </div>

            <div className="scale-base" />
            <div className="scale-floor-glow" />

          </div>

          {/* MARKET STATUS */}

          <div className="scale-status">
            <span className={leftUp ? 'status-green' : 'status-red'}>
              {pair.left.symbol} {leftUp ? 'RISING' : 'FALLING'}
            </span>

            <span className="scale-status-divider">
              /
            </span>

            <span className={leftUp ? 'status-red' : 'status-green'}>
              {pair.right.symbol} {leftUp ? 'FALLING' : 'RISING'}
            </span>

            <span className="scale-status-divider">
              ·
            </span>

            <span>
              typical daily r = {pair.r.toFixed(2)}
            </span>
          </div>

        </div>
      </div>
    </section>
  )
}



// --------------------------------------------------
// MARKET DASHBOARD
// --------------------------------------------------

function MarketDashboard({ onRunResearch }) {
  const [config, setConfig] = useState(null)
  const [locations, setLocations] = useState({})
  const [trades, setTrades] = useState({})
  const [feed, setFeed] = useState([])
  const [arcEvents, setArcEvents] = useState([])
  const [connection, setConnection] = useState('connecting')
  const [selected, setSelected] = useState(null)
  const [fakeMode, setFakeMode] = useState(false)

  const simulationTimer = useRef(null)
  const pendingTrades = useRef([])
  const pendingSimulatedTrades = useRef([])

  const handleModeChange = useCallback(() => {}, [])

  // Initial data and real Alpaca socket connection.
  useEffect(() => {
    let socket
    let cancelled = false
    let flushInterval = null
    let companyLocations = {}

    async function initialiseDashboard() {
      try {
        const [configResponse, snapshotResponse, locationsResponse] =
          await Promise.all([
            fetch(`${API}/api/market/config`),
            fetch(`${API}/api/market/snapshot`),
            fetch(`${API}/api/market/locations`)
          ])

        if (!configResponse.ok || !snapshotResponse.ok || !locationsResponse.ok) {
          throw new Error('Could not load market data.')
        }

        const [configData, snapshotData, locationsData] =
          await Promise.all([
            configResponse.json(),
            snapshotResponse.json(),
            locationsResponse.json()
          ])

        if (cancelled) return

        const allLocations = new Map()

        for (const location of Array.isArray(locationsData) ? locationsData : []) {
          allLocations.set(location.symbol, {
            ...location,
            name: location.company,
            lat: location.latitude,
            lon: location.longitude
          })
        }

        for (const [symbol, reference] of Object.entries(COMPANY_LOCATIONS)) {
          if (!allLocations.has(symbol)) {
            allLocations.set(symbol, {
              symbol,
              company: COMPANY_NAMES[symbol] || symbol,
              name: COMPANY_NAMES[symbol] || symbol,
              city: reference.city,
              country: reference.country,
              latitude: reference.lat,
              longitude: reference.lng,
              lat: reference.lat,
              lon: reference.lng,
              sector: null
            })
          }
        }

        companyLocations = Object.fromEntries(allLocations)

        setLocations(companyLocations)
        setConfig({ ...configData, locations: companyLocations })

        const initialTrades = Object.entries(snapshotData.symbols || {})
          .filter(([_, value]) => value.latest)
          .map(([symbol, value]) => ({
            ...(companyLocations[symbol] || {}),
            ...value.latest,
            symbol
          }))

        setTrades(Object.fromEntries(
          initialTrades.map(trade => [trade.symbol, trade])
        ))

        const recentTrades = (snapshotData.recent_trades || []).map(trade => ({
          ...trade,
          ...(companyLocations[trade.symbol] || {}),
          simulated: false
        }))

        setFeed(recentTrades.slice(0, MAX_FEED))
      } catch (error) {
        console.error('Dashboard initialisation failed:', error)

        if (!cancelled) {
          setConfig(current => ({
            ...(current || {}),
            error: error.message
          }))
        }
      }

      if (cancelled) return

      socket = io(`${API}/market`, {
        transports: ['websocket', 'polling'],
        reconnection: true
      })

      socket.on('connect', () => {
        if (!cancelled) setConnection('live')
      })

      socket.on('disconnect', () => {
        if (!cancelled) setConnection('reconnecting')
      })

      socket.on('connect_error', error => {
        console.error('Socket connection error:', error)
        if (!cancelled) setConnection('offline')
      })

      socket.on('market_trade', trade => {
        if (cancelled) return

        pendingTrades.current.push({
          ...trade,
          ...(companyLocations[trade.symbol] || {}),
          simulated: false
        })

        if (pendingTrades.current.length > 500) {
          pendingTrades.current = pendingTrades.current.slice(-500)
        }
      })

      flushInterval = window.setInterval(() => {
        if (cancelled || !pendingTrades.current.length) return

        const batch = pendingTrades.current.splice(0, 100)
        const newestFirst = batch.slice().reverse()

        setTrades(current => {
          const next = { ...current }

          for (const trade of batch) {
            next[trade.symbol] = {
              ...(next[trade.symbol] || {}),
              ...trade
            }
          }

          return next
        })

        setFeed(current => [
          ...newestFirst,
          ...current
        ].slice(0, MAX_FEED))
      }, SOCKET_FLUSH_INTERVAL)
    }

    initialiseDashboard()

    return () => {
      cancelled = true

      if (flushInterval) {
        window.clearInterval(flushInterval)
      }

      pendingTrades.current = []

      if (socket) {
        socket.removeAllListeners()
        socket.disconnect()
      }
    }
  }, [])

  // Generate one simulated trade at a time.
  // Arc events are published immediately; feed updates are buffered.
  const generateSimulatedTrade = useCallback(() => {
    const availableCompanies = Object.values(locations).filter(location => {
      const lat = Number(location.lat ?? location.latitude)
      const lng = Number(location.lon ?? location.longitude)

      return (
        location.symbol &&
        Number.isFinite(lat) &&
        Number.isFinite(lng)
      )
    })

    if (!availableCompanies.length) return

    const company = availableCompanies[
      Math.floor(Math.random() * availableCompanies.length)
    ]

    const headquarters = {
      city: company.city || 'Company HQ',
      country: company.country || '',
      lat: Number(company.lat ?? company.latitude),
      lng: Number(company.lon ?? company.longitude)
    }

    const buyer = BUYER_LOCATIONS[
      Math.floor(Math.random() * BUYER_LOCATIONS.length)
    ]

    const trade = makeSimulatedTrade(company, headquarters, null)

    const newTrade = {
      ...trade,
      createdAt: Date.now(),
      buyer_city: buyer.city,
      buyer_country: buyer.country,
      buyer_lat: buyer.lat,
      buyer_lng: buyer.lng
    }

    // Send the arc to the globe immediately.
    // This is separate from the slower feed update.
    setArcEvents(current => [
      newTrade,
      ...current
    ].slice(0, MAX_ARCS * 2))

    // Buffer the heavier dashboard data update.
    pendingSimulatedTrades.current.push(newTrade)
  }, [locations])

  // Flush simulated feed data at a lower frequency.
  useEffect(() => {
    const flushTimer = window.setInterval(() => {
      if (!pendingSimulatedTrades.current.length) return

      const batch = pendingSimulatedTrades.current.splice(0, 20)

      setTrades(current => {
        const next = { ...current }

        for (const trade of batch) {
          next[trade.symbol] = {
            ...(next[trade.symbol] || {}),
            ...trade
          }
        }

        return next
      })

      setFeed(current => [
        ...batch.slice().reverse(),
        ...current
      ].slice(0, MAX_FEED))
    }, SOCKET_FLUSH_INTERVAL)

    return () => {
      window.clearInterval(flushTimer)
      pendingSimulatedTrades.current = []
    }
  }, [])

  // Continuous simulation at a steady rate.
  useEffect(() => {
    if (!fakeMode) {
      if (simulationTimer.current) {
        window.clearInterval(simulationTimer.current)
        simulationTimer.current = null
      }

      return
    }

    const timer = window.setInterval(() => {
      generateSimulatedTrade()
    }, SIMULATION_INTERVAL)

    simulationTimer.current = timer

    return () => {
      window.clearInterval(timer)

      if (simulationTimer.current === timer) {
        simulationTimer.current = null
      }
    }
  }, [fakeMode, generateSimulatedTrade])

  const toggleFakeMode = useCallback(() => {
    if (fakeMode) {
      if (simulationTimer.current) {
        window.clearInterval(simulationTimer.current)
        simulationTimer.current = null
      }

      pendingSimulatedTrades.current = []

      const realFeed = feed.filter(trade => !trade.simulated)
      const latestRealTrades = {}

      for (const trade of realFeed) {
        if (!latestRealTrades[trade.symbol]) {
          latestRealTrades[trade.symbol] = trade
        }
      }

      setFeed(realFeed)

      setTrades(current => {
        const next = { ...current }

        for (const symbol of Object.keys(next)) {
          if (next[symbol]?.simulated) {
            delete next[symbol]
          }
        }

        return {
          ...next,
          ...latestRealTrades
        }
      })

      setArcEvents([])
      setFakeMode(false)
    } else {
      setFakeMode(true)
    }
  }, [fakeMode, feed])

  return (
    <>
      {config?.error && (
        <div className="notice">Market data: {config.error}</div>
      )}

      {config?.credentials_configured === false && (
        <div className="notice">
          Alpaca API keys are not configured. Check backend/.env.
        </div>
      )}

      <MarketGlobe
        locations={locations}
        trades={trades}
        feed={feed}
        arcEvents={arcEvents}
        selected={selected}
        onSelect={setSelected}
        connection={connection}
        onModeChange={handleModeChange}
        fakeMode={fakeMode}
        onToggleFakeMode={toggleFakeMode}
        onRunResearch={onRunResearch}
      />
    </>
  )
}

// --------------------------------------------------
// TASK MANAGER
// --------------------------------------------------

function TasksPage() {
  const [items, setItems] = useState([])
  const [name, setName] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [updatingStatusId, setUpdatingStatusId] = useState(null)

  async function parseResponse(response) {
    const data = response.status === 204 ? null : await response.json()

    if (!response.ok) {
      throw new Error(data?.error || `Request failed (${response.status})`)
    }

    return data
  }

  async function loadItems() {
    setLoading(true)
    setError('')

    try {
      const data = await parseResponse(await fetch(`${API}/api/items`))
      setItems(data.map(item => ({
        ...item,
        status: item.status || 'Not Started'
      })))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadItems()
  }, [])

  async function addItem(event) {
    event.preventDefault()
    if (!name.trim()) return

    try {
      await parseResponse(await fetch(`${API}/api/items`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: name.trim() })
      }))

      setName('')
      await loadItems()
    } catch (err) {
      setError(err.message)
    }
  }

  async function deleteItem(id) {
    try {
      await parseResponse(await fetch(`${API}/api/items/${id}`, {
        method: 'DELETE'
      }))

      await loadItems()
    } catch (err) {
      setError(err.message)
    }
  }

  async function changeStatus(id, status) {
    setUpdatingStatusId(id)

    try {
      const item = await parseResponse(await fetch(`${API}/api/items/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status })
      }))

      setItems(current =>
        current.map(existing => existing.id === id ? item : existing)
      )
    } catch (err) {
      setError(err.message)
    } finally {
      setUpdatingStatusId(null)
    }
  }

  return (
    <main className="tasks-page">
      <div className="tasks-header">
        <div>
          <div className="eyebrow">PROJECT WORKSPACE</div>
          <h1>Tasks</h1>
        </div>
        <span className="feed-count">{items.length} TASKS</span>
      </div>

      {error && <div className="notice warning">{error}</div>}

      <form className="task-add" onSubmit={addItem}>
        <input
          value={name}
          onChange={event => setName(event.target.value)}
          placeholder="Add a project task..."
        />
        <button className="primary-button" disabled={!name.trim()}>
          Add task
        </button>
      </form>

      <div className="task-panel">
        <table className="feed-table">
          <thead>
            <tr>
              <th>TASK</th>
              <th>DESCRIPTION</th>
              <th>STATUS</th>
              <th />
            </tr>
          </thead>

          <tbody>
            {items.length ? items.map(item => (
              <tr key={item.id}>
                <td>{item.title}</td>
                <td>{item.description || '—'}</td>
                <td>
                  <StatusSelect
                    id={`status-${item.id}`}
                    value={item.status}
                    options={STATUSES}
                    onChange={status => changeStatus(item.id, status)}
                    disabled={loading || updatingStatusId === item.id}
                  />
                </td>
                <td>
                  <button
                    className="delete-button"
                    onClick={() => deleteItem(item.id)}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            )) : (
              <tr>
                <td colSpan="4" className="empty-feed">
                  {loading ? 'Loading...' : 'No tasks yet.'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </main>
  )
}

// --------------------------------------------------
// MAIN APP
// --------------------------------------------------

export default function App() {
  const [page, setPage] = useState('market')
  const [researchPrefill, setResearchPrefill] = useState(null)

  // "Ask the globe" hands a question to the Research tab (pre-filled, not started).
  const openResearchWith = useCallback((question, tickers, pair) => {
    setResearchPrefill({ question, tickers, pair, nonce: Date.now() })
    setPage('research')
  }, [])

  return (
    <div className="app-shell">
      <style>{styles}</style>

      <nav className="app-nav">
        <button
          className={`nav-tab ${page === 'market' ? 'selected' : ''}`}
          onClick={() => setPage('market')}
        >
          ◉ &nbsp; Global intelligence
        </button>

        <button
          className={`nav-tab ${page === 'research' ? 'selected' : ''}`}
          onClick={() => setPage('research')}
        >
          ◇ &nbsp; Research
        </button>
      </nav>

      {page === 'market' ? (
        <div className="market-page">
          <ResearchHero />
          <MarketDashboard onRunResearch={openResearchWith} />
        </div>
      ) : (
        <main style={{
          width: 'min(1120px, calc(100% - 40px))',
          margin: '42px auto'
        }}>
          <div className="eyebrow">AUTONOMOUS RESEARCH</div>
          <h1>Research</h1>
          <ResearchLauncher prefill={researchPrefill} />
        </main>
      )}
    </div>
  )
}