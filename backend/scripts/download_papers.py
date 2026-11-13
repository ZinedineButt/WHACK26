"""Download up to 5 open-access research papers (PDF) per tracked stock into <repo>/papers/<SYMBOL>/.

Uses OpenAlex (open_access.oa_url / best_oa_location pdf) and falls back to arXiv. Only freely available PDFs are fetched;
papers with no open copy are listed in INDEX.md with their link. Run from backend/ with the venv python: scripts/download_papers.py"""
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2] / "papers"
# symbol -> (search name, words of which at least one must appear in the title/abstract)
COMPANY = {"AAPL": ("Apple Inc", ["apple inc", "aapl", "apple's", "apple stock"]), "MSFT": ("Microsoft", ["microsoft", "msft"]),
           "NVDA": ("NVIDIA", ["nvidia", "nvda"]), "AMZN": ("Amazon", ["amazon.com", "amazon inc", "amzn", "amazon stock", "amazon's"]),
           "GOOGL": ("Alphabet Google", ["alphabet inc", "googl", "google stock", "google inc", "alphabet's"]),
           "META": ("Meta Platforms Facebook", ["meta platforms", "facebook inc", "facebook stock", "facebook's", "fb stock"]),
           "TSLA": ("Tesla", ["tesla"]), "AMD": ("Advanced Micro Devices", ["advanced micro devices", "amd stock", "amd's", "(amd)"])}
FINANCE = re.compile(r"stock|share price|equity|equities|return|volatil|invest|market|earnings|valuation|trading", re.I)
HDR = {"User-Agent": "WHACK26-research-tool/1.0"}
WANT = 5


def slug(t):
    return re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_")[:80]


def _abstract(inv):
    if not inv:
        return ""
    words = {}
    for w, pos in inv.items():
        for i in pos:
            words[i] = w
    return " ".join(words[i] for i in sorted(words))


def _get(url, **kw):
    for i in range(5):
        r = requests.get(url, headers=HDR, timeout=60, **kw)
        if r.status_code not in (429, 503):
            return r
        time.sleep(3 * 2 ** i)
    return r


def relevant(text, terms):
    t = text.lower()
    return any(k in t for k in terms) and bool(FINANCE.search(t))


def openalex(q, terms):
    out = []
    r = _get("https://api.openalex.org/works", params={
        "search": q, "per-page": 50, "filter": "type:article,is_oa:true,has_abstract:true",
        "select": "title,publication_year,doi,cited_by_count,best_oa_location,open_access,abstract_inverted_index"})
    r.raise_for_status()
    for w in r.json().get("results", []):
        loc = w.get("best_oa_location") or {}
        pdf = loc.get("pdf_url") or (w.get("open_access") or {}).get("oa_url")
        text = f"{w.get('title') or ''} {_abstract(w.get('abstract_inverted_index'))}"
        if w.get("title") and pdf and relevant(text, terms):
            out.append({"title": w["title"], "year": w.get("publication_year"), "doi": w.get("doi"), "pdf": pdf,
                        "cited": w.get("cited_by_count") or 0})
    return sorted(out, key=lambda p: -p["cited"])


def arxiv(q, terms):
    out = []
    expr = "all:" + "+AND+all:".join(re.findall(r"[A-Za-z]+", q)[:2]) + "+AND+(cat:q-fin.ST+OR+cat:q-fin.GN+OR+cat:q-fin.PM+OR+cat:q-fin.CP+OR+cat:q-fin.EC)"
    r = _get(f"http://export.arxiv.org/api/query?search_query={expr}&max_results=40&sortBy=relevance")
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for e in ET.fromstring(r.text).findall("a:entry", ns):
        t = " ".join((e.findtext("a:title", "", ns) or "").split())
        link = e.findtext("a:id", "", ns)
        summ = " ".join((e.findtext("a:summary", "", ns) or "").split())
        if t and link and relevant(f"{t} {summ}", terms):
            out.append({"title": t, "year": (e.findtext("a:published", "", ns) or "")[:4], "doi": link,
                        "pdf": link.replace("/abs/", "/pdf/") + ".pdf", "cited": 0})
    return out


def fetch_pdf(url, dest):
    r = _get(url, allow_redirects=True)
    if r.status_code == 200 and r.content[:5] == b"%PDF-" and len(r.content) > 20_000:
        dest.write_bytes(r.content)
        return True
    return False


def main():
    ROOT.mkdir(exist_ok=True)
    index = ["# Downloaded papers\n"]
    for sym, (name, terms) in COMPANY.items():
        folder = ROOT / sym
        folder.mkdir(exist_ok=True)
        index.append(f"\n## {sym} ({name})\n")
        got, seen = 0, set()
        for label, fn, query in (("openalex", openalex, f"{name} stock returns"), ("openalex", openalex, f"{name} stock price prediction"),
                                 ("openalex", openalex, f"{name} shares investors"), ("arxiv", arxiv, f"{name} stock")):
            if got >= WANT:
                break
            try:
                cands = fn(query, terms)
            except Exception as e:
                print(f"  {sym} {label} search failed: {type(e).__name__}", file=sys.stderr)
                continue
            for p in cands:
                if got >= WANT:
                    break
                if p["title"].lower() in seen:
                    continue
                seen.add(p["title"].lower())
                dest = folder / f"{got + 1}_{p['year']}_{slug(p['title'])}.pdf"
                try:
                    ok = fetch_pdf(p["pdf"], dest)
                except requests.RequestException:
                    ok = False
                time.sleep(0.5)
                if ok:
                    got += 1
                    index.append(f"- {p['title']} ({p['year']}) · {p['doi'] or p['pdf']} · `{dest.name}`\n")
                    print(f"{sym} {got}/5 {dest.name}")
                time.sleep(0.3)
        if got < WANT:
            index.append(f"- _only {got} open-access PDFs could be downloaded_\n")
    (ROOT / "INDEX.md").write_text("".join(index), encoding="utf-8")


if __name__ == "__main__":
    main()
