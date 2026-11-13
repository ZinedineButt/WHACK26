"""Literature retrieval and grounded self-critique inputs.

Sources are tried in order until one returns papers: OpenAlex (optional free
OPENALEX_API_KEY; keyless use is rate-limited and often 429s), Semantic Scholar
(keyless, rate-limited), arXiv (keyless, finance/economics). Papers are only ever
cited from what an API actually returned: the model refers to papers by index and
the server joins the metadata, so references cannot be invented. Total failure
degrades to status 'unavailable' without affecting any statistics."""
import os
import re
import xml.etree.ElementTree as ET

import requests

from .providers import ProviderError, _get

OPENALEX = "https://api.openalex.org/works"
SEMANTIC = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV = "http://export.arxiv.org/api/query"
MAX_PAPERS = 5
_STOP = {"the", "and", "for", "with", "that", "this", "from", "does", "predict", "relationship", "between", "there", "stock", "market",
         "returns", "return", "predictability", "daily", "change", "level", "log", "index", "price", "demand", "future"}


def _abstract(inv):
    if not inv:
        return ""
    words = {}
    for w, pos in inv.items():
        for p in pos:
            words[p] = w
    return " ".join(words[i] for i in sorted(words))


def build_query(h):
    return clean_query(f"{h['predictor']['label']} {h['outcome']['label']} predictability returns")


def clean_query(text):
    q = re.sub(r"[^A-Za-z0-9 &.-]", " ", text or "")
    return re.sub(r"\s+", " ", q).strip()[:160]


def _paper(i, title, year, doi, url, venue, cited, abstract, source):
    return {"index": i, "title": title[:240], "year": year, "doi": doi, "url": url or doi, "venue": (venue or "")[:120],
            "cited_by": cited, "abstract": (abstract or "")[:700], "source": source}


def _openalex(q, limit):
    params = {"search": q, "per-page": limit, "filter": "type:article,has_abstract:true",
              "select": "id,title,publication_year,doi,cited_by_count,abstract_inverted_index,primary_location"}
    if os.environ.get("OPENALEX_API_KEY"):
        params["api_key"] = os.environ["OPENALEX_API_KEY"]
    r = _get(OPENALEX, params=params)
    if r.status_code != 200:
        raise ProviderError(f"OpenAlex HTTP {r.status_code}")
    out = []
    for w in r.json().get("results", [])[:limit]:
        title = (w.get("title") or "").strip()
        if title:
            loc = (w.get("primary_location") or {}).get("source") or {}
            out.append(_paper(len(out) + 1, title, w.get("publication_year"), w.get("doi"), w.get("doi") or w.get("id"),
                              loc.get("display_name"), w.get("cited_by_count"), _abstract(w.get("abstract_inverted_index")), "openalex"))
    return out


def _semantic(q, limit):
    r = _get(SEMANTIC, params={"query": q, "limit": limit, "fields": "title,year,abstract,externalIds,url,venue,citationCount"})
    if r.status_code != 200:
        raise ProviderError(f"Semantic Scholar HTTP {r.status_code}")
    out = []
    for w in r.json().get("data", [])[:limit]:
        title = (w.get("title") or "").strip()
        if title and w.get("abstract"):
            doi = (w.get("externalIds") or {}).get("DOI")
            out.append(_paper(len(out) + 1, title, w.get("year"), f"https://doi.org/{doi}" if doi else None, w.get("url"),
                              w.get("venue"), w.get("citationCount"), w.get("abstract"), "semantic_scholar"))
    return out


def _arxiv_terms(q):
    words = [w for w in re.findall(r"[A-Za-z0-9]+", q.lower()) if len(w) > 2 and w not in _STOP]
    return list(dict.fromkeys(words))[:4]


def _arxiv(q, limit):
    terms = _arxiv_terms(q)
    if not terms:
        raise ProviderError("no usable search terms")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    cat = "(cat:q-fin.ST OR cat:q-fin.GN OR cat:q-fin.PM OR cat:q-fin.EC OR cat:q-fin.CP OR cat:econ.GN)"
    for tlist in (terms, terms[:2]):                                  # progressively looser
        expr = " AND ".join(f"all:{t}" for t in tlist) + f" AND {cat}"
        r = _get(ARXIV, params={"search_query": expr, "max_results": limit, "sortBy": "relevance"})
        if r.status_code != 200:
            raise ProviderError(f"arXiv HTTP {r.status_code}")
        out = []
        for e in ET.fromstring(r.text).findall("a:entry", ns):
            title = " ".join((e.findtext("a:title", "", ns) or "").split())
            if title:
                pub = e.findtext("a:published", "", ns) or ""
                link = e.findtext("a:id", "", ns)
                out.append(_paper(len(out) + 1, title, int(pub[:4]) if pub[:4].isdigit() else None, None, link, "arXiv preprint",
                                  None, " ".join((e.findtext("a:summary", "", ns) or "").split()), "arxiv"))
        if out:
            return out
    return []


SOURCES = (("openalex", _openalex), ("semantic_scholar", _semantic), ("arxiv", _arxiv))


def search(h, limit=MAX_PAPERS, query=None):
    """Returns {status, query, provider, papers[], note}. Never raises. `query` overrides the hypothesis-derived query."""
    q = clean_query(query) if query else build_query(h)
    errors = []
    for name, fn in SOURCES:
        try:
            papers = fn(q, limit)
        except (ProviderError, requests.RequestException, ValueError, ET.ParseError) as e:
            errors.append(f"{name}: {type(e).__name__} {str(e)[:60]}")
            continue
        if papers:
            return {"status": "completed", "query": q, "papers": papers, "provider": name,
                    "note": "Papers are retrieved metadata; relevance is judged by the AI critique step, not guaranteed."
                            + (f" (earlier sources failed: {'; '.join(errors)})" if errors else "")}
        errors.append(f"{name}: no results")
    status = "no_results" if all("no results" in e for e in errors) else "unavailable"
    return {"status": status, "query": q, "papers": [], "provider": None,
            "note": "Literature search returned nothing (" + "; ".join(errors) + "); the analysis proceeds without papers."}
