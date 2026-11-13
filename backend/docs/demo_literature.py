"""Curated REAL background papers for the offline demo (the demo data is synthetic, the papers are not).

They are shown as related research and are never used to compute anything. DOIs link to the publisher record.
"""


def _p(i, title, authors, year, doi, venue, summary):
    return {"index": i, "title": title, "year": year, "doi": doi, "url": f"https://doi.org/{doi}", "venue": venue,
            "cited_by": None, "abstract": summary, "source": "curated", "authors": authors}


_HLZ = ("...and the Cross-Section of Expected Returns", "Harvey, Liu & Zhu", 2016, "10.1093/rfs/hhv059", "Review of Financial Studies",
        "Argues that after hundreds of published return predictors, a new finding needs a much higher bar (t-statistic above about 3) once repeated testing is accounted for.")
_DSR = ("The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality", "Bailey & Lopez de Prado", 2014,
        "10.3905/jpm.2014.40.5.094", "Journal of Portfolio Management",
        "Shows how to discount a backtest's Sharpe ratio for the number of strategies tried and for non-normal returns.")

_RAW = {
    "SYN_OIL": {
        "viability": 4, "expectation": "supports", "cited": [1, 2],
        "summary": "Published work finds that oil price moves can carry information about later stock returns, and that the effect depends on why oil moved. "
                   "That makes an oil-to-airline link plausible, but airlines also share the overall market, so a result needs checking against market moves.",
        "concerns": "Studies mostly use monthly data and broad indices; a one-day lead on a single industry is a harder claim.",
        "papers": [
            ("Striking oil: Another puzzle?", "Driesprong, Jacobsen & Maat", 2008, "10.1016/j.jfineco.2007.07.005", "Journal of Financial Economics",
             "Finds that changes in oil prices predict stock market returns in many countries, with a rise in oil prices followed by lower returns."),
            ("The impact of oil price shocks on the U.S. stock market", "Kilian & Park", 2009, "10.1111/j.1468-2354.2009.00568.x", "International Economic Review",
             "Shows that how oil prices affect US stocks depends on whether the move comes from demand or supply shocks."),
            _HLZ],
    },
    "SYN_RATE": {
        "viability": 3, "expectation": "mixed", "cited": [1, 2],
        "summary": "Interest rates move stock valuations, yet research finds that rate changes and similar variables rarely give reliable forecasts of next-day or next-month returns out of sample. "
                   "A null result here would match what the literature expects.",
        "concerns": "Many popular return predictors fail once tested on data they were not fitted to.",
        "papers": [
            ("What moves the stock and bond markets? A variance decomposition for long-term asset returns", "Campbell & Ammer", 1993, "10.1111/j.1540-6261.1993.tb04700.x", "Journal of Finance",
             "Studies what drives stock and long-term bond returns and how news about interest rates feeds into each."),
            ("A Comprehensive Look at the Empirical Performance of Equity Premium Prediction", "Welch & Goyal", 2008, "10.1093/rfs/hhm014", "Review of Financial Studies",
             "Finds that many variables thought to predict equity returns, including interest rates, have little out-of-sample forecasting power."),
            _HLZ],
    },
    "SYN_CONF_X": {
        "viability": 3, "expectation": "mixed", "cited": [1, 3],
        "summary": "When one stock seems to lead another, research shows the pattern often comes from both reacting to the same market news at different speeds, not from real predictability. "
                   "That is exactly the trap this pair is built to illustrate, and why a backtest alone is not enough.",
        "concerns": "Lead-lag patterns that vanish once a shared market factor is controlled for are common.",
        "papers": [
            ("When Are Contrarian Profits Due to Stock Market Overreaction?", "Lo & MacKinlay", 1990, "10.1093/rfs/3.2.175", "Review of Financial Studies",
             "Shows that lead-lag patterns between stocks can arise from how they respond to common news, not only from true predictability."),
            _HLZ, _DSR],
    },
}

DEMO_LITERATURE = {}
for _vid, _d in _RAW.items():
    _papers = [_p(i, *t) for i, t in enumerate(_d["papers"], 1)]
    DEMO_LITERATURE[_vid] = {
        "papers": _papers,
        "prior": {"viability": _d["viability"], "expectation": _d["expectation"], "summary": _d["summary"], "concerns": _d["concerns"],
                  "cited_papers": _d["cited"], "source": "curated", "papers_retrieved": len(_papers), "search_status": "completed"},
    }


def demo_lit(predictor_id):
    d = DEMO_LITERATURE.get(predictor_id)
    if not d:
        return {"status": "not_run", "query": None, "papers": [], "provider": None,
                "note": "Synthetic demo data: no literature search is performed."}
    return {"status": "completed", "query": None, "papers": [dict(p) for p in d["papers"]], "provider": "curated_reference",
            "note": "Demo only: real published papers chosen by hand as background for this topic. They were not retrieved by a search and nothing computed here depends on them."}
