"""query_parser.py — Extract company and fiscal-year signals from a query.

This is Phase 1 Step 1.2 of the LLM-Finance retrieval overhaul.

The parser returns a structured dict that downstream filters can use to
narrow the candidate pool before BM25 and dense retrieval, or to boost
scores for chunks that originate from the right company / year.

FinQA chunk ``example_id`` format:  ``"TICKER/YEAR/page_N.pdf-M"``
The parser extracts both the ticker(s) and the fiscal year(s) present in
the query so that a simple ``example_id.startswith(ticker + "/year")``
check is sufficient for pre-filtering.
"""

import re
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Fiscal year extraction
# ---------------------------------------------------------------------------

# Match 4-digit years in the range 1990-2030
_YEAR_PATTERN = re.compile(r"\b(19[9][0-9]|20[0-2][0-9]|2030)\b")


def extract_years(question: str) -> list[str]:
    """Return all fiscal years mentioned in *question*.

    Returns strings (not ints) so they can be compared directly against
    the string portions of ``example_id``.

    Examples
    --------
    >>> extract_years("What was revenue in 2009 versus 2008?")
    ['2008', '2009']
    """
    years = sorted(set(_YEAR_PATTERN.findall(question)))
    return years


# ---------------------------------------------------------------------------
# Company / ticker extraction
# ---------------------------------------------------------------------------

# Default ticker dictionary built from common S&P-500 companies.
# When the FinQA corpus is available, call ``build_ticker_dict`` to
# augment this at runtime with every unique ticker in the corpus.
_DEFAULT_TICKERS: dict[str, list[str]] = {
    # Maps canonical ticker -> list of name variants (lower-cased) to match
    "UNP": ["union pacific", "unp"],
    "GS": ["goldman sachs", "goldman", "gs"],
    "JPM": ["jpmorgan", "jp morgan", "jpm"],
    "BAC": ["bank of america", "bac"],
    "WFC": ["wells fargo", "wfc"],
    "C": ["citigroup", "citi", "citibank"],
    "MS": ["morgan stanley", "ms"],
    "AXP": ["american express", "amex", "axp"],
    "V": ["visa"],
    "MA": ["mastercard", "ma"],
    "XOM": ["exxon", "exxonmobil", "xom"],
    "CVX": ["chevron", "cvx"],
    "COP": ["conocophillips", "cop"],
    "SLB": ["schlumberger", "slb"],
    "HAL": ["halliburton", "hal"],
    "BA": ["boeing", "ba"],
    "LMT": ["lockheed martin", "lmt"],
    "RTX": ["raytheon", "rtx", "utc"],
    "GE": ["general electric", "ge"],
    "MMM": ["3m", "minnesota mining", "mmm"],
    "HON": ["honeywell", "hon"],
    "CAT": ["caterpillar", "cat"],
    "DE": ["deere", "john deere", "de"],
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft"],
    "GOOGL": ["google", "alphabet", "googl"],
    "AMZN": ["amazon", "amzn"],
    "META": ["meta", "facebook", "fb"],
    "NVDA": ["nvidia", "nvda"],
    "AMD": ["amd", "advanced micro devices"],
    "INTC": ["intel", "intc"],
    "IBM": ["ibm", "international business machines"],
    "ORCL": ["oracle", "orcl"],
    "CRM": ["salesforce", "crm"],
    "NOW": ["servicenow", "now"],
    "ADBE": ["adobe", "adbe"],
    "CSCO": ["cisco", "csco"],
    "QCOM": ["qualcomm", "qcom"],
    "TXN": ["texas instruments", "txn"],
    "AMAT": ["applied materials", "amat"],
    "MU": ["micron", "mu"],
    "WMT": ["walmart", "wmt"],
    "COST": ["costco", "cost"],
    "TGT": ["target", "tgt"],
    "HD": ["home depot", "hd"],
    "LOW": ["lowe's", "lowes", "low"],
    "AMGN": ["amgen", "amgn"],
    "GILD": ["gilead", "gild"],
    "BIIB": ["biogen", "biib"],
    "REGN": ["regeneron", "regn"],
    "VRTX": ["vertex", "vrtx"],
    "JNJ": ["johnson", "jnj"],
    "PFE": ["pfizer", "pfe"],
    "MRK": ["merck", "mrk"],
    "ABT": ["abbott", "abt"],
    "MDT": ["medtronic", "mdt"],
    "UNH": ["unitedhealth", "united health", "unh"],
    "CVS": ["cvs", "cvs health"],
    "CI": ["cigna", "ci"],
    "HUM": ["humana", "hum"],
    "NEE": ["nextera", "nee"],
    "DUK": ["duke energy", "duk"],
    "SO": ["southern company", "so"],
    "D": ["dominion", "d"],
    "PG": ["procter gamble", "procter & gamble", "pg"],
    "KO": ["coca-cola", "coca cola", "ko"],
    "PEP": ["pepsico", "pepsi", "pep"],
    "MCD": ["mcdonald's", "mcdonalds", "mcd"],
    "SBUX": ["starbucks", "sbux"],
    "YUM": ["yum brands", "yum"],
    "F": ["ford", "f"],
    "GM": ["general motors", "gm"],
    "TM": ["toyota", "tm"],
    "CMI": ["cummins", "cmi"],
    "DAL": ["delta air", "delta airlines", "dal"],
    "UAL": ["united airlines", "ual"],
    "AAL": ["american airlines", "aal"],
    "LUV": ["southwest airlines", "southwest", "luv"],
    "FDX": ["fedex", "fdx"],
    "UPS": ["ups", "united parcel service"],
    "CSX": ["csx"],
    "NSC": ["norfolk southern", "nsc"],
    "KSU": ["kansas city southern", "ksu"],
}


def build_ticker_dict(corpus_path: Optional[Path] = None) -> dict[str, list[str]]:
    """Build a ticker -> name-variants mapping from corpus ``example_id`` values.

    Each ``example_id`` in FinQA has the form ``"TICKER/YEAR/page_N.pdf-M"``.
    This function reads all unique tickers from the corpus and merges them
    into the default dictionary so that any company in the dataset is
    recognisable by its ticker abbreviation even if not listed above.

    Parameters
    ----------
    corpus_path :
        Path to the ``train_chunks.jsonl`` file.  If ``None`` or the file
        does not exist, only the default dictionary is returned.
    """
    import json

    tickers = dict(_DEFAULT_TICKERS)

    if corpus_path is None or not Path(corpus_path).exists():
        return tickers

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            example_id = chunk.get("example_id", "")
            parts = example_id.split("/")

            if not parts:
                continue

            ticker = parts[0].upper()

            if ticker and ticker not in tickers:
                # Register the ticker; it can be matched by its exact symbol
                tickers[ticker] = [ticker.lower()]

    return tickers


def extract_companies(
    question: str,
    ticker_dict: Optional[dict[str, list[str]]] = None
) -> list[str]:
    """Return canonical ticker symbols for companies mentioned in *question*.

    Matching is case-insensitive and covers both full company names and
    ticker symbols as listed in *ticker_dict*.

    Parameters
    ----------
    question :
        The raw natural-language question from the FinQA dataset.
    ticker_dict :
        Mapping of ``{TICKER: [name_variant, ...]}``.  Defaults to the
        built-in dictionary when ``None``.

    Returns
    -------
    list[str]
        Sorted list of matched ticker symbols (deduplicated).

    Examples
    --------
    >>> extract_companies("What were Union Pacific's revenues in 2009?")
    ['UNP']
    >>> extract_companies("Compare Goldman Sachs and Morgan Stanley profits.")
    ['GS', 'MS']
    """
    if ticker_dict is None:
        ticker_dict = _DEFAULT_TICKERS

    question_lower = question.lower()
    matched: set[str] = set()

    for ticker, variants in ticker_dict.items():
        for variant in variants:
            # Use word-boundary matching for short tickers to avoid false hits
            if len(variant) <= 4:
                pattern = r"\b" + re.escape(variant) + r"\b"
                if re.search(pattern, question_lower):
                    matched.add(ticker)
                    break
            else:
                if variant in question_lower:
                    matched.add(ticker)
                    break

    return sorted(matched)


# ---------------------------------------------------------------------------
# Top-level parse function
# ---------------------------------------------------------------------------

def parse_query(
    question: str,
    ticker_dict: Optional[dict[str, list[str]]] = None
) -> dict:
    """Parse a FinQA question and return company and fiscal-year signals.

    Parameters
    ----------
    question :
        The raw question string, e.g.
        ``"What is the change in interest expense from 2008 to 2009?"``
    ticker_dict :
        Optional pre-built ticker dictionary (see ``build_ticker_dict``).
        Defaults to the built-in dictionary.

    Returns
    -------
    dict with keys:

    ``company_ids`` : list[str]
        Canonical ticker symbols found in the question (may be empty).
    ``fiscal_years`` : list[str]
        4-digit year strings found in the question (sorted, may be empty).
    ``has_company`` : bool
        Whether at least one company was identified.
    ``has_year`` : bool
        Whether at least one fiscal year was identified.

    Examples
    --------
    >>> parse_query("What was Union Pacific's interest expense in 2009?")
    {
        'company_ids': ['UNP'],
        'fiscal_years': ['2009'],
        'has_company': True,
        'has_year': True,
    }
    """
    company_ids = extract_companies(question, ticker_dict)
    fiscal_years = extract_years(question)

    return {
        "company_ids": company_ids,
        "fiscal_years": fiscal_years,
        "has_company": bool(company_ids),
        "has_year": bool(fiscal_years),
    }


# ---------------------------------------------------------------------------
# Metadata filter / score-boost helpers
# ---------------------------------------------------------------------------

def chunk_matches_metadata(
    chunk: dict,
    company_ids: list[str],
    fiscal_years: list[str]
) -> bool:
    """Return True if *chunk* matches any parsed company or year.

    A chunk matches when its ``example_id`` starts with any of the
    provided company tickers, and (if years are provided) includes any
    of the fiscal years.

    Parameters
    ----------
    chunk :
        A single chunk dict with at least an ``"example_id"`` field.
    company_ids :
        Canonical ticker symbols to match (empty list = match all).
    fiscal_years :
        4-digit year strings to match (empty list = match all).

    Returns
    -------
    bool
    """
    example_id = chunk.get("example_id", "")

    # Split into parts: [TICKER, YEAR, page_N.pdf-M]
    parts = example_id.split("/")

    ticker_part = parts[0].upper() if parts else ""
    year_part = parts[1] if len(parts) > 1 else ""

    # Company filter
    if company_ids:
        if ticker_part not in company_ids:
            return False

    # Year filter
    if fiscal_years:
        if year_part not in fiscal_years:
            return False

    return True


def metadata_boost(
    chunk: dict,
    company_ids: list[str],
    fiscal_years: list[str],
    company_boost: float = 1.5,
    year_boost: float = 1.2,
) -> float:
    """Return a multiplicative score boost for metadata matches.

    Intended for use when pre-filtering is too aggressive — instead of
    dropping non-matching chunks, their retrieval scores are multiplied
    by this factor before ranking.

    Parameters
    ----------
    chunk :
        A chunk dict with ``"example_id"``.
    company_ids :
        Tickers from ``parse_query``.
    fiscal_years :
        Years from ``parse_query``.
    company_boost :
        Multiplier applied when the chunk's ticker matches.
    year_boost :
        Additional multiplier applied when the chunk's year also matches.

    Returns
    -------
    float
        ``1.0`` for no match, up to ``company_boost * year_boost`` for
        a full match.
    """
    example_id = chunk.get("example_id", "")
    parts = example_id.split("/")

    ticker_part = parts[0].upper() if parts else ""
    year_part = parts[1] if len(parts) > 1 else ""

    boost = 1.0

    if company_ids and ticker_part in company_ids:
        boost *= company_boost

    if fiscal_years and year_part in fiscal_years:
        boost *= year_boost

    return boost


def filter_chunks_by_metadata(
    chunks: list[dict],
    company_ids: list[str],
    fiscal_years: list[str],
    fallback_to_all: bool = True,
) -> list[dict]:
    """Return only the chunks that match parsed company / year metadata.

    Parameters
    ----------
    chunks :
        Full corpus chunk list.
    company_ids :
        Tickers from ``parse_query``.
    fiscal_years :
        Years from ``parse_query``.
    fallback_to_all :
        If ``True`` and the filtered result is empty, return the full
        corpus so retrieval always has candidates.  This prevents the
        degenerate case where an unrecognised company name would
        otherwise yield zero results.

    Returns
    -------
    list[dict]
        Filtered (or full) corpus.
    """
    if not company_ids and not fiscal_years:
        # Nothing to filter on; return everything
        return chunks

    filtered = [
        chunk
        for chunk in chunks
        if chunk_matches_metadata(chunk, company_ids, fiscal_years)
    ]

    if not filtered and fallback_to_all:
        return chunks

    return filtered
