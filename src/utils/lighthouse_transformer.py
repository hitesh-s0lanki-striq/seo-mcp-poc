"""Lighthouse JSON transformer to extract SEO-focused summaries from raw Lighthouse data."""

from typing import Any, Dict, Optional, List


# Which Lighthouse audits we consider important for SEO / CWV.
IMPORTANT_AUDITS = [
    # Core Web Vitals / performance
    "first-contentful-paint",
    "largest-contentful-paint",
    "speed-index",
    "total-blocking-time",
    "cumulative-layout-shift",
    "max-potential-fid",
    "interactive",              # TTI if present
    "server-response-time",
    "uses-http2",
    "redirects",

    # Technical SEO
    "is-on-https",
    "viewport",
    "http-status-code",
    "meta-description",
    "font-size",
    "link-text",
    "crawlable-anchors",
    "is-crawlable",
    "robots-txt",
    "hreflang",
    "canonical",
    "structured-data",

    # Diagnostics / quality signals
    "errors-in-console",
]


def _extract_audit(audits: Dict[str, Any], audit_id: str) -> Optional[Dict[str, Any]]:
    """
    Safely extract a compact view of a single Lighthouse audit.
    Returns None if the audit is missing.
    """
    raw = audits.get(audit_id)
    if not isinstance(raw, dict):
        return None

    return {
        "id": raw.get("id", audit_id),
        "title": raw.get("title"),
        "score": raw.get("score"),
        "score_display_mode": raw.get("scoreDisplayMode"),
        "display_value": raw.get("displayValue"),
        "numeric_value": raw.get("numericValue"),
        "numeric_unit": raw.get("numericUnit"),
    }


def extract_lighthouse_seo_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take the raw on_page_lighthouse JSON and return a small, SEO-focused summary.

    This is designed so your SEO analysis can rely ONLY on this optimized JSON.
    """
    data = payload.get("data", {})
    items: List[Dict[str, Any]] = data.get("items", [])
    item = items[0] if items else {}

    audits: Dict[str, Any] = item.get("audits", {})
    categories: Dict[str, Any] = item.get("categories", {})

    # --- Meta info about the run ---
    meta = {
        "tool_name": payload.get("tool_name"),
        "timestamp": payload.get("timestamp"),
        "lighthouse_version": item.get("lighthouseVersion"),
        "requested_url": item.get("requestedUrl"),
        "final_url": item.get("finalUrl") or item.get("finalDisplayedUrl"),
        "fetch_time": item.get("fetchTime"),
        "user_agent": item.get("userAgent"),
        "run_warnings": item.get("runWarnings", []),
    }

    # --- High-level category scores (0–1) ---
    def cat_score(cat_key: str) -> Optional[float]:
        cat = categories.get(cat_key) or {}
        return cat.get("score")

    scores = {
        "performance": cat_score("performance"),
        "seo": cat_score("seo"),
        "accessibility": cat_score("accessibility"),
        "best_practices": cat_score("best-practices"),
        "pwa": cat_score("pwa"),
    }

    # --- Core Web Vitals & timing metrics ---
    def num(aid: str) -> Optional[float]:
        a = audits.get(aid) or {}
        return a.get("numericValue")

    core_web_vitals = {
        "fcp_ms": num("first-contentful-paint"),
        "lcp_ms": num("largest-contentful-paint"),
        "speed_index_ms": num("speed-index"),
        "tbt_ms": num("total-blocking-time"),
        "cls": num("cumulative-layout-shift"),
        "tti_ms": num("interactive"),
        "max_potential_fid_ms": num("max-potential-fid"),
        "server_response_time_ms": num("server-response-time"),
    }

    # --- Technical SEO flags / booleans from audit scores ---
    def passed(aid: str) -> Optional[bool]:
        a = audits.get(aid) or {}
        score = a.get("score")
        if score is None:
            return None
        # Lighthouse uses 0/1 for binary, sometimes decimals; treat ≥0.9 as pass.
        return float(score) >= 0.9

    technical_seo = {
        "https": passed("is-on-https"),
        "viewport_meta": passed("viewport"),
        "http_status_ok": passed("http-status-code"),
        "has_meta_description": passed("meta-description"),
        "font_size_ok": passed("font-size"),
        "link_text_ok": passed("link-text"),
        "crawlable_anchors_ok": passed("crawlable-anchors"),
        "page_crawlable": passed("is-crawlable"),
        "robots_txt_valid": passed("robots-txt"),
        "hreflang_valid": passed("hreflang"),
        # canonical/structured-data can be N/A/manual; keep as raw scoreDisplayMode info
        "canonical_status": audits.get("canonical", {}).get("scoreDisplayMode"),
        "structured_data_status": audits.get("structured-data", {}).get("scoreDisplayMode"),
    }

    # --- Compact list of important audits (sorted by id) ---
    important_audits: List[Dict[str, Any]] = []
    for aid in sorted(IMPORTANT_AUDITS):
        a = _extract_audit(audits, aid)
        if a is not None:
            important_audits.append(a)

    # --- Console errors (just messages, trimmed) ---
    errors_raw = audits.get("errors-in-console", {})
    errors_details = errors_raw.get("details", {}) or {}
    error_items = errors_details.get("items", []) or []

    console_errors = [
        {
            "source": it.get("source"),
            "description": it.get("description"),
            "url": (it.get("sourceLocation") or {}).get("url"),
        }
        for it in error_items
    ]

    # Final optimized JSON – keys are ordered intentionally
    return {
        "meta": meta,
        "scores": scores,
        "core_web_vitals": core_web_vitals,
        "technical_seo": technical_seo,
        "important_audits": important_audits,
        "console_errors": console_errors,
    }

