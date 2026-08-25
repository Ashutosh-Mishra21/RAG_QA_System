from __future__ import annotations

import re

from backend.app.models import DocumentManifest, PageManifest, PageRoutingDecision


class PageRoutingPolicy:
    """Select bounded parser actions from cheap page observations."""

    def decide(self, manifest: DocumentManifest) -> list[PageRoutingDecision]:
        decisions = [self._decide_page(page) for page in manifest.pages]
        manifest.routing = decisions
        return decisions

    def _decide_page(self, page: PageManifest) -> PageRoutingDecision:
        actions = ["use_native_text"] if page.signals.text else []
        reasons = []
        priority = 0

        if page.signals.scanned:
            actions = ["parse_ocr"]
            reasons.append("page has images but no extractable text")
            priority = 10
        if page.signals.table:
            actions.append("parse_table")
            reasons.append("table signal detected")
            priority = max(priority, 8)
        if page.signals.chart or page.signals.formula:
            actions.append("parse_visual")
            reasons.append("visual or formula signal detected")
            priority = max(priority, 7)
        if page.signals.image and not page.signals.scanned:
            actions.append("inspect_visual")
            reasons.append("page contains an image")
            priority = max(priority, 5)

        return PageRoutingDecision(
            page_id=page.page_id,
            actions=list(dict.fromkeys(actions)),
            reason="; ".join(reasons) or "native text is sufficient",
            priority=priority,
        )


def detect_page_signals(
    text: str,
    image_count: int,
    block_count: int,
    page_area: float,
) -> tuple[bool, bool, bool, int, float]:
    """Conservatively infer table, chart, formula, numeric, and density signals."""
    normalized = text.casefold()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    numeric_tokens = len(re.findall(r"(?<!\w)\d+(?:[.,]\d+)?%?(?!\w)", text))
    table_terms = len(
        re.findall(r"\b(table|row|column|total|mean|median|n\s*=)\b", normalized)
    )
    chart_terms = len(
        re.findall(r"\b(figure|fig\.?|chart|plot|axis|legend|trend)\b", normalized)
    )
    formula = bool(re.search(r"[=∑√∫]|\b(eq(?:uation)?|formula)\b", text, re.I))
    short_numeric_lines = sum(
        1 for line in lines if len(line.split()) <= 8 and re.search(r"\d", line)
    )
    table = (
        table_terms >= 2
        or short_numeric_lines >= 4
        or (block_count >= 8 and numeric_tokens >= 6)
    )
    chart = chart_terms >= 1 and image_count > 0
    density = min(1.0, len(text) / max(page_area * 0.08, 1.0))
    return table, chart, formula, numeric_tokens, density
