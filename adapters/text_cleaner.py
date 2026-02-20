"""
text_cleaner.py - clean raw text from markdown/html markup before ingest.

Pipeline:
1) Markdown -> HTML
2) Remove tags while preserving meaningful line breaks
3) Normalize whitespace per line (without flattening newlines)
"""
from __future__ import annotations

import re

import markdown
from bs4 import BeautifulSoup

_BLOCK_TAGS = (
    "p", "div", "section", "article", "header", "footer",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "blockquote", "pre",
    "table", "thead", "tbody", "tr",
)


def clean_markup(text: str) -> str:
    """Return plain text with markdown/html removed and line breaks preserved."""
    # Keep explicit single-line breaks from markdown source.
    html = markdown.markdown(text, extensions=["extra", "nl2br"])

    soup = BeautifulSoup(html, "html.parser")

    # Preserve hard line breaks and block boundaries.
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.append("\n")

    plain = soup.get_text(separator="", strip=False)
    plain = plain.replace("\r\n", "\n").replace("\r", "\n")
    plain = re.sub(r"\n{2,}", "\n", plain)

    # Normalize spaces/tabs inside each line, keep newlines.
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in plain.split("\n")]

    # Trim blank lines on edges.
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)
