"""
text_cleaner.py — czyszczenie raw_text z markup przed ingestem.

Pipeline:
    1. markdown  →  Markdown → HTML  (np. **bold** → <strong>bold</strong>)
    2. BeautifulSoup  →  strip wszystkich tagów HTML/XML → czysty tekst
    3. normalizacja białych znaków
"""
import re

import markdown
from bs4 import BeautifulSoup


def clean_markup(text: str) -> str:
    """Zwraca czysty tekst pozbawiony znaczników Markdown, HTML i XML."""
    # 1. Markdown → HTML (na czystym tekście/surowym HTML jest idempotentne)
    html = markdown.markdown(text, extensions=["extra"])

    # 2. Stripowanie tagów HTML/XML
    soup = BeautifulSoup(html, "html.parser")
    plain = soup.get_text(separator=" ")

    # 3. Normalizacja białych znaków
    plain = re.sub(r"\s+", " ", plain).strip()

    return plain
