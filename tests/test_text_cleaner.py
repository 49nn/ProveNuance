from adapters.text_cleaner import clean_markup


def test_clean_markup_preserves_single_line_breaks():
    text = "Dodawanie to **laczenie** dwoch liczb.\nWynik to **suma**."
    cleaned = clean_markup(text)
    assert cleaned == "Dodawanie to laczenie dwoch liczb.\nWynik to suma."


def test_clean_markup_keeps_block_boundaries():
    text = "# Naglowek\n\nAkapit pierwszy.\n\n- punkt 1\n- punkt 2"
    cleaned = clean_markup(text)
    assert "Naglowek\nAkapit pierwszy." in cleaned
    assert "punkt 1\npunkt 2" in cleaned
