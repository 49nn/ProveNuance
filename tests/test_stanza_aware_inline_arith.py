from adapters.frame_extractor.stanza_aware_extractor import (
    _extract_inline_arith_frames_from_text,
)


def test_extract_inline_arith_from_answer_list_lines():
    text = (
        "4) Zadania do liczenia - odpowiedzi\n"
        "1 + 4 = 5\n"
        "2 + 6 = 8\n"
        "3 + 5 = 8\n"
    )
    frames = _extract_inline_arith_frames_from_text(text, span_id="s1")
    triples = [(f.operation, f.operands[0], f.operands[1], f.result) for f in frames]

    assert ("add", 1, 4, 5) in triples
    assert ("add", 2, 6, 8) in triples
    assert ("add", 3, 5, 8) in triples


def test_extract_inline_arith_ignores_chained_expression_middle_match():
    text = "5 + 1 + 2 = 8"
    frames = _extract_inline_arith_frames_from_text(text, span_id="s1")

    assert frames == []
