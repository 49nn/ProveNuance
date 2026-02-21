from adapters.frame_extractor.rule_based_extractor import RuleBasedExtractor
from contracts import DocSpan


def test_rule_based_extractor_parses_reversed_chained_equation():
    extractor = RuleBasedExtractor()
    span = DocSpan(
        doc_id="d1",
        version=1,
        surface_text="6 = 1 + 2 + 3",
        start_char=0,
        end_char=13,
    )

    frames = extractor.extract_frames(span)
    assert len(frames) == 1
    frame = frames[0]
    assert frame.operation == "add"
    assert frame.operands == [1, 2, 3]
    assert frame.result == 6
