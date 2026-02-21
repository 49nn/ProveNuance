from adapters.frame_mapper.math_grade1to3_mapper import MathGrade1to3Mapper
from contracts import ArithExampleFrame, Provenance


def test_mapper_decomposes_chained_addition_into_binary_facts():
    mapper = MathGrade1to3Mapper()
    frame = ArithExampleFrame(
        operation="add",
        operands=[1, 2, 3],
        result=6,
        source_span_id="s1",
    )

    result = mapper.map(frame, Provenance(span_ids=["s1"], doc_ref="doc1"))

    eq_facts = [f for f in result.facts if f.r == "eq"]
    atom_facts = [f for f in result.facts if f.r == "instance_of"]

    assert result.warnings == []
    assert len(eq_facts) == 2
    assert len(atom_facts) == 2
    assert (eq_facts[0].h, eq_facts[0].t) == ("add(1,2)", "3")
    assert (eq_facts[1].h, eq_facts[1].t) == ("add(3,3)", "6")
    assert (atom_facts[0].h, atom_facts[0].t) == ("add(1,2,3)", "add")
    assert (atom_facts[1].h, atom_facts[1].t) == ("add(3,3,6)", "add")


def test_mapper_returns_warning_when_chain_intermediate_cannot_be_computed():
    mapper = MathGrade1to3Mapper()
    frame = ArithExampleFrame(
        operation="add",
        operands=[1, "X", 3],
        result=6,
        source_span_id="s1",
    )

    result = mapper.map(frame, Provenance(span_ids=["s1"], doc_ref="doc1"))

    assert result.facts == []
    assert result.rules == []
    assert result.warnings
