# Stanza Math NER — Synthetic BIO Dataset (PL)

Synthetic starter dataset for training Stanza NER with a math-oriented taxonomy.

## Files
- `train.conll` — 160 sentences
- `dev.conll` — 40 sentences
- `test.conll` — 40 sentences

Format: one token per line, tab-separated: `TOKEN<TAB>NER_TAG`, sentences separated by a blank line.
NER tags follow BIO: `O`, `B-<LABEL>`, `I-<LABEL>`.

## Labels
THEOREM, LEMMA, DEF, STRUCTURE, FUNCTION, CONSTANT, AUTHOR, REF, CONJECTURE, AXIOM, PROPOSITION, COROLLARY, SET, SEQUENCE

## Notes
- Template-generated; intended for plumbing/testing, not for production quality.
- Replace with human-annotated examples from your target corpus for real performance.
