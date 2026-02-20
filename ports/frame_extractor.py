"""
Port: FrameExtractor
Odpowiedzialność: z DocSpan wyciąga "frames" w kontrolowanym DSL.
"""
from typing import Protocol, runtime_checkable

from contracts import DocSpan, Frame, ValidationIssue


@runtime_checkable
class FrameExtractor(Protocol):
    def extract_frames(self, span: DocSpan) -> list[Frame]:
        """
        Extracts structured frames from a DocSpan's surface_text.
        Returns a (possibly empty) list of Frame objects.
        Implementation: regex patterns (MVP) or LLM (later).
        """
        ...

    def validate_frame(self, frame: Frame) -> list[ValidationIssue]:
        """
        Validates a single frame for structural correctness.
        Returns list of ValidationIssue; empty = frame is valid.
        """
        ...
