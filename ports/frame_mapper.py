"""
Port: FrameMapper
Odpowiedzialność: deterministyczne mapowanie frame na fakty/reguły z provenance.
"""
from typing import Protocol, runtime_checkable

from contracts import Frame, MappingResult, Provenance


@runtime_checkable
class FrameMapper(Protocol):
    def map(self, frame: Frame, provenance: Provenance) -> MappingResult:
        """
        Maps a Frame to a set of Facts and/or Rules.
        Provenance is attached to every produced Fact/Rule.
        Returns MappingResult with facts, rules, and optional warnings.
        """
        ...
