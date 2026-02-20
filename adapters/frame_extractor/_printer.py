"""
_printer.py — wyświetlanie ramek na stdout (tryb verbose).

Używane przez wszystkie adaptery FrameExtractor gdy verbose=True.
"""
from __future__ import annotations

from contracts import Frame

_BAR = "─" * 64


def print_frames(surface_text: str, frames: list[Frame]) -> None:
    """Drukuje span i listę wyekstrahowanych ramek na stdout."""
    preview = surface_text[:80] + ("…" if len(surface_text) > 80 else "")
    print(_BAR)
    print(f"SPAN » {preview}")
    if not frames:
        print("  (brak ramek)")
        print(_BAR)
        return
    for i, frame in enumerate(frames, 1):
        ft = frame.frame_type.value
        data = frame.model_dump(exclude={"frame_type", "source_span_id"})
        slots = "  ".join(f"{k}={v!r}" for k, v in data.items())
        print(f"  [{i}] {ft}  {slots}")
    print(_BAR)
