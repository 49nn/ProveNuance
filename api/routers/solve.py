"""
Router: POST /solve
[Etap 3 + 4] — Wymaga MathProblemParser i Reasoner.
Na razie zwraca 501 z informacją.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.schemas import SolveRequest

router = APIRouter(prefix="/solve", tags=["solve"])


@router.post("")
async def solve(body: SolveRequest):
    return JSONResponse(
        status_code=501,
        content={
            "message": "Not implemented — requires Etap 3 (MathProblemParser) and Etap 4 (Reasoner).",
            "text": body.text,
        },
    )
