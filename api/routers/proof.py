"""
Router: GET /proof/{proof_id}
[Etap 4] — Wymaga Reasoner.
Na razie zwraca 501 z informacją.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/proof", tags=["proof"])


@router.get("/{proof_id}")
async def get_proof(proof_id: str):
    return JSONResponse(
        status_code=501,
        content={
            "message": "Not implemented — requires Etap 4 (Reasoner).",
            "proof_id": proof_id,
        },
    )
