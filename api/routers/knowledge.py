"""
Router: GET /facts, GET /rules
Listuje fakty i reguły z KnowledgeStore.
"""
from fastapi import APIRouter, Depends, Query

from api.dependencies import get_knowledge_store
from contracts import Fact, Rule

router = APIRouter(tags=["knowledge"])


@router.get("/facts", response_model=list[Fact])
async def list_facts(
    status: str = Query("asserted", description="hypothesis|asserted|retracted|all"),
    limit: int = Query(200, ge=1, le=1000),
    store=Depends(get_knowledge_store),
) -> list[Fact]:
    return await store.list_facts(status=status, limit=limit)


@router.get("/rules", response_model=list[Rule])
async def list_rules(
    status: str = Query("asserted", description="hypothesis|asserted|all"),
    limit: int = Query(200, ge=1, le=1000),
    store=Depends(get_knowledge_store),
) -> list[Rule]:
    return await store.list_rules(status=status, limit=limit)
