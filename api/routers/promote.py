"""
Router: POST /promote
Promuje podane fact_ids i/lub rule_ids z 'hypothesis' do 'asserted'.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_knowledge_store
from api.schemas import PromoteRequest, PromoteResponse

router = APIRouter(prefix="/promote", tags=["promote"])


@router.post("", response_model=PromoteResponse)
async def promote(
    body: PromoteRequest,
    store=Depends(get_knowledge_store),
) -> PromoteResponse:
    promoted_facts = 0
    promoted_rules = 0
    errors: list[str] = []

    for fact_id in body.fact_ids:
        try:
            await store.promote_fact(fact_id, reason=body.reason)
            promoted_facts += 1
        except KeyError as e:
            errors.append(str(e))

    for rule_id in body.rule_ids:
        try:
            await store.promote_rule(rule_id, reason=body.reason)
            promoted_rules += 1
        except KeyError as e:
            errors.append(str(e))

    return PromoteResponse(
        promoted_facts=promoted_facts,
        promoted_rules=promoted_rules,
        errors=errors,
    )
