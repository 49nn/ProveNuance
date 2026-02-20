"""
dependencies.py — FastAPI Dependency Injection.
Każda zależność zwraca odpowiedni adapter przez Request.app.state.
"""
from __future__ import annotations

from fastapi import Request

from adapters.document_store.postgres_document_store import PostgresDocumentStore
from adapters.entity_linker.dict_entity_linker import DictEntityLinker
from adapters.frame_extractor.llm_extractor import LLMFrameExtractor
from adapters.frame_extractor.stanza_aware_extractor import StanzaAwareExtractor
from adapters.frame_mapper.math_grade1to3_mapper import MathGrade1to3Mapper
from adapters.knowledge_store.postgres_knowledge_store import PostgresKnowledgeStore
from adapters.validator.simple_validator import SimpleValidator


def get_document_store(request: Request) -> PostgresDocumentStore:
    return request.app.state.document_store


def get_knowledge_store(request: Request) -> PostgresKnowledgeStore:
    return request.app.state.knowledge_store


def get_frame_extractor(request: Request) -> StanzaAwareExtractor:
    return request.app.state.frame_extractor


def get_frame_mapper(request: Request) -> MathGrade1to3Mapper:
    return request.app.state.frame_mapper


def get_entity_linker(request: Request) -> DictEntityLinker:
    return request.app.state.entity_linker


def get_validator(request: Request) -> SimpleValidator:
    return request.app.state.validator


def get_llm_extractor(request: Request) -> LLMFrameExtractor:
    return request.app.state.llm_extractor
