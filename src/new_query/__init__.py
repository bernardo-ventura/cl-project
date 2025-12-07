"""
New Query System: Sistema de consultas redesenhado SIMPLES

Etapas implementadas:
1. QuestionEntityExtractor: Extrai entidades de perguntas reutilizando sistema KG
2. EntityMapper: Mapeia candidatos para entidades exatas do KG
"""

__version__ = "2.0.0"
__author__ = "New Query System"

from .question_entity_extractor import QuestionEntityExtractor, create_question_entity_extractor
from .entity_mapper import EntityMapper, create_entity_mapper

__all__ = [
    'QuestionEntityExtractor',
    'create_question_entity_extractor',
    'EntityMapper', 
    'create_entity_mapper'
]