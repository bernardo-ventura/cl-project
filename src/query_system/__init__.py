"""
Sistema de Consultas para Knowledge Graph ML/DL

Este módulo implementa um sistema completo para consultar o Knowledge Graph
construído a partir da literatura de Machine Learning e Deep Learning.

Componentes:
- KGExecutor: Executa consultas SPARQL no grafo RDF
- QueryProcessor: Converte perguntas naturais em SPARQL
- ResponseFormatter: Formata resultados para apresentação
- QueryTemplates: Templates SPARQL reutilizáveis
"""

__version__ = "1.0.0"
__author__ = "Knowledge Graph Query System"

from .kg_executor import KGExecutor
from .query_processor import QueryProcessor
from .response_formatter import ResponseFormatter
from .query_templates import QueryTemplates

__all__ = [
    'KGExecutor',
    'QueryProcessor', 
    'ResponseFormatter',
    'QueryTemplates'
]