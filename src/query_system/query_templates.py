"""
Query Templates: Templates SPARQL reutilizáveis

Este módulo contém templates SPARQL pré-definidos para diferentes tipos
de consultas comuns no Knowledge Graph ML/DL.

Cada template é uma string com placeholders que serão substituídos
pelo QueryProcessor baseado na pergunta do usuário.
"""

from typing import Dict, List
from enum import Enum


class QueryType(Enum):
    """Tipos de consultas suportadas"""
    WHAT_IS = "what_is"                    # "O que é X?"
    WHAT_USES = "what_uses"               # "O que usa X?"
    WHAT_IS_TYPE_OF = "what_is_type_of"   # "X é um tipo de que?"
    WHO_CREATED = "who_created"           # "Quem criou X?"
    HOW_RELATED = "how_related"           # "Como X está relacionado com Y?"
    LIST_BY_TYPE = "list_by_type"         # "Liste todos os algoritmos"
    FIND_SIMILAR = "find_similar"         # "Encontre conceitos similares a X"


class QueryTemplates:
    """
    Coleção de templates SPARQL para diferentes tipos de consultas
    
    Cada template usa placeholders como {entity}, {relation}, {type}
    que são substituídos pelo QueryProcessor
    """
    
    # Prefixos comuns para todas as consultas
    PREFIXES = """
        PREFIX ml: <http://ml-kg.org/ontology/>
        PREFIX entity: <http://ml-kg.org/entity/>
        PREFIX relation: <http://ml-kg.org/relation/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    """
    
    @staticmethod
    def get_prefixes() -> str:
        """Retorna os prefixos padrão para todas as consultas"""
        return QueryTemplates.PREFIXES.strip()
    
    @staticmethod
    def what_is_entity(entity_name: str) -> str:
        """
        Template: "O que é X?"
        Busca definição, tipo e propriedades de uma entidade
        
        Args:
            entity_name: Nome da entidade (ex: "gradient_descent")
            
        Returns:
            Consulta SPARQL completa
        """
        return f"""
        {QueryTemplates.get_prefixes()}
        
        SELECT ?type ?label ?property ?value WHERE {{
            entity:{entity_name} rdf:type ?type .
            OPTIONAL {{ entity:{entity_name} rdfs:label ?label . }}
            OPTIONAL {{ entity:{entity_name} ?property ?value . }}
        }}
        """
    
    @staticmethod
    def what_uses_entity(entity_name: str) -> str:
        """
        Template: "O que usa X?" / "Quais algoritmos usam X?"
        Busca entidades que usam/implementam a entidade especificada
        
        Args:
            entity_name: Nome da entidade target
            
        Returns:
            Consulta SPARQL para encontrar usuários da entidade
        """
        return f"""
        {QueryTemplates.get_prefixes()}
        
        SELECT ?user ?userLabel ?userType ?relation WHERE {{
            ?user ?relation entity:{entity_name} .
            ?user rdf:type ?userType .
            ?user rdfs:label ?userLabel .
            
            FILTER(?relation IN (relation:uses, relation:implements, relation:applies_to))
        }}
        ORDER BY ?userType ?userLabel
        """
    
    @staticmethod
    def what_is_type_of(entity_name: str) -> str:
        """
        Template: "X é um tipo de que?" / "X é uma subclasse de que?"
        Busca hierarquia parent da entidade
        
        Args:
            entity_name: Nome da entidade
            
        Returns:
            Consulta para encontrar tipos parent
        """
        return f"""
        {QueryTemplates.get_prefixes()}
        
        SELECT ?parent ?parentLabel WHERE {{
            entity:{entity_name} relation:is_a ?parent .
            OPTIONAL {{ ?parent rdfs:label ?parentLabel . }}
        }}
        """
    
    @staticmethod
    def who_created_entity(entity_name: str) -> str:
        """
        Template: "Quem criou X?" / "Quem desenvolveu X?"
        Busca autores/criadores da entidade
        
        Args:
            entity_name: Nome da entidade
            
        Returns:
            Consulta para encontrar criadores
        """
        return f"""
        {QueryTemplates.get_prefixes()}
        
        SELECT ?creator ?creatorLabel WHERE {{
            {{ 
                entity:{entity_name} relation:developed_by ?creator .
                OPTIONAL {{ ?creator rdfs:label ?creatorLabel . }}
            }}
            UNION
            {{
                entity:{entity_name} relation:proposed_by ?creator .
                OPTIONAL {{ ?creator rdfs:label ?creatorLabel . }}
            }}
        }}
        """
    
    @staticmethod
    def how_entities_related(entity1: str, entity2: str) -> str:
        """
        Template: "Como X está relacionado com Y?"
        Busca todas as relações entre duas entidades
        
        Args:
            entity1: Primeira entidade
            entity2: Segunda entidade
            
        Returns:
            Consulta para encontrar relações entre entidades
        """
        return f"""
        {QueryTemplates.get_prefixes()}
        
        SELECT ?relation WHERE {{
            {{ entity:{entity1} ?relation entity:{entity2} . }}
            UNION
            {{ entity:{entity2} ?relation entity:{entity1} . }}
        }}
        """
    
    @staticmethod
    def list_entities_by_type(entity_type: str, limit: int = 20) -> str:
        """
        Template: "Liste todos os X" / "Quais são os algoritmos?"
        Lista entidades de um tipo específico
        
        Args:
            entity_type: Tipo da entidade (ex: "algorithm", "concept")
            limit: Número máximo de resultados
            
        Returns:
            Consulta para listar entidades por tipo
        """
        return f"""
        {QueryTemplates.get_prefixes()}
        
        SELECT ?entity ?label WHERE {{
            ?entity rdf:type ml:{entity_type} .
            ?entity rdfs:label ?label .
        }}
        ORDER BY ?label
        LIMIT {limit}
        """
    
    @staticmethod
    def find_similar_entities(entity_name: str, limit: int = 10) -> str:
        """
        Template: "Encontre conceitos similares a X"
        Busca entidades do mesmo tipo ou com relações similares
        
        Args:
            entity_name: Entidade de referência
            limit: Número máximo de resultados
            
        Returns:
            Consulta para encontrar entidades similares
        """
        return f"""
        {QueryTemplates.get_prefixes()}
        
        SELECT ?similar ?similarLabel ?commonType WHERE {{
            entity:{entity_name} rdf:type ?commonType .
            ?similar rdf:type ?commonType .
            ?similar rdfs:label ?similarLabel .
            
            FILTER(?similar != entity:{entity_name})
        }}
        ORDER BY ?similarLabel
        LIMIT {limit}
        """
    
    @staticmethod
    def get_entity_relations(entity_name: str) -> str:
        """
        Template: Busca todas as relações de uma entidade
        Útil para exploração geral
        
        Args:
            entity_name: Nome da entidade
            
        Returns:
            Consulta para todas as relações da entidade
        """
        return f"""
        {QueryTemplates.get_prefixes()}
        
        SELECT ?relation ?target ?targetLabel WHERE {{
            {{ entity:{entity_name} ?relation ?target . }}
            UNION  
            {{ ?target ?relation entity:{entity_name} . }}
            
            OPTIONAL {{ ?target rdfs:label ?targetLabel . }}
            
            # Filtra apenas relações do nosso domínio
            FILTER(STRSTARTS(STR(?relation), STR(relation:)))
        }}
        ORDER BY ?relation
        """


# Mapeamento de tipos de pergunta para templates
QUERY_TYPE_MAPPING = {
    QueryType.WHAT_IS: QueryTemplates.what_is_entity,
    QueryType.WHAT_USES: QueryTemplates.what_uses_entity,
    QueryType.WHAT_IS_TYPE_OF: QueryTemplates.what_is_type_of,
    QueryType.WHO_CREATED: QueryTemplates.who_created_entity,
    QueryType.HOW_RELATED: QueryTemplates.how_entities_related,
    QueryType.LIST_BY_TYPE: QueryTemplates.list_entities_by_type,
    QueryType.FIND_SIMILAR: QueryTemplates.find_similar_entities,
}


def get_template_for_query_type(query_type: QueryType) -> callable:
    """
    Retorna a função de template apropriada para um tipo de consulta
    
    Args:
        query_type: Tipo da consulta
        
    Returns:
        Função template correspondente
    """
    return QUERY_TYPE_MAPPING.get(query_type)


if __name__ == "__main__":
    # Teste dos templates
    print("🧪 Testando Query Templates...")
    
    # Exemplo de cada tipo de template
    examples = [
        ("WHAT_IS", QueryTemplates.what_is_entity("gradient_descent")),
        ("WHAT_USES", QueryTemplates.what_uses_entity("gradient_descent")),
        ("WHO_CREATED", QueryTemplates.who_created_entity("adam_optimizer")),
        ("LIST_BY_TYPE", QueryTemplates.list_entities_by_type("algorithm")),
    ]
    
    for name, query in examples:
        print(f"\n📋 Template {name}:")
        print(query.strip()[:200] + "...")
    
    print("\n✅ Templates funcionando!")