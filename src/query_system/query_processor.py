"""
Query Processor: Processador de consultas em linguagem natural

Este módulo converte perguntas em linguagem natural para consultas SPARQL
usando pattern matching e templates pré-definidos.

Fluxo:
1. Recebe pergunta do usuário
2. Identifica padrão da pergunta
3. Extrai entidades mencionadas
4. Seleciona template apropriado
5. Gera consulta SPARQL
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .query_templates import QueryTemplates, QueryType, get_template_for_query_type

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryIntent:
    """
    Representa a intenção extraída de uma pergunta do usuário
    
    Attributes:
        query_type: Tipo de consulta identificado
        entities: Entidades mencionadas na pergunta
        confidence: Confiança na interpretação (0-1)
        raw_question: Pergunta original do usuário
    """
    query_type: QueryType
    entities: List[str]
    confidence: float
    raw_question: str


class QueryProcessor:
    """
    Processador que converte perguntas naturais em consultas SPARQL
    
    O processador usa padrões regex para identificar tipos de perguntas
    e extrai entidades mencionadas para gerar a consulta apropriada.
    """
    
    def __init__(self):
        """Inicializa o processador com padrões de reconhecimento"""
        self._setup_patterns()
    
    def _setup_patterns(self) -> None:
        """
        Define padrões regex para diferentes tipos de perguntas
        Cada padrão mapeia para um QueryType específico
        """
        
        # Padrões para "O que é X?"
        self.what_is_patterns = [
            r"(?:o que é|what is|define|definição de|conceito de)\s+([a-zA-Z_\s]+)",
            r"([a-zA-Z_\s]+)\s+(?:é o que|é|significa o que)",
            r"(?:explique|explain)\s+([a-zA-Z_\s]+)",
        ]
        
        # Padrões para "O que usa X?" / "Quais algoritmos usam X?"
        self.what_uses_patterns = [
            r"(?:o que usa|what uses|quais?.*usam?|algoritmos? que usam?)\s+([a-zA-Z_\s]+)",
            r"(?:quais?|what).*(?:implementam?|implement)\s+([a-zA-Z_\s]+)",
            r"(?:find|encontre).*(?:que usa|that uses?)\s+([a-zA-Z_\s]+)",
        ]
        
        # Padrões para "X é um tipo de que?"
        self.type_of_patterns = [
            r"([a-zA-Z_\s]+)\s+(?:é um tipo de que|is a type of what|é uma subclasse de)",
            r"(?:que tipo de|what type of).*(?:é|is)\s+([a-zA-Z_\s]+)",
            r"([a-zA-Z_\s]+)\s+(?:extends|estende|herda de)",
        ]
        
        # Padrões para "Quem criou X?"
        self.who_created_patterns = [
            r"(?:quem criou|who created|quem desenvolveu|who developed)\s+([a-zA-Z_\s]+)",
            r"(?:autor de|author of|creator of)\s+([a-zA-Z_\s]+)",
            r"([a-zA-Z_\s]+)\s+(?:foi criado por|was created by|foi desenvolvido por)",
        ]
        
        # Padrões para "Como X está relacionado com Y?"
        self.how_related_patterns = [
            r"(?:como|how)\s+([a-zA-Z_\s]+)\s+(?:está relacionado com|is related to|se relaciona com)\s+([a-zA-Z_\s]+)",
            r"(?:relação entre|relationship between)\s+([a-zA-Z_\s]+)\s+(?:e|and)\s+([a-zA-Z_\s]+)",
        ]
        
        # Padrões para "Liste todos os X" / "Quais são os algoritmos?"
        self.list_by_type_patterns = [
            r"(?:liste|list|quais são|what are).*?(algoritmos?|algorithms?|conceitos?|concepts?|métricas?|metrics?)",
            r"(?:todos os|all|all the)\s+(algoritmos?|algorithms?|conceitos?|concepts?|métricas?|metrics?)",
            r"(?:show|mostre).*?(algoritmos?|algorithms?|conceitos?|concepts?|métricas?|metrics?)",
        ]
        
        # Padrões para "Encontre similares a X"
        self.find_similar_patterns = [
            r"(?:encontre|find|busque).*(?:similar|parecido|semelhante).*(?:a|to|with)\s+([a-zA-Z_\s]+)",
            r"(?:conceitos?|algorithms?).*(?:similar|parecido|semelhante).*(?:a|to)\s+([a-zA-Z_\s]+)",
        ]
    
    def process_question(self, question: str) -> QueryIntent:
        """
        Processa uma pergunta em linguagem natural e identifica a intenção
        
        Args:
            question: Pergunta do usuário
            
        Returns:
            QueryIntent com tipo de consulta e entidades identificadas
        """
        question = question.lower().strip()
        logger.info(f"🔍 Processando pergunta: '{question}'")
        
        # Tenta cada padrão em ordem de prioridade
        intent_checks = [
            (self.what_is_patterns, QueryType.WHAT_IS, self._extract_single_entity),
            (self.what_uses_patterns, QueryType.WHAT_USES, self._extract_single_entity),
            (self.type_of_patterns, QueryType.WHAT_IS_TYPE_OF, self._extract_single_entity),
            (self.who_created_patterns, QueryType.WHO_CREATED, self._extract_single_entity),
            (self.how_related_patterns, QueryType.HOW_RELATED, self._extract_two_entities),
            (self.list_by_type_patterns, QueryType.LIST_BY_TYPE, self._extract_type_entity),
            (self.find_similar_patterns, QueryType.FIND_SIMILAR, self._extract_single_entity),
        ]
        
        for patterns, query_type, entity_extractor in intent_checks:
            for pattern in patterns:
                match = re.search(pattern, question, re.IGNORECASE)
                if match:
                    entities = entity_extractor(match)
                    if entities:
                        logger.info(f"✅ Identificado: {query_type.value}, entidades: {entities}")
                        return QueryIntent(
                            query_type=query_type,
                            entities=entities,
                            confidence=0.8,  # Confidence básica
                            raw_question=question
                        )
        
        # Se nenhum padrão foi encontrado, assumir busca geral
        logger.warning(f"⚠️ Padrão não identificado, usando busca geral")
        return QueryIntent(
            query_type=QueryType.WHAT_IS,
            entities=[question.split()[-1]],  # Última palavra como entidade
            confidence=0.3,
            raw_question=question
        )
    
    def _extract_single_entity(self, match: re.Match) -> List[str]:
        """Extrai uma única entidade do match regex"""
        entity = match.group(1).strip()
        return [self._normalize_entity_name(entity)]
    
    def _extract_two_entities(self, match: re.Match) -> List[str]:
        """Extrai duas entidades do match regex"""
        entity1 = match.group(1).strip()
        entity2 = match.group(2).strip()
        return [
            self._normalize_entity_name(entity1),
            self._normalize_entity_name(entity2)
        ]
    
    def _extract_type_entity(self, match: re.Match) -> List[str]:
        """Extrai tipo de entidade do match regex"""
        entity_type = match.group(1).strip()
        # Mapeia tipos em português/inglês para tipos do KG
        type_mapping = {
            'algoritmos': 'algorithm',
            'algoritmo': 'algorithm', 
            'algorithms': 'algorithm',
            'algorithm': 'algorithm',
            'conceitos': 'concept',
            'conceito': 'concept',
            'concepts': 'concept',
            'concept': 'concept',
            'métricas': 'metric',
            'métrica': 'metric',
            'metrics': 'metric',
            'metric': 'metric',
        }
        
        normalized_type = type_mapping.get(entity_type.lower(), entity_type)
        return [normalized_type]
    
    def _normalize_entity_name(self, entity: str) -> str:
        """
        Normaliza o nome da entidade para o formato usado no KG
        
        Args:
            entity: Nome da entidade raw
            
        Returns:
            Nome normalizado (lowercase, underscores, etc.)
        """
        # Remove artigos e preposições comuns
        stop_words = ['o', 'a', 'os', 'as', 'de', 'da', 'do', 'das', 'dos', 'the', 'of', 'for']
        
        words = entity.lower().split()
        filtered_words = [word for word in words if word not in stop_words]
        
        # Junta com underscores e remove caracteres especiais
        normalized = '_'.join(filtered_words)
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)
        
        return normalized
    
    def generate_sparql_query(self, intent: QueryIntent) -> str:
        """
        Gera consulta SPARQL baseada na intenção identificada
        
        Args:
            intent: Intenção processada da pergunta
            
        Returns:
            Consulta SPARQL completa
        """
        template_func = get_template_for_query_type(intent.query_type)
        
        if not template_func:
            raise ValueError(f"Template não encontrado para {intent.query_type}")
        
        try:
            if intent.query_type == QueryType.HOW_RELATED:
                # Consultas de relação precisam de duas entidades
                if len(intent.entities) >= 2:
                    query = template_func(intent.entities[0], intent.entities[1])
                else:
                    raise ValueError("Consulta de relação precisa de duas entidades")
            
            elif intent.query_type == QueryType.LIST_BY_TYPE:
                # Consultas de listagem usam o tipo de entidade
                query = template_func(intent.entities[0], limit=15)
            
            else:
                # Consultas padrão usam uma entidade
                query = template_func(intent.entities[0])
            
            logger.info(f"✅ SPARQL gerado para {intent.query_type.value}")
            return query
            
        except Exception as e:
            logger.error(f"❌ Erro ao gerar SPARQL: {e}")
            raise
    
    def process_and_generate(self, question: str) -> Tuple[QueryIntent, str]:
        """
        Método conveniente que processa pergunta e gera SPARQL de uma vez
        
        Args:
            question: Pergunta em linguagem natural
            
        Returns:
            Tupla com (intenção, consulta SPARQL)
        """
        intent = self.process_question(question)
        sparql_query = self.generate_sparql_query(intent)
        
        return intent, sparql_query


# Função utilitária para uso direto
def create_query_processor() -> QueryProcessor:
    """
    Factory function para criar instância do QueryProcessor
    
    Returns:
        Instância configurada do QueryProcessor
    """
    return QueryProcessor()


if __name__ == "__main__":
    # Teste do processador
    print("🧪 Testando Query Processor...")
    
    processor = create_query_processor()
    
    # Exemplos de perguntas
    test_questions = [
        "O que é gradient descent?",
        "Quais algoritmos usam backpropagation?",
        "Adam optimizer é um tipo de que?",
        "Quem criou o algoritmo SVM?",
        "Liste todos os algoritmos",
        "Como neural network está relacionado com deep learning?",
    ]
    
    print("\n🔍 Testando perguntas:")
    for question in test_questions:
        try:
            intent, sparql = processor.process_and_generate(question)
            print(f"\n📝 '{question}'")
            print(f"   → Tipo: {intent.query_type.value}")
            print(f"   → Entidades: {intent.entities}")
            print(f"   → Confidence: {intent.confidence}")
            print(f"   → SPARQL: {len(sparql)} caracteres")
        except Exception as e:
            print(f"   ❌ Erro: {e}")
    
    print("\n✅ Processor funcionando!")