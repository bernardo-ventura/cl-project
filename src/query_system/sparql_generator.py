"""
SPARQL Generator: Gera consultas SPARQL usando templates pré-definidos

Objetivo: Criar SPARQL válido e executável baseado em entidades, intenção e predicados.
Garante confiabilidade, performance e manutenibilidade.

Abordagem: Templates pré-definidos por intenção com substituição de parâmetros.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

# Importar classes do pipeline
from .intent_classifier import IntentResult
from .predicate_selector import PredicateSelectionResult

logger = logging.getLogger(__name__)


@dataclass
class SPARQLGenerationResult:
    """Resultado da geração de consulta SPARQL."""
    sparql_query: str
    query_type: str  # 'describe', 'compare', 'relations', 'list', etc.
    template_used: str  # 'describe', 'comparison', 'relationship', etc.
    entities_count: int
    predicates_count: int
    generation_method: str  # 'template_based', 'fallback'


class SPARQLGenerator:
    """Gerador de consultas SPARQL usando templates pré-definidos"""
    
    def __init__(self):
        """Inicializa gerador de SPARQL"""
        logger.info("🔄 Inicializando SPARQLGenerator...")
        
        # Prefixos padrão para todas as consultas
        self.prefixes = """
PREFIX ml: <http://ml-kg.org/ontology/>
PREFIX entity: <http://ml-kg.org/entity/>
PREFIX relation: <http://ml-kg.org/relation/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
""".strip()
        
        # Mapeamento query_type → template methods
        self.template_mapping = {
            "describe": self._describe_template,
            "compare": self._comparison_template,
            "relations": self._relationship_template,
            "applications": self._applications_template,
            "list": self._listing_template,
            "process": self._process_template
        }
        
        logger.info(f"✅ SPARQLGenerator inicializado com {len(self.template_mapping)} templates")
    
    def generate_sparql(
        self, 
        entities: List[str], 
        intent_result: IntentResult,
        predicate_result: PredicateSelectionResult
    ) -> SPARQLGenerationResult:
        """
        Gera consulta SPARQL baseada nos resultados do pipeline
        
        Args:
            entities: Lista de entidades do KG
            intent_result: Resultado da classificação de intenção
            predicate_result: Resultado da seleção de predicados
            
        Returns:
            Resultado estruturado da geração SPARQL
        """
        if not entities:
            return self._create_empty_result()
        
        logger.info(f"🔍 Gerando SPARQL para {len(entities)} entidades")
        logger.info(f"    Query type: {intent_result.query_type}")
        logger.info(f"    Predicados: {len(predicate_result.selected_predicates)}")
        
        query_type = intent_result.query_type
        predicates = predicate_result.selected_predicates
        
        # Selecionar template baseado no query_type
        template_method = self.template_mapping.get(query_type, self._describe_template)
        template_name = query_type if query_type in self.template_mapping else "describe"
        
        # Gerar consulta usando template apropriado
        try:
            query_body = template_method(entities, predicates, intent_result)
            full_query = f"{self.prefixes}\n\n{query_body}"
            
            result = SPARQLGenerationResult(
                sparql_query=full_query,
                query_type=query_type,
                template_used=template_name,
                entities_count=len(entities),
                predicates_count=len(predicates),
                generation_method="template_based"
            )
            
            logger.info(f"✅ SPARQL gerado usando template '{template_name}'")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro gerando SPARQL: {e}")
            return self._create_fallback_result(entities, intent_result)
    
    def _describe_template(self, entities: List[str], predicates: List[str], intent_result: IntentResult) -> str:
        """Template para descrever entidades (query_type: describe)"""
        entity = entities[0]
        entity_uri = self._entity_uri(entity)
        
        # Construir padrões OPTIONAL para cada predicado
        predicate_patterns = []
        select_vars = ["?label"]
        
        for i, pred in enumerate(predicates[:5]):
            pred_uri = self._predicate_uri(pred)
            var_name = f"?value{i+1}"
            predicate_patterns.append(f"    OPTIONAL {{ {entity_uri} {pred_uri} {var_name} . }}")
            select_vars.append(var_name)
        
        patterns = "\n".join(predicate_patterns)
        select_clause = " ".join(select_vars)
        
        query = f"""
SELECT {select_clause}
WHERE {{
    {entity_uri} rdfs:label ?label .
{patterns}
}}
LIMIT 10""".strip()
        
        return query
    
    def _comparison_template(self, entities: List[str], predicates: List[str], intent_result: IntentResult) -> str:
        """Template para comparar entidades (query_type: compare)"""
        if len(entities) < 2:
            return self._describe_template(entities, predicates, intent_result)
        
        entity1 = entities[0]
        entity2 = entities[1]
        entity1_uri = self._entity_uri(entity1)
        entity2_uri = self._entity_uri(entity2)
        
        # Padrões para comparar valores dos mesmos predicados
        comparison_patterns = []
        select_vars = ["?label1", "?label2"]
        
        for i, pred in enumerate(predicates[:4]):
            pred_uri = self._predicate_uri(pred)
            var1 = f"?value1_{i+1}"
            var2 = f"?value2_{i+1}"
            comparison_patterns.append(f"    OPTIONAL {{ {entity1_uri} {pred_uri} {var1} . }}")
            comparison_patterns.append(f"    OPTIONAL {{ {entity2_uri} {pred_uri} {var2} . }}")
            select_vars.extend([var1, var2])
        
        patterns = "\n".join(comparison_patterns)
        select_clause = " ".join(select_vars)
        
        query = f"""
SELECT {select_clause}
WHERE {{
    {entity1_uri} rdfs:label ?label1 .
    {entity2_uri} rdfs:label ?label2 .
{patterns}
}}
LIMIT 15""".strip()
        
        return query
    
    def _relationship_template(self, entities: List[str], predicates: List[str], intent_result: IntentResult) -> str:
        """Template para relações entre entidades (query_type: relations)"""
        if len(entities) < 2:
            return self._describe_template(entities, predicates, intent_result)
        
        entity1 = entities[0]
        entity2 = entities[1] 
        entity1_uri = self._entity_uri(entity1)
        entity2_uri = self._entity_uri(entity2)
        
        # Buscar relacionamentos diretos
        relationship_patterns = []
        for pred in predicates[:6]:
            pred_uri = self._predicate_uri(pred)
            # Direção 1: entity1 -> entity2
            relationship_patterns.append(f"    OPTIONAL {{ {entity1_uri} {pred_uri} {entity2_uri} . BIND('{pred}' AS ?rel_type) BIND('direct' AS ?direction) }}")
            # Direção 2: entity2 -> entity1 
            relationship_patterns.append(f"    OPTIONAL {{ {entity2_uri} {pred_uri} {entity1_uri} . BIND('{pred}' AS ?rel_type) BIND('reverse' AS ?direction) }}")
        
        patterns = "\n".join(relationship_patterns)
        
        query = f"""
SELECT ?label1 ?label2 ?rel_type ?direction
WHERE {{
    {entity1_uri} rdfs:label ?label1 .
    {entity2_uri} rdfs:label ?label2 .
{patterns}
}}
LIMIT 20""".strip()
        
        return query
    
    def _applications_template(self, entities: List[str], predicates: List[str], intent_result: IntentResult) -> str:
        """Template para aplicações de entidades (query_type: applications)"""
        entity = entities[0]
        entity_uri = self._entity_uri(entity)
        
        # Focar em predicados de aplicação
        app_patterns = []
        select_vars = ["?label"]
        
        for i, pred in enumerate(predicates[:4]):
            if 'used_for' in pred or 'application' in pred or 'applies_to' in pred:
                pred_uri = self._predicate_uri(pred)
                var_name = f"?app{i+1}"
                app_patterns.append(f"    OPTIONAL {{ {entity_uri} {pred_uri} {var_name} . }}")
                select_vars.append(var_name)
        
        patterns = "\n".join(app_patterns) if app_patterns else f"    OPTIONAL {{ {entity_uri} ml:used_for ?app1 . }}"
        if not app_patterns:
            select_vars.append("?app1")
        
        select_clause = " ".join(select_vars)
        
        query = f"""
SELECT {select_clause}
WHERE {{
    {entity_uri} rdfs:label ?label .
{patterns}
}}
LIMIT 12""".strip()
        
        return query
    
    def _listing_template(self, entities: List[str], predicates: List[str], intent_result: IntentResult) -> str:
        """Template para listar itens relacionados (query_type: list)"""
        entity = entities[0] if entities else "algorithm"
        entity_uri = self._entity_uri(entity)
        
        # Buscar entidades relacionadas por tipo
        query = f"""
SELECT ?item ?label ?type
WHERE {{
    ?item rdfs:label ?label .
    OPTIONAL {{ ?item rdf:type ?type . }}
    FILTER(CONTAINS(LCASE(?label), "{entity.lower()}"))
}}
LIMIT 25""".strip()
        
        return query
    
    def _process_template(self, entities: List[str], predicates: List[str], intent_result: IntentResult) -> str:
        """Template para processos e implementações (query_type: process)"""
        entity = entities[0]
        entity_uri = self._entity_uri(entity)
        
        # Focar em predicados de processo
        process_patterns = []
        select_vars = ["?label"]
        
        for i, pred in enumerate(predicates[:4]):
            if 'process' in pred or 'step' in pred or 'implementation' in pred:
                pred_uri = self._predicate_uri(pred)
                var_name = f"?step{i+1}"
                process_patterns.append(f"    OPTIONAL {{ {entity_uri} {pred_uri} {var_name} . }}")
                select_vars.append(var_name)
        
        patterns = "\n".join(process_patterns) if process_patterns else f"    OPTIONAL {{ {entity_uri} ml:process ?step1 . }}"
        if not process_patterns:
            select_vars.append("?step1")
        
        select_clause = " ".join(select_vars)
        
        query = f"""
SELECT {select_clause}
WHERE {{
    {entity_uri} rdfs:label ?label .
{patterns}
}}
LIMIT 15""".strip()
        
        return query
    
    def _create_empty_result(self) -> SPARQLGenerationResult:
        """Cria resultado vazio para casos sem entidades"""
        empty_query = f"""
{self.prefixes}

SELECT ?entity ?label
WHERE {{
    ?entity rdfs:label ?label .
}}
LIMIT 5""".strip()
        
        return SPARQLGenerationResult(
            sparql_query=empty_query,
            query_type="empty",
            template_used="fallback",
            entities_count=0,
            predicates_count=0,
            generation_method="fallback"
        )
    
    def _create_fallback_result(self, entities: List[str], intent_result: IntentResult) -> SPARQLGenerationResult:
        """Cria resultado de fallback para casos de erro"""
        entity_uri = self._entity_uri(entities[0]) if entities else "?entity"
        
        fallback_query = f"""
{self.prefixes}

SELECT ?label ?value
WHERE {{
    {entity_uri} rdfs:label ?label .
    OPTIONAL {{ {entity_uri} ?predicate ?value . }}
}}
LIMIT 8""".strip()
        
        return SPARQLGenerationResult(
            sparql_query=fallback_query,
            query_type=intent_result.query_type,
            template_used="fallback",
            entities_count=len(entities),
            predicates_count=0,
            generation_method="fallback"
        )
    
    def _entity_uri(self, entity_name: str) -> str:
        """Converte nome de entidade para URI normalizada (100% compatível com KG Builder)"""
        import re
        # Usar EXATAMENTE o mesmo processo do KnowledgeGraphBuilder._create_entity_uri()
        clean_name = re.sub(r'[^\w\s-]', '', entity_name)      # Remove tudo exceto alfanumérico, espaços e hífens
        clean_name = re.sub(r'\s+', '_', clean_name)           # Qualquer sequência de espaços → _
        clean_name = clean_name.lower().strip('_')             # Minúsculas + remove _ do início/fim
        return f"entity:{clean_name}"
    
    def _predicate_uri(self, predicate: str) -> str:
        """Converte predicado para URI"""
        # Usar predicados diretamente se já têm prefixo
        if ':' in predicate:
            return predicate
        return f"relation:{predicate}"


def create_sparql_generator() -> SPARQLGenerator:
    """Factory function"""
    return SPARQLGenerator()


if __name__ == "__main__":
    # Teste integrado do gerador
    print("🧪 Testando SPARQL Generator integrado...")
    
    try:
        # Importar componentes necessários
        from .intent_classifier import create_intent_classifier
        from .predicate_selector import create_predicate_selector
        from .question_entity_extractor import create_question_entity_extractor
        
        print("Inicializando pipeline completo...")
        entity_extractor = create_question_entity_extractor()
        intent_classifier = create_intent_classifier()
        predicate_selector = create_predicate_selector()
        sparql_generator = create_sparql_generator()
        
        # Casos de teste para diferentes query_types
        test_questions = [
            "O que é CNN?",              # describe
            "SVM vs Random Forest",      # compare  
            "CNN utiliza backpropagation?",  # relations
            "Para que serve deep learning?", # applications
            "Liste algoritmos de clustering", # list
            "Como implementar k-means?"      # process
        ]
        
        print(f"\n🎯 Testando pipeline completo (Steps 1-5):")
        print("=" * 70)
        
        for i, question in enumerate(test_questions, 1):
            print(f"{i}. \"{question}\"")
            
            # Pipeline completo Steps 1-5
            entities = entity_extractor.extract_entities(question)
            intent_result = intent_classifier.classify_intent(question, entities)
            
            if entities:
                pred_result = predicate_selector.select_predicates(entities, intent_result)
                sparql_result = sparql_generator.generate_sparql(entities, intent_result, pred_result)
                
                print(f"   → Query type: {sparql_result.query_type} | Template: {sparql_result.template_used}")
                print(f"   → Entidades: {entities[:2]}...")
                print(f"   → SPARQL preview:")
                # Mostrar apenas algumas linhas do SPARQL
                lines = sparql_result.sparql_query.split('\n')
                preview_lines = lines[:8] + ['...'] if len(lines) > 8 else lines
                for line in preview_lines:
                    print(f"     {line}")
            else:
                print(f"   → Sem entidades detectadas")
            print()
        
        print("✅ Pipeline Steps 1-5 completos!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()