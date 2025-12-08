"""
SPARQL Executor: Executa consultas SPARQL no Knowledge Graph

Objetivo: Executar consultas SPARQL geradas pelo pipeline e retornar resultados estruturados.
Integra com kg_executor.py existente para reutilizar funcionalidade de carregamento do KG.

Abordagem: Wrapper que adiciona funcionalidades específicas do pipeline.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Importar classes do pipeline
from .sparql_generator import SPARQLGenerationResult

# Import direto para evitar problemas de path relativo
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from query_system.kg_executor import create_kg_executor

logger = logging.getLogger(__name__)


@dataclass
class SPARQLExecutionResult:
    """Resultado da execução de consulta SPARQL."""
    raw_results: List[Dict[str, Any]]
    results_count: int
    execution_status: str  # 'success', 'empty_results', 'error'
    execution_time_ms: float
    query_info: Dict[str, Any]  # Metadados da consulta original
    processed_results: List[Dict[str, Any]]  # Resultados processados/limpos


class SPARQLExecutor:
    """Executor de consultas SPARQL integrado com o pipeline"""
    
    def __init__(self, kg_path: Optional[Path] = None):
        """
        Inicializa executor SPARQL
        
        Args:
            kg_path: Caminho opcional para arquivo KG (padrão: data/ml_kg.turtle)
        """
        logger.info("🔄 Inicializando SPARQLExecutor...")
        
        try:
            # Reutilizar o executor existente
            self.kg_executor = create_kg_executor(kg_path)
            logger.info("✅ SPARQLExecutor inicializado com KG carregado")
            
        except Exception as e:
            logger.error(f"❌ Erro inicializando SPARQLExecutor: {e}")
            raise
    
    def execute_sparql(
        self, 
        sparql_result: SPARQLGenerationResult
    ) -> SPARQLExecutionResult:
        """
        Executa consulta SPARQL gerada pelo pipeline
        
        Args:
            sparql_result: Resultado da geração SPARQL
            
        Returns:
            Resultado estruturado da execução
        """
        logger.info(f"🔍 Executando SPARQL ({sparql_result.template_used})")
        logger.info(f"    Entidades: {sparql_result.entities_count}")
        logger.info(f"    Predicados: {sparql_result.predicates_count}")
        
        import time
        start_time = time.time()
        
        try:
            # Executar consulta usando o kg_executor
            raw_results = self.kg_executor.execute_sparql(sparql_result.sparql_query)
            
            execution_time = (time.time() - start_time) * 1000  # em ms
            
            # Processar resultados baseado no tipo de template
            processed_results = self._process_results(
                raw_results, 
                sparql_result.template_used
            )
            
            # Determinar status
            if not raw_results:
                status = "empty_results"
            else:
                status = "success"
            
            result = SPARQLExecutionResult(
                raw_results=raw_results,
                results_count=len(raw_results),
                execution_status=status,
                execution_time_ms=execution_time,
                query_info={
                    "query_type": sparql_result.query_type,
                    "template_used": sparql_result.template_used,
                    "entities_count": sparql_result.entities_count,
                    "predicates_count": sparql_result.predicates_count,
                    "generation_method": sparql_result.generation_method
                },
                processed_results=processed_results
            )
            
            logger.info(f"✅ SPARQL executado: {len(raw_results)} resultados ({execution_time:.1f}ms)")
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"❌ Erro executando SPARQL: {e}")
            
            # Retornar resultado de erro
            return SPARQLExecutionResult(
                raw_results=[],
                results_count=0,
                execution_status="error",
                execution_time_ms=execution_time,
                query_info={
                    "query_type": sparql_result.query_type,
                    "template_used": sparql_result.template_used,
                    "error": str(e)
                },
                processed_results=[]
            )
    
    def _process_results(
        self, 
        raw_results: List[Dict[str, Any]], 
        template_used: str
    ) -> List[Dict[str, Any]]:
        """
        Processa resultados baseado no tipo de template usado
        
        Args:
            raw_results: Resultados brutos do SPARQL
            template_used: Template que gerou a consulta
            
        Returns:
            Resultados processados e limpos
        """
        if not raw_results:
            return []
        
        processed = []
        
        for result in raw_results:
            # Limpar valores None e normalizar URIs
            clean_result = {}
            
            for key, value in result.items():
                if value is not None:
                    # Normalizar URIs para nomes legíveis
                    if isinstance(value, str):
                        if value.startswith("http://ml-kg.org/entity/"):
                            clean_value = value.replace("http://ml-kg.org/entity/", "").replace("_", " ")
                        elif value.startswith("http://ml-kg.org/relation/"):
                            clean_value = value.replace("http://ml-kg.org/relation/", "")
                        else:
                            clean_value = value
                    else:
                        clean_value = value
                    
                    clean_result[key] = clean_value
            
            # Adicionar só se tem conteúdo útil
            if clean_result:
                processed.append(clean_result)
        
        # Aplicar processamento específico por template
        if template_used == "compare":
            processed = self._process_comparison_results(processed)
        elif template_used == "relations":
            processed = self._process_relationship_results(processed)
        elif template_used == "list":
            processed = self._process_list_results(processed)
        
        return processed
    
    def _process_comparison_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processa resultados de comparação para agrupar por entidade"""
        # Agrupar valores por entidade para facilitar comparação
        grouped = {}
        
        for result in results:
            # Identificar entidades (label1, label2)
            entity1 = result.get('label1')
            entity2 = result.get('label2')
            
            if entity1 and entity1 not in grouped:
                grouped[entity1] = {"entity": entity1, "properties": {}}
            if entity2 and entity2 not in grouped:
                grouped[entity2] = {"entity": entity2, "properties": {}}
            
            # Adicionar propriedades
            for key, value in result.items():
                if key not in ['label1', 'label2'] and value:
                    if entity1 and key.startswith('value1_'):
                        prop_name = key.replace('value1_', 'property_')
                        grouped[entity1]["properties"][prop_name] = value
                    elif entity2 and key.startswith('value2_'):
                        prop_name = key.replace('value2_', 'property_')
                        grouped[entity2]["properties"][prop_name] = value
        
        return list(grouped.values())
    
    def _process_relationship_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processa resultados de relacionamentos para mostrar conexões claras"""
        relationships = []
        
        for result in results:
            entity1 = result.get('label1')
            entity2 = result.get('label2')
            rel_type = result.get('rel_type')
            direction = result.get('direction')
            
            if entity1 and entity2 and rel_type:
                relationship = {
                    "source": entity1 if direction == "direct" else entity2,
                    "target": entity2 if direction == "direct" else entity1,
                    "relationship": rel_type,
                    "direction": direction
                }
                relationships.append(relationship)
        
        return relationships
    
    def _process_list_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processa resultados de listagem para remover duplicatas e ordenar"""
        # Remover duplicatas baseado no label
        seen_labels = set()
        unique_results = []
        
        for result in results:
            label = result.get('label') or result.get('item')
            if label and label not in seen_labels:
                seen_labels.add(label)
                unique_results.append(result)
        
        # Ordenar alfabeticamente por label
        unique_results.sort(key=lambda x: (x.get('label') or x.get('item', '')).lower())
        
        return unique_results


def create_sparql_executor(kg_path: Optional[Path] = None) -> SPARQLExecutor:
    """Factory function"""
    return SPARQLExecutor(kg_path)


if __name__ == "__main__":
    # Teste integrado do executor
    print("🧪 Testando SPARQL Executor integrado...")
    
    try:
        # Importar componentes necessários para teste completo
        import sys
        sys.path.append('..')
        
        from .sparql_generator import create_sparql_generator
        from .predicate_selector import create_predicate_selector
        from .intent_classifier import create_intent_classifier
        from .question_entity_extractor import create_question_entity_extractor
        
        print("Inicializando pipeline completo + executor...")
        entity_extractor = create_question_entity_extractor()
        intent_classifier = create_intent_classifier()
        predicate_selector = create_predicate_selector()
        sparql_generator = create_sparql_generator()
        sparql_executor = create_sparql_executor()
        
        # Casos de teste
        test_questions = [
            "O que é CNN?",
            "SVM vs Random Forest",
            "CNN utiliza backpropagation?",
        ]
        
        print(f"\n🎯 Testando pipeline completo (Steps 1-6):")
        print("=" * 70)
        
        for i, question in enumerate(test_questions, 1):
            print(f"{i}. \"{question}\"")
            
            # Pipeline Steps 1-6
            entities = entity_extractor.extract_entities(question)
            intent_result = intent_classifier.classify_intent(question, entities)
            
            if entities:
                pred_result = predicate_selector.select_predicates(entities, intent_result)
                sparql_result = sparql_generator.generate_sparql(entities, intent_result, pred_result)
                
                # NOVO: Executar SPARQL
                execution_result = sparql_executor.execute_sparql(sparql_result)
                
                print(f"   → Template: {sparql_result.template_used}")
                print(f"   → Execução: {execution_result.execution_status} ({execution_result.execution_time_ms:.1f}ms)")
                print(f"   → Resultados: {execution_result.results_count} encontrados")
                
                # Mostrar alguns resultados
                if execution_result.processed_results:
                    for j, result in enumerate(execution_result.processed_results[:2], 1):
                        result_preview = str(result)[:60] + "..." if len(str(result)) > 60 else str(result)
                        print(f"      {j}. {result_preview}")
                else:
                    print(f"      (nenhum resultado encontrado no KG)")
            else:
                print(f"   → Sem entidades detectadas")
            print()
        
        print("✅ Pipeline Steps 1-6 funcionando!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()