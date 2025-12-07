"""
Complete Knowledge Graph Query Pipeline: Sistema integrado completo

Este módulo fornece uma interface unificada para o pipeline completo de 7 steps:
1. Entity Extraction - Extração de entidades da pergunta
2. Entity Linking - Mapeamento para entidades do KG  
3. Intent Classification - Classificação da intenção
4. Predicate Selection - Seleção de predicados SPARQL
5. SPARQL Generation - Geração de consultas SPARQL
6. SPARQL Execution - Execução no Knowledge Graph
7. Natural Language Generation - Resposta em linguagem natural

Uso simples:
    pipeline = create_complete_pipeline()
    result = pipeline.answer_question("O que é machine learning?")
    print(result.natural_answer)
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Configurar logging apenas para erros
logging.basicConfig(level=logging.ERROR)

# Importar todos os componentes do pipeline
from .question_entity_extractor import QuestionEntityExtractor, create_question_entity_extractor
from .entity_mapper import EntityMapper, create_entity_mapper
from .intent_classifier import IntentClassifier, IntentResult, create_intent_classifier
from .predicate_selector import PredicateSelector, PredicateSelectionResult, create_predicate_selector
from .sparql_generator import SPARQLGenerator, SPARQLGenerationResult, create_sparql_generator
from .sparql_executor import SPARQLExecutor, SPARQLExecutionResult, create_sparql_executor
from .natural_language_generator import NaturalLanguageGenerator, NaturalLanguageResult, create_natural_language_generator

logger = logging.getLogger(__name__)


@dataclass
class CompletePipelineResult:
    """Resultado completo do pipeline de 7 steps"""
    
    # Entrada
    original_question: str
    
    # Step 1: Entity Extraction
    extracted_entities: List[str]
    
    # Step 2: Entity Linking (implícito no entity_extractor)
    entity_linking_status: str
    
    # Step 3: Intent Classification
    intent_result: Optional[IntentResult]
    
    # Step 4: Predicate Selection
    predicate_result: Optional[PredicateSelectionResult]
    
    # Step 5: SPARQL Generation
    sparql_result: Optional[SPARQLGenerationResult]
    
    # Step 6: SPARQL Execution
    execution_result: Optional[SPARQLExecutionResult]
    
    # Step 7: Natural Language Answer
    nl_result: Optional[NaturalLanguageResult]
    
    # Metadados
    total_time_ms: float
    pipeline_status: str  # 'success', 'partial', 'failed', 'no_entities'
    step_completed: int  # Último step completado com sucesso
    error_info: Optional[str]
    
    @property
    def natural_answer(self) -> str:
        """Resposta final em linguagem natural"""
        if self.nl_result:
            return self.nl_result.answer
        elif self.pipeline_status == "no_entities":
            return "Não consegui identificar entidades relevantes de Machine Learning na sua pergunta. Tente usar termos mais específicos da área."
        else:
            return "Desculpe, houve um problema ao processar sua pergunta. Tente reformular ou use termos mais específicos."
    
    @property
    def confidence(self) -> str:
        """Nível de confiança da resposta"""
        if self.nl_result:
            return self.nl_result.confidence
        else:
            return "low"
    
    @property
    def answer_quality(self) -> str:
        """Qualidade da resposta"""
        if self.nl_result:
            return self.nl_result.answer_quality
        else:
            return "error"


class CompletePipeline:
    """Pipeline completo integrado de 7 steps para responder perguntas sobre ML/DL"""
    
    def __init__(self):
        """Inicializa todos os componentes do pipeline"""
        # Inicializar todos os componentes silenciosamente
        self.entity_extractor = create_question_entity_extractor()
        self.entity_mapper = create_entity_mapper()
        self.intent_classifier = create_intent_classifier()
        self.predicate_selector = create_predicate_selector()
        self.sparql_generator = create_sparql_generator()
        self.sparql_executor = create_sparql_executor()
        self.nl_generator = create_natural_language_generator()
    
    def answer_question(self, question: str) -> CompletePipelineResult:
        """
        Processa pergunta através de todo o pipeline de 7 steps
        
        Args:
            question: Pergunta do usuário
            
        Returns:
            Resultado completo com resposta em linguagem natural
        """
        start_time = time.time()
        
        # Inicializar resultado
        result = CompletePipelineResult(
            original_question=question,
            extracted_entities=[],
            entity_linking_status="not_attempted",
            intent_result=None,
            predicate_result=None,
            sparql_result=None,
            execution_result=None,
            nl_result=None,
            total_time_ms=0.0,
            pipeline_status="failed",
            step_completed=0,
            error_info=None
        )
        
        try:
            # Step 1: Entity Extraction
            entities = self.entity_extractor.extract_entities(question)
            result.extracted_entities = entities
            result.step_completed = 1
            
            if not entities:
                result.pipeline_status = "no_entities"
                result.entity_linking_status = "no_entities_found"
                return self._finalize_result(result, start_time)
            
            # Step 2: Entity Linking (implícito - entidades já mapeadas)
            result.entity_linking_status = "completed"
            result.step_completed = 2
            
            # Step 3: Intent Classification
            intent_result = self.intent_classifier.classify_intent(question, entities)
            result.intent_result = intent_result
            result.step_completed = 3
            
            # Step 4: Predicate Selection
            predicate_result = self.predicate_selector.select_predicates(entities, intent_result)
            result.predicate_result = predicate_result
            result.step_completed = 4
            
            # Step 5: SPARQL Generation
            sparql_result = self.sparql_generator.generate_sparql(entities, intent_result, predicate_result)
            result.sparql_result = sparql_result
            result.step_completed = 5
            
            # Step 6: SPARQL Execution
            execution_result = self.sparql_executor.execute_sparql(sparql_result)
            result.execution_result = execution_result
            result.step_completed = 6
            
            # Step 7: Natural Language Generation
            nl_result = self.nl_generator.generate_answer(question, intent_result, execution_result)
            result.nl_result = nl_result
            result.step_completed = 7
            
            # Determinar status final
            if nl_result.answer_quality in ["complete", "partial"]:
                result.pipeline_status = "success"
            else:
                result.pipeline_status = "partial"
            
        except Exception as e:
            result.error_info = str(e)
            result.pipeline_status = "failed"
        
        return self._finalize_result(result, start_time)
    
    def _finalize_result(self, result: CompletePipelineResult, start_time: float) -> CompletePipelineResult:
        """Finaliza resultado calculando tempo total"""
        result.total_time_ms = round((time.time() - start_time) * 1000, 1)
        return result
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o pipeline carregado"""
        
        # Tentar obter número de triplas do KG
        kg_triplas = "carregado"
        try:
            if hasattr(self.sparql_executor, 'kg_executor'):
                kg_executor = self.sparql_executor.kg_executor
                if hasattr(kg_executor, 'graph'):
                    kg_triplas = f"{len(kg_executor.graph):,}"
                elif hasattr(kg_executor, 'triplas_count'):
                    kg_triplas = f"{kg_executor.triplas_count:,}"
                elif hasattr(kg_executor, '_graph'):
                    kg_triplas = f"{len(kg_executor._graph):,}"
        except Exception:
            kg_triplas = "carregado"
        
        return {
            "steps": 7,
            "components": [
                "QuestionEntityExtractor",
                "EntityMapper", 
                "IntentClassifier",
                "PredicateSelector",
                "SPARQLGenerator",
                "SPARQLExecutor", 
                "NaturalLanguageGenerator"
            ],
            "kg_triplas": kg_triplas,
            "predicates": len(self.predicate_selector.available_predicates),
            "status": "ready"
        }


def create_complete_pipeline() -> CompletePipeline:
    """Factory function para criar pipeline completo"""
    return CompletePipeline()


if __name__ == "__main__":
    # Interface Interativa do Knowledge Graph
    print("🤖 Sistema de Perguntas e Respostas - Knowledge Graph ML/DL")
    print("=" * 65)
    
    try:
        print("🔄 Carregando sistema...")
        pipeline = create_complete_pipeline()
        
        info = pipeline.get_pipeline_info()
        print(f"✅ Sistema pronto!")
        print(f"   • Knowledge Graph: {info.get('kg_triplas', 'carregado')} triplas")
        print(f"   • Pipeline: {info['steps']} steps implementados")
        print()
        
        print("💡 Faça perguntas sobre Machine Learning e Deep Learning")
        print("   Exemplos: 'O que é SVM?', 'Diferença entre CNN e RNN', 'Tipos de clustering'")
        print("   Digite 'sair' para encerrar")
        print()
        
        while True:
            try:
                user_question = input("❓ Pergunta: ").strip()
                
                if user_question.lower() in ['quit', 'sair', 'exit', '']:
                    break
                
                if not user_question:
                    continue
                
                print("🔍 Processando...")
                result = pipeline.answer_question(user_question)
                
                print(f"\n🤖 Resposta:")
                print(f"{result.natural_answer}")
                
                # Mostrar info adicional se houver problemas
                if result.pipeline_status != "success":
                    if result.pipeline_status == "no_entities":
                        print("\n💡 Dica: Tente usar termos mais específicos de ML/DL")
                    else:
                        print(f"\n⚠️  Status: {result.pipeline_status}")
                
                print("\n" + "-" * 60 + "\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Erro processando pergunta: {e}")
                print("Tente reformular a pergunta.\n")
        
        print("👋 Até logo!")
        
    except Exception as e:
        print(f"❌ Erro inicializando sistema: {e}")
        print("Verifique se todas as dependências estão instaladas.")
        import traceback
        traceback.print_exc()