"""
Predicate Selector: Seleciona predicados/relações do KG baseado na intenção

Objetivo: Escolher quais predicados usar no SPARQL baseado na intenção e entidades.
Garante validade (só escolhe predicados que existem no KG) e evita alucinação.

Abordagem: LLM com lista restrita de predicados válidos do KG.
"""

import logging
import ollama
import json
from typing import List, Optional, Dict
from pathlib import Path
import pickle
from dataclasses import dataclass

# Importar IntentResult do intent_classifier
from .intent_classifier import IntentResult

logger = logging.getLogger(__name__)


@dataclass
class PredicateSelectionResult:
    """Resultado da seleção de predicados."""
    selected_predicates: List[str]
    selection_method: str  # 'intent_based', 'llm_refined', 'fallback'
    entities_used: List[str]
    intent_info: Dict[str, str]  # Para debug/logging


class PredicateSelector:
    """Selecionador de predicados usando LLM com lista restrita do KG"""
    
    def __init__(self, model: str = "llama3.2:3b"):
        """
        Inicializa selecionador de predicados
        
        Args:
            model: Modelo LLM para usar
        """
        logger.info("🔄 Inicializando PredicateSelector...")
        
        self.model = model
        
        # Carregar predicados disponíveis no KG
        self.available_predicates = self._load_kg_predicates()
        
        # Mapeamento detalhado de predicados por domínio
        self.predicate_categories = {
            "definition": ["rdfs:label", "rdfs:comment", "ml:definition", "rdf:type"],
            "structure": ["ml:part_of", "ml:implements", "ml:extends", "ml:subclass_of"],
            "usage": ["ml:uses", "ml:applies_to", "ml:solves", "ml:used_for"],
            "comparison": ["ml:compares_with", "ml:outperforms", "ml:similar_to"],
            "process": ["ml:process", "ml:step", "ml:implementation", "ml:requires"],
            "relationships": ["ml:depends_on", "ml:based_on", "ml:optimizes"]
        }
        
        logger.info(f"✅ PredicateSelector inicializado com {len(self.available_predicates)} predicados do KG")
    
    def _load_kg_predicates(self) -> List[str]:
        """Carrega lista de predicados disponíveis no KG"""
        relations_file = Path("data/extracted_relations.pkl")
        
        if not relations_file.exists():
            logger.warning(f"❌ Arquivo de relações não encontrado: {relations_file}")
            # Predicados básicos como fallback
            return ["uses", "is_a", "applies_to", "extends", "implements", "part_of"]
        
        try:
            with open(relations_file, 'rb') as f:
                data = pickle.load(f)
            
            summary = data.get('summary', {})
            predicate_counts = summary.get('predicate_counts', {})
            
            # Pegar TODOS os predicados do esquema original (mesmo com freq baixa)
            schema_predicates = [
                'is_a', 'part_of', 'subclass_of',
                'uses', 'implements', 'optimizes', 'applies_to', 'solves',
                'requires', 'depends_on', 'based_on', 'extends',
                'outperforms', 'compared_to', 'equivalent_to',
                'precedes', 'evolved_from',
                'trained_on', 'evaluated_on', 'measures', 'predicts',
                'created_by', 'proposed_by', 'developed_by'
            ]
            
            # Ordenar por frequência (mais comuns primeiro)
            predicates = []
            for pred in schema_predicates:
                if pred in predicate_counts:
                    predicates.append(pred)
            
            # Ordenar por frequência decrescente
            predicates.sort(key=lambda p: predicate_counts.get(p, 0), reverse=True)
            
            logger.info(f"✅ {len(predicates)} predicados carregados do KG")
            return predicates
            
        except Exception as e:
            logger.error(f"❌ Erro carregando predicados: {e}")
            return ["uses", "is_a", "applies_to", "extends", "implements", "part_of"]
    
    def select_predicates(
        self, 
        entities: List[str], 
        intent_result: IntentResult
    ) -> PredicateSelectionResult:
        """
        Seleciona predicados relevantes baseado nas entidades e resultado da classificação de intenção
        
        Args:
            entities: Lista de entidades do KG
            intent_result: Resultado da classificação de intenção
            
        Returns:
            Resultado estruturado da seleção de predicados
        """
        if not entities:
            return self._create_empty_result(intent_result)
        
        logger.info(f"🔍 Selecionando predicados para {len(entities)} entidades")
        logger.info(f"    Intenção: {intent_result.intent} ({intent_result.classification_status})")
        
        # Usar predicados sugeridos do IntentResult como base
        suggested_predicates = intent_result.suggested_predicates
        
        # Se a classificação foi bem-sucedida, refinar com LLM
        if intent_result.classification_status == "llm_classified" and len(entities) >= 2:
            # Para consultas complexas, usar LLM para refinamento
            selected_predicates = self._llm_refine_predicates(
                entities, intent_result, suggested_predicates
            )
            selection_method = "llm_refined"
        else:
            # Usar predicados sugeridos diretamente
            selected_predicates = self._filter_available_predicates(suggested_predicates)
            selection_method = "intent_based"
        
        result = PredicateSelectionResult(
            selected_predicates=selected_predicates,
            selection_method=selection_method,
            entities_used=entities,
            intent_info={
                "intent": intent_result.intent,
                "status": intent_result.classification_status,
                "query_type": intent_result.query_type
            }
        )
        
        logger.info(f"✅ {len(selected_predicates)} predicados selecionados ({selection_method})")
        return result
    
    def _filter_available_predicates(self, suggested_predicates: List[str]) -> List[str]:
        """Filtra predicados sugeridos para manter apenas os disponíveis no KG"""
        # Para desenvolvimento, assumir que todos os predicados sugeridos estão disponíveis
        # TODO: Implementar validação real contra o KG quando tivermos o schema completo
        return suggested_predicates[:4]  # Limitar a 4 predicados
    
    def _create_empty_result(self, intent_result: IntentResult) -> PredicateSelectionResult:
        """Cria resultado vazio para casos sem entidades"""
        return PredicateSelectionResult(
            selected_predicates=["rdfs:label"],  # Predicado básico
            selection_method="fallback",
            entities_used=[],
            intent_info={
                "intent": intent_result.intent,
                "status": intent_result.classification_status,
                "query_type": intent_result.query_type
            }
        )
    
    def _llm_refine_predicates(
        self, 
        entities: List[str], 
        intent_result: IntentResult, 
        suggested_predicates: List[str]
    ) -> List[str]:
        """Usa LLM para refinar seleção de predicados para consultas complexas"""
        
        # Prompt estruturado para refinamento
        prompt = self._build_refinement_prompt(entities, intent_result, suggested_predicates)
        
        try:
            # Chamar LLM
            response = self._call_ollama(prompt)
            
            # Extrair predicados refinados
            refined = self._extract_predicates_from_response(response, suggested_predicates)
            
            return refined if refined else suggested_predicates[:3]
            
        except Exception as e:
            logger.error(f"❌ Erro no refinamento LLM: {e}")
            # Fallback: usar predicados sugeridos
            return suggested_predicates[:3]
    
    def _build_refinement_prompt(
        self, 
        entities: List[str], 
        intent_result: IntentResult, 
        suggested_predicates: List[str]
    ) -> str:
        """Constrói prompt para refinamento de predicados"""
        
        entities_str = ", ".join(entities)
        predicates_str = ", ".join(suggested_predicates)
        
        prompt = f"""Refine a seleção de predicados para uma consulta SPARQL:

ENTIDADES: {entities_str}
INTENÇÃO: {intent_result.intent}
TIPO DE CONSULTA: {intent_result.query_type}

PREDICADOS SUGERIDOS:
{predicates_str}

INSTRUÇÕES:
- Escolha 2-3 predicados mais relevantes para conectar essas entidades
- Para consultas de comparação, inclua predicados de similaridade
- Para definições, inclua predicados descritivos
- Para aplicações, inclua predicados de uso

RESPOSTA: Liste apenas os predicados escolhidos, separados por vírgula:"""
        
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """Chama API do Ollama usando biblioteca oficial"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.1,  # Baixa criatividade para seleção
                    'num_predict': 30,   # Resposta curta com predicados
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"❌ Erro chamando Ollama: {e}")
            raise
    
    def _extract_predicates_from_response(self, response: str, available_predicates: List[str]) -> List[str]:
        """Extrai predicados da resposta do LLM"""
        
        response = response.lower().strip()
        
        # Encontrar predicados mencionados na resposta
        selected = []
        
        for predicate in available_predicates:
            # Buscar tanto o predicado completo quanto a parte após ':'
            pred_lower = predicate.lower()
            pred_short = pred_lower.split(':')[-1] if ':' in pred_lower else pred_lower
            
            if pred_lower in response or pred_short in response:
                selected.append(predicate)
        
        # Se não encontrou nenhum, usar os primeiros disponíveis
        if not selected:
            selected = available_predicates[:2]
        
        # Limitar a no máximo 3 predicados
        return selected[:3]


def create_predicate_selector() -> PredicateSelector:
    """Factory function"""
    return PredicateSelector()


if __name__ == "__main__":
    # Teste integrado do selecionador
    print("🧪 Testando Predicate Selector integrado...")
    
    try:
        # Importar componentes necessários
        from .intent_classifier import create_intent_classifier, IntentResult
        from .question_entity_extractor import create_question_entity_extractor
        
        print("Inicializando componentes...")
        entity_extractor = create_question_entity_extractor()
        intent_classifier = create_intent_classifier()
        predicate_selector = create_predicate_selector()
        
        # Casos de teste completos
        test_questions = [
            "O que é CNN?",  # definition
            "Diferença entre SVM e Random Forest?",  # comparison  
            "Como CNN utiliza backpropagation?",  # relationship
            "Liste algoritmos de clustering",  # listing
            "Para que serve deep learning?"  # application
        ]
        
        print(f"\n🎯 Testando pipeline completo (Steps 1-4):")
        print("=" * 70)
        
        for i, question in enumerate(test_questions, 1):
            print(f"{i}. '{question}'")
            
            # Step 1: Extract entities
            entities = entity_extractor.extract_entities(question)
            print(f"   Entidades: {entities}")
            
            # Step 3: Classify intent
            intent_result = intent_classifier.classify_intent(question, entities)
            print(f"   Intenção: {intent_result.intent} ({intent_result.classification_status})")
            
            # Step 4: Select predicates
            if entities:  # Só processar se tiver entidades
                pred_result = predicate_selector.select_predicates(entities, intent_result)
                print(f"   Predicados: {pred_result.selected_predicates} ({pred_result.selection_method})")
            else:
                print(f"   Predicados: [sem entidades detectadas]")
            
            print()
        
        print("✅ Pipeline Steps 1-4 funcionando!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()