"""
Natural Language Answer Generator: Transforma resultados SPARQL em respostas naturais

Objetivo: Gerar respostas em linguagem natural a partir dos resultados SPARQL.
Usando LLM para criar respostas contextualizadas, claras e informativas.

Abordagem: Templates + LLM com prompts especializados por tipo de consulta.
"""

import logging
import ollama
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Importar classes do pipeline
from .intent_classifier import IntentResult
from .sparql_executor import SPARQLExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class NaturalLanguageResult:
    """Resultado da geração de resposta em linguagem natural."""
    answer: str
    answer_quality: str  # 'complete', 'partial', 'empty', 'error'
    generation_method: str  # 'llm_generated', 'template_based', 'fallback'
    source_info: Dict[str, Any]  # Metadados dos resultados SPARQL
    confidence: str  # 'high', 'medium', 'low'
    reasoning: str  # Breve explicação de como a resposta foi construída


class NaturalLanguageGenerator:
    """Gerador de respostas em linguagem natural usando LLM"""
    
    def __init__(self, model: str = "llama3.2:3b"):
        """
        Inicializa gerador de linguagem natural
        
        Args:
            model: Modelo LLM para usar
        """
        logger.info("🔄 Inicializando NaturalLanguageGenerator...")
        
        self.model = model
        
        # Templates base para diferentes tipos de resposta
        self.response_templates = {
            "describe": {
                "success": "Com base no Knowledge Graph, {entity} é {description}.",
                "empty": "Não encontrei informações específicas sobre {entity} no Knowledge Graph.",
                "multiple": "Encontrei as seguintes informações sobre {entity}: {items}."
            },
            "compare": {
                "success": "Comparando {entity1} e {entity2}: {comparison}",
                "empty": "Não encontrei informações suficientes para comparar {entity1} e {entity2}.",
                "partial": "Encontrei informações parciais: {available_info}"
            },
            "list": {
                "success": "Encontrei {count} itens relacionados: {items}",
                "empty": "Não encontrei itens específicos para listar.",
                "limited": "Encontrei {count} itens (mostrando principais): {items}"
            },
            "relations": {
                "success": "A relação entre {entity1} e {entity2}: {relationships}",
                "empty": "Não encontrei relações diretas entre {entity1} e {entity2}.",
                "multiple": "Encontrei múltiplas relações: {relationships}"
            },
            "applications": {
                "success": "{entity} é usado para: {applications}",
                "empty": "Não encontrei aplicações específicas para {entity}.",
                "general": "{entity} tem as seguintes aplicações: {applications}"
            }
        }
        
        logger.info(f"✅ NaturalLanguageGenerator inicializado com {len(self.response_templates)} templates")
    
    def generate_answer(
        self,
        original_question: str,
        intent_result: IntentResult, 
        execution_result: SPARQLExecutionResult
    ) -> NaturalLanguageResult:
        """
        Gera resposta em linguagem natural
        
        Args:
            original_question: Pergunta original do usuário
            intent_result: Resultado da classificação de intenção
            execution_result: Resultado da execução SPARQL
            
        Returns:
            Resposta em linguagem natural estruturada
        """
        logger.info(f"🔍 Gerando resposta natural para: \"{original_question[:50]}...\"")
        logger.info(f"    Template: {execution_result.query_info.get('template_used')}")
        logger.info(f"    Resultados: {execution_result.results_count}")
        
        try:
            # Determinar estratégia de geração baseada nos resultados
            if execution_result.execution_status == "error":
                return self._create_error_response(original_question, execution_result)
            elif execution_result.results_count == 0:
                return self._create_empty_response(original_question, intent_result, execution_result)
            else:
                return self._generate_llm_response(original_question, intent_result, execution_result)
                
        except Exception as e:
            logger.error(f"❌ Erro gerando resposta: {e}")
            return self._create_fallback_response(original_question, str(e))
    
    def _generate_llm_response(
        self,
        question: str,
        intent_result: IntentResult,
        execution_result: SPARQLExecutionResult
    ) -> NaturalLanguageResult:
        """Gera resposta usando LLM com contexto dos resultados SPARQL"""
        
        # Construir contexto estruturado dos resultados
        context = self._build_context(execution_result)
        
        # Criar prompt especializado por tipo de consulta
        prompt = self._build_llm_prompt(question, intent_result, context)
        
        try:
            # Chamar LLM
            response = self._call_ollama(prompt)
            
            # Processar e validar resposta
            answer = self._process_llm_response(response, question, execution_result)
            
            # Determinar qualidade e confiança
            quality, confidence = self._assess_answer_quality(answer, execution_result)
            
            return NaturalLanguageResult(
                answer=answer,
                answer_quality=quality,
                generation_method="llm_generated",
                source_info={
                    "results_count": execution_result.results_count,
                    "template_used": execution_result.query_info.get("template_used"),
                    "execution_time": execution_result.execution_time_ms
                },
                confidence=confidence,
                reasoning=f"Gerada via LLM com {execution_result.results_count} resultados do KG"
            )
            
        except Exception as e:
            logger.error(f"❌ Erro na geração LLM: {e}")
            # Fallback para template
            return self._create_template_response(question, intent_result, execution_result)
    
    def _build_context(self, execution_result: SPARQLExecutionResult) -> str:
        """Constrói contexto estruturado dos resultados SPARQL"""
        
        if not execution_result.processed_results:
            return "Nenhum resultado encontrado no Knowledge Graph."
        
        template_used = execution_result.query_info.get("template_used", "unknown")
        results = execution_result.processed_results[:5]  # Limitar para não sobrecarregar
        
        context_parts = []
        
        if template_used == "describe":
            for result in results:
                label = result.get("label", "Item")
                properties = [f"{k}={v}" for k, v in result.items() if k != "label" and v]
                if properties:
                    context_parts.append(f"- {label}: {', '.join(properties[:3])}")
                else:
                    context_parts.append(f"- {label}")
                    
        elif template_used == "compare":
            for result in results:
                entity = result.get("entity", "Entidade")
                props = result.get("properties", {})
                if props:
                    prop_desc = ', '.join([f"{k}={v}" for k, v in props.items()][:3])
                    context_parts.append(f"- {entity}: {prop_desc}")
                else:
                    context_parts.append(f"- {entity}")
                    
        elif template_used == "list":
            for result in results:
                item = result.get("label") or result.get("item", "Item")
                item_type = result.get("type", "")
                if item_type:
                    context_parts.append(f"- {item} ({item_type})")
                else:
                    context_parts.append(f"- {item}")
                    
        elif template_used == "relations":
            for result in results:
                source = result.get("source", "Entidade1")
                target = result.get("target", "Entidade2") 
                relationship = result.get("relationship", "relaciona-se")
                direction = result.get("direction", "direct")
                context_parts.append(f"- {source} {relationship} {target} ({direction})")
                
        else:
            # Template genérico
            for result in results:
                main_field = None
                for field in ["label", "entity", "item", "source"]:
                    if field in result:
                        main_field = result[field]
                        break
                if main_field:
                    context_parts.append(f"- {main_field}")
        
        return "\n".join(context_parts) if context_parts else "Resultados disponíveis mas sem estrutura clara."
    
    def _build_llm_prompt(
        self,
        question: str, 
        intent_result: IntentResult,
        context: str
    ) -> str:
        """Constrói prompt especializado para o LLM"""
        
        query_type = intent_result.query_type
        intent = intent_result.intent
        
        # Instruções específicas por tipo de consulta
        instructions = {
            "describe": "Forneça uma definição clara e concisa baseada nas informações disponíveis.",
            "compare": "Compare as entidades destacando principais diferenças ou similaridades.",
            "list": "Liste os itens de forma organizada e clara.",
            "relations": "Explique as relações entre as entidades de forma clara.",
            "applications": "Descreva os usos práticos e aplicações."
        }
        
        instruction = instructions.get(query_type, "Responda de forma clara e informativa.")
        
        prompt = f"""Você é um assistente especializado em Machine Learning e Deep Learning.

PERGUNTA: "{question}"
INTENÇÃO: {intent}
TIPO DE CONSULTA: {query_type}

INFORMAÇÕES DO KNOWLEDGE GRAPH:
{context}

INSTRUÇÕES:
- {instruction}
- Use linguagem técnica mas acessível
- Seja conciso (2-4 frases)
- Base sua resposta apenas nas informações fornecidas
- Se as informações são limitadas, seja honesto sobre isso
- Use terminologia de ML/DL apropriada

RESPOSTA:"""
        
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """Chama API do Ollama usando biblioteca oficial"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.3,  # Criatividade moderada
                    'num_predict': 500,  # Resposta completa sem cortes
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"❌ Erro chamando Ollama: {e}")
            raise
    
    def _process_llm_response(
        self,
        response: str,
        question: str,
        execution_result: SPARQLExecutionResult
    ) -> str:
        """Processa e valida resposta do LLM"""
        
        # Limpar resposta
        answer = response.strip()
        
        # Verificar se resposta é muito curta ou vazia
        if len(answer) < 10:
            template_used = execution_result.query_info.get("template_used", "describe")
            return self._get_template_fallback(template_used, execution_result)
        
        # Verificar se resposta não é apenas repetição da pergunta
        if question.lower() in answer.lower() and len(answer) < 50:
            return self._get_template_fallback("describe", execution_result)
        
        return answer
    
    def _assess_answer_quality(
        self,
        answer: str,
        execution_result: SPARQLExecutionResult
    ) -> tuple[str, str]:
        """Avalia qualidade da resposta e confiança"""
        
        # Qualidade baseada em resultados e comprimento da resposta
        if execution_result.results_count == 0:
            quality = "empty"
            confidence = "low"
        elif execution_result.results_count >= 3 and len(answer) > 50:
            quality = "complete"
            confidence = "high"
        elif execution_result.results_count >= 1 and len(answer) > 30:
            quality = "partial"
            confidence = "medium"
        else:
            quality = "partial"
            confidence = "low"
        
        return quality, confidence
    
    def _get_template_fallback(self, template_used: str, execution_result: SPARQLExecutionResult) -> str:
        """Gera resposta usando template quando LLM falha"""
        
        if not execution_result.processed_results:
            return "Não encontrei informações específicas no Knowledge Graph."
        
        result = execution_result.processed_results[0]
        
        if template_used == "describe":
            label = result.get("label", "conceito")
            return f"Baseado no Knowledge Graph, {label} é um conceito de Machine Learning."
        elif template_used == "list":
            count = len(execution_result.processed_results)
            return f"Encontrei {count} itens relacionados no Knowledge Graph."
        else:
            return f"Encontrei {execution_result.results_count} resultados relevantes no Knowledge Graph."
    
    def _create_template_response(
        self,
        question: str,
        intent_result: IntentResult,
        execution_result: SPARQLExecutionResult
    ) -> NaturalLanguageResult:
        """Cria resposta baseada em template quando LLM falha"""
        
        template_used = execution_result.query_info.get("template_used", "describe")
        answer = self._get_template_fallback(template_used, execution_result)
        
        return NaturalLanguageResult(
            answer=answer,
            answer_quality="partial",
            generation_method="template_based",
            source_info={
                "results_count": execution_result.results_count,
                "template_used": template_used
            },
            confidence="medium",
            reasoning="Gerada via template (fallback do LLM)"
        )
    
    def _create_empty_response(
        self,
        question: str,
        intent_result: IntentResult,
        execution_result: SPARQLExecutionResult
    ) -> NaturalLanguageResult:
        """Cria resposta para casos sem resultados"""
        
        answer = f"Não encontrei informações específicas sobre sua pergunta no Knowledge Graph de Machine Learning. Talvez você possa reformular a pergunta ou usar termos mais específicos da área."
        
        return NaturalLanguageResult(
            answer=answer,
            answer_quality="empty",
            generation_method="template_based",
            source_info={"results_count": 0},
            confidence="low",
            reasoning="Nenhum resultado encontrado no KG"
        )
    
    def _create_error_response(
        self,
        question: str,
        execution_result: SPARQLExecutionResult
    ) -> NaturalLanguageResult:
        """Cria resposta para casos de erro"""
        
        answer = "Houve um problema técnico ao processar sua pergunta. Por favor, tente reformular ou faça uma pergunta mais específica."
        
        return NaturalLanguageResult(
            answer=answer,
            answer_quality="error",
            generation_method="fallback",
            source_info={"error": execution_result.query_info.get("error", "Unknown")},
            confidence="low",
            reasoning="Erro na execução SPARQL"
        )
    
    def _create_fallback_response(self, question: str, error: str) -> NaturalLanguageResult:
        """Cria resposta de fallback para erros inesperados"""
        
        answer = "Desculpe, houve um problema ao gerar a resposta. Por favor, tente novamente."
        
        return NaturalLanguageResult(
            answer=answer,
            answer_quality="error", 
            generation_method="fallback",
            source_info={"error": error},
            confidence="low",
            reasoning="Fallback por erro inesperado"
        )


def create_natural_language_generator() -> NaturalLanguageGenerator:
    """Factory function"""
    return NaturalLanguageGenerator()


if __name__ == "__main__":
    # Teste integrado completo do pipeline
    print("🧪 Testando Pipeline COMPLETO (Steps 1-7)...")
    
    try:
        # Importar todos os componentes
        import sys
        sys.path.append('..')
        
        from .question_entity_extractor import create_question_entity_extractor
        from .intent_classifier import create_intent_classifier
        from .predicate_selector import create_predicate_selector
        from .sparql_generator import create_sparql_generator
        from .sparql_executor import create_sparql_executor
        
        print("Inicializando pipeline completo...")
        entity_extractor = create_question_entity_extractor()
        intent_classifier = create_intent_classifier()
        predicate_selector = create_predicate_selector()
        sparql_generator = create_sparql_generator()
        sparql_executor = create_sparql_executor()
        nl_generator = create_natural_language_generator()
        
        # Casos de teste para pipeline completo
        test_questions = [
            "O que é linear regression?",      # describe
            "SVM vs Random Forest",            # compare  
            "Para que serve neural networks?", # applications
        ]
        
        print(f"\n🎯 Pipeline COMPLETO (Steps 1-7):")
        print("=" * 60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"{i}. \"{question}\"")
            
            # Pipeline completo Steps 1-7
            entities = entity_extractor.extract_entities(question)
            intent_result = intent_classifier.classify_intent(question, entities)
            
            if entities:
                pred_result = predicate_selector.select_predicates(entities, intent_result)
                sparql_result = sparql_generator.generate_sparql(entities, intent_result, pred_result)
                execution_result = sparql_executor.execute_sparql(sparql_result)
                
                # Step 7: Resposta em linguagem natural
                nl_result = nl_generator.generate_answer(question, intent_result, execution_result)
                
                print(f"   → Template: {sparql_result.template_used}")
                print(f"   → Resultados KG: {execution_result.results_count}")
                print(f"   → Resposta ({nl_result.answer_quality}, {nl_result.confidence}):") 
                print(f"     \"{nl_result.answer}\"")
                print(f"   → Método: {nl_result.generation_method}")
            else:
                print(f"   → Sem entidades detectadas")
            print()
        
        print("✅ Pipeline COMPLETO Steps 1-7 funcionando!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()