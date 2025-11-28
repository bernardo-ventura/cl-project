"""
Response Enhancer: Converte respostas estruturadas em linguagem natural

Este módulo usa o LLM (Ollama) para transformar respostas estruturadas
do Knowledge Graph em texto natural fluido e conversacional.
"""

import logging
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass
import ollama

from .response_formatter import FormattedResponse

logger = logging.getLogger(__name__)


@dataclass
class EnhancedResponse:
    """
    Resposta melhorada em linguagem natural
    
    Attributes:
        natural_answer: Resposta em linguagem natural fluida
        structured_data: Dados estruturados originais
        confidence: Confiança na resposta (0-1)
        processing_time: Tempo de processamento
    """
    natural_answer: str
    structured_data: str
    confidence: float
    processing_time: float


class ResponseEnhancer:
    """
    Enhancer que usa LLM para converter respostas estruturadas em linguagem natural
    
    Fluxo:
    1. Recebe resposta estruturada do ResponseFormatter
    2. Cria prompt contextualizado para o LLM
    3. Gera resposta em linguagem natural
    4. Combina dados estruturados + resposta natural
    """
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Inicializa o enhancer com modelo Ollama
        
        Args:
            model_name: Nome do modelo Ollama a usar
        """
        self.model_name = model_name
        self._test_llm_connection()
    
    def _test_llm_connection(self) -> None:
        """Testa conexão com Ollama"""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "Test"}],
                options={"num_predict": 10}
            )
            logger.info(f"✅ Conexão com Ollama ({self.model_name}) estabelecida")
        except Exception as e:
            logger.warning(f"⚠️ Ollama não disponível: {e}")
            raise
    
    def enhance_response(self, 
                        formatted_response: FormattedResponse,
                        original_question: str,
                        query_type: str) -> EnhancedResponse:
        """
        Converte resposta estruturada em linguagem natural
        
        Args:
            formatted_response: Resposta estruturada do ResponseFormatter
            original_question: Pergunta original do usuário
            query_type: Tipo de consulta executada
            
        Returns:
            Resposta melhorada em linguagem natural
        """
        import time
        start_time = time.time()
        
        try:
            # Cria prompt contextualizado
            prompt = self._create_enhancement_prompt(
                question=original_question,
                structured_answer=formatted_response.answer,
                query_type=query_type,
                confidence=formatted_response.confidence,
                metadata=formatted_response.metadata
            )
            
            # Gera resposta natural com LLM
            natural_answer = self._generate_natural_response(prompt)
            
            processing_time = time.time() - start_time
            
            return EnhancedResponse(
                natural_answer=natural_answer,
                structured_data=formatted_response.answer,
                confidence=formatted_response.confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"❌ Erro no enhancement: {e}")
            # Fallback: retorna resposta estruturada original
            return EnhancedResponse(
                natural_answer=formatted_response.answer,
                structured_data=formatted_response.answer,
                confidence=formatted_response.confidence * 0.8,  # Reduz confidence
                processing_time=time.time() - start_time
            )
    
    def _create_enhancement_prompt(self,
                                  question: str,
                                  structured_answer: str,
                                  query_type: str,
                                  confidence: float,
                                  metadata: Dict[str, Any]) -> str:
        """
        Cria prompt contextualizado para o LLM
        
        Adapta o prompt baseado no tipo de consulta e dados disponíveis
        """
        
        # Instruções base
        base_instructions = """Você é um assistente especializado em Machine Learning e Deep Learning. 
Sua tarefa é converter informações estruturadas do Knowledge Graph em respostas naturais e conversacionais.

REGRAS IMPORTANTES:
1. Use linguagem clara e didática
2. Mantenha precisão técnica
3. Seja conciso mas informativo
4. Use exemplos quando apropriado
5. Responda em português brasileiro

"""
        
        # Instruções específicas por tipo de consulta
        query_specific_instructions = {
            "what_is": "Explique o conceito de forma didática, incluindo definição, características principais e aplicações.",
            "what_uses": "Liste e explique brevemente cada item que usa o conceito mencionado.",
            "what_is_type_of": "Explique a hierarquia e classificação do conceito.",
            "who_created": "Forneça informações sobre os criadores e contexto histórico.",
            "how_related": "Explique as conexões e relações entre os conceitos.",
            "list_by_type": "Apresente a lista de forma organizada com breves descrições.",
            "find_similar": "Compare e explique as similaridades entre os conceitos."
        }
        
        specific_instruction = query_specific_instructions.get(
            query_type, 
            "Responda de forma clara e informativa."
        )
        
        # Monta prompt final
        prompt = f"""{base_instructions}

TIPO DE CONSULTA: {query_type}
INSTRUÇÃO ESPECÍFICA: {specific_instruction}

PERGUNTA DO USUÁRIO: "{question}"

DADOS DO KNOWLEDGE GRAPH:
{structured_answer}

METADADOS: {metadata.get('result_count', 0)} resultados encontrados, confiança: {confidence:.1%}

TAREFA: Transforme as informações estruturadas acima em uma resposta natural e conversacional que responda à pergunta do usuário. Mantenha a precisão técnica mas use linguagem acessível."""

        return prompt
    
    def _generate_natural_response(self, prompt: str) -> str:
        """
        Gera resposta natural usando Ollama
        
        Args:
            prompt: Prompt contextualizado
            
        Returns:
            Resposta em linguagem natural
        """
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }],
                options={
                    "num_predict": 300,  # Limite de tokens para resposta concisa
                    "temperature": 0.7,   # Criatividade moderada
                    "top_p": 0.9,        # Diversidade de vocabulário
                    "stop": ["TAREFA:", "PERGUNTA:", "DADOS:"]  # Stop tokens
                }
            )
            
            natural_answer = response['message']['content'].strip()
            
            # Validação básica
            if len(natural_answer) < 10:
                raise ValueError("Resposta muito curta")
            
            return natural_answer
            
        except Exception as e:
            logger.error(f"❌ Erro na geração LLM: {e}")
            raise
    
    def create_combined_response(self,
                               enhanced: EnhancedResponse,
                               show_structured: bool = True) -> str:
        """
        Combina resposta natural com dados estruturados opcionais
        
        Args:
            enhanced: Resposta melhorada
            show_structured: Se deve mostrar dados estruturados também
            
        Returns:
            Resposta final combinada
        """
        response_parts = [
            "🤖 **Resposta:**",
            enhanced.natural_answer
        ]
        
        if show_structured:
            response_parts.extend([
                "",
                "📊 **Dados Estruturados:**",
                enhanced.structured_data
            ])
        
        response_parts.extend([
            "",
            f"🎯 **Confiança**: {enhanced.confidence:.1%}",
            f"⏱️ **Tempo LLM**: {enhanced.processing_time:.2f}s"
        ])
        
        return "\n".join(response_parts)


# Factory function
def create_response_enhancer(model_name: str = "llama3.2:3b") -> ResponseEnhancer:
    """
    Factory function para criar instância do ResponseEnhancer
    
    Args:
        model_name: Modelo Ollama a usar
        
    Returns:
        Instância configurada do ResponseEnhancer
    """
    return ResponseEnhancer(model_name)


if __name__ == "__main__":
    # Teste do enhancer
    print("🧪 Testando Response Enhancer...")
    
    try:
        enhancer = create_response_enhancer()
        
        # Mock de resposta estruturada
        from .query_templates import QueryType
        
        mock_formatted = FormattedResponse(
            answer="""📋 **Gradient Descent**
🏷️ **Tipo**: Algorithm
📊 **Propriedades**:
   • **Uses**: backpropagation, optimization
   • **Is A**: optimization_algorithm""",
            metadata={'result_count': 5},
            raw_results=[],
            confidence=0.9
        )
        
        # Testa enhancement
        enhanced = enhancer.enhance_response(
            formatted_response=mock_formatted,
            original_question="O que é gradient descent?",
            query_type="what_is"
        )
        
        print(f"\n📝 Resposta Natural:")
        print(enhanced.natural_answer)
        print(f"\n🎯 Confidence: {enhanced.confidence:.1%}")
        print(f"⏱️ Tempo: {enhanced.processing_time:.2f}s")
        
        print("\n✅ Enhancer funcionando!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        print("💡 Certifique-se que o Ollama está rodando: ollama serve")