"""
Intent Classifier: Classifica a intenção da pergunta usando LLM

Objetivo: Entender o que o usuário quer fazer com as entidades identificadas.
Exemplos: definição, comparação, explicação, aplicações, etc.

Abordagem: LLM com prompt estruturado e categorias pré-definidas.
"""

import logging
import ollama
from typing import Optional, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    """Resultado da classificação de intenção."""
    intent: str
    classification_status: str  # 'llm_classified' ou 'fallback_default'
    suggested_predicates: List[str]  # Para integração com predicate_selector
    query_type: str  # Para integração com sparql_generator


class IntentClassifier:
    """Classificador de intenção usando LLM local (Ollama)"""
    
    def __init__(self, model: str = "llama3.2:3b"):
        """
        Inicializa classificador de intenção
        
        Args:
            model: Modelo LLM para usar
        """
        logger.info("🔄 Inicializando IntentClassifier...")
        
        self.model = model
        
        # Categorias expandidas para melhor integração com pipeline
        self.intent_categories = {
            "definition": {
                "description": "O que é X? Como funciona X? Explique X.",
                "predicates": ["rdfs:label", "rdfs:comment", "ml:definition"],
                "query_type": "describe"
            },
            "comparison": {
                "description": "Diferença entre X e Y? Compare X com Y. X vs Y.",
                "predicates": ["rdfs:label", "ml:compares_with", "ml:similar_to"],
                "query_type": "compare"
            },
            "application": {
                "description": "Para que serve X? Onde usar X? Vantagens de X?",
                "predicates": ["ml:used_for", "ml:application", "ml:advantage"],
                "query_type": "applications"
            },
            "relationship": {
                "description": "X utiliza Y? X implementa Y? Como X se relaciona com Y?",
                "predicates": ["ml:uses", "ml:implements", "ml:depends_on", "ml:part_of"],
                "query_type": "relations"
            },
            "listing": {
                "description": "Liste algoritmos. Quais são os tipos de X? Exemplos de Y.",
                "predicates": ["rdf:type", "ml:algorithm_type", "ml:example_of"],
                "query_type": "list"
            },
            "process": {
                "description": "Como implementar X? Passos para Y? Como aplicar Z?",
                "predicates": ["ml:process", "ml:step", "ml:implementation"],
                "query_type": "process"
            }
        }
        
        logger.info(f"✅ IntentClassifier inicializado com {len(self.intent_categories)} categorias")
    
    def classify_intent(self, question: str, entities: List[str] = None) -> IntentResult:
        """
        Classifica a intenção da pergunta
        
        Args:
            question: Pergunta do usuário
            entities: Entidades identificadas (opcional, para contexto)
            
        Returns:
            Resultado estruturado com intenção e metadados
        """
        if not question.strip():
            return self._create_fallback_result()
        
        logger.info(f"🔍 Classificando intenção da pergunta")
        
        # Prompt estruturado para classificação
        prompt = self._build_classification_prompt(question, entities)
        
        try:
            # Chamar LLM
            response = self._call_ollama(prompt)
            
            # Extrair intenção da resposta
            intent = self._extract_intent(response)
            
            # Criar resultado estruturado
            result = self._create_intent_result(intent)
            
            logger.info(f"✅ Intenção: {intent} (status: {result.classification_status})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erro classificando intenção: {e}")
            return self._create_fallback_result()
    
    def _build_classification_prompt(self, question: str, entities: List[str] = None) -> str:
        """Constrói prompt estruturado para classificação"""
        
        entities_context = ""
        if entities:
            entities_context = f"\nENTIDADES IDENTIFICADAS: {', '.join(entities)}"
        
        categories_text = "\n".join([
            f"- {intent}: {info['description']}"
            for intent, info in self.intent_categories.items()
        ])
        
        prompt = f"""Classifique a intenção desta pergunta escolhendo UMA categoria:

PERGUNTA: "{question}"{entities_context}

CATEGORIAS DISPONÍVEIS:
{categories_text}

EXEMPLOS:
- "O que é CNN?" → definition
- "Como funciona gradient descent?" → definition
- "CNN vs RNN" → comparison
- "Para que serve SVM?" → application
- "CNN utiliza backpropagation?" → relationship
- "Liste algoritmos de clustering" → listing
- "Como implementar PCA?" → process

RESPOSTA: Responda APENAS uma palavra da lista acima.

Categoria:"""
        
        return prompt
    
    def _call_ollama(self, prompt: str) -> str:
        """Chama API do Ollama usando biblioteca oficial"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.1,  # Baixa criatividade para classificação
                    'num_predict': 10,   # Resposta muito curta
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"❌ Erro chamando Ollama: {e}")
            raise
    
    def _extract_intent(self, response: str) -> str:
        """Extrai intenção limpa da resposta do LLM"""
        
        response = response.lower().strip()
        
        # Buscar categoria diretamente
        for intent in self.intent_categories.keys():
            if intent in response:
                return intent
        
        # Mapeamento português → inglês
        pt_mapping = {
            "definição": "definition", "explicação": "definition",
            "comparação": "comparison", "diferença": "comparison",
            "aplicação": "application", "uso": "application",
            "relação": "relationship", "relacionamento": "relationship",
            "lista": "listing", "listar": "listing", "exemplos": "listing",
            "processo": "process", "implementação": "process", "como": "process"
        }
        
        for pt_word, en_intent in pt_mapping.items():
            if pt_word in response:
                return en_intent
        
        return "definition"  # Fallback
    
    def _create_intent_result(self, intent: str) -> IntentResult:
        """Cria resultado estruturado para a intenção"""
        if intent not in self.intent_categories:
            intent = "definition"
        
        category_info = self.intent_categories[intent]
        
        return IntentResult(
            intent=intent,
            classification_status="llm_classified",  # LLM conseguiu classificar
            suggested_predicates=category_info["predicates"],
            query_type=category_info["query_type"]
        )
    
    def _create_fallback_result(self) -> IntentResult:
        """Cria resultado fallback padrão"""
        return IntentResult(
            intent="definition",
            classification_status="fallback_default",  # Fallback por erro/pergunta vazia
            suggested_predicates=["rdfs:label", "rdfs:comment"],
            query_type="describe"
        )


def create_intent_classifier() -> IntentClassifier:
    """Factory function"""
    return IntentClassifier()


if __name__ == "__main__":
    # Teste do classificador
    print("🧪 Testando Intent Classifier...")
    
    try:
        classifier = create_intent_classifier()
        
        # Casos de teste expandidos
        test_questions = [
            "O que é CNN?",  # definition
            "Como funciona gradient descent?",  # definition
            "Diferença entre SVM e Random Forest?",  # comparison
            "Para que serve deep learning?",  # application
            "CNN utiliza backpropagation?",  # relationship
            "Liste algoritmos de clustering",  # listing
            "Como implementar PCA?",  # process
            "Quais são os tipos de redes neurais?",  # listing
            "Vantagens do LSTM?",  # application
            "Como Adam se relaciona com gradient descent?",  # relationship
        ]
        
        print(f"\n🎯 Testando {len(test_questions)} casos:")
        print("=" * 60)
        
        for i, question in enumerate(test_questions, 1):
            result = classifier.classify_intent(question)
            print(f"{i:2d}. '{question}'")
            print(f"     → {result.intent} ({result.classification_status})")
            print(f"     → predicados: {result.suggested_predicates[:2]}...")  # Primeiros 2
            print(f"     → tipo query: {result.query_type}")
            print()
        
        print("✅ Intent Classifier funcionando!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()