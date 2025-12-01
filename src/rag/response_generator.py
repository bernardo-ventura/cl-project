"""
RAG Response Generator: Geração de respostas usando LLM com contexto recuperado
Integra documentos relevantes com Ollama LLM para respostas fundamentadas
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import ollama

try:
    from .retriever import RetrievedDocument, RetrievalResult
except ImportError:
    from retriever import RetrievedDocument, RetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configurações para geração de respostas."""
    model_name: str = "llama3.2:3b"
    max_context_length: int = 4000
    temperature: float = 0.1
    max_response_tokens: int = 800
    include_sources: bool = True
    citation_style: str = "bracket"  # bracket, footnote, inline
    response_style: str = "comprehensive"  # comprehensive, concise, technical
    
@dataclass
class GeneratedResponse:
    """Resposta gerada pelo LLM com metadados."""
    query: str
    answer: str
    sources_used: List[str]
    confidence_score: float
    generation_time: float
    token_count: Optional[int] = None
    model_used: str = ""
    context_length: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'query': self.query,
            'answer': self.answer,
            'sources_used': self.sources_used,
            'confidence_score': self.confidence_score,
            'generation_time': self.generation_time,
            'token_count': self.token_count,
            'model_used': self.model_used,
            'context_length': self.context_length
        }


class RAGResponseGenerator:
    """Gerador de respostas para sistema RAG usando Ollama LLM."""
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        """
        Inicializa o gerador de respostas.
        
        Args:
            config: Configurações de geração
        """
        self.config = config or GenerationConfig()
        self.ollama_available = False
        self.is_initialized = False
        
        logger.info(f"🤖 RAGResponseGenerator criado:")
        logger.info(f"   Modelo: {self.config.model_name}")
        logger.info(f"   Estilo: {self.config.response_style}")
        logger.info(f"   Temperatura: {self.config.temperature}")
    
    def initialize(self):
        """Inicializa e verifica conexão com Ollama."""
        if self.is_initialized:
            return
        
        try:
            # Testar conexão com Ollama
            response = ollama.list()
            
            # Verificar se o modelo está disponível
            available_models = [model.model for model in response.models]
            if self.config.model_name in available_models:
                self.ollama_available = True
                logger.info(f"✅ Ollama conectado, modelo {self.config.model_name} disponível")
            else:
                logger.warning(f"⚠️ Modelo {self.config.model_name} não encontrado")
                logger.info(f"📋 Modelos disponíveis: {available_models}")
                if available_models:
                    self.config.model_name = available_models[0]
                    self.ollama_available = True
                    logger.info(f"🔄 Usando modelo: {self.config.model_name}")
                else:
                    self.ollama_available = False
                
        except Exception as e:
            logger.error(f"❌ Erro conectando com Ollama: {e}")
            self.ollama_available = False
        
        self.is_initialized = True
    
    def _build_context(self, documents: List[RetrievedDocument]) -> str:
        """
        Constroi contexto para o LLM a partir dos documentos.
        
        Args:
            documents: Documentos recuperados
            
        Returns:
            Contexto formatado
        """
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(documents):
            # Formato: [1] Livro: Conteúdo...
            source_ref = f"[{i+1}]"
            doc_text = f"{source_ref} {doc.source_book}: {doc.content.strip()}"
            
            # Verificar se ainda cabe no limite
            if total_length + len(doc_text) > self.config.max_context_length:
                # Truncar o documento para caber
                remaining_space = self.config.max_context_length - total_length - len(source_ref) - len(doc.source_book) - 3
                if remaining_space > 100:  # Mínimo útil
                    truncated_content = doc.content[:remaining_space] + "..."
                    doc_text = f"{source_ref} {doc.source_book}: {truncated_content}"
                    context_parts.append(doc_text)
                break
            
            context_parts.append(doc_text)
            total_length += len(doc_text)
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Constroi prompt para o LLM baseado no estilo de resposta.
        
        Args:
            query: Pergunta do usuário
            context: Contexto dos documentos
            
        Returns:
            Prompt formatado
        """
        style_instructions = {
            "comprehensive": "Forneça uma resposta detalhada e completa",
            "concise": "Seja conciso mas informativo",
            "technical": "Foque nos aspectos técnicos e detalhes específicos"
        }
        
        style_instruction = style_instructions.get(
            self.config.response_style, 
            "Forneça uma resposta clara e informativa"
        )
        
        citation_instruction = ""
        if self.config.include_sources:
            if self.config.citation_style == "bracket":
                citation_instruction = "Cite as fontes usando [1], [2], etc. quando referenciar informações específicas."
            elif self.config.citation_style == "footnote":
                citation_instruction = "Inclua referências como notas no final da resposta."
            else:  # inline
                citation_instruction = "Mencione as fontes diretamente no texto."
        
        prompt = f"""Você é um assistente especializado em Machine Learning e Deep Learning. Responda à pergunta usando APENAS as informações fornecidas no contexto.

PERGUNTA: {query}

CONTEXTO:
{context}

INSTRUÇÕES:
- {style_instruction}
- Base sua resposta EXCLUSIVAMENTE no contexto fornecido
- Se o contexto não contém informação suficiente, diga isso claramente
- {citation_instruction}
- Seja factual e preciso
- Use linguagem clara e acessível

RESPOSTA:"""
        
        return prompt
    
    def _calculate_confidence(self, response_text: str, documents: List[RetrievedDocument]) -> float:
        """
        Calcula score de confiança baseado na resposta e documentos.
        
        Args:
            response_text: Texto da resposta gerada
            documents: Documentos usados como contexto
            
        Returns:
            Score de confiança (0-1)
        """
        # Fatores para confiança
        factors = []
        
        # 1. Qualidade dos documentos (similarity scores)
        if documents:
            avg_similarity = sum(doc.similarity_score for doc in documents) / len(documents)
            factors.append(min(avg_similarity, 1.0))
        
        # 2. Comprimento da resposta (nem muito curta nem muito longa)
        response_length = len(response_text.split())
        if 50 <= response_length <= 400:
            factors.append(1.0)
        elif 20 <= response_length <= 600:
            factors.append(0.8)
        else:
            factors.append(0.6)
        
        # 3. Presença de citations (se habilitado)
        if self.config.include_sources:
            citation_count = response_text.count('[') + response_text.count('fonte')
            if citation_count > 0:
                factors.append(0.9)
            else:
                factors.append(0.7)
        
        # 4. Não contém frases de incerteza
        uncertainty_phrases = [
            "não tenho informação", "não posso", "não está claro",
            "insuficiente", "não consegui", "não sei"
        ]
        has_uncertainty = any(phrase in response_text.lower() for phrase in uncertainty_phrases)
        factors.append(0.5 if has_uncertainty else 1.0)
        
        # Média ponderada
        if factors:
            return sum(factors) / len(factors)
        return 0.5
    
    def _extract_sources(self, response_text: str, documents: List[RetrievedDocument]) -> List[str]:
        """
        Extrai fontes mencionadas na resposta.
        
        Args:
            response_text: Texto da resposta
            documents: Documentos disponíveis
            
        Returns:
            Lista de fontes utilizadas
        """
        sources = set()
        
        # Buscar citações [1], [2], etc.
        import re
        citations = re.findall(r'\[(\d+)\]', response_text)
        
        for citation in citations:
            try:
                idx = int(citation) - 1  # Converter para índice 0-based
                if 0 <= idx < len(documents):
                    source = f"{documents[idx].source_book} (chunk: {documents[idx].chunk_id})"
                    sources.add(source)
            except ValueError:
                continue
        
        # Se não encontrou citações específicas, considerar todas as fontes usadas
        if not sources:
            sources = {f"{doc.source_book} (chunk: {doc.chunk_id})" for doc in documents}
        
        return list(sources)
    
    def generate_response(self, retrieval_result: RetrievalResult) -> GeneratedResponse:
        """
        Gera resposta baseada no resultado da recuperação.
        
        Args:
            retrieval_result: Resultado da recuperação de documentos
            
        Returns:
            Resposta gerada
        """
        if not self.is_initialized:
            self.initialize()
        
        start_time = time.time()
        
        logger.info(f"🤖 Gerando resposta para: '{retrieval_result.query}'")
        
        # Verificar se há documentos
        if not retrieval_result.documents:
            logger.warning("⚠️ Nenhum documento recuperado")
            return self._generate_fallback_response(retrieval_result.query, start_time)
        
        # Verificar se Ollama está disponível
        if not self.ollama_available:
            logger.warning("⚠️ Ollama não disponível, gerando resposta básica")
            return self._generate_fallback_response(retrieval_result.query, start_time)
        
        try:
            # Construir contexto
            context = self._build_context(retrieval_result.documents)
            logger.info(f"📄 Contexto construído: {len(context)} caracteres")
            
            # Construir prompt
            prompt = self._build_prompt(retrieval_result.query, context)
            
            # Chamar LLM
            logger.info(f"🔄 Chamando {self.config.model_name}...")
            response = ollama.generate(
                model=self.config.model_name,
                prompt=prompt,
                options={
                    'temperature': self.config.temperature,
                    'num_predict': self.config.max_response_tokens,
                }
            )
            
            answer_text = response['response'].strip()
            
            # Calcular métricas
            generation_time = time.time() - start_time
            confidence = self._calculate_confidence(answer_text, retrieval_result.documents)
            sources = self._extract_sources(answer_text, retrieval_result.documents)
            
            logger.info(f"✅ Resposta gerada em {generation_time:.2f}s")
            logger.info(f"📊 Confiança: {confidence:.2f}")
            
            return GeneratedResponse(
                query=retrieval_result.query,
                answer=answer_text,
                sources_used=sources,
                confidence_score=confidence,
                generation_time=generation_time,
                token_count=len(answer_text.split()),
                model_used=self.config.model_name,
                context_length=len(context)
            )
            
        except Exception as e:
            logger.error(f"❌ Erro na geração: {e}")
            return self._generate_fallback_response(retrieval_result.query, start_time)
    
    def _generate_fallback_response(self, query: str, start_time: float) -> GeneratedResponse:
        """Gera resposta de fallback quando LLM não está disponível."""
        generation_time = time.time() - start_time
        
        fallback_text = f"""Desculpe, não consigo gerar uma resposta completa para "{query}" no momento. 
O sistema LLM não está disponível ou não foram encontrados documentos relevantes suficientes. 
Por favor, tente reformular sua pergunta ou verifique se o serviço Ollama está executando."""
        
        return GeneratedResponse(
            query=query,
            answer=fallback_text,
            sources_used=[],
            confidence_score=0.1,
            generation_time=generation_time,
            model_used="fallback",
            context_length=0
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do gerador."""
        return {
            'ollama_available': self.ollama_available,
            'is_initialized': self.is_initialized,
            'config': {
                'model_name': self.config.model_name,
                'temperature': self.config.temperature,
                'max_context_length': self.config.max_context_length,
                'response_style': self.config.response_style,
                'include_sources': self.config.include_sources
            }
        }


def create_response_generator(config: Optional[GenerationConfig] = None) -> RAGResponseGenerator:
    """Factory function para criar RAGResponseGenerator."""
    return RAGResponseGenerator(config)


if __name__ == "__main__":
    # Teste do módulo
    print("🧪 Testando RAG Response Generator...")
    
    try:
        # Simular resultado de recuperação (mock)
        from retriever import RetrievedDocument, RetrievalResult, RetrievalConfig
        
        # Documentos mock para teste
        mock_docs = [
            RetrievedDocument(
                content="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions without being explicitly programmed. It uses algorithms to identify patterns in data.",
                source_book="intro_ml_python",
                chunk_id="test_chunk_001",
                similarity_score=0.85,
                rank=1,
                relevance_reason="Highly relevant definition",
                token_count=30
            ),
            RetrievedDocument(
                content="Supervised learning algorithms learn from labeled training data to make predictions on new, unseen data. Common examples include linear regression, decision trees, and neural networks.",
                source_book="ml_art_science",
                chunk_id="test_chunk_002",
                similarity_score=0.72,
                rank=2,
                relevance_reason="Good technical explanation",
                token_count=28
            )
        ]
        
        mock_result = RetrievalResult(
            query="What is machine learning?",
            documents=mock_docs,
            total_found=10,
            processing_time=0.05,
            config_used=RetrievalConfig(),
            query_analysis={'query_type': 'technical'}
        )
        
        # Criar gerador
        generator = create_response_generator()
        
        print("📋 Testando geração de resposta...")
        response = generator.generate_response(mock_result)
        
        print(f"\n📊 RESULTADO:")
        print(f"🔎 Query: {response.query}")
        print(f"⏱️  Tempo de geração: {response.generation_time:.2f}s")
        print(f"🤖 Modelo usado: {response.model_used}")
        print(f"📊 Confiança: {response.confidence_score:.2f}")
        print(f"📄 Tokens: {response.token_count}")
        print(f"📚 Fontes: {len(response.sources_used)}")
        
        print(f"\n💬 RESPOSTA:")
        print("-" * 50)
        print(response.answer)
        print("-" * 50)
        
        if response.sources_used:
            print(f"\n📖 FONTES UTILIZADAS:")
            for i, source in enumerate(response.sources_used, 1):
                print(f"   {i}. {source}")
        
        print("\n🎉 Teste concluído!")
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()