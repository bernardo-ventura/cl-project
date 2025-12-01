"""
RAG Pipeline: Orquestração completa do sistema RAG
Integra Retriever + Response Generator em um pipeline unificado
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import json
from pathlib import Path

try:
    from .retriever import RAGRetriever, create_retriever, RetrievalConfig
    from .response_generator import RAGResponseGenerator, create_response_generator, GenerationConfig
except ImportError:
    from retriever import RAGRetriever, create_retriever, RetrievalConfig
    from response_generator import RAGResponseGenerator, create_response_generator, GenerationConfig

logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuração completa do sistema RAG."""
    # Vector Store
    vector_store_path: str = "data/rag_vector_store"
    
    # Retrieval
    top_k: int = 5
    similarity_threshold: float = 0.3
    enable_reranking: bool = True
    book_diversity: bool = True
    max_tokens_per_doc: int = 500
    
    # Generation
    model_name: str = "llama3.2:3b"
    temperature: float = 0.1
    max_context_length: int = 4000
    max_response_tokens: int = 800
    response_style: str = "comprehensive"  # comprehensive, concise, technical
    include_sources: bool = True
    citation_style: str = "bracket"  # bracket, footnote, inline
    
    # Pipeline
    enable_caching: bool = False
    debug_mode: bool = False
    save_history: bool = True

@dataclass
class RAGResponse:
    """Resposta completa do sistema RAG com métricas."""
    query: str
    answer: str
    sources: List[str]
    confidence_score: float
    
    # Métricas detalhadas
    total_time: float
    retrieval_time: float
    generation_time: float
    
    # Metadados
    documents_found: int
    documents_used: int
    model_used: str
    config_used: Dict[str, Any]
    
    # Debug info
    retrieval_debug: Optional[Dict[str, Any]] = None
    generation_debug: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Converte para JSON."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class RAGPipeline:
    """Pipeline completo do sistema RAG."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Inicializa o pipeline RAG.
        
        Args:
            config: Configuração do sistema
        """
        self.config = config or RAGConfig()
        self.retriever: Optional[RAGRetriever] = None
        self.generator: Optional[RAGResponseGenerator] = None
        self.is_initialized = False
        self.query_history: List[Dict[str, Any]] = []
        
        logger.info(f"🚀 RAGPipeline criado:")
        logger.info(f"   Vector Store: {self.config.vector_store_path}")
        logger.info(f"   Modelo LLM: {self.config.model_name}")
        logger.info(f"   Top-K: {self.config.top_k}")
        logger.info(f"   Estilo: {self.config.response_style}")
    
    def initialize(self):
        """Inicializa todos os componentes do pipeline."""
        if self.is_initialized:
            return
        
        start_time = time.time()
        logger.info("🔄 Inicializando pipeline RAG...")
        
        # Configurar retriever
        retrieval_config = RetrievalConfig(
            top_k=self.config.top_k,
            similarity_threshold=self.config.similarity_threshold,
            enable_reranking=self.config.enable_reranking,
            book_diversity=self.config.book_diversity,
            max_tokens_per_doc=self.config.max_tokens_per_doc
        )
        
        self.retriever = create_retriever(
            vector_store_path=self.config.vector_store_path,
            config=retrieval_config
        )
        
        # Configurar gerador
        generation_config = GenerationConfig(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            max_context_length=self.config.max_context_length,
            max_response_tokens=self.config.max_response_tokens,
            response_style=self.config.response_style,
            include_sources=self.config.include_sources,
            citation_style=self.config.citation_style
        )
        
        self.generator = create_response_generator(generation_config)
        
        # Inicializar componentes
        self.retriever.initialize()
        self.generator.initialize()
        
        self.is_initialized = True
        init_time = time.time() - start_time
        
        logger.info(f"✅ Pipeline inicializado em {init_time:.2f}s")
    
    def query(self, question: str) -> RAGResponse:
        """
        Processa uma consulta completa no sistema RAG.
        
        Args:
            question: Pergunta do usuário
            
        Returns:
            Resposta completa do sistema RAG
        """
        if not self.is_initialized:
            self.initialize()
        
        start_time = time.time()
        logger.info(f"❓ Processando consulta: '{question}'")
        
        try:
            # 1. RECUPERAÇÃO
            logger.info("🔍 Fase 1: Recuperação de documentos")
            retrieval_start = time.time()
            
            retrieval_result = self.retriever.retrieve(question)
            
            retrieval_time = time.time() - retrieval_start
            logger.info(f"✅ Recuperação: {len(retrieval_result.documents)} docs em {retrieval_time:.3f}s")
            
            # 2. GERAÇÃO
            logger.info("🤖 Fase 2: Geração de resposta")
            generation_start = time.time()
            
            generation_result = self.generator.generate_response(retrieval_result)
            
            generation_time = time.time() - generation_start
            logger.info(f"✅ Geração: resposta em {generation_time:.3f}s")
            
            # 3. CONSOLIDAÇÃO
            total_time = time.time() - start_time
            
            # Criar resposta consolidada
            response = RAGResponse(
                query=question,
                answer=generation_result.answer,
                sources=generation_result.sources_used,
                confidence_score=generation_result.confidence_score,
                total_time=total_time,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                documents_found=retrieval_result.total_found,
                documents_used=len(retrieval_result.documents),
                model_used=generation_result.model_used,
                config_used={
                    'top_k': self.config.top_k,
                    'temperature': self.config.temperature,
                    'response_style': self.config.response_style
                }
            )
            
            # Debug info se habilitado
            if self.config.debug_mode:
                response.retrieval_debug = {
                    'query_analysis': retrieval_result.query_analysis,
                    'processing_time': retrieval_result.processing_time,
                    'documents': [doc.to_dict() for doc in retrieval_result.documents]
                }
                response.generation_debug = {
                    'context_length': generation_result.context_length,
                    'token_count': generation_result.token_count,
                    'generation_time': generation_result.generation_time
                }
            
            # Salvar no histórico
            if self.config.save_history:
                self.query_history.append({
                    'timestamp': time.time(),
                    'query': question,
                    'response_summary': {
                        'confidence': response.confidence_score,
                        'total_time': total_time,
                        'sources_count': len(response.sources)
                    }
                })
            
            logger.info(f"🎉 Consulta processada: {total_time:.2f}s total, confiança {response.confidence_score:.2f}")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Erro no pipeline: {e}")
            
            # Resposta de fallback
            total_time = time.time() - start_time
            return RAGResponse(
                query=question,
                answer=f"Desculpe, ocorreu um erro ao processar sua consulta: {str(e)}",
                sources=[],
                confidence_score=0.0,
                total_time=total_time,
                retrieval_time=0.0,
                generation_time=0.0,
                documents_found=0,
                documents_used=0,
                model_used="error_fallback",
                config_used=asdict(self.config)
            )
    
    def batch_query(self, questions: List[str]) -> List[RAGResponse]:
        """
        Processa múltiplas consultas em lote.
        
        Args:
            questions: Lista de perguntas
            
        Returns:
            Lista de respostas
        """
        logger.info(f"📊 Processando {len(questions)} consultas em lote...")
        
        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"🔄 Processando {i}/{len(questions)}")
            response = self.query(question)
            results.append(response)
        
        logger.info(f"✅ Lote processado: {len(results)} respostas")
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do pipeline."""
        stats = {
            'is_initialized': self.is_initialized,
            'config': asdict(self.config),
            'query_history': {
                'total_queries': len(self.query_history),
                'recent_queries': self.query_history[-5:] if self.query_history else []
            }
        }
        
        if self.is_initialized:
            if self.retriever:
                stats['retriever'] = self.retriever.get_statistics()
            if self.generator:
                stats['generator'] = self.generator.get_statistics()
        
        return stats
    
    def clear_history(self):
        """Limpa histórico de consultas."""
        self.query_history.clear()
        logger.info("🗑️ Histórico de consultas limpo")
    
    def save_history_to_file(self, filepath: str):
        """Salva histórico em arquivo."""
        if not self.query_history:
            logger.warning("⚠️ Nenhum histórico para salvar")
            return
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.query_history, f, indent=2, default=str)
        
        logger.info(f"💾 Histórico salvo: {filepath} ({len(self.query_history)} consultas)")
    
    def load_history_from_file(self, filepath: str):
        """Carrega histórico de arquivo."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.query_history = json.load(f)
            
            logger.info(f"📂 Histórico carregado: {filepath} ({len(self.query_history)} consultas)")
        except Exception as e:
            logger.error(f"❌ Erro carregando histórico: {e}")


def create_rag_pipeline(config: Optional[RAGConfig] = None) -> RAGPipeline:
    """Factory function para criar RAGPipeline."""
    return RAGPipeline(config)


if __name__ == "__main__":
    # Teste do pipeline
    print("🧪 Testando RAG Pipeline...")
    
    try:
        # Criar configuração para teste
        test_config = RAGConfig(
            top_k=3,
            response_style="concise",
            debug_mode=True,
            save_history=True
        )
        
        # Criar pipeline
        pipeline = create_rag_pipeline(test_config)
        
        # Consultas de teste
        test_queries = [
            "What is machine learning?",
            "How does gradient descent work?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔎 TESTE {i}: '{query}'")
            print("=" * 50)
            
            response = pipeline.query(query)
            
            print(f"⏱️  Tempo total: {response.total_time:.2f}s")
            print(f"   Recuperação: {response.retrieval_time:.3f}s")
            print(f"   Geração: {response.generation_time:.3f}s")
            print(f"📊 Confiança: {response.confidence_score:.2f}")
            print(f"📄 Documentos: {response.documents_used}/{response.documents_found}")
            print(f"🤖 Modelo: {response.model_used}")
            
            print(f"\n💬 RESPOSTA:")
            print("-" * 30)
            print(response.answer)
            print("-" * 30)
            
            if response.sources:
                print(f"\n📖 FONTES ({len(response.sources)}):")
                for j, source in enumerate(response.sources, 1):
                    print(f"   {j}. {source}")
        
        # Estatísticas
        print(f"\n📊 ESTATÍSTICAS DO PIPELINE:")
        stats = pipeline.get_statistics()
        print(f"   Consultas processadas: {stats['query_history']['total_queries']}")
        print(f"   Pipeline inicializado: {stats['is_initialized']}")
        
        # Salvar histórico
        pipeline.save_history_to_file("test_rag_history.json")
        
        print(f"\n🎉 Teste do RAG Pipeline concluído com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()