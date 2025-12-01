"""
RAG Retriever: Interface inteligente para recuperação de documentos
Combina Vector Store com processamento avançado de consultas
"""

import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    from .vector_store import VectorStore, SearchResult, create_vector_store
    from .document_processor import create_document_processor
except ImportError:
    from vector_store import VectorStore, SearchResult, create_vector_store
    from document_processor import create_document_processor

logger = logging.getLogger(__name__)

@dataclass
class RetrievalConfig:
    """Configurações para recuperação de documentos."""
    top_k: int = 5
    similarity_threshold: float = 0.3
    enable_reranking: bool = True
    diversity_factor: float = 0.7
    max_tokens_per_doc: int = 500
    book_diversity: bool = True
    
@dataclass 
class RetrievedDocument:
    """Documento recuperado com metadados enriquecidos."""
    content: str
    source_book: str
    chunk_id: str
    similarity_score: float
    rank: int
    relevance_reason: Optional[str] = None
    token_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'content': self.content,
            'source_book': self.source_book,
            'chunk_id': self.chunk_id,
            'similarity_score': float(self.similarity_score),
            'rank': self.rank,
            'relevance_reason': self.relevance_reason,
            'token_count': self.token_count
        }

@dataclass
class RetrievalResult:
    """Resultado completo de uma consulta."""
    query: str
    documents: List[RetrievedDocument]
    total_found: int
    processing_time: float
    config_used: RetrievalConfig
    query_analysis: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'query': self.query,
            'documents': [doc.to_dict() for doc in self.documents],
            'total_found': self.total_found,
            'processing_time': self.processing_time,
            'query_analysis': self.query_analysis
        }


class RAGRetriever:
    """Retriever inteligente para sistema RAG."""
    
    def __init__(self, vector_store_path: str, config: Optional[RetrievalConfig] = None):
        """
        Inicializa o retriever.
        
        Args:
            vector_store_path: Caminho para o vector store (sem extensão)
            config: Configurações de recuperação
        """
        self.vector_store_path = vector_store_path
        self.config = config or RetrievalConfig()
        self.vector_store: Optional[VectorStore] = None
        self.document_processor = None
        self.is_initialized = False
        
        logger.info(f"🔍 RAGRetriever criado com configurações:")
        logger.info(f"   Top-K: {self.config.top_k}")
        logger.info(f"   Threshold: {self.config.similarity_threshold}")
        logger.info(f"   Re-ranking: {self.config.enable_reranking}")
    
    def initialize(self):
        """Inicializa componentes (lazy loading)."""
        if self.is_initialized:
            return
        
        start_time = time.time()
        
        # Carregar vector store
        logger.info("📂 Carregando Vector Store...")
        self.vector_store = create_vector_store()
        self.vector_store.load(self.vector_store_path)
        
        # Carregar document processor para encoding de queries
        logger.info("🤖 Inicializando Document Processor...")
        self.document_processor = create_document_processor()
        self.document_processor._load_model()
        
        self.is_initialized = True
        init_time = time.time() - start_time
        
        logger.info(f"✅ RAGRetriever inicializado em {init_time:.2f}s")
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analisa a consulta para otimizar a recuperação.
        
        Args:
            query: Consulta em linguagem natural
            
        Returns:
            Análise da consulta
        """
        analysis = {
            'query_length': len(query.split()),
            'has_algorithm_terms': False,
            'has_concept_terms': False,
            'has_technical_terms': False,
            'query_type': 'general'
        }
        
        # Keywords para diferentes categorias
        algorithm_terms = {'svm', 'cnn', 'rnn', 'lstm', 'gru', 'transformer', 'bert', 
                          'gradient descent', 'backpropagation', 'adam', 'sgd'}
        concept_terms = {'overfitting', 'regularization', 'cross-validation', 
                        'supervised', 'unsupervised', 'reinforcement'}
        technical_terms = {'neural network', 'deep learning', 'machine learning',
                          'feature', 'embedding', 'optimization'}
        
        query_lower = query.lower()
        
        # Detectar categorias
        for term in algorithm_terms:
            if term in query_lower:
                analysis['has_algorithm_terms'] = True
                break
                
        for term in concept_terms:
            if term in query_lower:
                analysis['has_concept_terms'] = True
                break
                
        for term in technical_terms:
            if term in query_lower:
                analysis['has_technical_terms'] = True
                break
        
        # Determinar tipo da consulta
        if analysis['has_algorithm_terms']:
            analysis['query_type'] = 'algorithm_specific'
        elif analysis['has_concept_terms']:
            analysis['query_type'] = 'conceptual'
        elif analysis['has_technical_terms']:
            analysis['query_type'] = 'technical'
        elif any(word in query_lower for word in ['what', 'how', 'why', 'explain']):
            analysis['query_type'] = 'explanatory'
        elif any(word in query_lower for word in ['list', 'show', 'find', 'examples']):
            analysis['query_type'] = 'exploratory'
        
        return analysis
    
    def _rerank_results(self, results: List[SearchResult], query_analysis: Dict[str, Any]) -> List[SearchResult]:
        """
        Re-ranking inteligente baseado na análise da consulta.
        
        Args:
            results: Resultados brutos do vector store
            query_analysis: Análise da consulta
            
        Returns:
            Resultados re-rankeados
        """
        if not self.config.enable_reranking:
            return results
        
        # Aplicar book diversity se habilitado
        if self.config.book_diversity:
            results = self._apply_book_diversity(results)
        
        # Boosting por tipo de consulta
        boosted_results = []
        for result in results:
            score = result.similarity_score
            content_lower = result.document.content.lower()
            
            # Boost para queries sobre algoritmos específicos
            if query_analysis['query_type'] == 'algorithm_specific':
                if any(term in content_lower for term in ['algorithm', 'method', 'approach']):
                    score *= 1.2
            
            # Boost para queries conceituais
            elif query_analysis['query_type'] == 'conceptual':
                if any(term in content_lower for term in ['concept', 'idea', 'principle']):
                    score *= 1.15
            
            # Penalty para conteúdo muito curto
            if len(result.document.content.split()) < 50:
                score *= 0.9
            
            # Boost para conteúdo com good context
            if len(result.document.content.split()) > 200:
                score *= 1.05
            
            # Criar novo result com score ajustado
            boosted_result = SearchResult(
                document=result.document,
                similarity_score=score,
                rank=result.rank
            )
            boosted_results.append(boosted_result)
        
        # Re-ordenar por novo score
        boosted_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Atualizar ranks
        for i, result in enumerate(boosted_results):
            result.rank = i + 1
        
        return boosted_results
    
    def _apply_book_diversity(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Aplica diversidade de livros nos resultados.
        
        Args:
            results: Resultados originais
            
        Returns:
            Resultados com diversidade aplicada
        """
        if len(results) <= 3:
            return results
        
        diverse_results = []
        books_used = set()
        remaining_results = results.copy()
        
        # Primeira passada: um resultado por livro
        for result in results:
            book = result.document.source_book
            if book not in books_used:
                diverse_results.append(result)
                books_used.add(book)
                remaining_results.remove(result)
            
            if len(diverse_results) >= self.config.top_k:
                break
        
        # Segunda passada: preencher restantes com melhores scores
        while len(diverse_results) < self.config.top_k and remaining_results:
            diverse_results.append(remaining_results.pop(0))
        
        return diverse_results
    
    def _create_retrieved_documents(self, results: List[SearchResult], query_analysis: Dict[str, Any]) -> List[RetrievedDocument]:
        """
        Converte SearchResults em RetrievedDocuments com metadados.
        
        Args:
            results: Resultados da busca
            query_analysis: Análise da consulta
            
        Returns:
            Lista de RetrievedDocument
        """
        retrieved_docs = []
        
        for result in results:
            # Truncar conteúdo se necessário
            content = result.document.content
            tokens = content.split()
            
            if len(tokens) > self.config.max_tokens_per_doc:
                content = ' '.join(tokens[:self.config.max_tokens_per_doc]) + "..."
            
            # Gerar razão de relevância simples
            relevance_reason = self._generate_relevance_reason(result, query_analysis)
            
            retrieved_doc = RetrievedDocument(
                content=content,
                source_book=result.document.source_book,
                chunk_id=result.document.chunk_id,
                similarity_score=result.similarity_score,
                rank=result.rank,
                relevance_reason=relevance_reason,
                token_count=len(tokens)
            )
            
            retrieved_docs.append(retrieved_doc)
        
        return retrieved_docs
    
    def _generate_relevance_reason(self, result: SearchResult, query_analysis: Dict[str, Any]) -> str:
        """Gera explicação simples da relevância."""
        score = result.similarity_score
        
        if score > 0.8:
            return "Highly relevant content with strong semantic match"
        elif score > 0.6:
            return "Good semantic relevance to the query"
        elif score > 0.4:
            return "Moderate relevance with related concepts"
        else:
            return "Basic relevance, may contain useful context"
    
    def retrieve(self, query: str, custom_config: Optional[RetrievalConfig] = None) -> RetrievalResult:
        """
        Recupera documentos relevantes para uma consulta.
        
        Args:
            query: Consulta em linguagem natural
            custom_config: Configurações customizadas (opcional)
            
        Returns:
            Resultado completo da recuperação
        """
        if not self.is_initialized:
            self.initialize()
        
        start_time = time.time()
        config = custom_config or self.config
        
        logger.info(f"🔍 Recuperando documentos para: '{query}'")
        
        # 1. Analisar consulta
        query_analysis = self._analyze_query(query)
        logger.info(f"📊 Tipo da consulta: {query_analysis['query_type']}")
        
        # 2. Buscar no vector store
        search_results = self.vector_store.search_by_text(
            query, 
            self.document_processor.model, 
            top_k=min(config.top_k * 2, 20)  # Buscar mais para re-ranking
        )
        
        # 3. Filtrar por threshold
        filtered_results = [
            result for result in search_results 
            if result.similarity_score >= config.similarity_threshold
        ]
        
        logger.info(f"📋 {len(search_results)} → {len(filtered_results)} após threshold")
        
        # 4. Re-ranking inteligente
        if config.enable_reranking:
            filtered_results = self._rerank_results(filtered_results, query_analysis)
            logger.info("🔄 Re-ranking aplicado")
        
        # 5. Limitar ao top-k final
        final_results = filtered_results[:config.top_k]
        
        # 6. Converter para RetrievedDocuments
        retrieved_docs = self._create_retrieved_documents(final_results, query_analysis)
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Recuperação concluída: {len(retrieved_docs)} docs em {processing_time*1000:.1f}ms")
        
        return RetrievalResult(
            query=query,
            documents=retrieved_docs,
            total_found=len(search_results),
            processing_time=processing_time,
            config_used=config,
            query_analysis=query_analysis
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do retriever."""
        if not self.is_initialized:
            return {'status': 'not_initialized'}
        
        vs_stats = self.vector_store.get_statistics()
        
        return {
            'status': 'initialized',
            'vector_store': vs_stats,
            'config': {
                'top_k': self.config.top_k,
                'similarity_threshold': self.config.similarity_threshold,
                'enable_reranking': self.config.enable_reranking,
                'diversity_factor': self.config.diversity_factor,
                'book_diversity': self.config.book_diversity
            }
        }


def create_retriever(vector_store_path: str = "data/rag_vector_store", 
                    config: Optional[RetrievalConfig] = None) -> RAGRetriever:
    """Factory function para criar RAGRetriever."""
    return RAGRetriever(vector_store_path, config)


if __name__ == "__main__":
    # Teste do módulo
    print("🧪 Testando RAG Retriever...")
    
    try:
        # Criar retriever
        retriever = create_retriever()
        
        # Consultas de teste
        test_queries = [
            "What is machine learning?",
            "How does gradient descent work?",
            "Explain overfitting and regularization",
            "What are convolutional neural networks?",
            "Support vector machines algorithm"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔎 TESTE {i}: '{query}'")
            print("-" * 50)
            
            result = retriever.retrieve(query)
            
            print(f"📊 Análise: {result.query_analysis['query_type']}")
            print(f"⏱️  Tempo: {result.processing_time*1000:.1f}ms")
            print(f"📋 Documentos encontrados: {result.total_found}")
            print(f"📄 Documentos retornados: {len(result.documents)}")
            
            print(f"\n📚 Top documentos:")
            for doc in result.documents[:3]:
                print(f"   {doc.rank}. {doc.chunk_id}")
                print(f"      📖 Livro: {doc.source_book}")
                print(f"      🎯 Score: {doc.similarity_score:.3f}")
                print(f"      💭 Relevância: {doc.relevance_reason}")
                print(f"      📝 {len(doc.content.split())} palavras")
                content_preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
                print(f"      📄 Conteúdo: {content_preview}")
                print()
        
        print("🎉 Teste do RAG Retriever concluído com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()