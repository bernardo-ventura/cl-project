"""
Vector Store: Armazenamento vetorial usando FAISS para busca de similaridade
"""

import logging
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import faiss
import time

try:
    from .document_processor import ProcessedDocument
except ImportError:
    # Para execução direta
    from document_processor import ProcessedDocument

logger = logging.getLogger(__name__)

@dataclass 
class SearchResult:
    """Representa um resultado de busca."""
    document: ProcessedDocument
    similarity_score: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'chunk_id': self.document.chunk_id,
            'content': self.document.content,
            'source_book': self.document.source_book,
            'similarity_score': float(self.similarity_score),
            'rank': self.rank
        }


class VectorStore:
    """Armazenamento vetorial com FAISS para busca de similaridade."""
    
    def __init__(self, embedding_dimension: int = 384):
        """
        Inicializa o Vector Store.
        
        Args:
            embedding_dimension: Dimensão dos embeddings (384 para all-MiniLM-L6-v2)
        """
        self.embedding_dimension = embedding_dimension
        self.index = None
        self.documents: List[ProcessedDocument] = []
        self.is_trained = False
        
        logger.info(f"🔍 VectorStore inicializado com dimensão: {embedding_dimension}")
    
    def _create_index(self) -> faiss.Index:
        """
        Cria índice FAISS.
        
        Returns:
            Índice FAISS configurado
        """
        # IndexFlatIP: Inner Product (similar a cosine para vetores normalizados)
        # Ideal para datasets pequenos-médios (< 100K), busca exata
        index = faiss.IndexFlatIP(self.embedding_dimension)
        
        logger.info(f"📊 Criado IndexFlatIP com {self.embedding_dimension} dimensões")
        return index
    
    def add_documents(self, documents: List[ProcessedDocument]):
        """
        Adiciona documentos ao vector store.
        
        Args:
            documents: Lista de documentos processados com embeddings
        """
        if not documents:
            raise ValueError("Lista de documentos vazia")
        
        logger.info(f"📥 Adicionando {len(documents)} documentos ao vector store...")
        
        # Criar índice se não existir
        if self.index is None:
            self.index = self._create_index()
        
        # Extrair embeddings como matriz numpy
        embeddings = np.array([doc.embedding for doc in documents], dtype=np.float32)
        
        # Adicionar ao índice FAISS
        self.index.add(embeddings)
        
        # Armazenar documentos para recuperação de metadados
        self.documents.extend(documents)
        
        self.is_trained = True
        
        logger.info(f"✅ {len(documents)} documentos adicionados")
        logger.info(f"📊 Total no index: {self.index.ntotal} documentos")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[SearchResult]:
        """
        Busca documentos similares.
        
        Args:
            query_embedding: Embedding da query
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de SearchResult ordenada por similaridade
        """
        if not self.is_trained:
            raise ValueError("Vector store não foi treinado. Adicione documentos primeiro.")
        
        if query_embedding.shape[0] != self.embedding_dimension:
            raise ValueError(f"Dimensão do embedding ({query_embedding.shape[0]}) "
                           f"não compatível ({self.embedding_dimension})")
        
        start_time = time.time()
        
        # Buscar no índice FAISS
        # query_embedding deve ser 2D para FAISS
        query_2d = query_embedding.reshape(1, -1).astype(np.float32)
        
        scores, indices = self.index.search(query_2d, top_k)
        
        search_time = time.time() - start_time
        
        # Converter resultados
        results = []
        for rank, (doc_idx, score) in enumerate(zip(indices[0], scores[0])):
            if doc_idx < len(self.documents):  # Verificar índice válido
                result = SearchResult(
                    document=self.documents[doc_idx],
                    similarity_score=float(score),
                    rank=rank + 1
                )
                results.append(result)
        
        logger.info(f"🔍 Busca concluída: {len(results)} resultados em {search_time*1000:.1f}ms")
        
        return results
    
    def search_by_text(self, query_text: str, encoder_model, top_k: int = 5) -> List[SearchResult]:
        """
        Busca por texto (conveniência - codifica o texto automaticamente).
        
        Args:
            query_text: Texto da consulta
            encoder_model: Modelo sentence-transformer para encoding
            top_k: Número de resultados
            
        Returns:
            Lista de SearchResult
        """
        # Codificar query
        query_embedding = encoder_model.encode([query_text], convert_to_numpy=True)[0]
        
        # Buscar
        return self.search(query_embedding, top_k)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do vector store."""
        if not self.is_trained:
            return {'status': 'not_trained'}
        
        # Estatísticas por livro
        books_stats = {}
        for doc in self.documents:
            book = doc.source_book
            if book not in books_stats:
                books_stats[book] = 0
            books_stats[book] += 1
        
        return {
            'status': 'trained',
            'total_documents': len(self.documents),
            'embedding_dimension': self.embedding_dimension,
            'index_type': type(self.index).__name__,
            'faiss_total': self.index.ntotal,
            'books_distribution': books_stats
        }
    
    def save(self, filepath: str):
        """
        Salva vector store em arquivo.
        
        Args:
            filepath: Caminho para salvar (sem extensão)
        """
        if not self.is_trained:
            raise ValueError("Vector store não treinado")
        
        filepath = Path(filepath)
        
        # Salvar índice FAISS
        faiss_path = filepath.with_suffix('.faiss')
        faiss.write_index(self.index, str(faiss_path))
        
        # Salvar metadados (documentos)
        metadata_path = filepath.with_suffix('.pkl')
        metadata = {
            'documents': [doc.to_dict() for doc in self.documents],
            'embedding_dimension': self.embedding_dimension,
            'is_trained': self.is_trained
        }
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Estatísticas do arquivo
        faiss_size = faiss_path.stat().st_size / (1024 * 1024)  # MB
        metadata_size = metadata_path.stat().st_size / (1024 * 1024)  # MB
        
        logger.info(f"💾 Vector store salvo:")
        logger.info(f"   📄 Índice FAISS: {faiss_path} ({faiss_size:.1f} MB)")
        logger.info(f"   📋 Metadados: {metadata_path} ({metadata_size:.1f} MB)")
    
    def load(self, filepath: str):
        """
        Carrega vector store de arquivo.
        
        Args:
            filepath: Caminho base (sem extensão)
        """
        filepath = Path(filepath)
        faiss_path = filepath.with_suffix('.faiss')
        metadata_path = filepath.with_suffix('.pkl')
        
        if not faiss_path.exists():
            raise FileNotFoundError(f"Índice FAISS não encontrado: {faiss_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadados não encontrados: {metadata_path}")
        
        # Carregar índice FAISS
        self.index = faiss.read_index(str(faiss_path))
        
        # Carregar metadados
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Reconstruir documentos
        self.documents = [
            ProcessedDocument.from_dict(doc_data) 
            for doc_data in metadata['documents']
        ]
        
        self.embedding_dimension = metadata['embedding_dimension']
        self.is_trained = metadata['is_trained']
        
        logger.info(f"📂 Vector store carregado:")
        logger.info(f"   📊 {len(self.documents)} documentos")
        logger.info(f"   📏 Dimensão: {self.embedding_dimension}")
        logger.info(f"   🔍 Tipo índice: {type(self.index).__name__}")


def create_vector_store(embedding_dimension: int = 384) -> VectorStore:
    """Factory function para criar VectorStore."""
    return VectorStore(embedding_dimension)


if __name__ == "__main__":
    # Teste do módulo
    print("🧪 Testando Vector Store...")
    
    try:
        # Importar document processor para carregar dados
        import sys
        from pathlib import Path
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        sys.path.append(str(project_root))
        
        from src.rag.document_processor import create_document_processor
        
        # Criar vector store
        vector_store = create_vector_store()
        
        # Carregar documentos processados
        print("📂 Carregando documentos processados...")
        processor = create_document_processor()
        documents = processor.load_processed_docs("data/rag_processed_documents.pkl")
        
        print(f"✅ {len(documents)} documentos carregados")
        
        # Adicionar ao vector store
        print("📥 Adicionando documentos ao vector store...")
        vector_store.add_documents(documents)
        
        # Estatísticas
        stats = vector_store.get_statistics()
        print(f"\n📊 Estatísticas do Vector Store:")
        print(f"   Total: {stats['total_documents']} documentos")
        print(f"   Dimensão: {stats['embedding_dimension']}")
        print(f"   Tipo índice: {stats['index_type']}")
        
        # Teste de busca
        print(f"\n🔍 Testando busca...")
        
        # Usar primeiro documento como query de teste
        test_query_embedding = documents[0].embedding
        results = vector_store.search(test_query_embedding, top_k=3)
        
        print(f"📋 Resultados da busca teste:")
        for result in results:
            print(f"   {result.rank}. {result.document.chunk_id} "
                  f"(score: {result.similarity_score:.3f})")
            print(f"      {result.document.content[:100]}...")
        
        # Salvar vector store
        print(f"\n💾 Salvando vector store...")
        vector_store.save("data/test_vector_store")
        
        # Testar carregamento
        print(f"📂 Testando carregamento...")
        vector_store2 = create_vector_store()
        vector_store2.load("data/test_vector_store")
        
        stats2 = vector_store2.get_statistics()
        print(f"✅ Teste de carregamento: {stats2['total_documents']} documentos")
        
        print(f"\n🎉 Teste concluído com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()