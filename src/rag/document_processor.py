"""
Document Processor: Carregamento e processamento de documentos para RAG
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys

# Adicionar src ao path para imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.knowledge_graph.chunk_loader import TextChunk, ChunkLoader

logger = logging.getLogger(__name__)

@dataclass
class ProcessedDocument:
    """Representa um documento processado com embedding."""
    chunk_id: str
    content: str
    source_book: str
    chunk_number: int
    word_count: int
    embedding: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário serializável."""
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'source_book': self.source_book,
            'chunk_number': self.chunk_number,
            'word_count': self.word_count,
            'embedding': self.embedding.tolist()  # Converter numpy para list
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessedDocument':
        """Cria instância a partir de dicionário."""
        return cls(
            chunk_id=data['chunk_id'],
            content=data['content'],
            source_book=data['source_book'],
            chunk_number=data['chunk_number'],
            word_count=data['word_count'],
            embedding=np.array(data['embedding'])  # Converter list para numpy
        )


class DocumentProcessor:
    """Processador de documentos para sistema RAG."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Inicializa o processador.
        
        Args:
            model_name: Nome do modelo sentence-transformers
        """
        self.model_name = model_name
        self.model = None
        self.processed_docs: List[ProcessedDocument] = []
        
        logger.info(f"📚 DocumentProcessor inicializado com modelo: {model_name}")
    
    def _load_model(self):
        """Carrega o modelo de embeddings (lazy loading)."""
        if self.model is None:
            logger.info(f"🤖 Carregando modelo de embeddings: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✅ Modelo carregado com sucesso")
    
    def load_chunks(self) -> List[TextChunk]:
        """Carrega chunks usando o mesmo sistema do KG."""
        logger.info("📂 Carregando chunks de texto...")
        
        chunk_loader = ChunkLoader()
        chunks = chunk_loader.load_all_chunks()
        
        logger.info(f"✅ {len(chunks)} chunks carregados")
        return chunks
    
    def process_chunks(self, chunks: List[TextChunk]) -> List[ProcessedDocument]:
        """
        Processa chunks criando embeddings.
        
        Args:
            chunks: Lista de TextChunk para processar
            
        Returns:
            Lista de ProcessedDocument com embeddings
        """
        self._load_model()
        
        logger.info(f"🔄 Processando {len(chunks)} chunks...")
        
        # Extrair textos para processamento em batch
        texts = [chunk.content for chunk in chunks]
        
        # Gerar embeddings em batch (mais eficiente)
        logger.info("🤖 Gerando embeddings...")
        embeddings = self.model.encode(
            texts, 
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Criar ProcessedDocuments
        processed_docs = []
        for chunk, embedding in zip(chunks, embeddings):
            processed_doc = ProcessedDocument(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                source_book=chunk.source_book,
                chunk_number=chunk.chunk_number,
                word_count=chunk.word_count,
                embedding=embedding
            )
            processed_docs.append(processed_doc)
        
        self.processed_docs = processed_docs
        logger.info(f"✅ {len(processed_docs)} documentos processados")
        
        return processed_docs
    
    def save_processed_docs(self, filepath: str):
        """
        Salva documentos processados em arquivo.
        
        Args:
            filepath: Caminho para salvar os dados
        """
        if not self.processed_docs:
            raise ValueError("Nenhum documento processado para salvar")
        
        # Converter para formato serializável
        serializable_data = {
            'model_name': self.model_name,
            'total_docs': len(self.processed_docs),
            'embedding_dim': self.processed_docs[0].embedding.shape[0],
            'documents': [doc.to_dict() for doc in self.processed_docs]
        }
        
        # Salvar usando pickle (mais eficiente para numpy arrays)
        with open(filepath, 'wb') as f:
            pickle.dump(serializable_data, f)
        
        file_size = Path(filepath).stat().st_size / (1024 * 1024)  # MB
        logger.info(f"💾 Documentos salvos em: {filepath}")
        logger.info(f"📊 Tamanho do arquivo: {file_size:.1f} MB")
    
    def load_processed_docs(self, filepath: str) -> List[ProcessedDocument]:
        """
        Carrega documentos processados de arquivo.
        
        Args:
            filepath: Caminho do arquivo com dados
            
        Returns:
            Lista de ProcessedDocument carregados
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Reconstruir ProcessedDocuments
        processed_docs = [
            ProcessedDocument.from_dict(doc_data) 
            for doc_data in data['documents']
        ]
        
        self.model_name = data['model_name']
        self.processed_docs = processed_docs
        
        logger.info(f"📂 {len(processed_docs)} documentos carregados de: {filepath}")
        logger.info(f"🤖 Modelo usado: {self.model_name}")
        logger.info(f"📏 Dimensão dos embeddings: {data['embedding_dim']}")
        
        return processed_docs
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas dos documentos processados."""
        if not self.processed_docs:
            return {}
        
        books = {}
        total_words = 0
        
        for doc in self.processed_docs:
            if doc.source_book not in books:
                books[doc.source_book] = {'count': 0, 'words': 0}
            
            books[doc.source_book]['count'] += 1
            books[doc.source_book]['words'] += doc.word_count
            total_words += doc.word_count
        
        return {
            'total_documents': len(self.processed_docs),
            'total_words': total_words,
            'avg_words_per_doc': total_words / len(self.processed_docs),
            'embedding_dimension': self.processed_docs[0].embedding.shape[0],
            'model_name': self.model_name,
            'books': books
        }


def create_document_processor(model_name: str = "all-MiniLM-L6-v2") -> DocumentProcessor:
    """Factory function para criar DocumentProcessor."""
    return DocumentProcessor(model_name)


if __name__ == "__main__":
    # Teste do módulo
    print("🧪 Testando Document Processor...")
    
    try:
        # Criar processor
        processor = create_document_processor()
        
        # Carregar chunks
        chunks = processor.load_chunks()
        print(f"📊 Chunks carregados: {len(chunks)}")
        
        # Processar alguns chunks como teste (primeiros 50)
        test_chunks = chunks[:50]
        print(f"🔬 Testando com {len(test_chunks)} chunks...")
        
        processed = processor.process_chunks(test_chunks)
        
        # Estatísticas
        stats = processor.get_statistics()
        print(f"\n📈 Estatísticas do teste:")
        print(f"   Total: {stats['total_documents']} documentos")
        print(f"   Palavras: {stats['total_words']:,}")
        print(f"   Dimensão embedding: {stats['embedding_dimension']}")
        print(f"   Modelo: {stats['model_name']}")
        
        # Salvar teste
        test_path = "data/test_rag_docs.pkl"
        processor.save_processed_docs(test_path)
        print(f"💾 Teste salvo em: {test_path}")
        
        # Testar carregamento
        processor2 = create_document_processor()
        loaded = processor2.load_processed_docs(test_path)
        print(f"✅ Teste de carregamento: {len(loaded)} documentos")
        
        print("\n🎉 Teste concluído com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()