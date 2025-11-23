"""
Módulo para carregamento e gerenciamento dos chunks de texto.
Passo 2 do pipeline de construção do Knowledge Graph.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Representa um chunk de texto com metadados."""
    chunk_id: str
    content: str
    source_book: str
    chunk_number: int
    word_count: int
    
    def __str__(self):
        return f"Chunk {self.chunk_id}: {len(self.content)} chars, {self.word_count} words"

class ChunkLoader:
    """Carregador e gerenciador de chunks de texto."""
    
    def __init__(self, chunks_dir: str = None):
        """
        Inicializa o carregador de chunks.
        
        Args:
            chunks_dir: Diretório contendo os arquivos de chunks
        """
        if chunks_dir is None:
            # Caminho padrão relativo à raiz do projeto
            project_root = Path(__file__).parent.parent.parent
            chunks_dir = project_root / "data" / "processed_texts" / "chunks"
            
        self.chunks_dir = Path(chunks_dir)
        self.chunks: List[TextChunk] = []
        
        logger.info(f"Inicializando ChunkLoader com diretório: {self.chunks_dir}")
    
    def load_all_chunks(self) -> List[TextChunk]:
        """
        Carrega todos os chunks de todos os arquivos.
        
        Returns:
            Lista de objetos TextChunk com IDs únicos
        """
        logger.info("Iniciando carregamento de todos os chunks...")
        
        if not self.chunks_dir.exists():
            raise FileNotFoundError(f"Diretório de chunks não encontrado: {self.chunks_dir}")
        
        chunk_files = list(self.chunks_dir.glob("*_chunks.txt"))
        logger.info(f"Encontrados {len(chunk_files)} arquivos de chunks")
        
        total_chunks = 0
        
        for chunk_file in chunk_files:
            book_chunks = self._load_chunks_from_file(chunk_file)
            self.chunks.extend(book_chunks)
            total_chunks += len(book_chunks)
            logger.info(f"Carregados {len(book_chunks)} chunks de: {chunk_file.name}")
        
        logger.info(f"✅ Total de {total_chunks} chunks carregados de {len(chunk_files)} livros")
        return self.chunks
    
    def _load_chunks_from_file(self, file_path: Path) -> List[TextChunk]:
        """
        Carrega chunks de um arquivo específico.
        
        Args:
            file_path: Caminho para o arquivo de chunks
            
        Returns:
            Lista de TextChunk do arquivo
        """
        book_name = self._extract_book_name(file_path.name)
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Dividir por separador de chunks (formato: === CHUNK XXX ===)
            import re
            # Dividir usando regex para capturar o padrão === CHUNK XXX ===
            chunk_pattern = r'=== CHUNK \d+ ==='
            chunk_texts = re.split(chunk_pattern, content)
            
            for i, chunk_text in enumerate(chunk_texts):
                chunk_text = chunk_text.strip()
                if chunk_text and i > 0:  # Ignorar a primeira parte (antes do primeiro chunk)
                    # Remover metadados do início se existirem
                    lines = chunk_text.split('\n')
                    content_start = 0
                    for j, line in enumerate(lines):
                        if line.startswith('--------------------------------------------------'):
                            content_start = j + 1
                            break
                    
                    # Pegar apenas o conteúdo depois dos metadados
                    actual_content = '\n'.join(lines[content_start:]).strip()
                    
                    if actual_content:  # Verificar se há conteúdo real
                        chunk_id = f"{book_name}_chunk_{i:04d}"
                        word_count = len(actual_content.split())

                        chunk = TextChunk(
                            chunk_id=chunk_id,
                            content=actual_content,
                            source_book=book_name,
                            chunk_number=i,
                            word_count=word_count
                        )
                        chunks.append(chunk)
            
        except Exception as e:
            logger.error(f"Erro ao carregar arquivo {file_path}: {e}")
            raise
        
        return chunks
    
    def _extract_book_name(self, filename: str) -> str:
        """
        Extrai nome limpo do livro a partir do nome do arquivo.
        
        Args:
            filename: Nome do arquivo (ex: "Bishop-Pattern-Recognition-and-Machine-Learning-2006_chunks.txt")
            
        Returns:
            Nome limpo do livro (ex: "bishop_pattern_recognition")
        """
        # Remover sufixo _chunks.txt
        name = filename.replace('_chunks.txt', '')
        
        # Mapear nomes conhecidos para versões mais limpas
        name_mapping = {
            'Bishop-Pattern-Recognition-and-Machine-Learning-2006': 'bishop_pattern_recognition',
            'goodfellow2016deep_learning': 'goodfellow_deep_learning', 
            'deep-learning': 'deep_learning_book',
            'prince2023udl': 'prince_deep_learning',
            '698-machine-learning-the-art-and-science-of-algorithms-that-make-sense-of-data-(www.tawcer.com)': 'ml_art_science',
            'Introduction to Machine Learning with Python ( PDFDrive.com )-min': 'intro_ml_python',
            'Pattern Recognition - Concepts Methods and Applications - J. deSa (Springer, 2001) WW': 'pattern_recognition_concepts',
            'the-science-of-deep-learning-9781108835084-9781108891530_compress': 'science_deep_learning'
        }
        
        return name_mapping.get(name, name.lower().replace('-', '_').replace(' ', '_'))
    
    def get_chunks_by_book(self, book_name: str) -> List[TextChunk]:
        """
        Retorna chunks de um livro específico.
        
        Args:
            book_name: Nome do livro
            
        Returns:
            Lista de chunks do livro especificado
        """
        return [chunk for chunk in self.chunks if chunk.source_book == book_name]
    
    def get_chunk_statistics(self) -> Dict:
        """
        Retorna estatísticas sobre os chunks carregados.
        
        Returns:
            Dicionário com estatísticas
        """
        if not self.chunks:
            return {}
        
        total_chunks = len(self.chunks)
        total_words = sum(chunk.word_count for chunk in self.chunks)
        books = set(chunk.source_book for chunk in self.chunks)
        
        # Estatísticas por livro
        book_stats = {}
        for book in books:
            book_chunks = self.get_chunks_by_book(book)
            book_stats[book] = {
                'chunks': len(book_chunks),
                'words': sum(chunk.word_count for chunk in book_chunks),
                'avg_words_per_chunk': sum(chunk.word_count for chunk in book_chunks) / len(book_chunks) if book_chunks else 0
            }
        
        return {
            'total_chunks': total_chunks,
            'total_words': total_words,
            'total_books': len(books),
            'avg_words_per_chunk': total_words / total_chunks if total_chunks else 0,
            'books': sorted(books),
            'book_statistics': book_stats
        }
    
    def sample_chunks(self, n: int = 5) -> List[TextChunk]:
        """
        Retorna uma amostra aleatória de chunks para teste.
        
        Args:
            n: Número de chunks a amostrar
            
        Returns:
            Lista de chunks amostrados
        """
        import random
        return random.sample(self.chunks, min(n, len(self.chunks)))


# Função de conveniência para uso direto
def load_chunks(chunks_dir: str = None) -> Tuple[List[TextChunk], Dict]:
    """
    Função de conveniência para carregar chunks e obter estatísticas.
    
    Args:
        chunks_dir: Diretório de chunks (opcional)
        
    Returns:
        Tupla com (lista de chunks, estatísticas)
    """
    loader = ChunkLoader(chunks_dir)
    chunks = loader.load_all_chunks()
    stats = loader.get_chunk_statistics()
    
    return chunks, stats


if __name__ == "__main__":
    # Teste do módulo
    print("🔍 Testando carregamento de chunks...")
    
    try:
        chunks, stats = load_chunks()
        
        print(f"\n📊 Estatísticas:")
        print(f"Total de chunks: {stats['total_chunks']}")
        print(f"Total de palavras: {stats['total_words']:,}")
        print(f"Livros: {stats['total_books']}")
        print(f"Média de palavras por chunk: {stats['avg_words_per_chunk']:.1f}")
        
        print(f"\n📚 Livros processados:")
        for book, book_stats in stats['book_statistics'].items():
            print(f"  • {book}: {book_stats['chunks']} chunks, {book_stats['words']:,} palavras")
        
        print(f"\n📝 Amostras de chunks:")
        loader = ChunkLoader()
        loader.chunks = chunks
        for chunk in loader.sample_chunks(3):
            print(f"  • {chunk.chunk_id}: {chunk.content[:100]}...")
            
    except Exception as e:
        print(f"❌ Erro: {e}")