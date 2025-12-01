#!/usr/bin/env python3
"""
Script para processar TODOS os chunks e gerar embeddings para RAG
"""
import sys
import logging
from pathlib import Path

# Adicionar src ao path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.rag.document_processor import create_document_processor

def main():
    """Processa todos os chunks e gera embeddings."""
    
    logging.basicConfig(level=logging.INFO)
    print("🚀 PROCESSAMENTO DE DOCUMENTOS PARA RAG")
    print("=" * 50)
    
    try:
        # Criar processador
        print("🔧 Inicializando Document Processor...")
        processor = create_document_processor("all-MiniLM-L6-v2")
        
        # Carregar todos os chunks
        print("📂 Carregando chunks...")
        chunks = processor.load_chunks()
        print(f"✅ {len(chunks)} chunks carregados")
        
        # Processar TODOS os chunks
        print(f"⚙️ Processando {len(chunks)} chunks...")
        print("   Isso pode levar alguns minutos...")
        processed_docs = processor.process_chunks(chunks)
        
        # Salvar documentos processados
        output_path = "data/rag_processed_documents.pkl"
        print(f"💾 Salvando documentos processados...")
        processor.save_processed_docs(output_path)
        
        # Estatísticas finais
        stats = processor.get_statistics()
        print(f"\n📊 ESTATÍSTICAS FINAIS:")
        print(f"   📄 Total de documentos: {stats['total_documents']:,}")
        print(f"   📝 Total de palavras: {stats['total_words']:,}")
        print(f"   📏 Dimensão dos embeddings: {stats['embedding_dimension']}")
        print(f"   🤖 Modelo usado: {stats['model_name']}")
        print(f"   📈 Média palavras/doc: {stats['avg_words_per_doc']:.1f}")
        
        print(f"\n📚 Livros processados:")
        for book, book_stats in stats['books'].items():
            print(f"   • {book}: {book_stats['count']} chunks, {book_stats['words']:,} palavras")
        
        print(f"\n🎉 PROCESSAMENTO CONCLUÍDO!")
        print(f"📁 Dados salvos em: {output_path}")
        print(f"🚀 Pronto para Etapa 2: Vector Store!")
        
    except Exception as e:
        print(f"❌ Erro durante o processamento: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)