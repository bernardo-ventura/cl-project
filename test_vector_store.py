"""
Script para testar o Vector Store
"""

import sys
from pathlib import Path

# Adicionar o diretório raiz do projeto ao Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.rag.vector_store import create_vector_store
from src.rag.document_processor import create_document_processor

def test_vector_store():
    """Testa a implementação do Vector Store."""
    print("🧪 Testando Vector Store...")
    
    try:
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
        print(f"   FAISS total: {stats['faiss_total']}")
        
        # Distribuição por livros
        print(f"\n📚 Distribuição por livros:")
        for book, count in stats['books_distribution'].items():
            print(f"   {book}: {count} chunks")
        
        # Teste de busca
        print(f"\n🔍 Testando busca por similaridade...")
        
        # Usar primeiro documento como query de teste
        test_query_embedding = documents[0].embedding
        results = vector_store.search(test_query_embedding, top_k=5)
        
        print(f"📋 Resultados da busca (top 5):")
        for result in results:
            print(f"   {result.rank}. {result.document.chunk_id} "
                  f"(score: {result.similarity_score:.3f})")
            print(f"      Livro: {result.document.source_book}")
            print(f"      Conteúdo: {result.document.content[:100]}...")
            print()
        
        # Salvar vector store
        print(f"💾 Salvando vector store...")
        vector_store.save("data/rag_vector_store")
        print(f"✅ Vector store salvo em data/rag_vector_store.*")
        
        # Testar carregamento
        print(f"\n📂 Testando carregamento...")
        vector_store2 = create_vector_store()
        vector_store2.load("data/rag_vector_store")
        
        stats2 = vector_store2.get_statistics()
        print(f"✅ Teste de carregamento: {stats2['total_documents']} documentos")
        
        # Teste de busca no vector store carregado
        print(f"🔍 Testando busca no vector store carregado...")
        results2 = vector_store2.search(test_query_embedding, top_k=3)
        print(f"📋 Resultados consistentes: {len(results2) == 3}")
        
        print(f"\n🎉 Teste do Vector Store concluído com sucesso!")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vector_store()
    sys.exit(0 if success else 1)