"""
Demonstração de Busca Semântica no RAG
Teste de consultas de texto para encontrar documentos relevantes
"""

import sys
from pathlib import Path

# Adicionar o diretório raiz do projeto ao Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.rag.vector_store import create_vector_store
from src.rag.document_processor import create_document_processor

def demo_semantic_search():
    """Demonstra busca semântica com diferentes consultas."""
    print("🔍 DEMONSTRAÇÃO DE BUSCA SEMÂNTICA RAG")
    print("=" * 50)
    
    try:
        # Carregar vector store
        print("📂 Carregando Vector Store...")
        vector_store = create_vector_store()
        vector_store.load("data/rag_vector_store")
        
        # Carregar processador para encoding de consultas
        print("🤖 Carregando modelo de embeddings...")
        processor = create_document_processor()
        processor._load_model()  # Garantir que modelo está inicializado
        model = processor.model
        
        # Estatísticas
        stats = vector_store.get_statistics()
        print(f"✅ Vector Store carregado: {stats['total_documents']} documentos")
        print(f"📚 Livros: {len(stats['books_distribution'])}")
        print()
        
        # Consultas de teste
        queries = [
            "What is machine learning?",
            "deep neural networks and backpropagation",
            "supervised learning algorithms",
            "overfitting and regularization techniques",
            "support vector machines",
            "clustering and unsupervised learning",
            "feature selection and dimensionality reduction",
            "cross-validation and model evaluation"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"🔎 CONSULTA {i}: '{query}'")
            print("-" * 40)
            
            # Buscar documentos relevantes
            results = vector_store.search_by_text(query, model, top_k=3)
            
            print(f"📋 Top 3 resultados mais relevantes:")
            for result in results:
                print(f"\n   {result.rank}. {result.document.chunk_id}")
                print(f"      📖 Livro: {result.document.source_book}")
                print(f"      🎯 Similaridade: {result.similarity_score:.3f}")
                
                # Mostrar snippet do conteúdo (primeiras 200 chars)
                content = result.document.content.strip()
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"      📄 Conteúdo: {content}")
            
            print()
        
        print("🎉 Demonstração concluída com sucesso!")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na demonstração: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_semantic_search()
    sys.exit(0 if success else 1)