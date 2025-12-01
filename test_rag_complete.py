"""
Teste simplificado do Sistema RAG
"""

import sys
from pathlib import Path

# Adicionar diretório do projeto ao path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.rag.rag_pipeline import create_rag_pipeline, RAGConfig

def test_rag_system():
    """Testa o sistema RAG completo."""
    print("🚀 TESTE DO SISTEMA RAG COMPLETO")
    print("=" * 50)
    
    # Criar configuração
    config = RAGConfig(
        top_k=3,
        response_style="comprehensive",
        save_history=True,
        include_sources=True
    )
    
    # Criar e inicializar pipeline
    print("📋 Criando pipeline RAG...")
    pipeline = create_rag_pipeline(config)
    
    # Consultas de teste
    test_queries = [
        "What is machine learning?",
        "How does gradient descent work?",
        "Explain neural networks"
    ]
    
    print(f"\n🔍 Executando {len(test_queries)} consultas de teste:\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"🔎 CONSULTA {i}: '{query}'")
        print("-" * 50)
        
        response = pipeline.query(query)
        
        print(f"⏱️  Tempo: {response.total_time:.2f}s")
        print(f"📊 Confiança: {response.confidence_score:.2f}")
        print(f"📄 Documentos: {response.documents_used}")
        print(f"🤖 Modelo: {response.model_used}")
        
        print(f"\n💬 RESPOSTA:")
        # Mostrar apenas os primeiros 200 caracteres para teste
        answer = response.answer
        if len(answer) > 200:
            answer = answer[:200] + "..."
        print(f"   {answer}")
        
        if response.sources:
            print(f"\n📚 Fontes: {len(response.sources)} livros")
        
        print("\n")
    
    # Estatísticas finais
    stats = pipeline.get_statistics()
    print("📊 ESTATÍSTICAS FINAIS:")
    print(f"   Consultas processadas: {stats['query_history']['total_queries']}")
    print(f"   Sistema funcionando: ✅")
    
    print("\n🎉 Teste do sistema RAG CONCLUÍDO com sucesso!")
    print("💡 Para usar interativamente, execute: python demo_rag_interactive.py")

if __name__ == "__main__":
    test_rag_system()