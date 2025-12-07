"""
Sistema RAG - Interface de Pergunta e Resposta
Faça perguntas sobre Machine Learning e Deep Learning
"""

import sys
from pathlib import Path

# Adicionar diretório do projeto ao path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.rag.rag_pipeline import create_rag_pipeline, RAGConfig


def main():
    """Sistema principal de perguntas e respostas."""
    
    # Cabeçalho do sistema
    print("=" * 70)
    print("SISTEMA RAG - MACHINE LEARNING & DEEP LEARNING")
    print("Recuperação e Geração Aumentada por Documentos")
    print("=" * 70)
    print()
    print("Base de conhecimento: 3,219 chunks de 8 livros de ML/DL")
    print("Vector Store: FAISS com embeddings all-MiniLM-L6-v2") 
    print("LLM: Ollama (llama3.2:3b)")
    print()
    
    # Inicializar sistema
    print("🚀 Inicializando sistema...")
    config = RAGConfig(
        top_k=5,
        response_style="comprehensive",
        include_sources=True,
        debug_mode=False
    )
    pipeline = create_rag_pipeline(config)
    print("✅ Sistema pronto!")
    print()
    
    # Loop principal
    print("💡 Faça uma pergunta sobre Machine Learning ou Deep Learning")
    print("💡 Digite 'sair' para encerrar")
    print()
    
    while True:
        try:
            # Receber pergunta
            pergunta = input("Sua pergunta: ").strip()
            
            if not pergunta:
                continue
                
            if pergunta.lower() in ['sair', 'quit', 'exit']:
                print("Encerrando sistema. Obrigado!")
                break
            
            print()
            print("Processando sua pergunta...")
            print()
            
            # Processar pergunta
            response = pipeline.query(pergunta)
            
            # Mostrar resposta
            print("💬 RESPOSTA:")
            print("=" * 60)
            print(response.answer)
            print("=" * 60)
            print()
            
            # Métricas básicas
            print(f"📊 Confiança: {response.confidence_score:.2f}")
            print(f"⏱️  Tempo: {response.total_time:.1f}s")
            print(f"📄 Documentos consultados: {response.documents_used}")
            
            # Fontes
            if response.sources:
                print(f"\n📚 Fontes consultadas:")
                for i, source in enumerate(response.sources[:3], 1):
                    book_name = source.split('(')[0].strip()
                    print(f"   {i}. {book_name}")
            
            print()
            print("-" * 70)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Encerrando sistema. Obrigado!")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")
            print("💡 Tente reformular sua pergunta.")
            print()


if __name__ == "__main__":
    main()