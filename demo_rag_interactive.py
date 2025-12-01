"""
Demo Interativo do Sistema RAG
Interface CLI similar ao sistema KG para consultas em tempo real
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Adicionar diretório do projeto ao path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.rag.rag_pipeline import create_rag_pipeline, RAGConfig


class RAGInteractiveDemo:
    """Interface interativa para o sistema RAG."""
    
    def __init__(self):
        """Inicializa a demo."""
        self.pipeline = None
        self.config = RAGConfig(
            top_k=5,
            response_style="comprehensive",
            debug_mode=False,
            save_history=True,
            include_sources=True,
            citation_style="bracket"
        )
        self.is_initialized = False
        self.stats = {
            'queries_processed': 0,
            'total_time': 0.0,
            'avg_confidence': 0.0
        }
    
    def initialize(self):
        """Inicializa o pipeline RAG."""
        if self.is_initialized:
            return
        
        print("🚀 Inicializando sistema RAG...")
        print("⏳ Carregando componentes (isso pode demorar na primeira vez)...")
        
        start_time = time.time()
        self.pipeline = create_rag_pipeline(self.config)
        self.pipeline.initialize()
        init_time = time.time() - start_time
        
        self.is_initialized = True
        print(f"✅ Sistema RAG inicializado em {init_time:.1f}s")
        print()
    
    def print_header(self):
        """Imprime cabeçalho da aplicação."""
        print("🤖 " + "=" * 70)
        print("🤖 SISTEMA RAG - MACHINE LEARNING & DEEP LEARNING")
        print("🤖 Recuperação e Geração Aumentada por Documentos")
        print("🤖 " + "=" * 70)
        print()
        print("📚 Base de conhecimento: 3,219 chunks de 8 livros de ML/DL")
        print("🔍 Vector Store: FAISS com embeddings all-MiniLM-L6-v2") 
        print("🤖 LLM: Ollama (llama3.2:3b)")
        print()
    
    def print_help(self):
        """Imprime ajuda dos comandos."""
        print("📋 COMANDOS DISPONÍVEIS:")
        print("  help          - Mostra esta ajuda")
        print("  stats         - Estatísticas do sistema")
        print("  config        - Mostra/altera configurações")
        print("  debug on/off  - Liga/desliga modo debug")
        print("  style <style> - Altera estilo (comprehensive/concise/technical)")
        print("  topk <num>    - Define número de documentos (1-10)")
        print("  history       - Mostra histórico de consultas")
        print("  clear         - Limpa histórico")
        print("  demo          - Executa demonstração automática")
        print("  quit/exit     - Sair do sistema")
        print()
        print("💡 Exemplos de consultas:")
        print("  • What is machine learning?")
        print("  • How does gradient descent work?")
        print("  • Explain neural networks and backpropagation")
        print("  • What is the difference between supervised and unsupervised learning?")
        print("  • How do convolutional neural networks work?")
        print()
    
    def print_stats(self):
        """Imprime estatísticas do sistema."""
        if not self.is_initialized:
            print("⚠️ Sistema não inicializado")
            return
        
        system_stats = self.pipeline.get_statistics()
        
        print("📊 ESTATÍSTICAS DO SISTEMA RAG:")
        print(f"   Status: {'✅ Inicializado' if system_stats['is_initialized'] else '❌ Não inicializado'}")
        print()
        
        # Stats do retriever
        if 'retriever' in system_stats:
            ret_stats = system_stats['retriever']
            print(f"🔍 RETRIEVER:")
            print(f"   Documentos indexados: {ret_stats.get('vector_store', {}).get('total_documents', 'N/A')}")
            print(f"   Dimensão embeddings: {ret_stats.get('vector_store', {}).get('embedding_dimension', 'N/A')}")
            print(f"   Tipo de índice: {ret_stats.get('vector_store', {}).get('index_type', 'N/A')}")
            print()
        
        # Stats do generator
        if 'generator' in system_stats:
            gen_stats = system_stats['generator']
            print(f"🤖 GENERATOR:")
            print(f"   Ollama disponível: {'✅' if gen_stats.get('ollama_available', False) else '❌'}")
            print(f"   Modelo: {gen_stats.get('config', {}).get('model_name', 'N/A')}")
            print(f"   Temperatura: {gen_stats.get('config', {}).get('temperature', 'N/A')}")
            print()
        
        # Stats da sessão
        print(f"📈 SESSÃO ATUAL:")
        print(f"   Consultas processadas: {self.stats['queries_processed']}")
        if self.stats['queries_processed'] > 0:
            print(f"   Tempo médio: {self.stats['total_time']/self.stats['queries_processed']:.2f}s")
            print(f"   Confiança média: {self.stats['avg_confidence']/self.stats['queries_processed']:.2f}")
        print()
    
    def print_config(self):
        """Imprime configurações atuais."""
        print("⚙️ CONFIGURAÇÕES ATUAIS:")
        print(f"   Top-K documentos: {self.config.top_k}")
        print(f"   Estilo de resposta: {self.config.response_style}")
        print(f"   Threshold similaridade: {self.config.similarity_threshold}")
        print(f"   Modo debug: {'✅' if self.config.debug_mode else '❌'}")
        print(f"   Re-ranking: {'✅' if self.config.enable_reranking else '❌'}")
        print(f"   Diversidade de livros: {'✅' if self.config.book_diversity else '❌'}")
        print(f"   Incluir fontes: {'✅' if self.config.include_sources else '❌'}")
        print(f"   Temperatura LLM: {self.config.temperature}")
        print()
    
    def run_demo(self):
        """Executa demonstração automática."""
        demo_queries = [
            "What is machine learning?",
            "How does gradient descent work?",
            "Explain overfitting and regularization",
            "What are support vector machines?",
            "How do neural networks learn?"
        ]
        
        print("🎪 DEMONSTRAÇÃO AUTOMÁTICA")
        print(f"Executando {len(demo_queries)} consultas de exemplo...")
        print()
        
        for i, query in enumerate(demo_queries, 1):
            print(f"🔎 Demo {i}/{len(demo_queries)}: '{query}'")
            print("-" * 60)
            
            response = self.pipeline.query(query)
            
            print(f"⏱️  Tempo: {response.total_time:.2f}s")
            print(f"📊 Confiança: {response.confidence_score:.2f}")
            print(f"📄 Documentos: {response.documents_used}")
            print()
            print("💬 RESPOSTA:")
            print(response.answer[:300] + "..." if len(response.answer) > 300 else response.answer)
            print()
            
            if response.sources:
                print(f"📚 Fontes: {len(response.sources)} livros referenciados")
            print()
        
        print("🎉 Demonstração concluída!")
        print()
    
    def process_command(self, user_input: str) -> bool:
        """
        Processa comandos especiais.
        
        Returns:
            True se foi um comando especial, False se é consulta normal
        """
        command = user_input.lower().strip()
        
        if command in ['help', 'ajuda']:
            self.print_help()
            return True
        
        elif command == 'stats':
            self.print_stats()
            return True
        
        elif command == 'config':
            self.print_config()
            return True
        
        elif command.startswith('debug '):
            mode = command.split()[1]
            if mode == 'on':
                self.config.debug_mode = True
                print("🔍 Modo debug ATIVADO")
            elif mode == 'off':
                self.config.debug_mode = False
                print("🔍 Modo debug DESATIVADO")
            else:
                print("⚠️ Use: debug on/off")
            return True
        
        elif command.startswith('style '):
            style = command.split()[1]
            if style in ['comprehensive', 'concise', 'technical']:
                self.config.response_style = style
                print(f"📝 Estilo alterado para: {style}")
                if self.is_initialized:
                    print("ℹ️ Reinicialização necessária para aplicar mudança")
            else:
                print("⚠️ Estilos disponíveis: comprehensive, concise, technical")
            return True
        
        elif command.startswith('topk '):
            try:
                k = int(command.split()[1])
                if 1 <= k <= 10:
                    self.config.top_k = k
                    print(f"📊 Top-K alterado para: {k}")
                    if self.is_initialized:
                        print("ℹ️ Reinicialização necessária para aplicar mudança")
                else:
                    print("⚠️ Top-K deve estar entre 1 e 10")
            except ValueError:
                print("⚠️ Número inválido")
            return True
        
        elif command == 'history':
            if self.pipeline and hasattr(self.pipeline, 'query_history'):
                history = self.pipeline.query_history
                if history:
                    print(f"📋 HISTÓRICO ({len(history)} consultas):")
                    for entry in history[-5:]:  # Últimas 5
                        query = entry.get('query', 'N/A')
                        summary = entry.get('response_summary', {})
                        conf = summary.get('confidence', 0)
                        time_val = summary.get('total_time', 0)
                        print(f"   • {query[:50]}... (conf: {conf:.2f}, {time_val:.1f}s)")
                else:
                    print("📭 Histórico vazio")
            print()
            return True
        
        elif command == 'clear':
            if self.pipeline:
                self.pipeline.clear_history()
                self.stats = {'queries_processed': 0, 'total_time': 0.0, 'avg_confidence': 0.0}
            print("🗑️ Histórico limpo")
            return True
        
        elif command == 'demo':
            if not self.is_initialized:
                self.initialize()
            self.run_demo()
            return True
        
        elif command in ['quit', 'exit', 'sair']:
            return 'quit'
        
        return False
    
    def process_query(self, query: str):
        """Processa uma consulta normal."""
        if not self.is_initialized:
            self.initialize()
        
        print(f"🔍 Processando: '{query}'")
        print()
        
        start_time = time.time()
        response = self.pipeline.query(query)
        
        # Atualizar stats
        self.stats['queries_processed'] += 1
        self.stats['total_time'] += response.total_time
        self.stats['avg_confidence'] += response.confidence_score
        
        # Mostrar resposta
        print("💬 RESPOSTA:")
        print("=" * 60)
        print(response.answer)
        print("=" * 60)
        print()
        
        # Mostrar métricas
        print(f"⏱️  Tempo: {response.total_time:.2f}s")
        print(f"   Recuperação: {response.retrieval_time:.3f}s")
        print(f"   Geração: {response.generation_time:.2f}s")
        print(f"📊 Confiança: {response.confidence_score:.2f}")
        print(f"📄 Documentos: {response.documents_used}/{response.documents_found}")
        print(f"🤖 Modelo: {response.model_used}")
        
        # Mostrar fontes
        if response.sources:
            print(f"\n📚 FONTES ({len(response.sources)}):")
            for i, source in enumerate(response.sources, 1):
                print(f"   {i}. {source}")
        
        # Debug info
        if self.config.debug_mode and response.retrieval_debug:
            print(f"\n🔍 DEBUG INFO:")
            print(f"   Query analysis: {response.retrieval_debug.get('query_analysis', {})}")
            print(f"   Context length: {response.generation_debug.get('context_length', 'N/A')} chars")
        
        print()
    
    def run(self):
        """Executa a interface interativa."""
        self.print_header()
        
        print("💡 Digite 'help' para ver comandos disponíveis")
        print("💡 Digite 'demo' para ver demonstração automática")
        print("💡 Digite 'quit' para sair")
        print()
        
        while True:
            try:
                user_input = input("RAG> ").strip()
                
                if not user_input:
                    continue
                
                # Processar comandos especiais
                command_result = self.process_command(user_input)
                
                if command_result == 'quit':
                    print("👋 Encerrando sistema RAG...")
                    if self.pipeline:
                        self.pipeline.save_history_to_file("rag_session_history.json")
                        print("💾 Histórico salvo em rag_session_history.json")
                    print("🎯 Obrigado por usar o sistema RAG!")
                    break
                elif command_result:
                    continue  # Foi comando especial
                
                # Processar consulta normal
                self.process_query(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Encerrando sistema RAG...")
                break
            except Exception as e:
                print(f"❌ Erro: {e}")
                print("💡 Digite 'help' para ver comandos disponíveis")


def main():
    """Função principal."""
    demo = RAGInteractiveDemo()
    demo.run()


if __name__ == "__main__":
    main()