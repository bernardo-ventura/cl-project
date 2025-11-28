"""
Interactive Demo: Interface interativa para testar o sistema de consultas
"""

import os
import sys
import time
from pathlib import Path

# Adiciona o diretório src ao path para imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.query_system.kg_executor import create_kg_executor
from src.query_system.query_processor import create_query_processor
from src.query_system.response_formatter import create_response_formatter
from src.query_system.response_enhancer import create_response_enhancer


class InteractiveDemo:
    def __init__(self, kg_path=None):
        print("🚀 Inicializando Sistema de Consultas ao Knowledge Graph...")
        print("=" * 60)
        
        try:
            print("🔧 Carregando KG Executor...")
            self.kg_executor = create_kg_executor(kg_path)
            
            print("🧠 Carregando Query Processor...")
            self.query_processor = create_query_processor()
            
            print("🎨 Carregando Response Formatter...")
            self.response_formatter = create_response_formatter()
            
            print("🤖 Carregando Response Enhancer (LLM)...")
            self.response_enhancer = create_response_enhancer()
            
            stats = self.kg_executor.get_stats()
            print(f"✅ Sistema carregado com sucesso!")
            print(f"📊 Knowledge Graph: {stats.get('total_triples', 'N/A')} triplas")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Erro ao carregar sistema: {e}")
            raise
    
    def process_question(self, question, show_debug=False, use_natural_language=True):
        start_time = time.time()
        
        try:
            if show_debug:
                print(f"🔍 Processando pergunta: '{question}'")
            
            intent, sparql_query = self.query_processor.process_and_generate(question)
            
            if show_debug:
                print(f"🎯 Tipo identificado: {intent.query_type.value}")
                print(f"🏷️ Entidades: {intent.entities}")
                print(f"📝 SPARQL gerado ({len(sparql_query)} chars)")
            
            results = self.kg_executor.execute_sparql(sparql_query)
            
            if show_debug:
                print(f"📊 Resultados SPARQL: {len(results)} registros")
            
            formatted_response = self.response_formatter.format_response(
                results=results,
                query_type=intent.query_type,
                original_question=question,
                entities=intent.entities
            )
            
            # Nova funcionalidade: Enhancement com LLM
            if use_natural_language:
                try:
                    enhanced_response = self.response_enhancer.enhance_response(
                        formatted_response=formatted_response,
                        original_question=question,
                        query_type=intent.query_type.value
                    )
                    
                    main_answer = enhanced_response.natural_answer
                    llm_time = enhanced_response.processing_time
                    
                    if show_debug:
                        print(f"🤖 LLM Enhancement: {llm_time:.2f}s")
                    
                except Exception as e:
                    if show_debug:
                        print(f"⚠️ LLM Enhancement falhou: {e}")
                    # Fallback: usar resposta estruturada
                    main_answer = formatted_response.answer
                    llm_time = 0
            else:
                main_answer = formatted_response.answer
                llm_time = 0
            
            elapsed_time = time.time() - start_time
            
            response_parts = [
                main_answer,
                "",
                f"🔗 **Fonte**: Knowledge Graph ML/DL",
                f"⏱️ **Tempo**: {elapsed_time:.2f}s" + (f" (LLM: {llm_time:.2f}s)" if llm_time > 0 else ""),
                f"📊 **Resultados**: {formatted_response.metadata.get('result_count', 0)}",
                f"🎯 **Confiança**: {formatted_response.confidence:.1%}"
            ]
            
            if show_debug:
                response_parts.extend([
                    f"🔧 **Debug**: {intent.query_type.value}",
                    f"📝 **SPARQL**: {len(sparql_query)} caracteres"
                ])
                
                # Mostra dados estruturados também no debug
                if use_natural_language:
                    response_parts.extend([
                        "",
                        "📊 **Dados Estruturados Originais**:",
                        formatted_response.answer
                    ])
            
            return "\n".join(response_parts)
            
        except Exception as e:
            error_time = time.time() - start_time
            return f"❌ **Erro ao processar pergunta**: {str(e)}\n⏱️ **Tempo**: {error_time:.2f}s"
    
    def run_interactive_session(self):
        print("\n🎮 MODO INTERATIVO")
        print("Digite suas perguntas sobre Machine Learning e Deep Learning!")
        print("Comandos especiais:")
        print("  • 'help' - mostra exemplos de perguntas")
        print("  • 'debug on/off' - ativa/desativa modo debug")
        print("  • 'natural on/off' - liga/desliga respostas naturais (LLM)")
        print("  • 'stats' - estatísticas do Knowledge Graph") 
        print("  • 'quit' - sair")
        print("-" * 60)
        
        debug_mode = False
        natural_mode = True  # Default: usar LLM para respostas naturais
        print(f"🤖 Modo Natural Language: {'ATIVO' if natural_mode else 'INATIVO'}")
        print(f"🔧 Debug mode: {'ATIVO' if debug_mode else 'INATIVO'}")
        
        while True:
            try:
                question = input("\n🤔 Sua pergunta: ").strip()
                
                if not question:
                    continue
                    
                elif question.lower() == 'quit':
                    print("👋 Até logo!")
                    break
                    
                elif question.lower() == 'help':
                    self._show_help()
                    
                elif question.lower() == 'stats':
                    self._show_stats()
                    
                elif question.lower() == 'debug on':
                    debug_mode = True
                    print("🔧 Modo debug ATIVADO")
                    
                elif question.lower() == 'debug off':
                    debug_mode = False
                    print("🔧 Modo debug DESATIVADO")
                    
                elif question.lower() == 'natural on':
                    natural_mode = True
                    print("🤖 Modo Natural Language ATIVADO (usando LLM)")
                    
                elif question.lower() == 'natural off':
                    natural_mode = False
                    print("📊 Modo Natural Language DESATIVADO (respostas estruturadas)")
                    
                else:
                    print("\n🤖 **Resposta:**")
                    print("-" * 40)
                    
                    response = self.process_question(question, show_debug=debug_mode, use_natural_language=natural_mode)
                    print(response)
                    
            except KeyboardInterrupt:
                print("\n\n👋 Sessão interrompida. Até logo!")
                break
            except Exception as e:
                print(f"\n❌ Erro inesperado: {e}")
    
    def run_demo_questions(self):
        demo_questions = [
            "O que é gradient descent?",
            "Quais algoritmos usam backpropagation?",
            "Adam optimizer é um tipo de que?", 
            "Quem criou Support Vector Machine?",
            "Liste todos os algoritmos",
            "Como neural network está relacionado com deep learning?",
            "Encontre conceitos similares a CNN",
            "O que é overfitting?"
        ]
        
        print("\n🎬 DEMONSTRAÇÃO - Perguntas Exemplo")
        print("=" * 50)
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n📝 **Pergunta {i}**: {question}")
            print("-" * 30)
            
            response = self.process_question(question)
            print(response)
            
            input("\n⏎ Pressione Enter para continuar...")
    
    def _show_help(self):
        examples = [
            "📋 **Exemplos de perguntas que você pode fazer:**",
            "",
            "🔍 **Definições:**",
            "   • O que é gradient descent?",
            "   • Defina backpropagation",
            "   • Explique neural network",
            "",
            "🔧 **Uso e implementação:**", 
            "   • Quais algoritmos usam gradient descent?",
            "   • O que implementa backpropagation?",
            "   • Algoritmos que usam CNN",
            "",
            "🏷️ **Hierarquia e tipos:**",
            "   • Adam optimizer é um tipo de que?",
            "   • CNN é uma subclasse de que?",
            "",
            "👤 **Criadores e autores:**",
            "   • Quem criou Support Vector Machine?",
            "   • Quem desenvolveu LSTM?",
            "",
            "📊 **Listagens:**",
            "   • Liste todos os algoritmos",
            "   • Quais são as métricas?",
            "   • Mostre todos os conceitos",
            "",
            "🔗 **Relações:**",
            "   • Como CNN está relacionado com deep learning?",
            "   • Relação entre RNN e LSTM",
            "",
            "🔍 **Similaridade:**",
            "   • Encontre conceitos similares a CNN",
            "   • Algoritmos parecidos com SVM"
        ]
        
        print("\n" + "\n".join(examples))
    
    def _show_stats(self):
        try:
            stats = self.kg_executor.get_stats()
            
            print("\n📊 **Estatísticas do Knowledge Graph:**")
            print("-" * 35)
            print(f"🕸️ Total de triplas: {stats.get('total_triples', 'N/A'):,}")
            print(f"🏷️ Total de entidades: {stats.get('total_entities', 'N/A'):,}")
            print(f"🔗 Total de relações: {stats.get('total_relations', 'N/A'):,}")
            
        except Exception as e:
            print(f"❌ Erro ao obter estatísticas: {e}")


def main():
    print("🧠 SISTEMA DE CONSULTAS - KNOWLEDGE GRAPH ML/DL")
    print("=" * 60)
    
    try:
        demo = InteractiveDemo()
        
        print("\nEscolha uma opção:")
        print("1. Modo Interativo (você faz as perguntas)")
        print("2. Demonstração (perguntas exemplo)")
        print("3. Sair")
        
        while True:
            choice = input("\n👉 Sua escolha (1/2/3): ").strip()
            
            if choice == '1':
                demo.run_interactive_session()
                break
            elif choice == '2':
                demo.run_demo_questions()
                break
            elif choice == '3':
                print("👋 Até logo!")
                break
            else:
                print("❌ Opção inválida. Digite 1, 2 ou 3.")
    
    except Exception as e:
        print(f"❌ Erro fatal: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)