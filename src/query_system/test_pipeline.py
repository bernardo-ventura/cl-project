"""
Testes do Pipeline Completo KG

Arquivo separado para testar o pipeline sem interferir na interface interativa.
Use este arquivo quando quiser validar funcionalidades ou fazer debugging.
"""

from .interactive_kg import create_complete_pipeline


def test_pipeline_basic():
    """Testa funcionalidades básicas do pipeline"""
    print("🤖 Simulação de Usuário Real - 25 Perguntas sobre ML/DL")
    print("=" * 70)
    
    try:
        # Inicializar pipeline
        pipeline = create_complete_pipeline()
        
        # Mostrar informações
        info = pipeline.get_pipeline_info()
        print(f"✅ Sistema carregado:")
        print(f"   • Knowledge Graph: {info.get('kg_triplas', 'unknown')} triplas")
        print(f"   • Pipeline: {info['steps']} steps implementados")
        print()
        
        # Casos de teste expandidos - 25 perguntas diversas
        test_questions = [
            # Definições básicas
            "O que é neural network?",
            "O que é machine learning?",
            "Como funciona gradient descent?",
            "O que é deep learning?",
            "Explique linear regression",
            
            # Comparações
            "Diferença entre SVM e Random Forest",
            "CNN vs RNN qual é melhor?",
            "Supervised vs unsupervised learning",
            "Logistic regression ou decision tree?",
            "Adam vs SGD optimizer",
            
            # Listagens e tipos
            "Quais são os tipos de machine learning?",
            "Liste algoritmos de clustering",
            "Tipos de redes neurais",
            "Algoritmos de classificação",
            "Métricas de avaliação em ML",
            
            # Aplicações práticas
            "Para que serve deep learning?",
            "Aplicações de computer vision",
            "Onde usar reinforcement learning?",
            "NLP applications in practice",
            "Transfer learning uses",
            
            # Conceitos avançados
            "Como funciona backpropagation?",
            "O que é overfitting?",
            "Regularização em machine learning",
            "Feature engineering techniques",
            "Cross validation methods"
        ]
        
        print("💬 Perguntas e Respostas do Sistema:")
        print("=" * 70)
        
        success_count = 0
        total_time = 0
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 PERGUNTA {i}: {question}")
            print("-" * 60)
            
            try:
                result = pipeline.answer_question(question)
                
                print(f"🤖 RESPOSTA:")
                print(f"{result.natural_answer}")
                
                print(f"\n📊 Detalhes técnicos:")
                print(f"   • Status: {result.pipeline_status}")
                print(f"   • Tempo: {result.total_time_ms}ms")
                print(f"   • Entidades: {result.extracted_entities}")
                print(f"   • Qualidade: {result.answer_quality} ({result.confidence})")
                
                if result.execution_result:
                    print(f"   • Resultados KG: {result.execution_result.results_count}")
                
                if result.pipeline_status == "success":
                    success_count += 1
                
                total_time += result.total_time_ms
                
                print("=" * 70)
                    
            except Exception as e:
                print(f"❌ Erro: {e}")
                print("=" * 70)
        
        print(f"\n🎯 RESUMO DA SIMULAÇÃO:")
        print(f"   ✅ Perguntas respondidas com sucesso: {success_count}/{len(test_questions)}")
        print(f"   📊 Taxa de sucesso: {(success_count/len(test_questions))*100:.1f}%")
        print(f"   ⏱️  Tempo total: {total_time:.1f}ms")
        print(f"   ⏱️  Tempo médio por pergunta: {total_time/len(test_questions):.1f}ms")
        
        if success_count == len(test_questions):
            print(f"   🎉 Todas as perguntas foram respondidas com sucesso!")
        elif success_count >= len(test_questions) * 0.8:
            print(f"   ✅ Excelente performance do sistema!")
        else:
            print(f"   ⚠️  Sistema precisa de melhorias")
        
        return success_count == len(test_questions)
        
    except Exception as e:
        print(f"❌ Erro nos testes: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_custom(questions):
    """Testa perguntas customizadas"""
    print("🧪 Testando Perguntas Customizadas")
    print("=" * 40)
    
    try:
        pipeline = create_complete_pipeline()
        
        for i, question in enumerate(questions, 1):
            print(f"\n{i}. \"{question}\"")
            result = pipeline.answer_question(question)
            print(f"   📝 Resposta: {result.natural_answer}")
            print(f"   📊 Status: {result.pipeline_status}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")


if __name__ == "__main__":
    # Executar testes básicos
    success = test_pipeline_basic()
    
    if success:
        print(f"\n✅ Pipeline está funcionando corretamente!")
    else:
        print(f"\n❌ Pipeline tem problemas que precisam ser resolvidos.")