"""
Script para comparação KG vs RAG com formato específico
"""

import sys
import time
sys.path.append('/home/beventura/UC/cl-project')
sys.path.append('/home/beventura/UC/cl-project/src')

from src.query_system.interactive_kg import CompletePipeline
from src.rag.rag_pipeline import RAGPipeline

def run_kg_question(pipeline, question):
    """Executa pergunta no sistema KG e extrai detalhes"""
    try:
        start_time = time.time()
        result = pipeline.answer_question(question)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Extrair resposta final
        if hasattr(result, 'natural_answer') and result.natural_answer:
            final_answer = result.natural_answer
        elif hasattr(result, 'final_answer') and result.final_answer:
            final_answer = result.final_answer
        else:
            final_answer = str(result)
        
        # Extrair entidades
        entities = []
        if hasattr(result, 'extracted_entities'):
            entities = result.extracted_entities
        entities_str = ", ".join(entities) if entities else "Nenhuma"
        
        # Extrair intenção
        intent = ""
        if hasattr(result, 'intent_result') and result.intent_result:
            if hasattr(result.intent_result, 'predicted_intent'):
                intent = result.intent_result.predicted_intent
            elif hasattr(result.intent_result, 'intent'):
                intent = result.intent_result.intent
            elif hasattr(result.intent_result, 'classification'):
                intent = result.intent_result.classification
            else:
                intent = str(result.intent_result)
        
        # Extrair predicado
        predicate = ""
        if hasattr(result, 'predicate_result') and result.predicate_result:
            if hasattr(result.predicate_result, 'selected_template'):
                predicate = result.predicate_result.selected_template
            elif hasattr(result.predicate_result, 'template'):
                predicate = result.predicate_result.template
            elif hasattr(result.predicate_result, 'predicates'):
                predicate = str(result.predicate_result.predicates)
            else:
                predicate = str(result.predicate_result)
        

        
        return {
            'answer': final_answer,
            'time': response_time,
            'entities': entities_str,
            'intent': intent,
            'predicate': predicate
        }
        
    except Exception as e:
        return {
            'answer': f"Erro: {str(e)}",
            'time': 0.0,
            'entities': "Erro",
            'intent': "Erro", 
            'predicate': "Erro"
        }

def run_rag_question(pipeline, question):
    """Executa pergunta no sistema RAG"""
    try:
        start_time = time.time()
        response = pipeline.query(question)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Verificar se obteve resposta válida e extrair fonte
        if hasattr(response, 'response'):
            response_text = response.response
        elif hasattr(response, 'answer'):
            response_text = response.answer
        else:
            response_text = str(response) if response else "Nenhuma resposta gerada"
        

        
        return {
            'answer': response_text,
            'time': response_time
        }
        
    except Exception as e:
        return {
            'answer': f"Erro: {str(e)}",
            'time': 0.0
        }

def main():
    # Lista de perguntas organizadas por dificuldade e intenção
    questions = [
        # DEFINIÇÃO - FÁCIL (3 perguntas)
        "What is machine learning?",
        "What is a neural network?",
        "Define supervised learning",
        
        # DEFINIÇÃO - MÉDIO (3 perguntas)
        "What is deep learning and how is it different from machine learning?",
        "Define overfitting and explain why it happens",
        "What are support vector machines?",
        
        # DEFINIÇÃO - DIFÍCIL (2 perguntas)
        "What is deep reinforcement learning and how does it differ from traditional reinforcement learning?",
        "Define ensemble learning and explain the theoretical foundation behind combining multiple models",
        
        # COMPARAÇÃO - FÁCIL (3 perguntas)
        "What is the difference between classification and regression?",
        "Compare supervised and unsupervised learning",
        "What are the differences between training and testing data?",
        
        # COMPARAÇÃO - MÉDIO (3 perguntas)
        "Compare decision trees and random forests",
        "What are the differences between gradient descent and stochastic gradient descent?",
        "Compare precision and recall metrics",
        
        # COMPARAÇÃO - DIFÍCIL (2 perguntas)
        "Compare the computational efficiency and accuracy trade-offs between random forests and gradient boosting machines",
        "What are the fundamental differences between convolutional neural networks and recurrent neural networks in terms of architecture and applications?",
        
        # APLICAÇÃO - FÁCIL (3 perguntas)
        "When would you use linear regression?",
        "In what situations is k-means clustering useful?",
        "When should you use cross-validation?",
        
        # APLICAÇÃO - MÉDIO (2 perguntas)
        "How would you handle missing data in a dataset?",
        "When would you choose a neural network over a decision tree?",
        
        # APLICAÇÃO - DIFÍCIL (2 perguntas)
        "How would you apply transfer learning to improve performance on a small dataset for medical image classification?",
        "In what scenarios would you choose to implement a transformer architecture over an LSTM for natural language processing tasks?",
        
        # RELAÇÃO - FÁCIL (2 perguntas)
        "How does the size of training data affect model performance?",
        "What is the relationship between model complexity and overfitting?",
        
        # RELAÇÃO - MÉDIO (2 perguntas)
        "How does learning rate affect neural network training?",
        "What is the relationship between feature selection and model accuracy?",
        
        # RELAÇÃO - DIFÍCIL (2 perguntas)
        "How does the choice of activation function relate to the vanishing gradient problem in deep neural networks?",
        "What is the relationship between bias-variance tradeoff and model complexity in machine learning algorithms?",
        
        # LISTA - FÁCIL (2 perguntas)
        "List three types of machine learning",
        "What are the main components of a neural network?",
        
        # LISTA - MÉDIO (1 pergunta)
        "List the key evaluation metrics for classification problems",
        
        # LISTA - DIFÍCIL (1 pergunta)
        "List and explain the key hyperparameters that need to be tuned when training a support vector machine",
        
        # PROCESSO - FÁCIL (2 perguntas)
        "How does k-means clustering work?",
        "Explain the basic steps of training a machine learning model",
        
        # PROCESSO - MÉDIO (1 pergunta)
        "Describe the process of feature selection in machine learning",
        
        # PROCESSO - DIFÍCIL (1 pergunta)
        "Describe the complete process of training a neural network from initialization to convergence, including key considerations at each step"
    ]
    
    print("🔧 Inicializando sistemas...")
    
    # Inicializar sistemas
    try:
        kg_pipeline = CompletePipeline()
        rag_pipeline = RAGPipeline()
        print("✅ Sistemas inicializados!")
    except Exception as e:
        print(f"❌ Erro ao inicializar: {e}")
        return
    
    # Executar comparação
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/37] Processando: {question[:60]}...")
        
        # Testar KG
        print("  🧠 Testando KG...")
        kg_result = run_kg_question(kg_pipeline, question)
        
        # Testar RAG
        print("  📚 Testando RAG...")
        rag_result = run_rag_question(rag_pipeline, question)
        
        results.append({
            'question': question,
            'kg': kg_result,
            'rag': rag_result
        })
        
        print(f"  ✅ KG: {kg_result['time']:.1f}s | RAG: {rag_result['time']:.1f}s")
    
    # Gerar documento
    print("\n📝 Gerando documento de comparação...")
    
    output_lines = []
    
    for i, result in enumerate(results, 1):
        output_lines.append(f"Pergunta {i}: \"{result['question']}\"\n")
        
        # KG
        kg = result['kg']
        output_lines.append("KG: \"" + kg['answer'] + "\"\n")
        output_lines.append(f"Tempo de resposta: {kg['time']:.2f}s")
        output_lines.append(f"Entidades reconhecidas: \"{kg['entities']}\"")
        output_lines.append(f"Intenção: \"{kg['intent']}\"")
        output_lines.append(f"Predicado: \"{kg['predicate']}\"\n")
        
        # RAG  
        rag = result['rag']
        output_lines.append("RAG: \"" + rag['answer'] + "\"\n")
        output_lines.append(f"Tempo de resposta: {rag['time']:.2f}s\n")
        
        output_lines.append("=" * 70 + "\n")
    
    # Salvar arquivo
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"/home/beventura/UC/cl-project/comparison/results/comparacao_detalhada_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"✅ Documento salvo em: {filename}")

if __name__ == "__main__":
    main()