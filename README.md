# Projeto: Comparação Knowledge Graphs vs RAG

## Objetivo
Este projeto tem como objetivo **comparar e analisar duas abordagens complementares para representação e recuperação de conhecimento**:
1. **Knowledge Graphs (KG)** — representação estruturada e simbólica do conhecimento
2. **Retrieval-Augmented Generation (RAG)** — arquitetura neural de recuperação que fundamenta respostas de LLM em documentos externos

O domínio de estudo é **Machine Learning e Deep Learning**, escolhido por conter entidades bem definidas, relações hierárquicas e recursos textuais abundantes.

O objetivo final é entender como cada paradigma representa, recupera e raciocina sobre conhecimento, e como uma **abordagem híbrida (KG + RAG)** pode melhorar a factualidade, coerência e interpretabilidade em sistemas de IA baseados em conhecimento.

## Estrutura do Projeto Refatorada

```
Cl-project/
├── README.md                    # Este arquivo
├── requirements.txt             # Dependências Python
├── src/
│   ├── preprocessing/          # ✅ Módulo de preprocessamento (COMPLETO)
│   │   ├── extract_sample.py   # Extração de amostras (teste)
│   │   ├── extract_full.py     # Extração completa ✅
│   │   └── chunking.py         # Divisão em chunks ✅
│   ├── knowledge_graph/        # ✅ Módulo KG (COMPLETO)
│   │   ├── chunk_loader.py     # Carregamento de chunks ✅
│   │   ├── entity_extractor.py # Extração de entidades ✅
│   │   ├── entity_normalizer.py # Normalização LLM ✅
│   │   ├── relation_extractor.py # Extração de relações ✅
│   │   ├── kg_constructor.py   # Construção RDF ✅
│   │   ├── entities.py         # Classes de entidades ✅
│   │   ├── relations.py        # Classes de relações ✅
│   │   └── execute_*.py        # Scripts de execução ✅
│   ├── query_system/           # ✅ Sistema de Consultas KG (COMPLETO)
│   │   ├── kg_executor.py      # Executor SPARQL ✅
│   │   ├── query_processor.py  # Processador NL→SPARQL ✅
│   │   ├── query_templates.py  # Templates SPARQL ✅
│   │   ├── response_formatter.py # Formatador de respostas ✅
│   │   ├── response_enhancer.py # Enhancement LLM ✅
│   │   └── interactive_demo.py # Interface interativa ✅
│   ├── rag/                    # ✅ Sistema RAG (CONCLUÍDO)
   │   ├── __init__.py         # Módulo RAG ✅
   │   ├── document_processor.py # Geração de embeddings ✅
   │   ├── vector_store.py     # FAISS Vector Store ✅
   │   ├── retriever.py        # Recuperação inteligente ✅
   │   ├── response_generator.py # LLM Response Generation ✅
   │   └── rag_pipeline.py     # Pipeline completo ✅
│   └── experiments/            # Experimentos comparativos (Fase 4)
├── data/
│   ├── raw_pdfs/              # 8 PDFs originais
│   ├── samples/               # Amostras de teste
│   ├── processed_texts/       # Textos completos ✅
│   │   └── chunks/            # Chunks divididos ✅
│   ├── *.pkl                  # Dados intermediários ✅
│   ├── ml_kg.turtle           # Knowledge Graph principal ✅
│   ├── ml_kg.xml              # KG formato XML ✅
│   ├── ml_kg.json-ld          # KG formato JSON-LD ✅
│   └── kg_construction_report.txt # Relatório final ✅
```

## Fases do Projeto

### Fase 1: Preprocessamento
- ✅ **Extração**: PyPDF2 (padrão) + PyMuPDF (fallback) - 8/8 PDFs extraídos com sucesso (extract_full.py)
- ✅ **Chunking**: Divisão inteligente por sentenças (~350 palavras) - **Concluído** (chunking.py)
- ⏸️ **Limpeza**: Normalização de texto e caracteres - Planejado (Não é essencial, pular por enquanto)

### Fase 2: Knowledge Graphs ✅ CONCLUÍDO
**Pipeline Híbrido (spaCy + Ollama LLM) - FINALIZADO**

**📋 Pipeline de Construção do Knowledge Graph:**

**0. ✅ Configuração do LLM Local (Ollama)**
   - ✅ Instalação e configuração do Ollama no WSL Ubuntu
   - ✅ Download do modelo Llama 3.2 3B (2.0 GB)
   - ✅ Teste de integração Python com biblioteca ollama

**1. 🔧 Inicialização do Ambiente**
   - ✅ spaCy instalado com modelo en_core_web_sm
   - ✅ RDFLib para construção do grafo
   - ✅ Ambiente virtual configurado

**2. ✅ Carregamento dos Chunks de Texto**
   - ✅ Ler todos os arquivos de chunks de `data/processed_texts/chunks/`
   - ✅ Armazenar em lista com IDs únicos para rastreabilidade
   - ✅ **3.219 chunks carregados** de 8 livros (~1.2M palavras)

**3. ✅ Extração de Entidades (spaCy)**
   - ✅ Executar Named Entity Recognition (NER) em cada chunk
   - ✅ Usar padrões customizados (Matcher) para termos específicos de ML/DL
   - ✅ Coletar candidatos a entidades brutas por chunk
   - ✅ **Resultado**: 3.219 chunks → 44.183 entidades (16.325 únicas)

**4. ✅ Normalização de Entidades (LLM)**
   - ✅ Enviar lotes de candidatos a entidades para o LLM
   - ✅ Deduplicar, normalizar e unificar formato/capitalização  
   - ✅ Classificar cada entidade (algoritmo, modelo, conceito, técnica, métrica, etc.)
   - ✅ **Resultado**: 44.183 → 5.993 entidades (86.4% redução, 795 calls LLM)

**5. ✅ Extração de Relações (LLM)**
   - ✅ Para cada chunk: passar texto + lista de entidades canônicas
   - ✅ Extrair relações entre entidades usando esquema controlado
   - ✅ Esquema: is_a, part_of, uses, implements, optimizes, depends_on, etc.
   - ✅ **Concluído**: 2.747 chunks → **3.056 relações extraídas**

**6. ✅ Construção do Knowledge Graph (RDF)**
   - ✅ Criar grafo RDF usando rdflib
   - ✅ Criar namespace para conceitos de ML
   - ✅ Converter entidades em nós
   - ✅ Converter relações em triplas RDF
   - ✅ Serializar para múltiplos formatos

**📤 Saída Final:**
- ✅ `ml_kg.turtle` (2.5MB) - Knowledge Graph em formato Turtle
- ✅ `ml_kg.xml` (5.2MB) - Formato XML para compatibilidade
- ✅ `ml_kg.json-ld` - JSON-LD para web semântica
- ✅ `kg_construction_report.txt` - Relatório detalhado de estatísticas
- ✅ **64.124 triplas RDF, 5.993 entidades, 3.056 relações**

### Fase 3: Sistema de Consultas KG ✅ CONCLUÍDO
**Sistema Inteligente de Consultas com Enhancement LLM**

**🎯 Componentes do Sistema:**

**1. ✅ KG Executor** 
   - ✅ Carregamento e execução SPARQL no Knowledge Graph
   - ✅ Otimização para 64K+ triplas
   - ✅ Namespace management e estatísticas

**2. ✅ Query Processor**
   - ✅ Conversão linguagem natural → SPARQL
   - ✅ 7 tipos de consultas suportadas:
     - `what_is` - "O que é gradient descent?"
     - `what_uses` - "Quais algoritmos usam backpropagation?"
     - `who_created` - "Quem criou Support Vector Machine?"
     - `list_by_type` - "Liste todos os algoritmos"
     - `how_related` - "Como CNN está relacionado com deep learning?"
     - `find_similar` - "Encontre conceitos similares a CNN"
     - `is_type_of` - "Adam optimizer é um tipo de que?"

**3. ✅ Query Templates**
   - ✅ 7 templates SPARQL otimizados
   - ✅ Sintaxe UNION corrigida
   - ✅ Estrutura flexível para diferentes tipos de consultas

**4. ✅ Response Formatter**
   - ✅ Formatação estruturada dos resultados SPARQL
   - ✅ Cálculo de confiança baseado em resultados
   - ✅ Limpeza de URIs e metadados

**5. ✅ Response Enhancer (LLM)**
   - ✅ **Enhancement com Ollama (llama3.2:3b)**
   - ✅ Conversão de respostas estruturadas → **respostas naturais conversacionais**
   - ✅ Prompts contextualizados por tipo de consulta
   - ✅ Fallback automático se LLM falhar
   - ✅ Medição de tempo de processamento LLM

**6. ✅ Interactive Demo**
   - ✅ Interface CLI completa e intuitiva
   - ✅ Modo interativo + demonstração
   - ✅ Toggle `natural on/off` - liga/desliga LLM
   - ✅ Toggle `debug on/off` - modo debug detalhado
   - ✅ Comandos `help`, `stats`, `quit`
   - ✅ Métricas em tempo real (tempo total + LLM)

**🚀 Como usar o Sistema:**
```bash
cd /home/beventura/UC/cl-project
source venv/bin/activate
python src/query_system/interactive_demo.py
```

**Exemplos de consultas testadas:**
- ✅ "O que é gradient descent?" → Resposta natural completa (9.49s, 90% confiança)
- ✅ "Quais algoritmos usam backpropagation?" → Lista contextualizada
- ✅ "Liste todos os algoritmos" → Lista organizada e categorizada
- ✅ "Como neural network está relacionado com deep learning?" → Análise de relações

### Fase 4: Sistema RAG ✅ CONCLUÍDO
**Pipeline RAG Completo com Sentence-Transformers + FAISS + Ollama LLM**

**📋 Implementação RAG Completa:**

**✅ Fase 1: Document Processing (CONCLUÍDA)**
   - ✅ Reutilização dos chunks existentes (3.219 chunks)
   - ✅ Embedding generation com all-MiniLM-L6-v2 (384 dims)
   - ✅ GPU acceleration com CUDA
   - ✅ Processamento completo: 1.175M palavras em 17 segundos
   - ✅ Arquivo persistente: data/rag_processed_documents.pkl (19.4 MB)

**✅ Fase 2: Vector Store (CONCLUÍDA)**
   - ✅ FAISS IndexFlatIP para busca exata por similaridade
   - ✅ 3.219 documentos indexados (4.7 MB index)
   - ✅ Performance sub-milissegundo: ~0.3-0.5ms por busca
   - ✅ Persistência com save/load (.faiss + .pkl)
   - ✅ Interface de busca por embedding e texto
   - ✅ Testes com 8 consultas ML de diferentes domínios

**✅ Fase 3: Retriever (CONCLUÍDA)**
   - ✅ Interface inteligente para recuperação de documentos
   - ✅ Análise automática de consultas (algorithm_specific, conceptual, technical, etc.)
   - ✅ Re-ranking inteligente com book diversity
   - ✅ Filtering por threshold de similaridade
   - ✅ Configurações flexíveis (top-k, style, diversity)

**✅ Fase 4: Response Generator (CONCLUÍDA)**
   - ✅ Integração completa com Ollama LLM (llama3.2:3b)
   - ✅ Prompt engineering otimizado para contexto RAG
   - ✅ Múltiplos estilos de resposta (comprehensive, concise, technical)
   - ✅ Sistema de citações e referências automáticas
   - ✅ Cálculo de confiança baseado em múltiplos fatores

**✅ Fase 5: Pipeline Completo (CONCLUÍDA)**
   - ✅ Orquestração end-to-end: Query → Retriever → Generator
   - ✅ Configuração unificada e flexível
   - ✅ Métricas detalhadas (tempo de recuperação + geração)
   - ✅ Error handling robusto com fallbacks
   - ✅ Histórico de consultas e persistência

**✅ Fase 6: Interface Interativa (CONCLUÍDA)**
   - ✅ Demo CLI completa similar ao sistema KG
   - ✅ Comandos especiais (help, stats, config, debug, demo)
   - ✅ Configuração dinâmica (style, top-k, debug mode)
   - ✅ Demonstração automática com consultas pré-definidas
   - ✅ Métricas em tempo real

**🎯 Arquitetura Final:**
```
Query → Document Processor (Embeddings) → Vector Store (FAISS) → Retriever (Re-ranking) → Response Generator (LLM) → Natural Language Answer
```

**🚀 Como usar o Sistema RAG:**
```bash
cd /home/beventura/UC/cl-project
source venv/bin/activate

# Interface interativa completa
python demo_rag_interactive.py

# Teste rápido do sistema
python test_rag_complete.py

# Testes individuais dos componentes
python src/rag/retriever.py
python src/rag/response_generator.py
python src/rag/rag_pipeline.py
```

**📊 Performance Observada:**
   - Inicialização: ~3s (carregamento de modelos)
   - Recuperação: ~0.3-0.5ms para 3K+ documentos
   - Geração LLM: ~5-25s dependendo da complexidade
   - Confiança média: 0.8-0.9 em consultas técnicas
   - GPU acceleration: ✅ (CUDA para embeddings)

**📁 Arquivos Criados:**
   - src/rag/__init__.py - Módulo RAG
   - src/rag/document_processor.py - Geração de embeddings ✅
   - src/rag/vector_store.py - FAISS Vector Store ✅
   - src/rag/retriever.py - Recuperação inteligente ✅
   - src/rag/response_generator.py - Geração com LLM ✅
   - src/rag/rag_pipeline.py - Pipeline completo ✅
   - demo_rag_interactive.py - Interface interativa ✅
   - test_rag_complete.py - Teste do sistema ✅

**🔍 Exemplos de Consultas Testadas:**
   - "What is machine learning?" → Definições introdutórias
   - "How does gradient descent work?" → Explicações algorítmicas
   - "Explain neural networks" → Conceitos técnicos detalhados
   - "What is overfitting and regularization?" → Conceitos avançados
   - "How do support vector machines work?" → Algoritmos específicos

### Fase 5: Experimentos Comparativos
- Métricas de avaliação KG vs RAG
- Testes de perguntas e respostas
- Análise comparativa detalhada

### Fase 6: Análise Híbrida
- Combinação de KG + RAG
- Otimizações e melhorias

## 🔧 Dependências Principais

**Fases 1, 2 & 3 (Concluídas):**
- ✅ **PyPDF2 & PyMuPDF**: Extração de PDFs
- ✅ **NLTK**: Tokenização de sentenças
- ✅ **spaCy**: Named Entity Recognition (en_core_web_sm)
- ✅ **RDFLib**: Construção e SPARQL no Knowledge Graph
- ✅ **ollama**: Interface para LLM local (Llama 3.2 3B)
- ✅ **tqdm**: Barras de progresso
- ✅ **pathlib**: Manipulação de caminhos
- ✅ **logging**: Sistema de logs

**Próximas fases:**
- ✅ **FAISS**: Busca vetorial (CONCLUÍDO)
- ✅ **sentence-transformers**: Embeddings (CONCLUÍDO)
- ✅ **Ollama**: LLM local integrado (CONCLUÍDO)

## 📈 Status do Projeto

### ✅ CONCLUÍDO:
1. **Preprocessamento completo** (8 PDFs → 3.219 chunks)
2. **Knowledge Graph completo** (64.124 triplas RDF)
   - 5.993 entidades normalizadas
   - 3.056 relações extraídas
   - Múltiplos formatos de saída
   - Ontologia ML/DL estruturada
3. **Sistema de Consultas KG completo** com Enhancement LLM
   - 7 tipos de consultas suportadas
   - Interface interativa com modo natural/estruturado
   - Integração Ollama para respostas conversacionais
   - Sistema robusto com fallbacks e debug
4. **Sistema RAG completo** (Pipeline end-to-end)
   - Document Processing: 3.219 docs com embeddings (all-MiniLM-L6-v2)
   - Vector Store: FAISS IndexFlatIP para busca sub-milissegundo
   - Retriever: Recuperação inteligente com re-ranking
   - Response Generator: Integração Ollama LLM com prompt engineering
   - Pipeline: Orquestração completa com métricas e configurações
   - Interface: Demo interativa com comandos e demonstrações

### ⏳ PRÓXIMO:
5. **Experimentos comparativos KG vs RAG** (Fase final)
6. **Análise híbrida KG+RAG** (Fase final)
7. **Relatório final e conclusões** (Documentação)

## 📚 Corpus de Dados
1. ✅ Pattern Recognition and Machine Learning (Bishop) - 758 páginas
2. ✅ Deep Learning (Goodfellow, Bengio, Courville) - 800 páginas  
3. ❌ Pattern Classification (Duda, Hart, Stork) - **Removido** (PDF escaneado)
4. ✅ Introduction to Machine Learning with Python - 392 páginas
5. ✅ Machine Learning: The Art and Science of Algorithms - 416 páginas
6. ✅ Deep Learning: Foundations and Concepts (Prince) - 541 páginas
7. ✅ Pattern Recognition: Concepts, Methods and Applications - 328 páginas
8. ✅ The Science of Deep Learning - 362 páginas
9. ✅ Deep Learning (outro livro) - 656 páginas

