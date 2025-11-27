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
│   ├── rag/                    # ⏳ Módulo RAG (PRÓXIMO)
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

### Fase 3: Sistema RAG ⏳ PRÓXIMO
- Implementação com LangChain + FAISS
- Embeddings com sentence-transformers
- Sistema de recuperação
- **Status**: Pronto para iniciar após conclusão do KG ✅

### Fase 4: Experimentos Comparativos
- Métricas de avaliação
- Testes de perguntas e respostas
- Análise comparativa

### Fase 5: Análise Híbrida
- Combinação de KG + RAG
- Otimizações

## 🔧 Dependências Principais

**Fase 1 & 2 (Concluídas):**
- ✅ **PyPDF2**: Extração de PDFs (método principal)
- ✅ **PyMuPDF**: Extração alternativa para PDFs problemáticos
- ✅ **NLTK**: Tokenização de sentenças
- ✅ **spaCy**: Named Entity Recognition (en_core_web_sm)
- ✅ **RDFLib**: Construção e serialização do Knowledge Graph
- ✅ **ollama**: Interface para LLM local (Llama 3.2 3B)
- ✅ **tqdm**: Barras de progresso
- ✅ **pathlib**: Manipulação de caminhos
- ✅ **logging**: Sistema de logs

**Próximas fases:**
- **LangChain**: Framework RAG
- **FAISS**: Busca vetorial
- **sentence-transformers**: Embeddings

## 📈 Status do Projeto

### ✅ CONCLUÍDO:
1. **Preprocessamento completo** (8 PDFs → 3.219 chunks)
2. **Knowledge Graph completo** (64.124 triplas RDF)
   - 5.993 entidades normalizadas
   - 3.056 relações extraídas
   - Múltiplos formatos de saída
   - Ontologia ML/DL estruturada

### ⏳ PRÓXIMO:
3. **Sistema RAG** (Fase 3)
4. **Experimentos comparativos** (Fase 4)
5. **Análise híbrida KG+RAG** (Fase 5)

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

