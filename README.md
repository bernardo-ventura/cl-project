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
│   ├── preprocessing/          # Módulo de preprocessamento
│   │   ├── extract_sample.py   # Extração de amostras (teste)
│   │   ├── extract_full.py     # Extração completa ✅
│   │   └── chunking.py         # Divisão em chunks (atual)
│   ├── knowledge_graph/        # Módulo KG (Fase 2)
│   ├── rag/                    # Módulo RAG (Fase 3)
│   └── experiments/            # Experimentos comparativos (Fase 4)
├── data/
│   ├── raw_pdfs/              # 8 PDFs originais
│   ├── samples/               # Amostras de teste
│   └── processed_texts/       # Textos completos ✅
       └── chunks/             # Chunks divididos ✅
```

## Fases do Projeto

### Fase 1: Preprocessamento
- ✅ **Extração**: PyPDF2 (padrão) + PyMuPDF (fallback) - 8/8 PDFs extraídos com sucesso (extract_full.py)
- ✅ **Chunking**: Divisão inteligente por sentenças (~350 palavras) - **Concluído** (chunking.py)
- ⏸️ **Limpeza**: Normalização de texto e caracteres - Planejado (Não é essencial, pular por enquanto)

### Fase 2: Knowledge Graphs
- Construção usando RDFLib
- Extração de entidades e relações
- Ontologia específica para ML/DL

### Fase 3: Sistema RAG
- Implementação com LangChain + FAISS
- Embeddings com sentence-transformers
- Sistema de recuperação

### Fase 4: Experimentos Comparativos
- Métricas de avaliação
- Testes de perguntas e respostas
- Análise comparativa

### Fase 5: Análise Híbrida
- Combinação de KG + RAG
- Otimizações

## 🔧 Dependências Principais

- **PyPDF2**: Extração de PDFs (método principal)
- **PyMuPDF**: Extração alternativa para PDFs problemáticos
- **NLTK**: Tokenização de sentenças
- **tqdm**: Barras de progresso
- **pathlib**: Manipulação de caminhos
- **logging**: Sistema de logs

## 📚 Corpus de Dados

**8 livros processados com sucesso** (Machine Learning e Deep Learning):
1. ✅ Pattern Recognition and Machine Learning (Bishop) - 758 páginas
2. ✅ Deep Learning (Goodfellow, Bengio, Courville) - 800 páginas  
3. ❌ Pattern Classification (Duda, Hart, Stork) - **Removido** (PDF escaneado)
4. ✅ Introduction to Machine Learning with Python - 392 páginas
5. ✅ Machine Learning: The Art and Science of Algorithms - 416 páginas
6. ✅ Deep Learning: Foundations and Concepts (Prince) - 541 páginas
7. ✅ Pattern Recognition: Concepts, Methods and Applications - 328 páginas
8. ✅ The Science of Deep Learning - 362 páginas
9. ✅ Deep Learning (outro livro) - 656 páginas

