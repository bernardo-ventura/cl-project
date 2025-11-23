"""
Módulo para extração de entidades usando spaCy NER + padrões customizados.
Passo 3 do pipeline de construção do Knowledge Graph.
"""

import spacy
import logging
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from spacy.matcher import Matcher
from spacy.tokens import Doc
from pathlib import Path
import sys

# Adicionar src ao path para imports
sys.path.append(str(Path(__file__).parent.parent))
from knowledge_graph.chunk_loader import load_chunks, TextChunk

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EntityCandidate:
    """Representa um candidato a entidade extraído."""
    text: str
    label: str
    start_char: int
    end_char: int
    chunk_id: str
    source: str  # 'spacy_ner' ou 'custom_pattern'
    confidence: float = 1.0

class EntityExtractor:
    """Extrator de entidades para domínio ML/DL."""
    
    def __init__(self):
        """Inicializa o extrator carregando spaCy e configurando padrões."""
        logger.info("Inicializando EntityExtractor...")
        
        # Carregar modelo spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ Modelo spaCy en_core_web_sm carregado")
        except OSError:
            logger.error("❌ Modelo spaCy não encontrado. Execute: python -m spacy download en_core_web_sm")
            raise
        
        # Configurar matcher para padrões customizados
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_ml_patterns()
        
        # Estatísticas
        self.stats = {
            'chunks_processed': 0,
            'spacy_entities': 0,
            'custom_entities': 0,
            'total_entities': 0
        }
    
    def _setup_ml_patterns(self):
        """Configura padrões customizados para termos de ML/DL."""
        logger.info("Configurando padrões customizados para ML/DL...")
        
        # Algoritmos de Machine Learning
        ml_algorithms = [
            # Redes Neurais
            [{"LOWER": "neural"}, {"LOWER": "network"}],
            [{"LOWER": "neural"}, {"LOWER": "networks"}],
            [{"LOWER": "deep"}, {"LOWER": "learning"}],
            [{"LOWER": "convolutional"}, {"LOWER": "neural"}, {"LOWER": "network"}],
            [{"LOWER": "recurrent"}, {"LOWER": "neural"}, {"LOWER": "network"}],
            [{"LOWER": "artificial"}, {"LOWER": "neural"}, {"LOWER": "network"}],
            [{"TEXT": {"REGEX": "^(CNN|RNN|LSTM|GRU|ANN)$"}}],
            
            # Algoritmos Clássicos
            [{"LOWER": "support"}, {"LOWER": "vector"}, {"LOWER": "machine"}],
            [{"LOWER": "support"}, {"LOWER": "vector"}, {"LOWER": "machines"}],
            [{"TEXT": {"REGEX": "^SVM$"}}],
            [{"LOWER": "random"}, {"LOWER": "forest"}],
            [{"LOWER": "decision"}, {"LOWER": "tree"}],
            [{"LOWER": "decision"}, {"LOWER": "trees"}],
            [{"LOWER": "k"}, {"LOWER": "-"}, {"LOWER": "means"}],
            [{"LOWER": "k"}, {"LOWER": "means"}],
            [{"LOWER": "logistic"}, {"LOWER": "regression"}],
            [{"LOWER": "linear"}, {"LOWER": "regression"}],
            [{"LOWER": "naive"}, {"LOWER": "bayes"}],
            [{"LOWER": "principal"}, {"LOWER": "component"}, {"LOWER": "analysis"}],
            [{"TEXT": {"REGEX": "^(PCA|LDA|ICA)$"}}],
            
            # Técnicas de Otimização
            [{"LOWER": "gradient"}, {"LOWER": "descent"}],
            [{"LOWER": "stochastic"}, {"LOWER": "gradient"}, {"LOWER": "descent"}],
            [{"LOWER": "backpropagation"}],
            [{"LOWER": "back"}, {"LOWER": "propagation"}],
            [{"TEXT": {"REGEX": "^(SGD|Adam|RMSprop|Adagrad)$"}}],
            
            # Funções de Ativação
            [{"LOWER": "activation"}, {"LOWER": "function"}],
            [{"LOWER": "relu"}],
            [{"LOWER": "sigmoid"}],
            [{"LOWER": "tanh"}],
            [{"LOWER": "softmax"}],
            
            # Métricas e Loss Functions
            [{"LOWER": "cross"}, {"LOWER": "-"}, {"LOWER": "entropy"}],
            [{"LOWER": "cross"}, {"LOWER": "entropy"}],
            [{"LOWER": "mean"}, {"LOWER": "squared"}, {"LOWER": "error"}],
            [{"TEXT": {"REGEX": "^(MSE|RMSE|MAE)$"}}],
            [{"LOWER": "accuracy"}],
            [{"LOWER": "precision"}],
            [{"LOWER": "recall"}],
            [{"LOWER": "f1"}, {"LOWER": "score"}],
            [{"LOWER": "f1"}, {"LOWER": "-"}, {"LOWER": "score"}],
            [{"TEXT": {"REGEX": "^(AUC|ROC)$"}}],
        ]
        
        # Conceitos Gerais
        ml_concepts = [
            # Aprendizado
            [{"LOWER": "supervised"}, {"LOWER": "learning"}],
            [{"LOWER": "unsupervised"}, {"LOWER": "learning"}],
            [{"LOWER": "reinforcement"}, {"LOWER": "learning"}],
            [{"LOWER": "machine"}, {"LOWER": "learning"}],
            [{"LOWER": "pattern"}, {"LOWER": "recognition"}],
            [{"LOWER": "feature"}, {"LOWER": "extraction"}],
            [{"LOWER": "feature"}, {"LOWER": "selection"}],
            [{"LOWER": "dimensionality"}, {"LOWER": "reduction"}],
            
            # Dados
            [{"LOWER": "training"}, {"LOWER": "set"}],
            [{"LOWER": "test"}, {"LOWER": "set"}],
            [{"LOWER": "validation"}, {"LOWER": "set"}],
            [{"LOWER": "dataset"}],
            [{"LOWER": "data"}, {"LOWER": "preprocessing"}],
            [{"LOWER": "data"}, {"LOWER": "augmentation"}],
            
            # Problemas
            [{"LOWER": "overfitting"}],
            [{"LOWER": "underfitting"}],
            [{"LOWER": "bias"}, {"LOWER": "variance"}, {"LOWER": "tradeoff"}],
            [{"LOWER": "curse"}, {"LOWER": "of"}, {"LOWER": "dimensionality"}],
        ]
        
        # Adicionar padrões ao matcher
        self.matcher.add("ML_ALGORITHM", ml_algorithms)
        self.matcher.add("ML_CONCEPT", ml_concepts)
        
        logger.info(f"✅ {len(ml_algorithms)} padrões de algoritmos adicionados")
        logger.info(f"✅ {len(ml_concepts)} padrões de conceitos adicionados")
    
    def extract_entities_from_chunk(self, chunk: TextChunk) -> List[EntityCandidate]:
        """
        Extrai entidades de um chunk usando spaCy NER + padrões customizados.
        
        Args:
            chunk: Chunk de texto para processar
            
        Returns:
            Lista de candidatos a entidades
        """
        entities = []
        
        # Processar texto com spaCy
        doc = self.nlp(chunk.content)
        
        # 1. Entidades do spaCy NER (pessoas, organizações, etc.)
        spacy_entities = self._extract_spacy_entities(doc, chunk.chunk_id)
        entities.extend(spacy_entities)
        self.stats['spacy_entities'] += len(spacy_entities)
        
        # 2. Padrões customizados para ML/DL
        custom_entities = self._extract_custom_entities(doc, chunk.chunk_id)
        entities.extend(custom_entities)
        self.stats['custom_entities'] += len(custom_entities)
        
        # 3. Filtrar duplicatas (mesmo texto, mesma posição)
        entities = self._remove_duplicates(entities)
        
        self.stats['chunks_processed'] += 1
        self.stats['total_entities'] += len(entities)
        
        return entities
    
    def _extract_spacy_entities(self, doc: Doc, chunk_id: str) -> List[EntityCandidate]:
        """Extrai entidades usando spaCy NER padrão."""
        entities = []
        
        for ent in doc.ents:
            # Filtrar apenas entidades relevantes
            if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'EVENT', 'LANGUAGE']:
                entity = EntityCandidate(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    chunk_id=chunk_id,
                    source='spacy_ner',
                    confidence=1.0  # spaCy não fornece confidence score facilmente
                )
                entities.append(entity)
        
        return entities
    
    def _extract_custom_entities(self, doc: Doc, chunk_id: str) -> List[EntityCandidate]:
        """Extrai entidades usando padrões customizados."""
        entities = []
        
        matches = self.matcher(doc)
        
        for match_id, start, end in matches:
            span = doc[start:end]
            label = self.nlp.vocab.strings[match_id]  # 'ML_ALGORITHM' ou 'ML_CONCEPT'
            
            entity = EntityCandidate(
                text=span.text,
                label=label,
                start_char=span.start_char,
                end_char=span.end_char,
                chunk_id=chunk_id,
                source='custom_pattern',
                confidence=1.0
            )
            entities.append(entity)
        
        return entities
    
    def _remove_duplicates(self, entities: List[EntityCandidate]) -> List[EntityCandidate]:
        """Remove entidades duplicadas (mesmo texto e posição)."""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # Chave única baseada em texto e posição
            key = (entity.text.lower(), entity.start_char, entity.end_char)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_entities_from_chunks(self, chunks: List[TextChunk], 
                                   max_chunks: int = None) -> Dict[str, List[EntityCandidate]]:
        """
        Extrai entidades de múltiplos chunks.
        
        Args:
            chunks: Lista de chunks para processar
            max_chunks: Limite de chunks para teste (opcional)
            
        Returns:
            Dicionário {chunk_id: [entidades]}
        """
        logger.info(f"Iniciando extração de entidades de {len(chunks)} chunks...")
        
        if max_chunks:
            chunks = chunks[:max_chunks]
            logger.info(f"Limitando processamento a {max_chunks} chunks para teste")
        
        results = {}
        
        for i, chunk in enumerate(chunks):
            if (i + 1) % 100 == 0:
                logger.info(f"Processado {i + 1}/{len(chunks)} chunks...")
            
            entities = self.extract_entities_from_chunk(chunk)
            results[chunk.chunk_id] = entities
        
        logger.info("✅ Extração de entidades concluída!")
        return results
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas da extração."""
        return {
            **self.stats,
            'avg_entities_per_chunk': (
                self.stats['total_entities'] / self.stats['chunks_processed'] 
                if self.stats['chunks_processed'] > 0 else 0
            )
        }
    
    def get_entity_summary(self, entities_by_chunk: Dict[str, List[EntityCandidate]]) -> Dict:
        """
        Gera resumo das entidades extraídas.
        
        Args:
            entities_by_chunk: Resultado da extração
            
        Returns:
            Resumo com contadores e exemplos
        """
        # Contar por tipo
        label_counts = {}
        source_counts = {}
        all_entities = []
        
        for chunk_id, entities in entities_by_chunk.items():
            for entity in entities:
                # Contadores por label
                label_counts[entity.label] = label_counts.get(entity.label, 0) + 1
                # Contadores por fonte
                source_counts[entity.source] = source_counts.get(entity.source, 0) + 1
                # Lista geral
                all_entities.append(entity)
        
        # Entidades únicas por texto
        unique_texts = set(entity.text.lower() for entity in all_entities)
        
        # Exemplos por categoria
        examples = {}
        for label in label_counts.keys():
            examples[label] = [
                entity.text for entity in all_entities 
                if entity.label == label
            ][:5]  # Primeiros 5 exemplos
        
        return {
            'total_entities': len(all_entities),
            'unique_entities': len(unique_texts),
            'label_counts': label_counts,
            'source_counts': source_counts,
            'examples': examples
        }


def extract_entities(chunks: List[TextChunk] = None, max_chunks: int = None) -> Tuple[Dict, Dict, Dict]:
    """
    Função de conveniência para extrair entidades.
    
    Args:
        chunks: Lista de chunks (carrega automaticamente se None)
        max_chunks: Limite para teste
        
    Returns:
        Tupla com (entidades_por_chunk, estatísticas, resumo)
    """
    if chunks is None:
        logger.info("Carregando chunks...")
        chunks, _ = load_chunks()
    
    extractor = EntityExtractor()
    entities_by_chunk = extractor.extract_entities_from_chunks(chunks, max_chunks)
    stats = extractor.get_statistics()
    summary = extractor.get_entity_summary(entities_by_chunk)
    
    return entities_by_chunk, stats, summary


if __name__ == "__main__":
    # Teste do módulo
    print("🔍 Testando extração de entidades...")
    
    try:
        # Teste com sample pequeno primeiro
        print("\n1️⃣ Carregando chunks...")
        chunks, chunk_stats = load_chunks()
        print(f"✅ {len(chunks)} chunks carregados")
        
        # Teste com 10 chunks primeiro
        print("\n2️⃣ Testando com 10 chunks...")
        entities_by_chunk, stats, summary = extract_entities(chunks, max_chunks=10)
        
        print(f"\n📊 Estatísticas da extração:")
        for key, value in stats.items():
            print(f"  • {key}: {value}")
        
        print(f"\n📋 Resumo das entidades:")
        print(f"  • Total de entidades: {summary['total_entities']}")
        print(f"  • Entidades únicas: {summary['unique_entities']}")
        
        print(f"\n🏷️ Por tipo:")
        for label, count in summary['label_counts'].items():
            print(f"  • {label}: {count}")
        
        print(f"\n🔧 Por fonte:")
        for source, count in summary['source_counts'].items():
            print(f"  • {source}: {count}")
        
        print(f"\n📝 Exemplos por categoria:")
        for label, examples in summary['examples'].items():
            print(f"  • {label}: {', '.join(examples[:3])}...")
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()