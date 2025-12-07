"""
Question Entity Extractor: Extração de entidades de perguntas com spaCy

Objetivo: Extrair candidatos a entidades de perguntas usando spaCy diretamente.
Abordagem: spaCy NER + padrões customizados ML/DL reutilizados do sistema KG.
Responsabilidade: APENAS passo 1 - extração de candidatos a entidades.
"""

import spacy
import logging
from typing import List, Set
from spacy.matcher import Matcher
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EntityCandidate:
    """Candidato a entidade extraído de uma pergunta."""
    text: str
    label: str
    start_char: int
    end_char: int
    source: str  # 'spacy_ner' ou 'ml_pattern'


class QuestionEntityExtractor:
    """Extrator de entidades para perguntas usando spaCy diretamente."""
    
    def __init__(self):
        """Inicializa spaCy e padrões ML/DL."""
        logger.info("🔄 Inicializando QuestionEntityExtractor...")
        
        # Carregar modelo spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ Modelo spaCy carregado")
        except OSError:
            logger.error("❌ Modelo spaCy não encontrado. Execute: python -m spacy download en_core_web_sm")
            raise
        
        # Configurar matcher para padrões ML/DL
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_ml_patterns()
        
        logger.info("✅ QuestionEntityExtractor inicializado")
    
    def _setup_ml_patterns(self):
        """Configura padrões ML/DL reutilizados do sistema KG."""
        
        # Algoritmos de Machine Learning (reutilizado do sistema KG)
        ml_algorithms = [
            # Redes Neurais
            [{"LOWER": "neural"}, {"LOWER": "network"}],
            [{"LOWER": "neural"}, {"LOWER": "networks"}],
            [{"LOWER": "deep"}, {"LOWER": "learning"}],
            [{"LOWER": "convolutional"}, {"LOWER": "neural"}, {"LOWER": "network"}],
            [{"LOWER": "recurrent"}, {"LOWER": "neural"}, {"LOWER": "network"}],
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
        
        # Conceitos Gerais (reutilizado do sistema KG)
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
        
        logger.info(f"✅ {len(ml_algorithms)} padrões de algoritmos configurados")
        logger.info(f"✅ {len(ml_concepts)} padrões de conceitos configurados")
    
    def extract_entities(self, question: str) -> List[str]:
        """
        Extrai candidatos a entidades de uma pergunta.
        
        Args:
            question: Pergunta do usuário
            
        Returns:
            Lista de textos de entidades candidatas (strings únicas)
        """
        # Processar pergunta com spaCy
        doc = self.nlp(question.strip())
        
        candidates = []
        
        # 1. Entidades spaCy NER (substantivos próprios, organizações, etc.)
        spacy_entities = self._extract_spacy_entities(doc)
        candidates.extend(spacy_entities)
        
        # 2. Padrões customizados ML/DL (PRIORIDADE ALTA)
        ml_entities = self._extract_ml_patterns(doc)
        candidates.extend(ml_entities)
        
        # 3. Substantivos simples como fallback (PRIORIDADE BAIXA)
        noun_entities = self._extract_nouns(doc)
        candidates.extend(noun_entities)
        
        # 4. Remover overlaps (priorizar entidades compostas)
        candidates = self._remove_overlaps(candidates)
        
        # 5. Filtrar e limpar
        entity_texts = self._filter_and_clean(candidates)
        
        logger.info(f"✅ {len(entity_texts)} entidades extraídas: {entity_texts}")
        return entity_texts
    
    def _extract_spacy_entities(self, doc) -> List[EntityCandidate]:
        """Extrai entidades usando spaCy NER."""
        entities = []
        
        for ent in doc.ents:
            # Filtrar apenas entidades relevantes para perguntas
            if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'LANGUAGE', 'NORP']:
                entities.append(EntityCandidate(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    source='spacy_ner'
                ))
        
        return entities
    
    def _extract_ml_patterns(self, doc) -> List[EntityCandidate]:
        """Extrai entidades usando padrões ML/DL."""
        entities = []
        matches = self.matcher(doc)
        
        for match_id, start, end in matches:
            span = doc[start:end]
            label = self.nlp.vocab.strings[match_id]  # 'ML_ALGORITHM' ou 'ML_CONCEPT'
            
            entities.append(EntityCandidate(
                text=span.text,
                label=label,
                start_char=span.start_char,
                end_char=span.end_char,
                source='ml_pattern'
            ))
        
        return entities
    
    def _extract_nouns(self, doc) -> List[EntityCandidate]:
        """Extrai substantivos como candidatos (fallback)."""
        entities = []
        
        for token in doc:
            # Substantivos próprios ou comuns importantes
            if (token.pos_ in ['NOUN', 'PROPN'] and 
                len(token.text) > 2 and 
                not token.is_stop and 
                not token.is_punct and
                token.text.isalnum()):
                
                entities.append(EntityCandidate(
                    text=token.text,
                    label='NOUN',
                    start_char=token.idx,
                    end_char=token.idx + len(token.text),
                    source='noun_fallback'
                ))
        
        return entities
    
    def _remove_overlaps(self, candidates: List[EntityCandidate]) -> List[EntityCandidate]:
        """Remove overlaps priorizando entidades compostas e de alta prioridade."""
        # Ordenar por prioridade: ml_pattern > spacy_ner > noun_fallback
        priority_order = {'ml_pattern': 0, 'spacy_ner': 1, 'noun_fallback': 2}
        sorted_candidates = sorted(candidates, key=lambda x: (
            priority_order.get(x.source, 3),
            -len(x.text)  # Preferir textos mais longos
        ))
        
        filtered = []
        used_ranges = []
        
        for candidate in sorted_candidates:
            # Verificar se overlap com alguma entidade já aceita
            overlaps = False
            for start, end in used_ranges:
                if not (candidate.end_char <= start or candidate.start_char >= end):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(candidate)
                used_ranges.append((candidate.start_char, candidate.end_char))
        
        return filtered
    
    def _filter_and_clean(self, candidates: List[EntityCandidate]) -> List[str]:
        """Filtra, limpa e remove duplicatas dos candidatos."""
        seen: Set[str] = set()
        clean_entities = []
        
        # Palavras de parada para filtrar
        stop_words = {'que', 'como', 'entre', 'explique', 'what', 'how', 'is', 'does', 'vs', 'and', 'or', 'the', 'a', 'an'}
        
        for candidate in candidates:
            # Limpar texto
            text = candidate.text.strip().lower()
            
            # Filtros básicos
            if (len(text) > 1 and 
                text not in seen and 
                text not in stop_words and
                not text.isdigit()):
                
                clean_entities.append(text)
                seen.add(text)
        
        return clean_entities


def create_question_entity_extractor() -> QuestionEntityExtractor:
    """Factory function para criar QuestionEntityExtractor."""
    return QuestionEntityExtractor()


if __name__ == "__main__":
    # Teste focado em extração de entidades
    print("🧪 Testando Question Entity Extractor...")
    
    try:
        extractor = create_question_entity_extractor()
        
        test_questions = [
            # Teste original
            "O que é CNN?",
            "Como funciona gradient descent?", 
            "Diferença entre SVM e Random Forest?",
            "Explique overfitting e regularização",
            "What is neural network?",
            "How does backpropagation work?",
            "Compare supervised vs unsupervised learning",
            
            # Novos 20 testes
            "How does LSTM work?",
            "What is deep learning?",
            "Explain convolutional neural network",
            "What is the difference between precision and recall?",
            "How to prevent overfitting?",
            "What is activation function?",
            "Compare ReLU vs sigmoid?",
            "What is cross entropy loss?",
            "How does principal component analysis work?",
            "Explain k-means clustering",
            "What is reinforcement learning?",
            "How does Adam optimizer work?",
            "What is feature selection?",
            "Explain bias variance tradeoff",
            "What is training set vs test set?",
            "How does RNN work?",
            "What is supervised learning?",
            "Explain artificial neural network",
            "What is machine learning?",
            "How does support vector machine work?",
            "What is regularization?",
            "Explain dimensionality reduction",
            "What is pattern recognition?",
            "How does decision tree work?",
            "What is unsupervised learning?",
            "Explain data preprocessing",
            "What is validation set?"
        ]
        
        print(f"\n📋 Testando {len(test_questions)} perguntas:")
        print("=" * 60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"{i}. '{question}'")
            entities = extractor.extract_entities(question)
            print(f"   → {entities}")
            print()
        
        print("✅ Teste concluído - Extração funcionando!")
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()