"""
Entity Mapper: Mapeia candidatos de entidades para entidades do KG usando embeddings

Objetivo: Converter candidatos extraídos ("cnn", "svm") para nomes exatos do KG 
("Convolutional Neural Network", "Support Vector Machine").

Abordagem: Similaridade semântica com embeddings para mapear com precisão.
"""

import logging
import pickle
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class EntityMapper:
    """Mapper para entidades do KG usando similaridade semântica com embeddings"""
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Inicializa mapper carregando entidades do KG e modelo de embeddings
        
        Args:
            similarity_threshold: Limiar mínimo de similaridade para aceitar match (0.7 = 70%)
        """
        logger.info("🔄 Inicializando EntityMapper com embeddings...")
        
        self.similarity_threshold = similarity_threshold
        
        # Carregar modelo de embeddings
        logger.info("📥 Carregando modelo de embeddings...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo rápido e eficiente
        
        # Carregar entidades normalizadas do KG
        self.kg_entities = self._load_kg_entities()
        
        # Pré-computar embeddings das entidades do KG
        self.entity_embeddings = self._compute_entity_embeddings()
        
        logger.info(f"✅ EntityMapper inicializado com {len(self.kg_entities)} entidades do KG")
    
    def _load_kg_entities(self) -> List[str]:
        """
        Carrega entidades normalizadas do KG
        
        Returns:
            Lista com nomes canônicos das entidades
        """
        kg_file = Path("data/normalized_entities.pkl")
        
        if not kg_file.exists():
            logger.warning(f"❌ Arquivo KG não encontrado: {kg_file}")
            return []
        
        try:
            # Adicionar path para importar classes
            sys.path.append(str(Path(__file__).parent.parent))
            
            with open(kg_file, 'rb') as f:
                data = pickle.load(f)
            
            normalized_entities = data.get('normalized_entities', {})
            
            # Extrair apenas nomes canônicos (chaves do dicionário)
            entity_names = list(normalized_entities.keys())
            
            logger.info(f"✅ {len(entity_names)} entidades canônicas carregadas")
            return entity_names
            
        except Exception as e:
            logger.error(f"❌ Erro carregando entidades do KG: {e}")
            return []
    
    def _compute_entity_embeddings(self) -> np.ndarray:
        """
        Pré-computa embeddings para todas as entidades do KG
        
        Returns:
            Array numpy com embeddings das entidades (shape: [n_entities, embedding_dim])
        """
        if not self.kg_entities:
            return np.array([])
        
        logger.info("🧮 Computando embeddings das entidades do KG...")
        
        # Gerar embeddings para todas as entidades
        embeddings = self.model.encode(self.kg_entities)
        
        logger.info(f"✅ Embeddings computados: {embeddings.shape}")
        return embeddings
    
    def map_entities(self, candidates: List[str]) -> List[str]:
        """
        Mapeia candidatos para entidades do KG usando similaridade semântica
        
        Args:
            candidates: Lista de candidatos extraídos ["cnn", "gradient descent"]
            
        Returns:
            Lista de entidades mapeadas do KG ["CNNs", "Gradient descent"]
        """
        if not candidates or not self.kg_entities:
            return []
        
        # Filtrar candidatos vazios
        clean_candidates = [c.strip() for c in candidates if c.strip()]
        if not clean_candidates:
            return []
        
        logger.info(f"🔍 Mapeando {len(clean_candidates)} candidatos para entidades do KG")
        
        # Gerar embeddings em batch (mais eficiente)
        candidate_embeddings = self.model.encode(clean_candidates)
        
        mapped = []
        
        for i, candidate in enumerate(clean_candidates):
            # Calcular similaridades
            similarities = cosine_similarity(
                candidate_embeddings[i:i+1], 
                self.entity_embeddings
            )[0]
            
            # Encontrar entidade mais similar
            max_similarity_idx = np.argmax(similarities)
            max_similarity = similarities[max_similarity_idx]
            
            # Verificar threshold
            if max_similarity >= self.similarity_threshold:
                best_match = self.kg_entities[max_similarity_idx]
                mapped.append(best_match)
        
        # Remover duplicatas mantendo ordem
        unique_mapped = list(dict.fromkeys(mapped))  # Preserva ordem, remove duplicatas
        
        logger.info(f"✅ {len(unique_mapped)} entidades mapeadas com sucesso")
        return unique_mapped
    
    def find_similar_entities(self, candidate: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Encontra as K entidades mais similares a um candidato
        
        Args:
            candidate: Candidato para buscar
            top_k: Número de resultados mais similares
            
        Returns:
            Lista de (entidade, similaridade) ordenada por similaridade decrescente
        """
        if not candidate or not self.kg_entities:
            return []
        
        # Gerar embedding do candidato
        candidate_embedding = self.model.encode([candidate])
        
        # Calcular similaridades
        similarities = cosine_similarity(candidate_embedding, self.entity_embeddings)[0]
        
        # Ordenar por similaridade decrescente e pegar top K
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            entity = self.kg_entities[idx]
            similarity = similarities[idx]
            results.append((entity, similarity))
        
        return results


def create_entity_mapper(similarity_threshold: float = 0.7) -> EntityMapper:
    """Factory function"""
    return EntityMapper(similarity_threshold)


if __name__ == "__main__":
    # Teste do mapper com embeddings
    print("🧪 Testando Entity Mapper...")
    
    try:
        mapper = create_entity_mapper(similarity_threshold=0.6)  # Threshold mais baixo para teste
        
        # Testes
        test_cases = [
            # Originais
            ["cnn"],
            ["svm", "gradient descent"], 
            ["overfitting", "deep learning"],
            ["lstm", "backpropagation"],
            ["convolutional network"],  # Teste com forma expandida
            ["neural net"],  # Teste com abreviação
            ["regression"],  # Termo genérico
            
            # Mais 20 casos
            ["random forest"],
            ["decision tree"],
            ["k means", "clustering"],
            ["pca", "dimensionality reduction"],
            ["logistic regression", "classification"],
            ["reinforcement learning"],
            ["supervised learning", "unsupervised learning"],
            ["cross validation"],
            ["feature selection"],
            ["bias", "variance"],
            ["activation function", "relu"],
            ["sigmoid", "tanh"],
            ["adam", "optimizer"],
            ["batch normalization"],
            ["dropout", "regularization"],
            ["ensemble methods"],
            ["bagging", "boosting"],
            ["naive bayes"],
            ["support vector"],
            ["artificial intelligence"],
            ["machine learning", "pattern recognition"],
            ["training set", "test set"],
            ["precision", "recall"],
            ["f1 score"],
            ["confusion matrix"],
            ["roc curve"]
        ]
        
        print(f"\n📝 Executando {len(test_cases)} casos de teste:")
        print("=" * 50)
        
        for i, candidates in enumerate(test_cases, 1):
            print(f"{i}. {candidates}")
            mapped = mapper.map_entities(candidates)
            print(f"   → {mapped}")
            print()
        
        print("✅ Entity Mapper funcionando!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()