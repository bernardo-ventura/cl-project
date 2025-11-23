"""
Módulo para normalização de entidades usando LLM local (Ollama).
Passo 4 do pipeline de construção do Knowledge Graph.
"""

import ollama
import logging
import json
import pickle
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import sys
from collections import Counter, defaultdict
import re

# Adicionar src ao path para imports
sys.path.append(str(Path(__file__).parent.parent))
from knowledge_graph.entity_extractor import EntityCandidate

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NormalizedEntity:
    """Entidade normalizada após processamento LLM."""
    canonical_name: str
    entity_type: str
    aliases: List[str]
    frequency: int
    confidence: float
    source_chunks: List[str]
    original_labels: List[str]

class EntityNormalizer:
    """Normalizador de entidades usando LLM local."""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Inicializa o normalizador.
        
        Args:
            model_name: Nome do modelo Ollama a usar
        """
        self.model_name = model_name
        self.normalized_entities: Dict[str, NormalizedEntity] = {}
        
        # Teste de conectividade
        try:
            response = ollama.chat(model=model_name, messages=[
                {'role': 'user', 'content': 'Hello'}
            ])
            logger.info(f"✅ Conectado ao modelo {model_name}")
        except Exception as e:
            logger.error(f"❌ Erro conectando ao Ollama: {e}")
            raise
        
        # Estatísticas
        self.stats = {
            'batches_processed': 0,
            'entities_input': 0,
            'entities_output': 0,
            'llm_calls': 0
        }
    
    def _create_normalization_prompt(self, entities: List[str]) -> str:
        """
        Cria prompt para normalização de entidades.
        
        Args:
            entities: Lista de entidades brutas para normalizar
            
        Returns:
            Prompt formatado para o LLM
        """
        entities_text = '\n'.join(f"- {entity}" for entity in entities)
        
        prompt = f"""You are an expert in Machine Learning and Deep Learning terminology. 

I have extracted the following entities from academic ML/DL texts. Please normalize and classify them:

ENTITIES TO NORMALIZE:
{entities_text}

TASKS:
1. Deduplicate: Group similar entities (e.g., "SVM", "Support Vector Machine", "support vector machines" → one canonical form)
2. Normalize: Use standard academic terminology and consistent capitalization
3. Classify: Assign each to ONE category: ALGORITHM, CONCEPT, PERSON, ORGANIZATION, SOFTWARE, METRIC, OTHER
4. Filter: Remove obvious noise/errors

RESPONSE FORMAT (JSON):
{{
  "normalized_entities": [
    {{
      "canonical_name": "Support Vector Machine",
      "type": "ALGORITHM", 
      "aliases": ["SVM", "support vector machines", "Support Vector Machines"]
    }},
    {{
      "canonical_name": "Geoffrey Hinton",
      "type": "PERSON",
      "aliases": ["Hinton", "G. Hinton"]
    }}
  ]
}}

Important: Only return valid JSON. Be conservative - if unsure about an entity, classify as OTHER."""

        return prompt
    
    def _parse_llm_response(self, response: str) -> List[Dict]:
        """
        Parse a resposta JSON do LLM (versão robusta).
        
        Args:
            response: Resposta do LLM
            
        Returns:
            Lista de entidades normalizadas
        """
        try:
            # Remover markdown code blocks se existirem
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*$', '', response)
            
            # Tentar extrair JSON da resposta
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("Nenhum JSON encontrado na resposta do LLM")
                return []
            
            json_text = response[json_start:json_end]
            
            # Limpar problemas comuns de JSON
            json_text = json_text.replace('\n', ' ').replace('\r', '')
            json_text = re.sub(r',\s*}', '}', json_text)  # Remover vírgulas antes de }
            json_text = re.sub(r',\s*]', ']', json_text)  # Remover vírgulas antes de ]
            
            data = json.loads(json_text)
            
            return data.get('normalized_entities', [])
            
        except json.JSONDecodeError as e:
            logger.warning(f"Erro parsing JSON: {e}")
            # Tentar parser mais permissivo
            try:
                # Extrair entidades manualmente se JSON falhar
                entities = []
                lines = response.split('\n')
                current_entity = {}
                
                for line in lines:
                    if 'canonical_name' in line:
                        name = re.search(r'"canonical_name":\s*"([^"]+)"', line)
                        if name:
                            current_entity['canonical_name'] = name.group(1)
                    elif 'type' in line and 'canonical_name' in current_entity:
                        entity_type = re.search(r'"type":\s*"([^"]+)"', line)
                        if entity_type:
                            current_entity['type'] = entity_type.group(1)
                            current_entity['aliases'] = []  # Simplificado
                            entities.append(current_entity.copy())
                            current_entity = {}
                
                if entities:
                    logger.info(f"Recuperadas {len(entities)} entidades com parser manual")
                    return entities
                    
            except Exception:
                pass
                
            logger.error(f"Parser manual também falhou. Resposta: {response[:200]}...")
            return []
        except Exception as e:
            logger.error(f"Erro inesperado parsing resposta: {e}")
            return []
    
    def _group_similar_entities(self, entities: List[EntityCandidate], 
                               batch_size: int = 20) -> List[List[str]]:
        """
        Agrupa entidades similares em lotes para processamento.
        
        Args:
            entities: Lista de candidatos a entidades
            batch_size: Tamanho do lote para enviar ao LLM
            
        Returns:
            Lista de lotes (cada lote é lista de strings)
        """
        # Contar frequência das entidades
        entity_counts = Counter(entity.text.strip() for entity in entities)
        
        # Pegar apenas entidades que aparecem pelo menos 2 vezes ou são importantes
        frequent_entities = []
        for entity_text, count in entity_counts.items():
            if count >= 2 or len(entity_text) > 3:  # Filtro básico
                frequent_entities.append(entity_text)
        
        # Dividir em lotes
        batches = []
        for i in range(0, len(frequent_entities), batch_size):
            batch = frequent_entities[i:i + batch_size]
            batches.append(batch)
        
        logger.info(f"Criados {len(batches)} lotes de entidades para normalização")
        return batches
    
    def normalize_entities(self, entities: List[EntityCandidate]) -> Dict[str, NormalizedEntity]:
        """
        Normaliza lista de entidades usando LLM.
        
        Args:
            entities: Lista de candidatos a entidades
            
        Returns:
            Dicionário de entidades normalizadas
        """
        logger.info(f"Iniciando normalização de {len(entities)} entidades...")
        
        # Agrupar em lotes
        entity_batches = self._group_similar_entities(entities)
        
        self.stats['entities_input'] = len(entities)
        all_normalized = {}
        
        for i, batch in enumerate(entity_batches):
            logger.info(f"Processando lote {i+1}/{len(entity_batches)} ({len(batch)} entidades)...")
            
            try:
                # Criar prompt
                prompt = self._create_normalization_prompt(batch)
                
                # Chamar LLM
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                
                self.stats['llm_calls'] += 1
                
                # Parse da resposta
                normalized_batch = self._parse_llm_response(response['message']['content'])
                
                # Processar entidades normalizadas
                for norm_entity in normalized_batch:
                    try:
                        canonical_name = norm_entity.get('canonical_name', '').strip()
                        if not canonical_name:
                            continue
                        
                        # Criar objeto NormalizedEntity
                        aliases = norm_entity.get('aliases', [])
                        entity_type = norm_entity.get('type', 'OTHER')
                        
                        # Calcular frequência total (soma de todas as aliases)
                        total_freq = sum(
                            sum(1 for e in entities if e.text.strip().lower() == alias.lower()) 
                            for alias in [canonical_name] + aliases
                        )
                        
                        # Coletar chunks onde aparece
                        source_chunks = list(set(
                            e.chunk_id for e in entities 
                            if e.text.strip().lower() in [alias.lower() for alias in [canonical_name] + aliases]
                        ))
                        
                        # Coletar labels originais
                        original_labels = list(set(
                            e.label for e in entities 
                            if e.text.strip().lower() in [alias.lower() for alias in [canonical_name] + aliases]
                        ))
                        
                        normalized_entity = NormalizedEntity(
                            canonical_name=canonical_name,
                            entity_type=entity_type,
                            aliases=aliases,
                            frequency=total_freq,
                            confidence=1.0,  # Simplificado por agora
                            source_chunks=source_chunks,
                            original_labels=original_labels
                        )
                        
                        all_normalized[canonical_name] = normalized_entity
                        
                    except Exception as e:
                        logger.warning(f"Erro processando entidade normalizada: {e}")
                        continue
                
            except Exception as e:
                logger.error(f"Erro processando lote {i+1}: {e}")
                continue
            
            self.stats['batches_processed'] += 1
            
            # Log de progresso
            if (i + 1) % 10 == 0:
                logger.info(f"Processados {i+1}/{len(entity_batches)} lotes...")
        
        self.stats['entities_output'] = len(all_normalized)
        logger.info("✅ Normalização concluída!")
        
        return all_normalized
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas da normalização."""
        reduction_pct = 0
        if self.stats['entities_input'] > 0:
            reduction_pct = (1 - self.stats['entities_output'] / self.stats['entities_input']) * 100
        
        return {
            **self.stats,
            'reduction_percentage': reduction_pct
        }
    
    def get_normalization_summary(self, normalized_entities: Dict[str, NormalizedEntity]) -> Dict:
        """
        Gera resumo das entidades normalizadas.
        
        Args:
            normalized_entities: Entidades normalizadas
            
        Returns:
            Resumo com estatísticas
        """
        if not normalized_entities:
            return {}
        
        # Contar por tipo
        type_counts = Counter(entity.entity_type for entity in normalized_entities.values())
        
        # Top entidades por frequência
        top_entities = sorted(
            normalized_entities.values(),
            key=lambda x: x.frequency,
            reverse=True
        )[:20]
        
        # Aliases mais comuns
        total_aliases = sum(len(entity.aliases) for entity in normalized_entities.values())
        
        return {
            'total_normalized': len(normalized_entities),
            'type_distribution': dict(type_counts),
            'total_aliases': total_aliases,
            'avg_aliases_per_entity': total_aliases / len(normalized_entities),
            'top_entities': [
                {
                    'name': entity.canonical_name,
                    'type': entity.entity_type,
                    'frequency': entity.frequency,
                    'aliases_count': len(entity.aliases)
                }
                for entity in top_entities
            ]
        }


def load_extracted_entities(file_path: str = None) -> List[EntityCandidate]:
    """
    Carrega entidades extraídas do arquivo pickle.
    
    Args:
        file_path: Caminho do arquivo (opcional)
        
    Returns:
        Lista de entidades candidatas
    """
    if file_path is None:
        file_path = Path("data/extracted_entities.pkl")
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    logger.info(f"Carregando entidades extraídas de: {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Converter para lista plana de EntityCandidate
    all_entities = []
    entities_by_chunk = data['entities_by_chunk']
    
    for chunk_id, entities in entities_by_chunk.items():
        all_entities.extend(entities)
    
    logger.info(f"✅ {len(all_entities)} entidades carregadas")
    return all_entities


def normalize_entities(entities: List[EntityCandidate] = None, 
                      model_name: str = "llama3.2:3b") -> Tuple[Dict, Dict, Dict]:
    """
    Função de conveniência para normalizar entidades.
    
    Args:
        entities: Lista de entidades (carrega automaticamente se None)
        model_name: Modelo Ollama a usar
        
    Returns:
        Tupla com (entidades_normalizadas, estatísticas, resumo)
    """
    if entities is None:
        logger.info("Carregando entidades extraídas...")
        entities = load_extracted_entities()
    
    normalizer = EntityNormalizer(model_name)
    normalized = normalizer.normalize_entities(entities)
    stats = normalizer.get_statistics()
    summary = normalizer.get_normalization_summary(normalized)
    
    return normalized, stats, summary


if __name__ == "__main__":
    # Teste do módulo
    print("🧠 Testando normalização de entidades com LLM...")
    
    try:
        # Carregar entidades extraídas
        print("\n1️⃣ Carregando entidades extraídas...")
        entities = load_extracted_entities()
        print(f"✅ {len(entities)} entidades carregadas")
        
        # Teste com amostra pequena primeiro
        sample_size = 100
        sample_entities = entities[:sample_size]
        print(f"\n2️⃣ Testando normalização com amostra de {sample_size} entidades...")
        
        normalized, stats, summary = normalize_entities(sample_entities)
        
        print(f"\n📊 Estatísticas da normalização:")
        for key, value in stats.items():
            print(f"  • {key}: {value}")
        
        print(f"\n📋 Resumo das entidades normalizadas:")
        for key, value in summary.items():
            if key != 'top_entities':
                print(f"  • {key}: {value}")
        
        print(f"\n🏆 Top entidades por frequência:")
        for entity in summary.get('top_entities', [])[:10]:
            print(f"  • {entity['name']} ({entity['type']}): {entity['frequency']} ocorrências")
        
        # Salvar resultados do teste
        output_file = Path("data/normalized_entities_sample.pkl")
        print(f"\n💾 Salvando amostra em: {output_file}")
        
        with open(output_file, 'wb') as f:
            pickle.dump({
                'normalized_entities': normalized,
                'statistics': stats,
                'summary': summary,
                'sample_size': sample_size
            }, f)
        
        print(f"✅ Teste concluído! Arquivo: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()