"""
Módulo para extração de relações entre entidades usando LLM local (Ollama).
Passo 5 do pipeline de construção do Knowledge Graph.
"""

import ollama
import logging
import json
import pickle
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import sys
import re
from collections import defaultdict

# Adicionar src ao path para imports
sys.path.append(str(Path(__file__).parent.parent))
from knowledge_graph.chunk_loader import load_chunks, TextChunk

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Relation:
    """Representa uma relação extraída entre duas entidades."""
    subject: str
    predicate: str
    object: str
    chunk_id: str
    confidence: float
    context: str  # Frase/contexto onde a relação foi encontrada

class RelationExtractor:
    """Extrator de relações entre entidades usando LLM."""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Inicializa o extrator de relações.
        
        Args:
            model_name: Nome do modelo Ollama a usar
        """
        self.model_name = model_name
        self.relations: List[Relation] = []
        
        # Teste de conectividade
        try:
            response = ollama.chat(model=model_name, messages=[
                {'role': 'user', 'content': 'Hello'}
            ])
            logger.info(f"✅ Conectado ao modelo {model_name}")
        except Exception as e:
            logger.error(f"❌ Erro conectando ao Ollama: {e}")
            raise
        
        # Esquema de relações ML/DL
        self.relation_schema = {
            # Relações hierárquicas
            'is_a': 'X is a type of Y',
            'part_of': 'X is part of Y',
            'subclass_of': 'X is a subclass of Y',
            
            # Relações funcionais
            'uses': 'X uses Y',
            'implements': 'X implements Y',
            'optimizes': 'X optimizes Y',
            'applies_to': 'X applies to Y',
            'solves': 'X solves Y',
            
            # Relações de dependência
            'requires': 'X requires Y',
            'depends_on': 'X depends on Y',
            'based_on': 'X is based on Y',
            'extends': 'X extends Y',
            
            # Relações de performance/comparação
            'outperforms': 'X outperforms Y',
            'compared_to': 'X compared to Y',
            'equivalent_to': 'X is equivalent to Y',
            
            # Relações temporais
            'precedes': 'X precedes Y',
            'evolved_from': 'X evolved from Y',
            
            # Relações específicas ML
            'trained_on': 'X is trained on Y',
            'evaluated_on': 'X is evaluated on Y',
            'measures': 'X measures Y',
            'predicts': 'X predicts Y',
            
            # Relações de autoria
            'created_by': 'X was created by Y',
            'proposed_by': 'X was proposed by Y',
            'developed_by': 'X was developed by Y'
        }
        
        # Estatísticas
        self.stats = {
            'chunks_processed': 0,
            'relations_extracted': 0,
            'llm_calls': 0,
            'failed_extractions': 0
        }
    
    def _create_relation_extraction_prompt(self, chunk_text: str, entities: List[str]) -> str:
        """
        Cria prompt para extração de relações.
        
        Args:
            chunk_text: Texto do chunk
            entities: Lista de entidades encontradas no chunk
            
        Returns:
            Prompt formatado para o LLM
        """
        entities_text = '\n'.join(f"- {entity}" for entity in entities)
        relations_text = '\n'.join(f"- {rel}: {desc}" for rel, desc in self.relation_schema.items())
        
        prompt = f"""You are an expert in Machine Learning and Deep Learning. Extract semantic relations between entities from the given text.

TEXT:
{chunk_text[:1500]}  # Limitar tamanho do contexto

ENTITIES FOUND:
{entities_text}

RELATION TYPES TO EXTRACT:
{relations_text}

TASK:
1. Find semantic relationships between the entities in the text
2. Use ONLY the relation types listed above
3. Extract relations that are explicitly or clearly implied in the text
4. Include the specific sentence/phrase that supports each relation

RESPONSE FORMAT (JSON):
{{
  "relations": [
    {{
      "subject": "Neural Network",
      "predicate": "uses", 
      "object": "Gradient Descent",
      "context": "Neural networks are trained using gradient descent optimization"
    }},
    {{
      "subject": "Support Vector Machine",
      "predicate": "solves",
      "object": "Classification Problem", 
      "context": "SVM is effective for classification tasks"
    }}
  ]
}}

Important: 
- Only extract relations that are clearly supported by the text
- Use entity names EXACTLY as they appear in the entities list
- Include meaningful context for each relation
- Return valid JSON only"""

        return prompt
    
    def _parse_relations_response(self, response: str) -> List[Dict]:
        """
        Parse a resposta JSON do LLM para relações.
        
        Args:
            response: Resposta do LLM
            
        Returns:
            Lista de relações extraídas
        """
        try:
            # Remover markdown code blocks se existirem
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*$', '', response)
            
            # Tentar extrair JSON da resposta
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("Nenhum JSON encontrado na resposta de relações")
                return []
            
            json_text = response[json_start:json_end]
            
            # Limpar problemas comuns de JSON
            json_text = json_text.replace('\n', ' ').replace('\r', '')
            json_text = re.sub(r',\s*}', '}', json_text)
            json_text = re.sub(r',\s*]', ']', json_text)
            
            data = json.loads(json_text)
            
            return data.get('relations', [])
            
        except json.JSONDecodeError as e:
            logger.warning(f"Erro parsing JSON de relações: {e}")
            # Parser manual simplificado
            try:
                relations = []
                lines = response.split('\n')
                current_relation = {}
                
                for line in lines:
                    if 'subject' in line:
                        subject = re.search(r'"subject":\s*"([^"]+)"', line)
                        if subject:
                            current_relation['subject'] = subject.group(1)
                    elif 'predicate' in line and 'subject' in current_relation:
                        predicate = re.search(r'"predicate":\s*"([^"]+)"', line)
                        if predicate:
                            current_relation['predicate'] = predicate.group(1)
                    elif 'object' in line and 'predicate' in current_relation:
                        obj = re.search(r'"object":\s*"([^"]+)"', line)
                        if obj:
                            current_relation['object'] = obj.group(1)
                            current_relation['context'] = "Extracted from text"
                            relations.append(current_relation.copy())
                            current_relation = {}
                
                if relations:
                    logger.info(f"Recuperadas {len(relations)} relações com parser manual")
                    return relations
                    
            except Exception:
                pass
                
            return []
        except Exception as e:
            logger.error(f"Erro inesperado parsing relações: {e}")
            return []
    
    def _filter_valid_relations(self, relations: List[Dict], entities: List[str]) -> List[Dict]:
        """
        Filtra relações válidas baseado nas entidades disponíveis.
        
        Args:
            relations: Lista de relações brutas
            entities: Lista de entidades válidas
            
        Returns:
            Lista de relações filtradas
        """
        valid_relations = []
        entities_lower = [e.lower() for e in entities]
        
        for relation in relations:
            try:
                subject = relation.get('subject', '').strip()
                predicate = relation.get('predicate', '').strip()
                obj = relation.get('object', '').strip()
                context = relation.get('context', '').strip()
                
                # Verificar se os campos essenciais existem
                if not (subject and predicate and obj):
                    continue
                
                # Verificar se subject e object estão na lista de entidades (case insensitive)
                subject_valid = any(subject.lower() in entity.lower() or entity.lower() in subject.lower() 
                                  for entity in entities)
                object_valid = any(obj.lower() in entity.lower() or entity.lower() in obj.lower() 
                                 for entity in entities)
                
                # Verificar se predicado está no esquema
                predicate_valid = predicate.lower() in [p.lower() for p in self.relation_schema.keys()]
                
                if subject_valid and object_valid and predicate_valid:
                    # Encontrar entidades exatas
                    subject_exact = next((e for e in entities if subject.lower() in e.lower() or e.lower() in subject.lower()), subject)
                    object_exact = next((e for e in entities if obj.lower() in e.lower() or e.lower() in obj.lower()), obj)
                    predicate_exact = next((p for p in self.relation_schema.keys() if p.lower() == predicate.lower()), predicate)
                    
                    relation['subject'] = subject_exact
                    relation['predicate'] = predicate_exact
                    relation['object'] = object_exact
                    relation['context'] = context or "Extracted from text"
                    
                    valid_relations.append(relation)
                
            except Exception as e:
                logger.warning(f"Erro validando relação: {e}")
                continue
        
        return valid_relations
    
    def extract_relations_from_chunk(self, chunk: TextChunk, entities_in_chunk: List[str]) -> List[Relation]:
        """
        Extrai relações de um chunk específico.
        
        Args:
            chunk: Chunk de texto para processar
            entities_in_chunk: Lista de entidades encontradas neste chunk
            
        Returns:
            Lista de relações extraídas
        """
        if len(entities_in_chunk) < 2:
            # Não há entidades suficientes para formar relações
            return []
        
        relations = []
        
        try:
            # Criar prompt
            prompt = self._create_relation_extraction_prompt(chunk.content, entities_in_chunk)
            
            # Chamar LLM
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            self.stats['llm_calls'] += 1
            
            # Parse da resposta
            raw_relations = self._parse_relations_response(response['message']['content'])
            
            # Filtrar relações válidas
            valid_relations = self._filter_valid_relations(raw_relations, entities_in_chunk)
            
            # Converter para objetos Relation
            for rel_data in valid_relations:
                relation = Relation(
                    subject=rel_data['subject'],
                    predicate=rel_data['predicate'],
                    object=rel_data['object'],
                    chunk_id=chunk.chunk_id,
                    confidence=1.0,  # Simplificado
                    context=rel_data['context']
                )
                relations.append(relation)
            
            self.stats['relations_extracted'] += len(relations)
            
        except Exception as e:
            logger.error(f"Erro extraindo relações do chunk {chunk.chunk_id}: {e}")
            self.stats['failed_extractions'] += 1
        
        self.stats['chunks_processed'] += 1
        return relations
    
    def extract_relations_from_chunks(self, chunks_entities: Dict[str, List[str]], 
                                    max_chunks: int = None) -> List[Relation]:
        """
        Extrai relações de múltiplos chunks.
        
        Args:
            chunks_entities: Dicionário {chunk_id: [entidades]}
            max_chunks: Limite de chunks para teste (opcional)
            
        Returns:
            Lista de todas as relações extraídas
        """
        logger.info(f"Iniciando extração de relações de {len(chunks_entities)} chunks...")
        
        # Carregar chunks
        chunks, _ = load_chunks()
        chunks_dict = {chunk.chunk_id: chunk for chunk in chunks}
        
        # Limitar para teste se necessário
        chunk_ids = list(chunks_entities.keys())
        if max_chunks:
            chunk_ids = chunk_ids[:max_chunks]
            logger.info(f"Limitando processamento a {max_chunks} chunks para teste")
        
        all_relations = []
        
        for i, chunk_id in enumerate(chunk_ids):
            if (i + 1) % 50 == 0:
                logger.info(f"Processado {i + 1}/{len(chunk_ids)} chunks...")
            
            chunk = chunks_dict.get(chunk_id)
            entities = chunks_entities.get(chunk_id, [])
            
            if not chunk or len(entities) < 2:
                continue
            
            chunk_relations = self.extract_relations_from_chunk(chunk, entities)
            all_relations.extend(chunk_relations)
        
        logger.info(f"✅ Extração de relações concluída!")
        logger.info(f"Total de relações extraídas: {len(all_relations)}")
        
        return all_relations
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas da extração de relações."""
        avg_relations_per_chunk = 0
        if self.stats['chunks_processed'] > 0:
            avg_relations_per_chunk = self.stats['relations_extracted'] / self.stats['chunks_processed']
        
        return {
            **self.stats,
            'avg_relations_per_chunk': avg_relations_per_chunk
        }
    
    def get_relations_summary(self, relations: List[Relation]) -> Dict:
        """
        Gera resumo das relações extraídas.
        
        Args:
            relations: Lista de relações
            
        Returns:
            Resumo com estatísticas
        """
        if not relations:
            return {}
        
        # Contar por predicado
        predicate_counts = defaultdict(int)
        subject_counts = defaultdict(int)
        object_counts = defaultdict(int)
        
        for relation in relations:
            predicate_counts[relation.predicate] += 1
            subject_counts[relation.subject] += 1
            object_counts[relation.object] += 1
        
        # Top predicados
        top_predicates = sorted(predicate_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Top entidades como subject
        top_subjects = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Top entidades como object
        top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Exemplo de relações por predicado
        predicate_examples = {}
        for predicate, count in top_predicates[:10]:
            examples = [r for r in relations if r.predicate == predicate][:3]
            predicate_examples[predicate] = [
                f"{ex.subject} {ex.predicate} {ex.object}"
                for ex in examples
            ]
        
        return {
            'total_relations': len(relations),
            'unique_predicates': len(predicate_counts),
            'unique_subjects': len(subject_counts),
            'unique_objects': len(object_counts),
            'predicate_counts': dict(top_predicates),
            'top_subjects': top_subjects,
            'top_objects': top_objects,
            'predicate_examples': predicate_examples
        }


def load_normalized_entities() -> Dict:
    """
    Carrega entidades normalizadas do arquivo pickle.
    
    Returns:
        Dicionário com entidades normalizadas
    """
    file_path = Path("data/normalized_entities.pkl")
    
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo de entidades normalizadas não encontrado: {file_path}")
    
    logger.info(f"Carregando entidades normalizadas de: {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['normalized_entities']


def map_entities_to_chunks(normalized_entities: Dict) -> Dict[str, List[str]]:
    """
    Mapeia entidades normalizadas para seus chunks de origem.
    
    Args:
        normalized_entities: Dicionário de entidades normalizadas
        
    Returns:
        Dicionário {chunk_id: [entidades_no_chunk]}
    """
    chunks_entities = defaultdict(list)
    
    for entity_name, entity_data in normalized_entities.items():
        for chunk_id in entity_data.source_chunks:
            chunks_entities[chunk_id].append(entity_name)
    
    # Filtrar chunks com pelo menos 2 entidades
    chunks_entities = {k: v for k, v in chunks_entities.items() if len(v) >= 2}
    
    logger.info(f"Mapeados {len(chunks_entities)} chunks com 2+ entidades")
    return dict(chunks_entities)


def extract_relations(max_chunks: int = None) -> Tuple[List[Relation], Dict, Dict]:
    """
    Função de conveniência para extrair relações.
    
    Args:
        max_chunks: Limite de chunks para teste
        
    Returns:
        Tupla com (relações, estatísticas, resumo)
    """
    # Carregar entidades normalizadas
    logger.info("Carregando entidades normalizadas...")
    normalized_entities = load_normalized_entities()
    
    # Mapear entidades para chunks
    chunks_entities = map_entities_to_chunks(normalized_entities)
    
    # Extrair relações
    extractor = RelationExtractor()
    relations = extractor.extract_relations_from_chunks(chunks_entities, max_chunks)
    stats = extractor.get_statistics()
    summary = extractor.get_relations_summary(relations)
    
    return relations, stats, summary


if __name__ == "__main__":
    # Teste do módulo
    print("🔗 Testando extração de relações...")
    
    try:
        # Teste com amostra pequena primeiro
        print("\n1️⃣ Testando com 10 chunks...")
        relations, stats, summary = extract_relations(max_chunks=10)
        
        print(f"\n📊 Estatísticas da extração:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  • {key}: {value:.2f}")
            else:
                print(f"  • {key}: {value}")
        
        print(f"\n📋 Resumo das relações:")
        for key, value in summary.items():
            if key in ['predicate_examples', 'top_subjects', 'top_objects']:
                continue
            print(f"  • {key}: {value}")
        
        print(f"\n🔗 Top predicados:")
        for predicate, count in list(summary.get('predicate_counts', {}).items())[:10]:
            print(f"  • {predicate}: {count}")
        
        print(f"\n📝 Exemplos de relações:")
        for predicate, examples in list(summary.get('predicate_examples', {}).items())[:5]:
            print(f"  • {predicate}:")
            for example in examples:
                print(f"    - {example}")
        
        # Salvar resultados do teste
        output_file = Path("data/relations_sample.pkl")
        print(f"\n💾 Salvando amostra em: {output_file}")
        
        with open(output_file, 'wb') as f:
            pickle.dump({
                'relations': relations,
                'statistics': stats,
                'summary': summary,
                'sample_size': 10
            }, f)
        
        print(f"✅ Teste concluído! Arquivo: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()