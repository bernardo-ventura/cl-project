"""
Response Formatter: Formatador de respostas do Knowledge Graph
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

from .query_templates import QueryType

logger = logging.getLogger(__name__)


@dataclass
class FormattedResponse:
    answer: str
    metadata: Dict[str, Any]
    raw_results: List[Dict[str, Any]]
    confidence: float


class ResponseFormatter:
    def __init__(self):
        self.formatters = {
            QueryType.WHAT_IS: self._format_what_is_response,
            QueryType.WHAT_USES: self._format_what_uses_response,
            QueryType.WHAT_IS_TYPE_OF: self._format_type_of_response,
            QueryType.WHO_CREATED: self._format_who_created_response,
            QueryType.HOW_RELATED: self._format_how_related_response,
            QueryType.LIST_BY_TYPE: self._format_list_by_type_response,
            QueryType.FIND_SIMILAR: self._format_find_similar_response,
        }
    
    def format_response(self, results: List[Dict[str, Any]], query_type: QueryType,
                       original_question: str = "", entities: List[str] = None) -> FormattedResponse:
        logger.info(f"🎨 Formatando resposta para {query_type.value}: {len(results)} resultados")
        
        if not results:
            return self._format_empty_response(query_type, original_question)
        
        formatter_func = self.formatters.get(query_type)
        if not formatter_func:
            return self._format_generic_response(results, query_type)
        
        try:
            return formatter_func(results, entities or [], original_question)
        except Exception as e:
            logger.error(f"❌ Erro na formatação: {e}")
            return self._format_error_response(str(e), results)
    
    def _format_what_is_response(self, results, entities, question):
        entity_name = entities[0] if entities else "entidade"
        entity_info = defaultdict(list)
        entity_type = None
        entity_label = None
        
        for result in results:
            if 'type' in result:
                entity_type = self._clean_uri(result['type'])
            if 'label' in result:
                entity_label = result['label']
            if 'property' in result and 'value' in result:
                try:
                    prop = self._clean_uri(result['property'])
                    value = self._clean_uri(result['value'])
                    entity_info[prop].append(value)
                except Exception as e:
                    # Skip problematic properties
                    continue
        
        answer_parts = []
        if entity_label:
            answer_parts.append(f"📋 **{entity_label}**")
        else:
            answer_parts.append(f"📋 **{entity_name.title().replace('_', ' ')}**")
        
        if entity_type:
            type_display = entity_type.replace('_', ' ').title()
            answer_parts.append(f"🏷️ **Tipo**: {type_display}")
        
        # Adiciona propriedades se houver
        if entity_info:
            answer_parts.append("📊 **Propriedades**:")
            for prop, values in entity_info.items():
                if str(prop) not in ['type', 'label']:  # Skip já mostradas (convert to str)
                    prop_display = str(prop).replace('_', ' ').title()
                    unique_values = list(set(str(v) for v in values))[:5]  # Max 5 valores
                    if unique_values:  # Only show if has values
                        values_display = ', '.join(unique_values)
                        answer_parts.append(f"   • **{prop_display}**: {values_display}")
        
        answer = '\n'.join(answer_parts)
        
        return FormattedResponse(
            answer=answer,
            metadata={'entity_type': entity_type, 'result_count': len(results)},
            raw_results=results,
            confidence=0.9 if entity_label else 0.6
        )
    
    def _format_what_uses_response(self, results, entities, question):
        entity_name = entities[0] if entities else "entidade"
        users_by_type = defaultdict(list)
        
        for result in results:
            user_label = result.get('userLabel', 'N/A')
            user_type = result.get('userType', 'unknown')
            relation = result.get('relation', '')
            
            user_type_clean = self._clean_uri(user_type)
            relation_clean = self._clean_uri(relation)
            
            users_by_type[user_type_clean].append({
                'label': user_label,
                'relation': relation_clean
            })
        
        answer_parts = []
        entity_display = entity_name.replace('_', ' ').title()
        answer_parts.append(f"🔍 **Entidades que usam {entity_display}:**")
        answer_parts.append("")
        
        total_users = 0
        for user_type, users in users_by_type.items():
            type_display = user_type.replace('_', ' ').title()
            answer_parts.append(f"📂 **{type_display}s:**")
            
            seen = set()
            unique_users = []
            for user in users:
                if user['label'] not in seen:
                    unique_users.append(user)
                    seen.add(user['label'])
            
            for user in unique_users[:10]:
                emoji = self._get_relation_emoji(user['relation'])
                answer_parts.append(f"   {emoji} {user['label']}")
                total_users += 1
        
        if total_users == 0:
            answer_parts = [f"🔍 Nenhuma entidade encontrada que use **{entity_display}** diretamente."]
        
        answer = '\n'.join(answer_parts)
        
        return FormattedResponse(
            answer=answer,
            metadata={'total_users': total_users, 'result_count': len(results)},
            raw_results=results,
            confidence=0.8 if total_users > 0 else 0.3
        )
    
    def _format_type_of_response(self, results, entities, question):
        entity_name = entities[0] if entities else "entidade"
        entity_display = entity_name.replace('_', ' ').title()
        
        parents = []
        for result in results:
            parent = result.get('parent', '')
            parent_label = result.get('parentLabel', self._clean_uri(parent))
            if parent_label and parent_label != 'N/A':
                parents.append(parent_label)
        
        unique_parents = list(set(parents))
        answer_parts = []
        
        if unique_parents:
            answer_parts.append(f"🎯 **{entity_display}** é um tipo de:")
            answer_parts.append("")
            for parent in unique_parents:
                answer_parts.append(f"   🔗 {parent}")
        else:
            answer_parts.append(f"🎯 **{entity_display}** não possui tipos parent identificados.")
        
        return FormattedResponse(
            answer='\n'.join(answer_parts),
            metadata={'parent_count': len(unique_parents), 'result_count': len(results)},
            raw_results=results,
            confidence=0.8 if unique_parents else 0.4
        )
    
    def _format_who_created_response(self, results, entities, question):
        entity_name = entities[0] if entities else "entidade"
        entity_display = entity_name.replace('_', ' ').title()
        
        creators = []
        for result in results:
            creator_label = result.get('creatorLabel', self._clean_uri(result.get('creator', '')))
            if creator_label and creator_label != 'N/A':
                creators.append(creator_label)
        
        unique_creators = list(set(creators))
        answer_parts = []
        
        if unique_creators:
            answer_parts.append(f"👤 **{entity_display}** foi criado/desenvolvido por:")
            answer_parts.append("")
            for creator in unique_creators:
                answer_parts.append(f"   📝 {creator}")
        else:
            answer_parts.append(f"👤 Criador de **{entity_display}** não identificado.")
        
        return FormattedResponse(
            answer='\n'.join(answer_parts),
            metadata={'creator_count': len(unique_creators), 'result_count': len(results)},
            raw_results=results,
            confidence=0.9 if unique_creators else 0.3
        )
    
    def _format_how_related_response(self, results, entities, question):
        if len(entities) >= 2:
            entity1 = entities[0].replace('_', ' ').title()
            entity2 = entities[1].replace('_', ' ').title()
        else:
            entity1, entity2 = "Entidade 1", "Entidade 2"
        
        relations = []
        for result in results:
            relation = self._clean_uri(result.get('relation', ''))
            if relation:
                relations.append(relation)
        
        unique_relations = list(set(relations))
        answer_parts = []
        
        if unique_relations:
            answer_parts.append(f"🔗 **{entity1}** e **{entity2}** estão relacionados através de:")
            answer_parts.append("")
            for relation in unique_relations:
                emoji = self._get_relation_emoji(relation)
                answer_parts.append(f"   {emoji} {relation.replace('_', ' ').title()}")
        else:
            answer_parts.append(f"🔗 Nenhuma relação direta encontrada entre **{entity1}** e **{entity2}**.")
        
        return FormattedResponse(
            answer='\n'.join(answer_parts),
            metadata={'relation_count': len(unique_relations), 'result_count': len(results)},
            raw_results=results,
            confidence=0.8 if unique_relations else 0.2
        )
    
    def _format_list_by_type_response(self, results, entities, question):
        entity_type = entities[0] if entities else "entidades"
        type_display = entity_type.replace('_', ' ').title()
        
        entities_list = []
        for result in results:
            label = result.get('label', 'N/A')
            entity = result.get('entity', '')
            
            if label and label != 'N/A':
                entities_list.append(label)
            elif entity:
                entities_list.append(self._clean_uri(entity).replace('_', ' ').title())
        
        unique_entities = sorted(list(set(entities_list)))
        answer_parts = []
        
        if unique_entities:
            answer_parts.append(f"📂 **{type_display}s** no Knowledge Graph:")
            answer_parts.append("")
            
            for i, entity in enumerate(unique_entities[:20], 1):
                answer_parts.append(f"   {i:2d}. {entity}")
            
            if len(unique_entities) > 20:
                answer_parts.append(f"")
                answer_parts.append(f"   ... e mais {len(unique_entities) - 20} {type_display.lower()}s")
        else:
            answer_parts.append(f"📂 Nenhum **{type_display}** encontrado.")
        
        return FormattedResponse(
            answer='\n'.join(answer_parts),
            metadata={'total_entities': len(unique_entities), 'result_count': len(results)},
            raw_results=results,
            confidence=0.9 if unique_entities else 0.3
        )
    
    def _format_find_similar_response(self, results, entities, question):
        target_entity = entities[0] if entities else "entidade"
        target_display = target_entity.replace('_', ' ').title()
        
        similar_entities = []
        for result in results:
            similar_label = result.get('similarLabel', self._clean_uri(result.get('similar', '')))
            if similar_label and similar_label != 'N/A':
                similar_entities.append(similar_label)
        
        unique_similar = list(set(similar_entities))
        answer_parts = []
        
        if unique_similar:
            answer_parts.append(f"🔍 **Entidades similares a {target_display}**:")
            answer_parts.append("")
            for i, entity in enumerate(unique_similar[:15], 1):
                answer_parts.append(f"   {i:2d}. {entity}")
        else:
            answer_parts.append(f"🔍 Nenhuma entidade similar a **{target_display}** encontrada.")
        
        return FormattedResponse(
            answer='\n'.join(answer_parts),
            metadata={'similar_count': len(unique_similar), 'result_count': len(results)},
            raw_results=results,
            confidence=0.7 if unique_similar else 0.3
        )
    
    def _format_empty_response(self, query_type, question):
        return FormattedResponse(
            answer=f"❌ Nenhum resultado encontrado para: \"{question}\"",
            metadata={'result_count': 0, 'query_type': query_type.value},
            raw_results=[],
            confidence=0.0
        )
    
    def _format_error_response(self, error_msg, results):
        return FormattedResponse(
            answer=f"⚠️ Erro ao formatar resposta: {error_msg}",
            metadata={'error': error_msg, 'result_count': len(results)},
            raw_results=results,
            confidence=0.0
        )
    
    def _format_generic_response(self, results, query_type):
        answer_parts = [f"📊 Resultados para {query_type.value}:"]
        answer_parts.append("")
        
        for i, result in enumerate(results[:10], 1):
            result_str = ', '.join([f"{k}: {v}" for k, v in result.items()])
            answer_parts.append(f"   {i}. {result_str}")
        
        return FormattedResponse(
            answer='\n'.join(answer_parts),
            metadata={'result_count': len(results), 'query_type': query_type.value},
            raw_results=results,
            confidence=0.5
        )
    
    def _clean_uri(self, uri) -> str:
        if not uri:
            return ""
        
        # Convert to string first
        uri_str = str(uri)
        
        # Remove namespace URIs
        if '#' in uri_str:
            return uri_str.split('#')[-1]
        elif '/' in uri_str:
            return uri_str.split('/')[-1]
        else:
            return uri_str
    
    def _get_relation_emoji(self, relation: str) -> str:
        emoji_map = {
            'uses': '🔧',
            'implements': '⚙️', 
            'is_a': '🏷️',
            'part_of': '🧩',
            'extends': '📈',
            'optimizes': '⚡',
            'measures': '📊',
            'developed_by': '👤',
            'proposed_by': '💡',
            'applies_to': '🎯'
        }
        return emoji_map.get(relation.lower(), '🔗')


def create_response_formatter() -> ResponseFormatter:
    return ResponseFormatter()