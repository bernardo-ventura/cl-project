"""
KG Executor: Motor de execução de consultas SPARQL no Knowledge Graph

Este módulo é responsável por:
1. Carregar o Knowledge Graph (ml_kg.turtle) 
2. Executar consultas SPARQL
3. Gerenciar cache para performance
4. Retornar resultados estruturados
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import rdflib
from rdflib import Graph, Namespace
from rdflib.plugins.sparql import prepareQuery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KGExecutor:
    """
    Executor de consultas SPARQL para o Knowledge Graph ML/DL
    
    Fluxo:
    1. Carrega ml_kg.turtle na inicialização
    2. Mantém grafo em memória (cache)
    3. Executa consultas SPARQL
    4. Retorna resultados estruturados
    """
    
    def __init__(self, kg_path: Optional[Path] = None):
        """
        Inicializa o executor e carrega o Knowledge Graph
        
        Args:
            kg_path: Caminho para o arquivo .turtle (padrão: data/ml_kg.turtle)
        """
        if kg_path is None:
            # Assume que está sendo executado da raiz do projeto
            self.kg_path = Path("data/ml_kg.turtle")
        else:
            self.kg_path = Path(kg_path)
            
        # Namespaces do nosso Knowledge Graph
        self.ml = Namespace("http://ml-kg.org/ontology/")
        self.entity = Namespace("http://ml-kg.org/entity/")
        self.relation = Namespace("http://ml-kg.org/relation/")
        
        # Inicializar grafo
        self.graph = None
        self._load_knowledge_graph()
    
    def _load_knowledge_graph(self) -> None:
        """
        Carrega o Knowledge Graph do arquivo Turtle
        Isso acontece uma vez na inicialização para performance
        """
        logger.info(f"🔍 Carregando Knowledge Graph de: {self.kg_path}")
        
        if not self.kg_path.exists():
            raise FileNotFoundError(f"Knowledge Graph não encontrado: {self.kg_path}")
        
        self.graph = Graph()
        
        try:
            # Bind namespaces para consultas mais legíveis
            self.graph.bind("ml", self.ml)
            self.graph.bind("entity", self.entity)
            self.graph.bind("relation", self.relation)
            
            # Carrega o grafo (pode demorar alguns segundos)
            self.graph.parse(str(self.kg_path), format="turtle")
            
            logger.info(f"✅ Knowledge Graph carregado: {len(self.graph)} triplas")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar Knowledge Graph: {e}")
            raise
    
    def execute_sparql(self, query: str) -> List[Dict[str, Any]]:
        """
        Executa uma consulta SPARQL no Knowledge Graph
        
        Args:
            query: Consulta SPARQL como string
            
        Returns:
            Lista de resultados como dicionários
            
        Example:
            query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 5"
            results = executor.execute_sparql(query)
        """
        if self.graph is None:
            raise RuntimeError("Knowledge Graph não foi carregado")
        
        logger.info(f"🔍 Executando consulta SPARQL...")
        logger.debug(f"Query: {query}")
        
        try:
            # Executa a consulta
            results = self.graph.query(query)
            
            # Converte resultados para formato estruturado
            formatted_results = []
            for row in results:
                result_dict = {}
                for var_name in results.vars:
                    value = row[var_name]
                    if value:
                        # Converte URIs para strings legíveis
                        if hasattr(value, 'toPython'):
                            result_dict[str(var_name)] = value.toPython()
                        else:
                            result_dict[str(var_name)] = str(value)
                
                formatted_results.append(result_dict)
            
            logger.info(f"✅ Consulta executada: {len(formatted_results)} resultados")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Erro na consulta SPARQL: {e}")
            raise
    
    def get_entity_info(self, entity_name: str) -> Dict[str, Any]:
        """
        Busca informações completas sobre uma entidade específica
        
        Args:
            entity_name: Nome da entidade (ex: "gradient_descent")
            
        Returns:
            Dicionário com propriedades da entidade
        """
        query = f"""
        PREFIX ml: <{self.ml}>
        PREFIX entity: <{self.entity}>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?property ?value WHERE {{
            entity:{entity_name} ?property ?value .
        }}
        """
        
        results = self.execute_sparql(query)
        return results
    
    def get_related_entities(self, entity_name: str, relation_type: str = None) -> List[Dict[str, Any]]:
        """
        Busca entidades relacionadas a uma entidade específica
        
        Args:
            entity_name: Nome da entidade central
            relation_type: Tipo de relação (opcional, ex: "uses", "is_a")
            
        Returns:
            Lista de entidades relacionadas
        """
        if relation_type:
            relation_filter = f"FILTER(?relation = relation:{relation_type})"
        else:
            relation_filter = ""
        
        query = f"""
        PREFIX ml: <{self.ml}>
        PREFIX entity: <{self.entity}>
        PREFIX relation: <{self.relation}>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT ?related ?relation ?label WHERE {{
            {{ entity:{entity_name} ?relation ?related . }}
            UNION
            {{ ?related ?relation entity:{entity_name} . }}
            
            OPTIONAL {{ ?related rdfs:label ?label . }}
            {relation_filter}
        }}
        """
        
        results = self.execute_sparql(query)
        return results
    
    def get_stats(self) -> Dict[str, int]:
        """
        Retorna estatísticas básicas do Knowledge Graph
        
        Returns:
            Dicionário com contagens de triplas, entidades, etc.
        """
        stats = {
            'total_triples': len(self.graph),
            'total_entities': 0,
            'total_relations': 0
        }
        
        # Conta entidades únicas
        entity_query = f"""
        PREFIX entity: <{self.entity}>
        SELECT (COUNT(DISTINCT ?entity) as ?count) WHERE {{
            ?entity ?p ?o .
            FILTER(STRSTARTS(STR(?entity), STR(entity:)))
        }}
        """
        
        results = self.execute_sparql(entity_query)
        if results:
            stats['total_entities'] = int(results[0]['count'])
        
        return stats


# Função utilitária para uso direto
def create_kg_executor(kg_path: Optional[Path] = None) -> KGExecutor:
    """
    Factory function para criar instância do KGExecutor
    
    Args:
        kg_path: Caminho opcional para o arquivo KG
        
    Returns:
        Instância configurada do KGExecutor
    """
    return KGExecutor(kg_path)


if __name__ == "__main__":
    # Teste básico do executor
    print("🧪 Testando KG Executor...")
    
    executor = create_kg_executor()
    stats = executor.get_stats()
    
    print(f"📊 Estatísticas do KG:")
    for key, value in stats.items():
        print(f"   • {key}: {value}")