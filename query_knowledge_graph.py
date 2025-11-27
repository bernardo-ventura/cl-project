#!/usr/bin/env python3
"""
Script para fazer consultas SPARQL no Knowledge Graph construído.
Demonstra como usar o KG para extrair informações.
"""

import sys
from pathlib import Path
from rdflib import Graph, Namespace
from rdflib.plugins.sparql import prepareQuery
import json

# Adicionar src ao path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

# Namespaces do KG
ML = Namespace("http://ml-kg.org/ontology/")
ENTITY = Namespace("http://ml-kg.org/entity/")
RELATION = Namespace("http://ml-kg.org/relation/")

def load_knowledge_graph(kg_path: str = "data/ml_kg.turtle") -> Graph:
    """Carrega o Knowledge Graph."""
    print(f"📖 Carregando Knowledge Graph de: {kg_path}")
    
    g = Graph()
    g.parse(kg_path, format="turtle")
    
    print(f"✅ KG carregado com {len(g):,} triplas")
    return g

def run_sparql_query(graph: Graph, query: str, title: str = "Consulta"):
    """Executa consulta SPARQL e exibe resultados."""
    print(f"\n🔍 {title}")
    print("-" * 50)
    
    try:
        results = graph.query(query)
        
        if len(results) == 0:
            print("❌ Nenhum resultado encontrado")
            return
        
        # Converter para lista para contar
        result_list = list(results)
        print(f"📊 Encontrados {len(result_list)} resultados:\n")
        
        for i, row in enumerate(result_list[:10], 1):  # Mostrar até 10 resultados
            values = [str(val) for val in row if val]
            print(f"{i:2d}. {' | '.join(values)}")
        
        if len(result_list) > 10:
            print(f"\n... e mais {len(result_list) - 10} resultados")
    
    except Exception as e:
        print(f"❌ Erro na consulta: {e}")

def main():
    """Executa consultas SPARQL de demonstração."""
    
    print("🔍 CONSULTAS SPARQL NO KNOWLEDGE GRAPH")
    print("=" * 50)
    
    # Carregar KG
    try:
        graph = load_knowledge_graph()
    except Exception as e:
        print(f"❌ Erro carregando KG: {e}")
        return 1
    
    # Query 1: Algoritmos mais mencionados
    query1 = """
    PREFIX ml: <http://ml-kg.org/ontology/>
    PREFIX entity: <http://ml-kg.org/entity/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?algorithm ?label ?frequency
    WHERE {
        ?algorithm a ml:algorithm .
        ?algorithm rdfs:label ?label .
        ?algorithm ml:frequency ?frequency .
    }
    ORDER BY DESC(?frequency)
    LIMIT 20
    """
    
    run_sparql_query(graph, query1, "TOP 20 ALGORITMOS POR FREQUÊNCIA")
    
    # Query 2: Relações "uses"
    query2 = """
    PREFIX ml: <http://ml-kg.org/ontology/>
    PREFIX entity: <http://ml-kg.org/entity/>
    PREFIX relation: <http://ml-kg.org/relation/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?subject_label ?object_label
    WHERE {
        ?subject relation:uses ?object .
        ?subject rdfs:label ?subject_label .
        ?object rdfs:label ?object_label .
    }
    LIMIT 15
    """
    
    run_sparql_query(graph, query2, "RELAÇÕES 'USES' (QUEM USA O QUÊ)")
    
    # Query 3: Pesquisadores mais citados
    query3 = """
    PREFIX ml: <http://ml-kg.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?person ?label ?frequency
    WHERE {
        ?person a ml:person .
        ?person rdfs:label ?label .
        ?person ml:frequency ?frequency .
    }
    ORDER BY DESC(?frequency)
    LIMIT 15
    """
    
    run_sparql_query(graph, query3, "TOP 15 PESQUISADORES POR FREQUÊNCIA")
    
    # Query 4: Conceitos fundamentais
    query4 = """
    PREFIX ml: <http://ml-kg.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?concept ?label ?frequency
    WHERE {
        ?concept a ml:concept .
        ?concept rdfs:label ?label .
        ?concept ml:frequency ?frequency .
        FILTER(?frequency > 10)
    }
    ORDER BY DESC(?frequency)
    LIMIT 20
    """
    
    run_sparql_query(graph, query4, "CONCEITOS FUNDAMENTAIS (freq > 10)")
    
    # Query 5: Relações hierárquicas (is_a)
    query5 = """
    PREFIX relation: <http://ml-kg.org/relation/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?specific ?general
    WHERE {
        ?specific_entity relation:is_a ?general_entity .
        ?specific_entity rdfs:label ?specific .
        ?general_entity rdfs:label ?general .
    }
    LIMIT 15
    """
    
    run_sparql_query(graph, query5, "HIERARQUIAS (X É UM TIPO DE Y)")
    
    # Query 6: Métricas e avaliação
    query6 = """
    PREFIX ml: <http://ml-kg.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?metric ?label ?frequency
    WHERE {
        ?metric a ml:metric .
        ?metric rdfs:label ?label .
        ?metric ml:frequency ?frequency .
    }
    ORDER BY DESC(?frequency)
    LIMIT 15
    """
    
    run_sparql_query(graph, query6, "MÉTRICAS DE AVALIAÇÃO")
    
    # Query 7: Organizações de pesquisa
    query7 = """
    PREFIX ml: <http://ml-kg.org/ontology/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?org ?label ?frequency
    WHERE {
        ?org a ml:organization .
        ?org rdfs:label ?label .
        ?org ml:frequency ?frequency .
    }
    ORDER BY DESC(?frequency)
    LIMIT 10
    """
    
    run_sparql_query(graph, query7, "ORGANIZAÇÕES/UNIVERSIDADES")
    
    # Query 8: Estatísticas gerais
    query8 = """
    PREFIX ml: <http://ml-kg.org/ontology/>
    
    SELECT ?class (COUNT(?entity) as ?count)
    WHERE {
        ?entity a ?class .
        FILTER(STRSTARTS(STR(?class), "http://ml-kg.org/ontology/"))
    }
    GROUP BY ?class
    ORDER BY DESC(?count)
    """
    
    run_sparql_query(graph, query8, "DISTRIBUIÇÃO POR CLASSES")
    
    print(f"\n🎯 RESUMO DAS CONSULTAS:")
    print(f"✅ Knowledge Graph analisado com sucesso")
    print(f"📊 Total de triplas: {len(graph):,}")
    print(f"🔍 8 consultas SPARQL executadas")
    print(f"\n💡 Para análises mais detalhadas, você pode:")
    print(f"   • Carregar o KG em Apache Jena Fuseki")
    print(f"   • Usar GraphDB ou Stardog")
    print(f"   • Criar visualizações em Gephi")
    print(f"   • Desenvolver aplicações que consomem o KG")

if __name__ == "__main__":
    main()