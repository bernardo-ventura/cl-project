"""
Módulo para construção do Knowledge Graph em formato RDF.
Passo 6 do pipeline de construção do Knowledge Graph.
"""

from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal, URIRef
from rdflib.namespace import XSD, DCTERMS, FOAF
import logging
import pickle
from typing import Dict, List, Set
from pathlib import Path
import sys
import re
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """Construtor do Knowledge Graph em RDF."""
    
    def __init__(self):
        """Inicializa o construtor do KG."""
        # Criar grafo RDF
        self.graph = Graph()
        
        # Definir namespaces
        self.ML = Namespace("http://ml-kg.org/ontology/")
        self.ENTITY = Namespace("http://ml-kg.org/entity/")
        self.RELATION = Namespace("http://ml-kg.org/relation/")
        
        # Bind namespaces
        self.graph.bind("ml", self.ML)
        self.graph.bind("entity", self.ENTITY)
        self.graph.bind("relation", self.RELATION)
        self.graph.bind("rdf", RDF)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("owl", OWL)
        self.graph.bind("foaf", FOAF)
        self.graph.bind("dcterms", DCTERMS)
        
        # Estatísticas
        self.stats = {
            'entities_added': 0,
            'relations_added': 0,
            'triples_total': 0,
            'entity_types': {},
            'relation_types': {}
        }
        
        logger.info("✅ Knowledge Graph builder inicializado")
    
    def _create_entity_uri(self, entity_name: str) -> URIRef:
        """
        Cria URI única para uma entidade.
        
        Args:
            entity_name: Nome da entidade
            
        Returns:
            URIRef da entidade
        """
        # Limpar nome para URI válida
        clean_name = re.sub(r'[^\w\s-]', '', entity_name)
        clean_name = re.sub(r'\s+', '_', clean_name)
        clean_name = clean_name.lower().strip('_')
        
        return self.ENTITY[clean_name]
    
    def _create_relation_uri(self, relation_name: str) -> URIRef:
        """
        Cria URI para um tipo de relação.
        
        Args:
            relation_name: Nome da relação
            
        Returns:
            URIRef da relação
        """
        return self.RELATION[relation_name.lower()]
    
    def _add_entity_type_class(self, entity_type: str) -> URIRef:
        """
        Adiciona classe de tipo de entidade ao grafo.
        
        Args:
            entity_type: Tipo da entidade (ALGORITHM, CONCEPT, etc.)
            
        Returns:
            URIRef da classe
        """
        class_uri = self.ML[entity_type.lower().replace(' ', '_')]
        
        # Adicionar classe apenas uma vez
        if (class_uri, RDF.type, OWL.Class) not in self.graph:
            self.graph.add((class_uri, RDF.type, OWL.Class))
            self.graph.add((class_uri, RDFS.label, Literal(entity_type)))
            self.graph.add((class_uri, RDFS.subClassOf, self.ML.Entity))
        
        return class_uri
    
    def _add_relation_property(self, relation_type: str) -> URIRef:
        """
        Adiciona propriedade de relação ao grafo.
        
        Args:
            relation_type: Tipo da relação (is_a, uses, etc.)
            
        Returns:
            URIRef da propriedade
        """
        property_uri = self._create_relation_uri(relation_type)
        
        # Adicionar propriedade apenas uma vez
        if (property_uri, RDF.type, OWL.ObjectProperty) not in self.graph:
            self.graph.add((property_uri, RDF.type, OWL.ObjectProperty))
            self.graph.add((property_uri, RDFS.label, Literal(relation_type.replace('_', ' '))))
        
        return property_uri
    
    def add_ontology_schema(self):
        """Adiciona schema básico da ontologia ML/DL."""
        logger.info("Adicionando schema da ontologia...")
        
        # Classe principal Entity
        self.graph.add((self.ML.Entity, RDF.type, OWL.Class))
        self.graph.add((self.ML.Entity, RDFS.label, Literal("Machine Learning Entity")))
        
        # Classes principais
        main_classes = [
            ('Algorithm', 'Machine Learning Algorithm'),
            ('Concept', 'Machine Learning Concept'),
            ('Person', 'Person or Researcher'),
            ('Organization', 'Organization or Institution'),
            ('Software', 'Software or Tool'),
            ('Metric', 'Evaluation Metric'),
            ('Dataset', 'Dataset or Data Source'),
            ('Publication', 'Academic Publication')
        ]
        
        for class_name, description in main_classes:
            class_uri = self.ML[class_name.lower()]
            self.graph.add((class_uri, RDF.type, OWL.Class))
            self.graph.add((class_uri, RDFS.label, Literal(class_name)))
            self.graph.add((class_uri, RDFS.comment, Literal(description)))
            self.graph.add((class_uri, RDFS.subClassOf, self.ML.Entity))
        
        # Propriedades principais
        main_properties = [
            ('is_a', 'is a type of'),
            ('part_of', 'is part of'),
            ('uses', 'uses or utilizes'),
            ('implements', 'implements'),
            ('optimizes', 'optimizes'),
            ('measures', 'measures or evaluates'),
            ('created_by', 'was created by'),
            ('applies_to', 'applies to')
        ]
        
        for prop_name, description in main_properties:
            prop_uri = self._create_relation_uri(prop_name)
            self.graph.add((prop_uri, RDF.type, OWL.ObjectProperty))
            self.graph.add((prop_uri, RDFS.label, Literal(prop_name.replace('_', ' '))))
            self.graph.add((prop_uri, RDFS.comment, Literal(description)))
        
        logger.info("✅ Schema da ontologia adicionado")
    
    def add_entities(self, normalized_entities: Dict):
        """
        Adiciona entidades normalizadas ao grafo.
        
        Args:
            normalized_entities: Dicionário de entidades normalizadas
        """
        logger.info(f"Adicionando {len(normalized_entities)} entidades ao grafo...")
        
        for entity_name, entity_data in normalized_entities.items():
            try:
                # Criar URI da entidade
                entity_uri = self._create_entity_uri(entity_name)
                
                # Adicionar classe do tipo
                entity_type = entity_data.entity_type
                class_uri = self._add_entity_type_class(entity_type)
                
                # Triplas básicas da entidade
                self.graph.add((entity_uri, RDF.type, class_uri))
                self.graph.add((entity_uri, RDFS.label, Literal(entity_name)))
                self.graph.add((entity_uri, self.ML.canonicalName, Literal(entity_name)))
                
                # Aliases
                for alias in entity_data.aliases:
                    if alias and alias != entity_name:
                        self.graph.add((entity_uri, self.ML.alias, Literal(alias)))
                
                # Metadados
                self.graph.add((entity_uri, self.ML.frequency, Literal(entity_data.frequency, datatype=XSD.integer)))
                self.graph.add((entity_uri, self.ML.confidence, Literal(entity_data.confidence, datatype=XSD.float)))
                
                # Source chunks (proveniência)
                for chunk_id in entity_data.source_chunks[:5]:  # Limitar para não sobrecarregar
                    self.graph.add((entity_uri, self.ML.sourceChunk, Literal(chunk_id)))
                
                self.stats['entities_added'] += 1
                self.stats['entity_types'][entity_type] = self.stats['entity_types'].get(entity_type, 0) + 1
                
            except Exception as e:
                logger.error(f"Erro adicionando entidade {entity_name}: {e}")
                continue
        
        logger.info(f"✅ {self.stats['entities_added']} entidades adicionadas")
    
    def add_relations(self, relations: List):
        """
        Adiciona relações ao grafo.
        
        Args:
            relations: Lista de objetos Relation
        """
        logger.info(f"Adicionando {len(relations)} relações ao grafo...")
        
        for relation in relations:
            try:
                # Criar URIs
                subject_uri = self._create_entity_uri(relation.subject)
                object_uri = self._create_entity_uri(relation.object)
                predicate_uri = self._add_relation_property(relation.predicate)
                
                # Adicionar tripla principal
                self.graph.add((subject_uri, predicate_uri, object_uri))
                
                # Criar URI para a instância da relação (para metadados)
                relation_instance_uri = self.ENTITY[f"rel_{self.stats['relations_added']}"]
                self.graph.add((relation_instance_uri, RDF.type, self.ML.Relation))
                self.graph.add((relation_instance_uri, self.ML.subject, subject_uri))
                self.graph.add((relation_instance_uri, self.ML.predicate, predicate_uri))
                self.graph.add((relation_instance_uri, self.ML.object, object_uri))
                self.graph.add((relation_instance_uri, self.ML.context, Literal(relation.context)))
                self.graph.add((relation_instance_uri, self.ML.sourceChunk, Literal(relation.chunk_id)))
                self.graph.add((relation_instance_uri, self.ML.confidence, Literal(relation.confidence, datatype=XSD.float)))
                
                self.stats['relations_added'] += 1
                self.stats['relation_types'][relation.predicate] = self.stats['relation_types'].get(relation.predicate, 0) + 1
                
            except Exception as e:
                logger.error(f"Erro adicionando relação {relation.subject} {relation.predicate} {relation.object}: {e}")
                continue
        
        logger.info(f"✅ {self.stats['relations_added']} relações adicionadas")
    
    def add_metadata(self):
        """Adiciona metadados sobre o KG."""
        logger.info("Adicionando metadados do Knowledge Graph...")
        
        # URI do grafo
        kg_uri = self.ML.MLKnowledgeGraph
        
        # Metadados básicos
        self.graph.add((kg_uri, RDF.type, self.ML.KnowledgeGraph))
        self.graph.add((kg_uri, RDFS.label, Literal("Machine Learning Knowledge Graph")))
        self.graph.add((kg_uri, DCTERMS.title, Literal("ML/DL Knowledge Graph from Academic Literature")))
        self.graph.add((kg_uri, DCTERMS.description, Literal("Knowledge graph extracted from machine learning and deep learning academic texts")))
        self.graph.add((kg_uri, DCTERMS.created, Literal(datetime.now().isoformat(), datatype=XSD.dateTime)))
        
        # Estatísticas
        self.graph.add((kg_uri, self.ML.totalEntities, Literal(self.stats['entities_added'], datatype=XSD.integer)))
        self.graph.add((kg_uri, self.ML.totalRelations, Literal(self.stats['relations_added'], datatype=XSD.integer)))
        self.graph.add((kg_uri, self.ML.totalTriples, Literal(len(self.graph), datatype=XSD.integer)))
        
        logger.info("✅ Metadados adicionados")
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas do grafo construído."""
        self.stats['triples_total'] = len(self.graph)
        return self.stats.copy()
    
    def save_graph(self, output_path: str = None, format: str = 'turtle'):
        """
        Salva o grafo em arquivo.
        
        Args:
            output_path: Caminho do arquivo (opcional)
            format: Formato de saída ('turtle', 'xml', 'nt', 'json-ld')
        """
        if output_path is None:
            output_path = f"data/ml_kg.{format}"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Salvando Knowledge Graph em: {output_path}")
        
        try:
            self.graph.serialize(destination=str(output_path), format=format)
            
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(f"✅ Knowledge Graph salvo! Tamanho: {file_size_mb:.1f} MB")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Erro salvando grafo: {e}")
            raise
    
    def generate_summary_report(self) -> str:
        """Gera relatório resumo do KG construído."""
        report = f"""
🕸️ KNOWLEDGE GRAPH - RELATÓRIO FINAL
{'=' * 50}

📊 ESTATÍSTICAS GERAIS:
   • Total de triplas RDF: {len(self.graph):,}
   • Entidades adicionadas: {self.stats['entities_added']:,}
   • Relações adicionadas: {self.stats['relations_added']:,}

📚 DISTRIBUIÇÃO DE ENTIDADES:
"""
        for entity_type, count in sorted(self.stats['entity_types'].items(), key=lambda x: x[1], reverse=True):
            report += f"   • {entity_type}: {count:,}\n"
        
        report += f"\n🔗 DISTRIBUIÇÃO DE RELAÇÕES:\n"
        for relation_type, count in sorted(self.stats['relation_types'].items(), key=lambda x: x[1], reverse=True)[:10]:
            report += f"   • {relation_type}: {count:,}\n"
        
        report += f"\n🎯 NAMESPACES UTILIZADOS:\n"
        for prefix, namespace in self.graph.namespaces():
            report += f"   • {prefix}: {namespace}\n"
        
        report += f"\n✅ Knowledge Graph construído com sucesso!"
        
        return report


def load_normalized_entities(file_path: str = "data/normalized_entities.pkl") -> Dict:
    """Carrega entidades normalizadas."""
    logger.info(f"Carregando entidades normalizadas de: {file_path}")
    
    # Importar classes necessárias
    sys.path.append(str(Path(__file__).parent.parent))
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['normalized_entities']


def load_extracted_relations(file_path: str = "data/extracted_relations.pkl") -> List:
    """Carrega relações extraídas."""
    logger.info(f"Carregando relações extraídas de: {file_path}")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['relations']


def build_knowledge_graph(output_format: str = 'turtle') -> Dict:
    """
    Função principal para construir o Knowledge Graph.
    
    Args:
        output_format: Formato de saída ('turtle', 'xml', 'nt', 'json-ld')
        
    Returns:
        Dicionário com estatísticas e caminhos dos arquivos
    """
    logger.info("🚀 Iniciando construção do Knowledge Graph...")
    
    try:
        # Inicializar builder
        builder = KnowledgeGraphBuilder()
        
        # Adicionar schema da ontologia
        builder.add_ontology_schema()
        
        # Carregar dados
        normalized_entities = load_normalized_entities()
        relations = load_extracted_relations()
        
        logger.info(f"Dados carregados: {len(normalized_entities)} entidades, {len(relations)} relações")
        
        # Construir grafo
        builder.add_entities(normalized_entities)
        builder.add_relations(relations)
        builder.add_metadata()
        
        # Salvar grafo
        output_file = builder.save_graph(format=output_format)
        
        # Estatísticas
        stats = builder.get_statistics()
        
        # Relatório
        report = builder.generate_summary_report()
        print(report)
        
        # Salvar relatório
        report_file = Path("data/kg_construction_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        return {
            'statistics': stats,
            'output_file': str(output_file),
            'report_file': str(report_file),
            'graph_size': len(builder.graph)
        }
        
    except Exception as e:
        logger.error(f"❌ Erro construindo Knowledge Graph: {e}")
        raise


if __name__ == "__main__":
    # Teste do módulo
    print("🕸️ Testando construção do Knowledge Graph...")
    
    try:
        result = build_knowledge_graph()
        
        print(f"\n✅ CONSTRUÇÃO CONCLUÍDA!")
        print(f"📊 Triplas RDF: {result['graph_size']:,}")
        print(f"📁 Arquivo: {result['output_file']}")
        print(f"📄 Relatório: {result['report_file']}")
        
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()