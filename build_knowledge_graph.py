#!/usr/bin/env python3
"""
Script para executar a construção do Knowledge Graph (Passo 6).
Converte entidades e relações em formato RDF.
"""

import sys
from pathlib import Path
import logging

# Adicionar src ao path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from knowledge_graph.kg_builder import build_knowledge_graph

def main():
    """Executa a construção do Knowledge Graph."""
    
    print("🕸️ PASSO 6: CONSTRUÇÃO DO KNOWLEDGE GRAPH")
    print("=" * 50)
    
    try:
        # Construir Knowledge Graph
        result = build_knowledge_graph(output_format='turtle')
        
        print(f"\n🎉 KNOWLEDGE GRAPH CONSTRUÍDO COM SUCESSO!")
        print(f"📊 Total de triplas RDF: {result['graph_size']:,}")
        print(f"📁 Arquivo principal: {result['output_file']}")
        print(f"📄 Relatório detalhado: {result['report_file']}")
        
        # Criar versões em outros formatos
        print(f"\n🔄 Gerando formatos adicionais...")
        
        formats = ['xml', 'n3', 'json-ld']
        for fmt in formats:
            try:
                build_knowledge_graph(output_format=fmt)
                print(f"✅ Formato {fmt} criado")
            except Exception as e:
                print(f"⚠️  Erro criando formato {fmt}: {e}")
        
        print(f"\n🎯 PIPELINE COMPLETO!")
        print(f"Você agora tem um Knowledge Graph completo em RDF.")
        print(f"Próximos passos possíveis:")
        print(f"• Carregar em Apache Jena ou GraphDB")
        print(f"• Fazer consultas SPARQL")
        print(f"• Visualizar com Gephi ou Cytoscape")
        print(f"• Comparar com abordagem RAG")
        
    except Exception as e:
        print(f"❌ Erro na construção do KG: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())