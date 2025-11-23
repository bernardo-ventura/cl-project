#!/usr/bin/env python3
"""
Script para processar TODOS os chunks e extrair entidades.
"""
import sys
from pathlib import Path
import pickle

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

from knowledge_graph.entity_extractor import extract_entities

def main():
    print("🚀 Iniciando extração de entidades de TODOS os chunks...")
    print("⏱️ Isso pode levar alguns minutos...")
    
    try:
        # Processar todos os chunks
        entities_by_chunk, stats, summary = extract_entities()
        
        print(f"\n🎉 EXTRAÇÃO COMPLETA CONCLUÍDA!")
        print(f"\n📊 Estatísticas finais:")
        for key, value in stats.items():
            print(f"  • {key}: {value:,}")
        
        print(f"\n📋 Resumo das entidades:")
        print(f"  • Total de entidades: {summary['total_entities']:,}")
        print(f"  • Entidades únicas: {summary['unique_entities']:,}")
        
        print(f"\n🏷️ Por tipo:")
        for label, count in sorted(summary['label_counts'].items(), key=lambda x: x[1], reverse=True):
            print(f"  • {label}: {count:,}")
        
        print(f"\n🔧 Por fonte:")
        for source, count in summary['source_counts'].items():
            print(f"  • {source}: {count:,}")
        
        print(f"\n📝 Top exemplos por categoria:")
        for label, examples in summary['examples'].items():
            if examples:
                print(f"  • {label}: {', '.join(examples[:5])}")
        
        # Salvar resultados
        output_file = Path("data/extracted_entities.pkl")
        print(f"\n💾 Salvando resultados em: {output_file}")
        
        with open(output_file, 'wb') as f:
            pickle.dump({
                'entities_by_chunk': entities_by_chunk,
                'statistics': stats,
                'summary': summary,
                'total_chunks_processed': len(entities_by_chunk)
            }, f)
        
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"✅ Resultados salvos! Arquivo: {output_file} ({file_size_mb:.1f} MB)")
        
    except Exception as e:
        print(f"❌ Erro durante processamento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()