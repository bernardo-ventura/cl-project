#!/usr/bin/env python3
"""
Script para normalizar TODAS as entidades extraídas usando LLM.
"""
import sys
from pathlib import Path
import pickle

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

from knowledge_graph.entity_normalizer import normalize_entities, load_extracted_entities

def main():
    print("🧠 Iniciando normalização de TODAS as entidades com LLM...")
    print("⏱️ Isso pode levar 15-30 minutos...")
    
    try:
        # Carregar todas as entidades
        print("\n1️⃣ Carregando entidades extraídas...")
        entities = load_extracted_entities()
        print(f"✅ {len(entities)} entidades carregadas")
        
        # Processar todas as entidades
        print(f"\n2️⃣ Iniciando normalização completa...")
        print(f"📊 Estimativa: ~{len(entities)//20} chamadas LLM necessárias")
        
        normalized, stats, summary = normalize_entities(entities)
        
        print(f"\n🎉 NORMALIZAÇÃO COMPLETA CONCLUÍDA!")
        print(f"\n📊 Estatísticas finais:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  • {key}: {value:.1f}")
            else:
                print(f"  • {key}: {value:,}")
        
        print(f"\n📋 Resumo das entidades normalizadas:")
        for key, value in summary.items():
            if key not in ['top_entities', 'type_distribution']:
                if isinstance(value, float):
                    print(f"  • {key}: {value:.1f}")
                else:
                    print(f"  • {key}: {value:,}")
        
        print(f"\n🏷️ Distribuição por tipo:")
        for entity_type, count in sorted(summary['type_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True):
            print(f"  • {entity_type}: {count:,}")
        
        print(f"\n🏆 Top 15 entidades por frequência:")
        for i, entity in enumerate(summary.get('top_entities', [])[:15], 1):
            aliases_info = f" ({entity['aliases_count']} aliases)" if entity['aliases_count'] > 0 else ""
            print(f"  {i:2}. {entity['name']} ({entity['type']}): {entity['frequency']} ocorrências{aliases_info}")
        
        # Salvar resultados
        output_file = Path("data/normalized_entities.pkl")
        print(f"\n💾 Salvando resultados finais em: {output_file}")
        
        with open(output_file, 'wb') as f:
            pickle.dump({
                'normalized_entities': normalized,
                'statistics': stats,
                'summary': summary,
                'total_entities_processed': len(entities)
            }, f)
        
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"✅ Resultados salvos! Arquivo: {output_file} ({file_size_mb:.1f} MB)")
        
        # Resumo final
        reduction_pct = stats.get('reduction_percentage', 0)
        print(f"\n🎯 RESUMO FINAL:")
        print(f"   📥 Entidades de entrada: {len(entities):,}")
        print(f"   📤 Entidades normalizadas: {len(normalized):,}")
        print(f"   📉 Redução: {reduction_pct:.1f}%")
        print(f"   🤖 Chamadas LLM: {stats.get('llm_calls', 0):,}")
        
    except Exception as e:
        print(f"❌ Erro durante processamento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()