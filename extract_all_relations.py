#!/usr/bin/env python3
"""
Script para extrair relações de TODOS os chunks com entidades.
"""
import sys
from pathlib import Path
import pickle

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent / "src"))

from knowledge_graph.relation_extractor import extract_relations

def main():
    print("🔗 Iniciando extração de relações de TODOS os chunks...")
    print("⏱️ Isso pode levar 30-45 minutos (2747 chunks com entidades)...")
    
    try:
        # Processar todos os chunks
        print(f"\n1️⃣ Iniciando extração completa de relações...")
        relations, stats, summary = extract_relations()
        
        print(f"\n🎉 EXTRAÇÃO DE RELAÇÕES COMPLETA!")
        print(f"\n📊 Estatísticas finais:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  • {key}: {value:.2f}")
            else:
                print(f"  • {key}: {value:,}")
        
        print(f"\n📋 Resumo das relações extraídas:")
        for key, value in summary.items():
            if key in ['predicate_examples', 'top_subjects', 'top_objects']:
                continue
            if isinstance(value, (int, float)):
                print(f"  • {key}: {value:,}")
            else:
                print(f"  • {key}: {len(value)} items")
        
        print(f"\n🔗 Top 10 predicados:")
        for predicate, count in list(summary.get('predicate_counts', {}).items())[:10]:
            print(f"  • {predicate:<20}: {count:>4,} relações")
        
        print(f"\n👑 Top 10 subjects (entidades que mais aparecem como subject):")
        for subject, count in summary.get('top_subjects', [])[:10]:
            print(f"  • {subject:<30}: {count:>3,} vezes")
        
        print(f"\n🎯 Top 10 objects (entidades que mais aparecem como object):")
        for obj, count in summary.get('top_objects', [])[:10]:
            print(f"  • {obj:<30}: {count:>3,} vezes")
        
        print(f"\n📝 Exemplos de relações por predicado:")
        for predicate, examples in list(summary.get('predicate_examples', {}).items())[:8]:
            print(f"  • {predicate}:")
            for example in examples:
                print(f"    - {example}")
        
        # Salvar resultados
        output_file = Path("data/extracted_relations.pkl")
        print(f"\n💾 Salvando resultados finais em: {output_file}")
        
        with open(output_file, 'wb') as f:
            pickle.dump({
                'relations': relations,
                'statistics': stats,
                'summary': summary,
                'total_chunks_processed': stats.get('chunks_processed', 0)
            }, f)
        
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"✅ Resultados salvos! Arquivo: {output_file} ({file_size_mb:.1f} MB)")
        
        # Resumo final
        print(f"\n🎯 RESUMO FINAL DA EXTRAÇÃO DE RELAÇÕES:")
        print(f"   📊 Chunks processados: {stats.get('chunks_processed', 0):,}")
        print(f"   🔗 Total de relações: {stats.get('relations_extracted', 0):,}")
        print(f"   📈 Média relações/chunk: {stats.get('avg_relations_per_chunk', 0):.2f}")
        print(f"   🤖 Chamadas LLM: {stats.get('llm_calls', 0):,}")
        print(f"   🎭 Tipos de relações: {summary.get('unique_predicates', 0)}")
        
    except Exception as e:
        print(f"❌ Erro durante processamento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()