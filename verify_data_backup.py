#!/usr/bin/env python3
"""
Script para verificar integridade e backup de todos os dados processados.
"""
import pickle
from pathlib import Path
import json
from datetime import datetime

def check_data_integrity():
    """Verifica integridade de todos os arquivos de dados."""
    
    print("🔍 VERIFICAÇÃO DE INTEGRIDADE DOS DADOS")
    print("=" * 50)
    
    data_dir = Path("data")
    backup_info = []
    
    # Lista de arquivos esperados
    expected_files = [
        ("extracted_entities.pkl", "Entidades brutas extraídas pelo spaCy"),
        ("normalized_entities.pkl", "Entidades normalizadas pelo LLM"), 
        ("extracted_relations.pkl", "Relações extraídas pelo LLM"),
        ("processed_chunks.pkl", "Chunks de texto estruturados")
    ]
    
    print(f"\n📂 Diretório de dados: {data_dir.absolute()}")
    
    for filename, description in expected_files:
        filepath = data_dir / filename
        
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1024 / 1024
            mod_time = datetime.fromtimestamp(filepath.stat().st_mtime)
            
            print(f"\n✅ {filename}")
            print(f"   📄 Descrição: {description}")
            print(f"   📊 Tamanho: {size_mb:.1f} MB")
            print(f"   🕐 Modificado: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Tentar carregar para verificar integridade
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    
                if 'statistics' in data:
                    stats = data['statistics']
                    print(f"   🔢 Estatísticas: {stats}")
                elif 'summary' in data:
                    summary = data['summary']
                    print(f"   📋 Items: {len(data.get('normalized_entities', {}))}")
                elif isinstance(data, dict) and 'entities_by_chunk' in data:
                    print(f"   📈 Entidades: {data['summary']['total_entities']:,}")
                elif isinstance(data, dict) and 'chunks' in data:
                    print(f"   📝 Chunks: {len(data['chunks']):,}")
                
                print(f"   ✅ Arquivo íntegro e carregável")
                
                backup_info.append({
                    'file': filename,
                    'size_mb': round(size_mb, 1),
                    'modified': mod_time.isoformat(),
                    'status': 'OK',
                    'description': description
                })
                
            except Exception as e:
                print(f"   ❌ ERRO ao carregar: {e}")
                backup_info.append({
                    'file': filename,
                    'size_mb': round(size_mb, 1),
                    'modified': mod_time.isoformat(),
                    'status': f'ERROR: {e}',
                    'description': description
                })
        else:
            print(f"\n❌ {filename} - ARQUIVO NÃO ENCONTRADO!")
            backup_info.append({
                'file': filename,
                'status': 'MISSING',
                'description': description
            })
    
    # Salvar relatório de backup
    backup_report = {
        'verification_time': datetime.now().isoformat(),
        'total_files': len(expected_files),
        'files_found': len([info for info in backup_info if info['status'] == 'OK']),
        'total_size_mb': sum(info.get('size_mb', 0) for info in backup_info if 'size_mb' in info),
        'files': backup_info
    }
    
    backup_file = Path("backup_verification_report.json")
    with open(backup_file, 'w') as f:
        json.dump(backup_report, f, indent=2)
    
    print(f"\n📋 RESUMO FINAL:")
    print(f"   📁 Total de arquivos esperados: {len(expected_files)}")
    print(f"   ✅ Arquivos encontrados e íntegros: {backup_report['files_found']}")
    print(f"   💾 Tamanho total dos dados: {backup_report['total_size_mb']:.1f} MB")
    print(f"   📄 Relatório salvo em: {backup_file}")
    
    if backup_report['files_found'] == len(expected_files):
        print(f"\n🎉 TODOS OS DADOS ESTÃO SEGUROS! ✅")
        print(f"   Você pode prosseguir sem preocupações.")
    else:
        print(f"\n⚠️ ATENÇÃO: Alguns arquivos estão faltando ou corrompidos!")
    
    return backup_report

if __name__ == "__main__":
    report = check_data_integrity()