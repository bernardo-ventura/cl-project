#!/usr/bin/env python3
"""
Script para dividir textos em chunks inteligentes
"""

import os
import re
from pathlib import Path
import time

def simple_sentence_split(text):
    """Divisão simples de sentenças usando regex"""
    # Padrão para detectar fim de sentenças
    sentence_endings = r'[.!?]+(?:\s+|$)'
    sentences = re.split(sentence_endings, text)
    # Filtrar sentenças vazias e muito pequenas
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return sentences

def create_chunks(text, target_words=350, max_words=500):
    """
    Cria chunks inteligentes baseados em sentenças
    
    Args:
        text: texto completo
        target_words: tamanho alvo do chunk em palavras
        max_words: tamanho máximo do chunk
    
    Returns:
        lista de chunks
    """
    # Dividir em sentenças
    sentences = simple_sentence_split(text)
    
    chunks = []
    current_chunk = []
    current_words = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # Se a sentença sozinha já ultrapassa o máximo, dividir ela
        if sentence_words > max_words:
            # Salvar chunk atual se não estiver vazio
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_words = 0
            
            # Dividir sentença longa em partes menores
            words = sentence.split()
            for i in range(0, len(words), target_words):
                chunk_part = ' '.join(words[i:i + target_words])
                chunks.append(chunk_part)
            continue
        
        # Se adicionar esta sentença ultrapassaria o limite, finalizar chunk atual
        if current_words + sentence_words > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_words = sentence_words
        else:
            current_chunk.append(sentence)
            current_words += sentence_words
        
        # Se chegou ao tamanho alvo, finalizar chunk
        if current_words >= target_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_words = 0
    
    # Adicionar último chunk se não estiver vazio
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def process_text_file(file_path, output_dir, target_words=350):
    """Processa um arquivo de texto e cria chunks"""
    filename = os.path.basename(file_path)
    print(f"\n📖 Processando: {filename}")
    
    # Ler arquivo
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    original_words = len(text.split())
    print(f"   📊 Texto original: {len(text):,} chars, {original_words:,} palavras")
    
    # Criar chunks
    start_time = time.time()
    chunks = create_chunks(text, target_words=target_words)
    elapsed = time.time() - start_time
    
    print(f"   ✂️  Criados {len(chunks)} chunks em {elapsed:.1f}s")
    
    # Salvar chunks
    base_name = Path(file_path).stem
    chunks_file = output_dir / f"{base_name}_chunks.txt"
    
    with open(chunks_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks, 1):
            f.write(f"=== CHUNK {i:03d} ===\n")
            f.write(f"Palavras: {len(chunk.split())}\n")
            f.write(f"Caracteres: {len(chunk)}\n")
            f.write("-" * 50 + "\n")
            f.write(chunk)
            f.write("\n\n" + "="*60 + "\n\n")
    
    # Estatísticas dos chunks
    chunk_sizes = [len(chunk.split()) for chunk in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes)
    min_size = min(chunk_sizes)
    max_size = max(chunk_sizes)
    
    print(f"   💾 Salvo: {chunks_file.name}")
    print(f"   📈 Stats: {avg_size:.0f} palavras/chunk (min:{min_size}, max:{max_size})")
    
    return len(chunks), sum(chunk_sizes)

def main():
    # Diretórios
    input_dir = Path("data/processed_texts")
    output_dir = Path("data/processed_texts/chunks")
    output_dir.mkdir(exist_ok=True)
    
    print("=== CHUNKING DOS TEXTOS ===")
    
    # Listar arquivos .txt
    txt_files = list(input_dir.glob("*.txt"))
    
    if not txt_files:
        print("❌ Nenhum arquivo .txt encontrado em data/processed_texts/")
        return
    
    print(f"📁 Encontrados {len(txt_files)} arquivos de texto")
    
    total_chunks = 0
    total_words = 0
    start_total = time.time()
    
    for i, txt_file in enumerate(txt_files, 1):
        print(f"\n[{i}/{len(txt_files)}]", end="")
        try:
            chunks_count, words_count = process_text_file(txt_file, output_dir)
            total_chunks += chunks_count
            total_words += words_count
        except Exception as e:
            print(f"   ❌ Erro: {e}")
    
    elapsed_total = time.time() - start_total
    
    print(f"\n" + "="*50)
    print(f"=== RESUMO FINAL ===")
    print(f"✅ {len(txt_files)} arquivos processados")
    print(f"📊 Total: {total_chunks:,} chunks criados")
    print(f"📝 Total: {total_words:,} palavras processadas")
    print(f"⏱️  Tempo total: {elapsed_total:.1f} segundos")
    print(f"📁 Chunks salvos em: {output_dir}")
    print(f"="*50)

if __name__ == "__main__":
    main()