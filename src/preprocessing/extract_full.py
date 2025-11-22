#!/usr/bin/env python3
"""
Script para extração completa de texto dos PDFs
"""

import os
from pathlib import Path
import time

# Tentar diferentes bibliotecas
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

def extract_full_pypdf2(pdf_path):
    """Extrai todo o texto com PyPDF2"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            
            print(f"   📄 Total de páginas: {len(reader.pages)}")
            
            for i, page in enumerate(reader.pages):
                if i % 50 == 0:  # Progress update every 50 pages
                    print(f"   📖 Processando página {i+1}/{len(reader.pages)}")
                
                page_text = page.extract_text()
                text += page_text + "\n"
            
            return text.strip()
    except Exception as e:
        print(f"   ❌ Erro PyPDF2: {e}")
        return None

def extract_full_pymupdf(pdf_path):
    """Extrai todo o texto com PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        print(f"   📄 Total de páginas: {doc.page_count}")
        
        for i in range(doc.page_count):
            if i % 50 == 0:  # Progress update every 50 pages
                print(f"   📖 Processando página {i+1}/{doc.page_count}")
            
            page = doc[i]
            page_text = page.get_text()
            text += page_text + "\n"
        
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"   ❌ Erro PyMuPDF: {e}")
        return None

def extract_full_from_pdf(pdf_path):
    """Extrai todo o texto do PDF usando as bibliotecas disponíveis"""
    filename = os.path.basename(pdf_path)
    print(f"\n📖 Processando: {filename}")
    
    start_time = time.time()
    
    # Tentar PyPDF2 primeiro
    if PYPDF2_AVAILABLE:
        print("   🔧 Usando PyPDF2...")
        text = extract_full_pypdf2(pdf_path)
        if text and len(text.strip()) > 1000:  # Minimum threshold for valid extraction
            elapsed = time.time() - start_time
            print(f"   ✅ Sucesso: {len(text):,} caracteres em {elapsed:.1f}s")
            return text
    
    # Tentar PyMuPDF se PyPDF2 falhou
    if PYMUPDF_AVAILABLE:
        print("   🔧 Tentando PyMuPDF como fallback...")
        text = extract_full_pymupdf(pdf_path)
        if text and len(text.strip()) > 1000:
            elapsed = time.time() - start_time
            print(f"   ✅ Sucesso: {len(text):,} caracteres em {elapsed:.1f}s")
            return text
    
    print("   ❌ Falha na extração completa")
    return None

def main():
    # Diretórios
    pdfs_dir = Path("data/raw_pdfs")
    output_dir = Path("data/processed_texts")
    output_dir.mkdir(exist_ok=True)
    
    print("=== EXTRAÇÃO COMPLETA DOS PDFs ===")
    print(f"PyPDF2 disponível: {PYPDF2_AVAILABLE}")
    print(f"PyMuPDF disponível: {PYMUPDF_AVAILABLE}")
    
    if not PYPDF2_AVAILABLE and not PYMUPDF_AVAILABLE:
        print("❌ Nenhuma biblioteca de PDF disponível!")
        return
    
    # Listar PDFs (excluindo o Duda que sabemos que falha)
    pdf_files = [f for f in pdfs_dir.glob("*.pdf") 
                if "Duda" not in f.name]
    
    print(f"\nEncontrados {len(pdf_files)} arquivos PDF para processar")
    
    success_count = 0
    total_chars = 0
    
    start_total = time.time()
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processando arquivo...")
        
        # Extrair texto completo
        full_text = extract_full_from_pdf(pdf_path)
        
        if full_text:
            # Criar nome do arquivo de saída
            output_file = output_dir / f"{pdf_path.stem}.txt"
            
            # Salvar texto
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            chars_count = len(full_text)
            words_count = len(full_text.split())
            
            print(f"   💾 Salvo: {output_file.name}")
            print(f"   📊 Stats: {chars_count:,} chars, {words_count:,} palavras")
            
            success_count += 1
            total_chars += chars_count
        else:
            print(f"   ❌ Falha ao processar {pdf_path.name}")
    
    elapsed_total = time.time() - start_total
    
    print(f"\n" + "="*50)
    print(f"=== RESUMO FINAL ===")
    print(f"✅ {success_count}/{len(pdf_files)} PDFs processados com sucesso")
    print(f"📊 Total: {total_chars:,} caracteres extraídos")
    print(f"⏱️  Tempo total: {elapsed_total:.1f} segundos")
    print(f"📁 Arquivos salvos em: {output_dir}")
    print(f"="*50)

if __name__ == "__main__":
    main()