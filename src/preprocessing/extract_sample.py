#!/usr/bin/env python3
"""
Script simples para extrair uma amostra pequena de cada PDF
"""

import os
from pathlib import Path

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

def extract_sample_pypdf2(pdf_path, max_pages=3):
    """Extrai primeiras páginas com PyPDF2"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            
            # Extrair no máximo as primeiras 3 páginas
            pages_to_extract = min(len(reader.pages), max_pages)
            
            for i in range(pages_to_extract):
                page = reader.pages[i]
                text += page.extract_text() + "\n"
            
            return text.strip()
    except Exception as e:
        print(f"Erro PyPDF2: {e}")
        return None

def extract_sample_pymupdf(pdf_path, max_pages=3):
    """Extrai primeiras páginas com PyMuPDF"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        
        # Extrair no máximo as primeiras 3 páginas
        pages_to_extract = min(doc.page_count, max_pages)
        
        for i in range(pages_to_extract):
            page = doc[i]
            text += page.get_text() + "\n"
        
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"Erro PyMuPDF: {e}")
        return None

def extract_sample_from_pdf(pdf_path):
    """Tenta extrair amostra do PDF usando as bibliotecas disponíveis"""
    filename = os.path.basename(pdf_path)
    print(f"\n📖 Processando: {filename}")
    
    # Tentar PyPDF2 primeiro
    if PYPDF2_AVAILABLE:
        print("   Tentando PyPDF2...")
        text = extract_sample_pypdf2(pdf_path)
        if text and len(text.strip()) > 100:
            print(f"   ✅ PyPDF2: {len(text)} caracteres extraídos")
            return text
    
    # Tentar PyMuPDF se PyPDF2 falhou
    if PYMUPDF_AVAILABLE:
        print("   Tentando PyMuPDF...")
        text = extract_sample_pymupdf(pdf_path)
        if text and len(text.strip()) > 100:
            print(f"   ✅ PyMuPDF: {len(text)} caracteres extraídos")
            return text
    
    print("   ❌ Falha na extração")
    return None

def main():
    # Diretórios
    pdfs_dir = Path("data/raw_pdfs")
    samples_dir = Path("data/samples")
    samples_dir.mkdir(exist_ok=True)
    
    print("=== EXTRAÇÃO DE AMOSTRAS DOS PDFs ===")
    print(f"PyPDF2 disponível: {PYPDF2_AVAILABLE}")
    print(f"PyMuPDF disponível: {PYMUPDF_AVAILABLE}")
    
    if not PYPDF2_AVAILABLE and not PYMUPDF_AVAILABLE:
        print("❌ Nenhuma biblioteca de PDF disponível!")
        return
    
    # Listar PDFs
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    print(f"\nEncontrados {len(pdf_files)} arquivos PDF")
    
    success_count = 0
    
    for pdf_path in pdf_files:
        # Extrair amostra
        sample_text = extract_sample_from_pdf(pdf_path)
        
        if sample_text:
            # Salvar amostra
            sample_file = samples_dir / f"{pdf_path.stem}_sample.txt"
            with open(sample_file, 'w', encoding='utf-8') as f:
                f.write(f"AMOSTRA DE: {pdf_path.name}\n")
                f.write("="*50 + "\n\n")
                f.write(sample_text)
            
            print(f"   💾 Salvo: {sample_file.name}")
            success_count += 1
        else:
            print(f"   ❌ Não foi possível extrair texto")
    
    print(f"\n=== RESUMO ===")
    print(f"✅ {success_count}/{len(pdf_files)} PDFs processados com sucesso")
    print(f"📁 Amostras salvas em: {samples_dir}")

if __name__ == "__main__":
    main()