#!/usr/bin/env python3
"""
PDF Research Paper Processing System

This module provides functionality to:
1. Extract text from PDF research papers
2. Chunk the text intelligently (preserving sentence boundaries)
3. Generate embeddings using the all-MiniLM-L6-v2 model
4. Save embeddings for later use in RAG systems
5. Analyze the embeddings and chunk statistics

"""

import os
import re
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

# PDF processing
import PyPDF2
import fitz  # PyMuPDF for better text extraction

# Text processing and embeddings
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data with fallback for different versions."""
    resources_to_download = ['punkt', 'punkt_tab']
    
    for resource in resources_to_download:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            logger.info(f"NLTK {resource} already available")
        except LookupError:
            try:
                print(f"Downloading NLTK {resource} tokenizer...")
                nltk.download(resource, quiet=True)
                logger.info(f"Successfully downloaded {resource}")
            except Exception as e:
                logger.warning(f"Could not download {resource}: {e}")

# Initialize NLTK data
download_nltk_data()

class PDFProcessor:
    """
    A class to process PDF research papers and generate embeddings.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the PDF processor with embedding model.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str, method: str = "pymupdf") -> str:
        """
        Extract text from PDF using specified method.
        
        Args:
            pdf_path: Path to the PDF file
            method: Extraction method ("pymupdf" or "pypdf2")
            
        Returns:
            Extracted text as string
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if method == "pymupdf":
            return self._extract_with_pymupdf(pdf_path)
        elif method == "pypdf2":
            return self._extract_with_pypdf2(pdf_path)
        else:
            raise ValueError("Method must be 'pymupdf' or 'pypdf2'")
    
    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF (better quality)."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                text += page_text + "\n"
            
            doc.close()
            logger.info(f"Extracted {len(text)} characters using PyMuPDF")
            return text
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            # Fallback to PyPDF2
            return self._extract_with_pypdf2(pdf_path)
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2 (fallback method)."""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    text += page_text + "\n"
            
            logger.info(f"Extracted {len(text)} characters using PyPDF2")
            return text
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing excessive whitespace and formatting artifacts.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single
        
        # Remove common PDF artifacts
        text = re.sub(r'\x0c', '', text)  # Form feed characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Non-ASCII characters
        
        # Remove page numbers and headers/footers (basic heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip likely page numbers
            if re.match(r'^\d+$', line):
                continue
            # Skip very short lines (likely artifacts)
            if len(line) < 3:
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = 512, 
        overlap: int = 50,
        method: str = "sentence_aware"
    ) -> List[Dict[str, any]]:
        """
        Chunk text into smaller pieces for embedding.
        
        Args:
            text: Input text to chunk
            chunk_size: Target size of each chunk (in tokens)
            overlap: Number of tokens to overlap between chunks
            method: Chunking method ("sentence_aware" or "sliding_window")
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if method == "sentence_aware":
            return self._chunk_sentence_aware(text, chunk_size, overlap)
        elif method == "sliding_window":
            return self._chunk_sliding_window(text, chunk_size, overlap)
        else:
            raise ValueError("Method must be 'sentence_aware' or 'sliding_window'")
    
    def _chunk_sentence_aware(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, any]]:
        """Chunk text while preserving sentence boundaries."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_tokens = word_tokenize(sentence)
            sentence_size = len(sentence_tokens)
            
            # If adding this sentence would exceed chunk size
            if current_size + sentence_size > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'id': chunk_id,
                    'text': chunk_text,
                    'token_count': current_size,
                    'sentence_count': len(current_chunk)
                })
                chunk_id += 1
                
                # Start new chunk with overlap
                if overlap > 0:
                    # Keep last few sentences for overlap
                    overlap_sentences = []
                    overlap_size = 0
                    for sent in reversed(current_chunk):
                        sent_size = len(word_tokenize(sent))
                        if overlap_size + sent_size <= overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_size += sent_size
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'token_count': current_size,
                'sentence_count': len(current_chunk)
            })
        
        logger.info(f"Created {len(chunks)} sentence-aware chunks")
        return chunks
    
    def _chunk_sliding_window(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, any]]:
        """Chunk text using sliding window approach."""
        words = word_tokenize(text)
        chunks = []
        chunk_id = 0
        
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'id': chunk_id,
                'text': chunk_text,
                'token_count': len(chunk_words),
                'start_idx': start,
                'end_idx': end
            })
            
            chunk_id += 1
            start += chunk_size - overlap
            
            if end >= len(words):
                break
        
        logger.info(f"Created {len(chunks)} sliding-window chunks")
        return chunks
    
    def generate_embeddings(self, chunks: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of chunks with embeddings added
        """
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        texts = [chunk['text'] for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()  # Convert numpy to list
            chunk['embedding_dim'] = len(embeddings[i])
        
        logger.info("Embeddings generated successfully")
        return chunks
    
    def process_pdf(
        self, 
        pdf_path: str, 
        output_dir: Optional[str] = None,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> Dict[str, any]:
        """
        Complete PDF processing pipeline.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save outputs (optional)
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            
        Returns:
            Dictionary containing all processing results
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text
        raw_text = self.extract_text_from_pdf(pdf_path)
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        
        # Chunk text
        chunks = self.chunk_text(cleaned_text, chunk_size, overlap)
        
        # Generate embeddings
        chunks_with_embeddings = self.generate_embeddings(chunks)
        
        # Prepare results
        results = {
            'pdf_path': pdf_path,
            'raw_text_length': len(raw_text),
            'cleaned_text_length': len(cleaned_text),
            'num_chunks': len(chunks_with_embeddings),
            'embedding_model': self.model_name,
            'chunks': chunks_with_embeddings,
            'metadata': {
                'chunk_size': chunk_size,
                'overlap': overlap,
                'processing_timestamp': str(pd.Timestamp.now())
            }
        }
        
        # Save results if output directory specified
        if output_dir:
            self.save_results(results, output_dir)
        
        logger.info("PDF processing completed successfully")
        return results
    
    def save_results(self, results: Dict[str, any], output_dir: str):
        """Save processing results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        pdf_name = Path(results['pdf_path']).stem
        
        # Save full results as JSON
        json_path = output_path / f"{pdf_name}_embeddings.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save embeddings as numpy array
        embeddings = np.array([chunk['embedding'] for chunk in results['chunks']])
        numpy_path = output_path / f"{pdf_name}_embeddings.npy"
        np.save(numpy_path, embeddings)
        
        # Save text chunks
        chunks_path = output_path / f"{pdf_name}_chunks.txt"
        with open(chunks_path, 'w') as f:
            for i, chunk in enumerate(results['chunks']):
                f.write(f"=== CHUNK {i} ===\n")
                f.write(chunk['text'])
                f.write("\n\n")
        
        logger.info(f"Results saved to {output_dir}")

# Import pandas for timestamp
try:
    import pandas as pd
except ImportError:
    # Fallback if pandas not available
    import datetime as pd
    pd.Timestamp = lambda: datetime.datetime.now()

def main():
    """Example usage of the PDF processor."""
    processor = PDFProcessor()
    
    # Example PDF path (replace with your actual PDF)
    pdf_path = "example_research_paper.pdf"
    
    if os.path.exists(pdf_path):
        results = processor.process_pdf(
            pdf_path=pdf_path,
            output_dir="./output",
            chunk_size=512,
            overlap=50
        )
        
        print(f"Processed PDF successfully!")
        print(f"Number of chunks: {results['num_chunks']}")
        print(f"Embedding dimension: {results['chunks'][0]['embedding_dim']}")
    else:
        print(f"PDF file not found: {pdf_path}")
        print("Please provide a valid PDF file path.")

if __name__ == "__main__":
    main()