"""

Enhanced Chunking Strategies for Better RAG Performance

This module provides improved text chunking strategies specifically designed
for research papers to improve RAG retrieval quality.
"""

import re
from typing import List, Dict, Any, Tuple
import nltk
from nltk.tokenize import sent_tokenize
from collections import defaultdict

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class EnhancedChunker:
    """Enhanced text chunking for research papers."""
    
    def __init__(self):
        self.section_patterns = [
            r'^(?:abstract|introduction|background|related work|methodology|methods|approach|results|discussion|conclusion|references)$',
            r'^\d+\.?\s+(?:abstract|introduction|background|related work|methodology|methods|approach|results|discussion|conclusion|references)',
            r'^[ivx]+\.?\s+(?:abstract|introduction|background|related work|methodology|methods|approach|results|discussion|conclusion|references)'
        ]
    
    def semantic_chunking(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Create semantically coherent chunks that respect paragraph and sentence boundaries.
        """
        chunks = []
        
        # First, try to identify sections
        sections = self._identify_sections(text)
        
        if sections:
            # Process each section separately
            for section_name, section_text in sections.items():
                section_chunks = self._chunk_section(section_text, section_name, chunk_size, overlap)
                chunks.extend(section_chunks)
        else:
            # Fallback to paragraph-based chunking
            chunks = self._chunk_by_paragraphs(text, chunk_size, overlap)
        
        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk['chunk_id'] = i
            chunk['chunk_type'] = chunk.get('chunk_type', 'content')
            chunk['importance_score'] = self._calculate_importance(chunk['text'])
        
        return chunks
    
    def _identify_sections(self, text: str) -> Dict[str, str]:
        """Identify paper sections like Abstract, Introduction, etc."""
        sections = {}
        lines = text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            is_section = False
            for pattern in self.section_patterns:
                if re.match(pattern, line.lower()):
                    is_section = True
                    # Save previous section
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content)
                    
                    current_section = line.lower()
                    current_content = []
                    break
            
            if not is_section and current_section:
                current_content.append(line)
        
        # Add final section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _chunk_section(self, text: str, section_name: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Chunk a specific section with appropriate metadata."""
        chunks = []
        
        # Different strategies for different sections
        if 'abstract' in section_name:
            # Abstract should typically be one chunk
            chunks.append({
                'text': text,
                'section': section_name,
                'chunk_type': 'abstract',
                'priority': 'high'
            })
        elif 'introduction' in section_name or 'background' in section_name:
            # Introduction can be chunked by paragraphs
            para_chunks = self._chunk_by_paragraphs(text, chunk_size, overlap)
            for chunk in para_chunks:
                chunk['section'] = section_name
                chunk['chunk_type'] = 'introduction'
                chunk['priority'] = 'high'
            chunks.extend(para_chunks)
        elif 'method' in section_name or 'approach' in section_name:
            # Methods are important for technical queries
            para_chunks = self._chunk_by_paragraphs(text, chunk_size, overlap)
            for chunk in para_chunks:
                chunk['section'] = section_name
                chunk['chunk_type'] = 'methodology'
                chunk['priority'] = 'high'
            chunks.extend(para_chunks)
        elif 'result' in section_name or 'discussion' in section_name:
            # Results are crucial for findings
            para_chunks = self._chunk_by_paragraphs(text, chunk_size, overlap)
            for chunk in para_chunks:
                chunk['section'] = section_name
                chunk['chunk_type'] = 'results'
                chunk['priority'] = 'high'
            chunks.extend(para_chunks)
        else:
            # Default chunking for other sections
            para_chunks = self._chunk_by_paragraphs(text, chunk_size, overlap)
            for chunk in para_chunks:
                chunk['section'] = section_name
                chunk['chunk_type'] = 'content'
                chunk['priority'] = 'medium'
            chunks.extend(para_chunks)
        
        return chunks
    
    def _chunk_by_paragraphs(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Chunk text by paragraphs while respecting size limits."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # If paragraph alone exceeds chunk size, split by sentences
            if para_size > chunk_size:
                if current_chunk:
                    chunks.append({
                        'text': '\n\n'.join(current_chunk),
                        'word_count': current_size
                    })
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph by sentences
                sentence_chunks = self._chunk_by_sentences(para, chunk_size, overlap)
                chunks.extend(sentence_chunks)
                continue
            
            # If adding this paragraph exceeds chunk size, finalize current chunk
            if current_size + para_size > chunk_size and current_chunk:
                chunks.append({
                    'text': '\n\n'.join(current_chunk),
                    'word_count': current_size
                })
                
                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1]  # Take last paragraph as overlap
                    current_chunk = [overlap_text, para]
                    current_size = len(overlap_text) + para_size
                else:
                    current_chunk = [para]
                    current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': '\n\n'.join(current_chunk),
                'word_count': current_size
            })
        
        return chunks
    
    def _chunk_by_sentences(self, text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """Chunk text by sentences when paragraphs are too large."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'word_count': current_size
                })
                
                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    overlap_sentences = current_chunk[-1:]  # Take last sentence as overlap
                    current_chunk = overlap_sentences + [sentence]
                    current_size = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'word_count': current_size
            })
        
        return chunks
    
    def _calculate_importance(self, text: str) -> float:
        """Calculate importance score for a chunk based on content patterns."""
        score = 0.5  # Base score
        text_lower = text.lower()
        
        # Boost for key academic terms
        important_terms = [
            'results', 'findings', 'conclusion', 'significant', 'demonstrate',
            'novel', 'propose', 'method', 'approach', 'algorithm', 'model',
            'performance', 'accuracy', 'evaluation', 'comparison', 'analysis'
        ]
        
        for term in important_terms:
            if term in text_lower:
                score += 0.1
        
        # Boost for numerical results
        if re.search(r'\d+\.?\d*%', text) or re.search(r'\d+\.?\d*\s*accuracy', text_lower):
            score += 0.2
        
        # Boost for citations (indicates important content)
        citation_pattern = r'\[[0-9,\s-]+\]|\([A-Za-z]+\s+et\s+al\.?,?\s+\d{4}\)'
        if re.search(citation_pattern, text):
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0

    def create_contextual_chunks(self, text: str, chunk_size: int = 512) -> List[Dict[str, Any]]:
        """Create chunks with additional context from surrounding text."""
        base_chunks = self.semantic_chunking(text, chunk_size, overlap=50)
        
        # Add context to each chunk
        for i, chunk in enumerate(base_chunks):
            context_before = ""
            context_after = ""
            
            # Add context from previous chunk
            if i > 0:
                prev_text = base_chunks[i-1]['text']
                context_before = prev_text[-100:]  # Last 100 chars
            
            # Add context from next chunk
            if i < len(base_chunks) - 1:
                next_text = base_chunks[i+1]['text']
                context_after = next_text[:100]  # First 100 chars
            
            chunk['context_before'] = context_before
            chunk['context_after'] = context_after
            chunk['full_context'] = f"{context_before} {chunk['text']} {context_after}".strip()
        
        return base_chunks