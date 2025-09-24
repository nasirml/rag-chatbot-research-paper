#!/usr/bin/env python3
"""
Research Paper Metadata Extraction Module

This module extracts structured metadata from research papers including:
- Title
- Authors
- Publication year
- Abstract
- DOI
- Venue/Journal
- Keywords

Author: Generated with GitHub Copilot
Date: September 2025
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class PaperMetadata:
    """Structured metadata for a research paper."""
    title: str
    authors: List[str]
    year: Optional[int] = None
    abstract: str = ""
    doi: str = ""
    venue: str = ""
    keywords: List[str] = None
    page_count: int = 0
    extraction_confidence: float = 0.0
    raw_title_match: str = ""
    raw_author_match: str = ""
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    def to_citation_apa(self) -> str:
        """Generate APA style citation."""
        if not self.authors or not self.title:
            return "Incomplete citation information"
        
        # Format authors
        if len(self.authors) == 1:
            author_str = self.authors[0]
        elif len(self.authors) == 2:
            author_str = f"{self.authors[0]} & {self.authors[1]}"
        else:
            author_str = f"{self.authors[0]} et al."
        
        # Build citation
        citation_parts = [author_str]
        
        if self.year:
            citation_parts.append(f"({self.year})")
        
        citation_parts.append(f"{self.title}.")
        
        if self.venue:
            citation_parts.append(f"{self.venue}.")
        
        if self.doi:
            citation_parts.append(f"https://doi.org/{self.doi}")
        
        return " ".join(citation_parts)

class MetadataExtractor:
    """Extract metadata from research paper text."""
    
    def __init__(self):
        """Initialize the metadata extractor."""
        self.title_patterns = [
            # Common title patterns - improved for research papers
            r'^(.{10,200})\n([A-Z][a-z]+\s+[A-Z])',  # Title followed by author name
            r'^(.{15,200})\n\n',  # First substantial line followed by blank line
            r'(?i)^title:?\s*(.+?)(?:\n|$)',
            r'(?i)^(.{20,150})$\s*\n\n',  # Standalone title line
            r'^([A-Z][^a-z]*[A-Z]\s*[A-Z][^a-z]*)',  # All caps pattern
            r'^(.{20,200})\n[A-Z][a-z]+.*[A-Z][a-z]+',  # Title before author names
        ]
        
        self.author_patterns = [
            # Author patterns - improved for various formats
            r'(?i)authors?:?\s*(.+?)(?:\n\n|\n[A-Z])',
            r'(?i)by:?\s*(.+?)(?:\n\n|\n[A-Z])',
            # Pattern for names with middle initials or names
            r'\n([A-Z][a-z]+\s+[A-Z][a-z]*\s*[A-Z][a-z]+(?:\s*,?\s*[A-Z][a-z]+\s+[A-Z][a-z]*\s*[A-Z][a-z]+)*)\n',
            # Pattern for names with middle initials (dots)
            r'([A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z][a-z]+)*)',
            # Pattern for simple first last names
            r'\n([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*,?\s*[A-Z][a-z]+\s+[A-Z][a-z]+)*)\n',
            # Pattern following title
            r'^.{20,200}\n([A-Z][a-z]+.*?[A-Z][a-z]+.*?)(?:\n[A-Z][a-z]+.*Lab|Department|\n\n)',
        ]
        
        self.year_patterns = [
            r'(?i)(?:published|year|copyright|©)\s*(?:in\s*)?(\d{4})',
            r'\((\d{4})\)',
            r'arXiv:.*?(\d{4})',  # ArXiv papers
            r'(\d{4})\.\d+\.\d+',  # ArXiv numbering format
            r'(?:19|20)(\d{2})',  # Last resort - any 4-digit year
        ]
        
        self.doi_patterns = [
            r'(?i)doi:?\s*(10\.\d{4,}/[^\s]+)',
            r'https?://doi\.org/(10\.\d{4,}/[^\s]+)',
            r'dx\.doi\.org/(10\.\d{4,}/[^\s]+)',
        ]
        
        self.abstract_patterns = [
            r'(?i)abstract:?\s*\n(.{50,2000}?)(?:\n\n|\n(?:1\.|introduction|keywords))',
            r'(?i)summary:?\s*\n(.{50,2000}?)(?:\n\n|\n(?:1\.|introduction|keywords))',
            r'^(.{100,2000}?)(?:\n\n|\n(?:1\.|introduction|keywords|I\.))',  # First paragraph if substantial
        ]
        
        self.venue_patterns = [
            r'(?i)(?:published in|appeared in|conference|journal):?\s*(.+?)(?:\n|,|\.|$)',
            r'(?i)proceedings of (.+?)(?:\n|,|\.|$)',
            r'(?i)(CVPR|ICCV|ECCV|NeurIPS|ICML|ICLR|AAAI|IJCAI|ACL|EMNLP)',
        ]
    
    def extract_metadata(self, text: str, filename: str = "") -> PaperMetadata:
        """
        Extract metadata from paper text.
        
        Args:
            text: Raw text from PDF
            filename: Original filename for fallback
            
        Returns:
            PaperMetadata object with extracted information
        """
        # Clean text for better extraction
        cleaned_text = self._clean_text_for_extraction(text)
        
        # Extract each component
        title, title_confidence = self._extract_title(cleaned_text, filename)
        authors, author_confidence = self._extract_authors(cleaned_text)
        year = self._extract_year(cleaned_text, filename)
        abstract = self._extract_abstract(cleaned_text)
        doi = self._extract_doi(cleaned_text)
        venue = self._extract_venue(cleaned_text)
        keywords = self._extract_keywords(cleaned_text)
        
        # Calculate overall confidence
        confidence = (title_confidence + author_confidence) / 2
        
        # Count pages (rough estimate)
        page_count = max(1, text.count('\n\n') // 20)  # Rough estimate
        
        metadata = PaperMetadata(
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            doi=doi,
            venue=venue,
            keywords=keywords,
            page_count=page_count,
            extraction_confidence=confidence
        )
        
        logger.info(f"Extracted metadata with confidence: {confidence:.2f}")
        return metadata
    
    def _clean_text_for_extraction(self, text: str) -> str:
        """Clean text to improve extraction accuracy."""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove page headers/footers
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip page numbers and short lines
            if re.match(r'^\d+$', line) or len(line) < 3:
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _extract_title(self, text: str, filename: str = "") -> Tuple[str, float]:
        """Extract paper title."""
        lines = text.split('\n')
        first_lines = '\n'.join(lines[:10])  # Focus on first few lines
        
        best_title = ""
        best_confidence = 0.0
        
        # Special case: Try to detect multi-line titles at the beginning
        # Look for 2-3 consecutive lines that form a title before author names
        for i in range(min(4, len(lines))):
            if len(lines[i].strip()) > 15:  # Substantial first line
                # Check if next 1-2 lines continue the title
                title_parts = [lines[i].strip()]
                
                # Check next line
                if i+1 < len(lines) and len(lines[i+1].strip()) > 10:
                    # If next line looks like continuation (no capital name pattern)
                    next_line = lines[i+1].strip()
                    if not re.match(r'^[A-Z][a-z]+\s+[A-Z]', next_line):
                        title_parts.append(next_line)
                        
                        # Check third line too
                        if i+2 < len(lines) and len(lines[i+2].strip()) > 5:
                            third_line = lines[i+2].strip()
                            if not re.match(r'^[A-Z][a-z]+\s+[A-Z]', third_line):
                                title_parts.append(third_line)
                
                candidate_title = ' '.join(title_parts)
                confidence = self._score_title_candidate(candidate_title)
                
                if confidence > best_confidence:
                    best_title = candidate_title
                    best_confidence = confidence
                break  # Only try from the very beginning
        
        # If multi-line detection didn't work, try patterns
        if best_confidence < 0.5:
            for pattern in self.title_patterns:
                match = re.search(pattern, first_lines, re.MULTILINE)
                if match:
                    candidate = match.group(1).strip()
                    confidence = self._score_title_candidate(candidate)
                    
                    if confidence > best_confidence:
                        best_title = candidate
                        best_confidence = confidence
        
        # Fallback to filename
        if best_confidence < 0.3 and filename:
            title_from_filename = re.sub(r'[_-]', ' ', filename)
            title_from_filename = re.sub(r'\.pdf$', '', title_from_filename, re.IGNORECASE)
            if len(title_from_filename) > 10:
                best_title = title_from_filename
                best_confidence = 0.2
        
        # Final fallback
        if not best_title:
            # Try first substantial line
            for line in lines[:5]:
                if len(line.strip()) > 15:
                    best_title = line.strip()
                    best_confidence = 0.1
                    break
        
        return best_title[:200], min(best_confidence, 1.0)  # Limit title length
    
    def _score_title_candidate(self, candidate: str) -> float:
        """Score a potential title candidate."""
        if len(candidate) < 10:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length scoring
        if 20 <= len(candidate) <= 150:
            score += 0.3
        
        # Capitalization patterns
        if candidate[0].isupper():
            score += 0.1
        
        # No excessive punctuation
        if candidate.count('.') <= 1 and candidate.count(',') <= 2:
            score += 0.1
        
        # Not all caps (likely header)
        if not candidate.isupper():
            score += 0.2
        
        # Has meaningful words
        words = candidate.split()
        if len(words) >= 3 and any(len(word) > 4 for word in words):
            score += 0.2
        
        return min(score, 1.0)
    
    def _extract_authors(self, text: str) -> Tuple[List[str], float]:
        """Extract author names."""
        lines = text.split('\n')
        first_section = '\n'.join(lines[:20])  # Focus on top section
        
        best_authors = []
        best_confidence = 0.0
        
        # Special case: Look for consecutive lines with author names after title
        # Skip first few lines that are likely title
        authors_from_lines = []
        start_looking = False
        
        for i, line in enumerate(lines[:15]):
            line = line.strip()
            if not line:
                continue
                
            # Start looking for authors after we see some substantial content
            if len(line) > 20 and not start_looking:
                start_looking = True
                continue
                
            if start_looking and self._is_valid_author_name(line):
                authors_from_lines.append(line)
            elif start_looking and authors_from_lines:
                # Stop when we hit non-author content
                break
        
        if authors_from_lines:
            confidence = 0.7 + (0.1 * min(len(authors_from_lines), 3))  # Higher confidence for multiple authors
            if confidence > best_confidence:
                best_authors = authors_from_lines
                best_confidence = confidence
        
        # Try pattern-based extraction as fallback
        if best_confidence < 0.5:
            for pattern in self.author_patterns:
                match = re.search(pattern, first_section, re.MULTILINE | re.DOTALL)
                if match:
                    author_text = match.group(1).strip()
                    authors, confidence = self._parse_author_text(author_text)
                    
                    if confidence > best_confidence:
                        best_authors = authors
                        best_confidence = confidence
        
        return best_authors, best_confidence
    
    def _parse_author_text(self, author_text: str) -> Tuple[List[str], float]:
        """Parse author text into individual names."""
        if len(author_text) > 300:  # Too long, probably not authors
            return [], 0.0
        
        # Clean up
        author_text = re.sub(r'[^\w\s,\.\-]', ' ', author_text)
        
        # Split by common separators
        authors = []
        for separator in [',', ' and ', ' & ']:
            if separator in author_text:
                candidates = [a.strip() for a in author_text.split(separator)]
                break
        else:
            candidates = [author_text.strip()]
        
        # Validate and clean author names
        for candidate in candidates:
            if self._is_valid_author_name(candidate):
                authors.append(candidate)
        
        # Score confidence
        confidence = 0.0
        if authors:
            confidence = 0.5
            # Higher confidence for multiple valid authors
            if len(authors) > 1:
                confidence += 0.3
            # Higher confidence for proper name patterns
            if all(re.match(r'^[A-Z][a-z]+\s+[A-Z]', author) for author in authors):
                confidence += 0.2
        
        return authors, min(confidence, 1.0)
    
    def _is_valid_author_name(self, name: str) -> bool:
        """Check if a string looks like a valid author name."""
        if not name or len(name) < 3 or len(name) > 60:
            return False
        
        # Remove asterisks (corresponding author markers)
        clean_name = name.replace('*', '').strip()
        
        # Should have at least first and last name
        parts = clean_name.split()
        if len(parts) < 2:
            return False
        
        # Should start with capital letter
        if not clean_name[0].isupper():
            return False
        
        # Should not contain numbers or excessive punctuation (allow asterisks, dots, hyphens)
        if re.search(r'\d|[^\w\s\.\-\*]', name):
            return False
        
        # All parts should look like name parts (capitalized)
        for part in parts:
            part = part.replace('*', '').replace('.', '')
            if part and not (part[0].isupper() and part[1:].islower()):
                # Allow single letters (middle initials)
                if not (len(part) == 1 and part.isupper()):
                    return False
        
        # Avoid common non-name patterns
        if any(word.lower() in clean_name.lower() for word in ['university', 'department', 'lab', 'institute', 'college']):
            return False
            
        return True
    
    def _extract_year(self, text: str, filename: str = "") -> Optional[int]:
        """Extract publication year."""
        current_year = datetime.now().year
        
        # First try filename for arXiv papers
        if filename:
            # arXiv format: YYMM.NNNNN where YY is 2-digit year
            arxiv_match = re.search(r'(\d{2})(\d{2})\.(\d{4,5})', filename)
            if arxiv_match:
                yy = int(arxiv_match.group(1))
                # Convert 2-digit year to 4-digit
                if yy <= 25:  # Assuming papers after 2025 don't exist yet
                    return 2000 + yy
                else:  # 92-99 would be 1992-1999
                    return 1900 + yy
        
        for pattern in self.year_patterns:
            matches = re.findall(pattern, text[:1000])  # Search in first part
            for match in matches:
                try:
                    year = int(match) if len(match) == 4 else int('20' + match)
                    if 1950 <= year <= current_year:  # Reasonable range
                        return year
                except ValueError:
                    continue
        
        return None
    
    def _extract_doi(self, text: str) -> str:
        """Extract DOI."""
        for pattern in self.doi_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return ""
    
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract."""
        for pattern in self.abstract_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up abstract
                abstract = re.sub(r'\n+', ' ', abstract)
                abstract = re.sub(r'\s+', ' ', abstract)
                if len(abstract) > 50:  # Minimum length for abstract
                    return abstract
        return ""
    
    def _extract_venue(self, text: str) -> str:
        """Extract publication venue."""
        for pattern in self.venue_patterns:
            match = re.search(pattern, text[:1000])
            if match:
                venue = match.group(1).strip()
                if len(venue) > 3 and len(venue) < 100:
                    return venue
        return ""
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords."""
        keyword_match = re.search(r'(?i)keywords?:?\s*(.{10,200}?)(?:\n\n|\n[A-Z])', text)
        if keyword_match:
            keyword_text = keyword_match.group(1)
            keywords = [kw.strip() for kw in re.split(r'[,;]', keyword_text)]
            return [kw for kw in keywords if kw and len(kw) > 2]
        return []

def test_metadata_extraction():
    """Test the metadata extraction with sample text."""
    sample_text = """
    Deep Learning for Computer Vision: A Comprehensive Survey
    
    John Doe, Jane Smith, Alice Johnson
    University of Technology
    
    Abstract: This paper provides a comprehensive survey of deep learning techniques 
    applied to computer vision tasks. We review the evolution from traditional methods 
    to modern neural network architectures including CNNs, transformers, and attention 
    mechanisms. Our analysis covers image classification, object detection, and semantic 
    segmentation across multiple benchmark datasets.
    
    Keywords: deep learning, computer vision, neural networks, CNN, transformer
    
    1. Introduction
    Computer vision has undergone a revolutionary transformation with the advent of 
    deep learning techniques...
    
    DOI: 10.1000/xyz123
    Published in: Conference on Computer Vision and Pattern Recognition (CVPR) 2023
    """
    
    extractor = MetadataExtractor()
    metadata = extractor.extract_metadata(sample_text)
    
    print("=== Metadata Extraction Test ===")
    print(f"Title: {metadata.title}")
    print(f"Authors: {metadata.authors}")
    print(f"Year: {metadata.year}")
    print(f"Abstract: {metadata.abstract[:100]}...")
    print(f"DOI: {metadata.doi}")
    print(f"Venue: {metadata.venue}")
    print(f"Keywords: {metadata.keywords}")
    print(f"Confidence: {metadata.extraction_confidence:.2f}")
    print(f"\nAPA Citation: {metadata.to_citation_apa()}")

if __name__ == "__main__":
    test_metadata_extraction()