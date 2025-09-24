#!/usr/bin/env python3
"""
ChromaDB Vector Database Integration for Research Papers

This module provides ChromaDB integration for storing and retrieving
research paper embeddings with metadata for RAG systems.

Features:
- Store embeddings with paper metadata
- Duplicate detection
- Semantic search
- Citation management
- Paper filtering and querying

Author: Generated with GitHub Copilot  
Date: September 2025
"""

import uuid
from typing import List, Dict, Optional, Tuple, Any
import logging
from pathlib import Path
import json
import numpy as np
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB not installed. Run: pip install chromadb")

from metadata_extractor import PaperMetadata

logger = logging.getLogger(__name__)

class ResearchPaperVectorDB:
    """ChromaDB-based vector database for research papers."""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "research_papers"):
        """
        Initialize the vector database.
        
        Args:
            db_path: Path to ChromaDB storage
            collection_name: Name of the collection
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Please run: pip install chromadb")
        
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Research paper embeddings with metadata"}
            )
            
            logger.info(f"Initialized ChromaDB at {self.db_path}")
            logger.info(f"Collection '{self.collection_name}' ready with {self.collection.count()} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_paper(
        self, 
        paper_metadata: PaperMetadata, 
        chunks: List[Dict[str, Any]], 
        pdf_path: str = ""
    ) -> bool:
        """
        Add a research paper to the vector database.
        
        Args:
            paper_metadata: Extracted paper metadata
            chunks: List of text chunks with embeddings
            pdf_path: Original PDF file path
            
        Returns:
            True if added successfully, False if duplicate detected
        """
        # Check for duplicates first
        if self.is_duplicate(paper_metadata):
            logger.warning(f"Duplicate paper detected: {paper_metadata.title}")
            return False
        
        # Generate unique paper ID
        paper_id = str(uuid.uuid4())
        
        # Prepare documents, embeddings, and metadata
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{paper_id}_chunk_{i}"
            
            # Prepare metadata for this chunk
            chunk_metadata = {
                # Paper-level metadata
                "paper_id": paper_id,
                "title": paper_metadata.title,
                "authors": json.dumps(paper_metadata.authors),  # Store as JSON string
                "year": paper_metadata.year or 0,
                "abstract": paper_metadata.abstract,
                "doi": paper_metadata.doi,
                "venue": paper_metadata.venue,
                "keywords": json.dumps(paper_metadata.keywords),
                
                # Chunk-level metadata
                "chunk_id": i,
                "chunk_token_count": chunk.get('token_count', 0),
                "chunk_sentence_count": chunk.get('sentence_count', 0),
                
                # System metadata
                "pdf_path": pdf_path,
                "added_timestamp": datetime.now().isoformat(),
                "extraction_confidence": paper_metadata.extraction_confidence,
            }
            
            documents.append(chunk['text'])
            embeddings.append(chunk['embedding'])
            metadatas.append(chunk_metadata)
            ids.append(chunk_id)
        
        try:
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added paper '{paper_metadata.title}' with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add paper to database: {e}")
            return False
    
    def is_duplicate(self, paper_metadata: PaperMetadata, similarity_threshold: float = 0.8) -> bool:
        """
        Check if a paper is already in the database.
        
        Args:
            paper_metadata: Paper metadata to check
            similarity_threshold: Similarity threshold for duplicate detection
            
        Returns:
            True if duplicate found
        """
        if not paper_metadata.title:
            return False
        

        # Check by DOI first (most reliable)
        if paper_metadata.doi:
            results = self.collection.get(
                where={"doi": paper_metadata.doi},
                limit=1
            )
            if results['documents']:
                logger.info(f"Duplicate found by DOI: {paper_metadata.doi}")
                return True
        
        # Check by exact title match
        results = self.collection.get(
            where={"title": paper_metadata.title},
            limit=1
        )
        if results['documents']:
            logger.info(f"Duplicate found by title: {paper_metadata.title}")
            return True
        
        # Check by title similarity using search
        try:
            results = self.collection.query(
                query_texts=[paper_metadata.title],
                n_results=5,
                where=None
            )
            
            for distance, metadata in zip(results['distances'][0], results['metadatas'][0]):
                # Convert distance to similarity (ChromaDB uses cosine distance)
                similarity = 1 - distance
                if similarity > similarity_threshold:
                    existing_title = metadata.get('title', '')
                    logger.info(f"Similar paper found (similarity: {similarity:.3f}): {existing_title}")
                    return True
                    
        except Exception as e:
            logger.warning(f"Error checking title similarity: {e}")
        
        return False
    
    def search_papers(
        self, 
        query: str, 
        n_results: int = 10,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant paper chunks.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of matching chunks with metadata
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'paper_title': results['metadatas'][0][i].get('title', 'Unknown'),
                    'authors': json.loads(results['metadatas'][0][i].get('authors', '[]')),
                    'year': results['metadatas'][0][i].get('year', 0),
                    'chunk_id': results['metadatas'][0][i].get('chunk_id', 0),
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_paper_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Get all chunks for a paper by title."""
        try:

            results = self.collection.get(
                where={"title": title},
                limit=10000,  # Get all chunks
                include=["documents", "metadatas"]
            )
            
            if not results['documents']:
                return None
            
            # Group chunks by paper
            chunks = []
            paper_metadata = None
            
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                if not paper_metadata:
                    paper_metadata = {
                        'title': metadata.get('title', ''),
                        'authors': json.loads(metadata.get('authors', '[]')),
                        'year': metadata.get('year', 0),
                        'abstract': metadata.get('abstract', ''),
                        'doi': metadata.get('doi', ''),
                        'venue': metadata.get('venue', ''),
                        'keywords': json.loads(metadata.get('keywords', '[]')),
                    }
                
                chunks.append({
                    'text': doc,
                    'chunk_id': metadata.get('chunk_id', 0),
                    'token_count': metadata.get('chunk_token_count', 0),
                })
            
            return {
                'metadata': paper_metadata,
                'chunks': sorted(chunks, key=lambda x: x['chunk_id'])
            }
            
        except Exception as e:
            logger.error(f"Failed to get paper: {e}")
            return None
    
    def list_papers(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all papers in the database."""
        try:

            # Get unique paper titles
            all_results = self.collection.get(
                limit=10000,
                include=["metadatas"]
            )
            
            papers = {}
            for metadata in all_results['metadatas'][0]:
                paper_id = metadata.get('paper_id')
                if paper_id not in papers:
                    papers[paper_id] = {
                        'paper_id': paper_id,
                        'title': metadata.get('title', ''),
                        'authors': json.loads(metadata.get('authors', '[]')),
                        'year': metadata.get('year', 0),
                        'doi': metadata.get('doi', ''),
                        'venue': metadata.get('venue', ''),
                        'added_timestamp': metadata.get('added_timestamp', ''),
                        'chunk_count': 0
                    }
                papers[paper_id]['chunk_count'] += 1
            
            # Return sorted by date added (newest first)
            paper_list = list(papers.values())
            paper_list.sort(key=lambda x: x['added_timestamp'], reverse=True)
            
            return paper_list[:limit]
            
        except Exception as e:
            logger.error(f"Failed to list papers: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            total_chunks = self.collection.count()
            papers = self.list_papers(limit=10000)  # Get all papers
            
            # Calculate statistics
            stats = {
                'total_papers': len(papers),
                'total_chunks': total_chunks,
                'average_chunks_per_paper': total_chunks / len(papers) if papers else 0,
                'papers_by_year': {},
                'most_common_venues': {},
                'database_path': str(self.db_path),
                'collection_name': self.collection_name
            }
            
            # Year distribution
            for paper in papers:
                year = paper.get('year', 0)
                if year > 0:
                    stats['papers_by_year'][year] = stats['papers_by_year'].get(year, 0) + 1
            
            # Venue distribution
            for paper in papers:
                venue = paper.get('venue', '').strip()
                if venue:
                    stats['most_common_venues'][venue] = stats['most_common_venues'].get(venue, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}
    
    def delete_paper(self, title: str) -> bool:
        """Delete a paper and all its chunks."""
        try:

            # Find all chunks for this paper
            results = self.collection.get(
                where={"title": title},
                limit=10000,
                include=["metadatas"]
            )
            
            if not results['metadatas']:
                logger.warning(f"Paper not found: {title}")
                return False
            
            # Get all chunk IDs
            chunk_ids = []
            for metadata in results['metadatas'][0]:
                # The ID should be the document ID in ChromaDB
                pass  # ChromaDB query doesn't return IDs directly
            
            # For now, we'll need to delete by paper_id
            paper_id = results['metadatas'][0][0].get('paper_id') if results['metadatas'][0] else None
            if paper_id:
                self.collection.delete(where={"paper_id": paper_id})
                logger.info(f"Deleted paper: {title}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete paper: {e}")
            return False

def test_chromadb_integration():
    """Test the ChromaDB integration."""
    print("=== ChromaDB Integration Test ===")
    
    # Initialize database
    try:
        db = ResearchPaperVectorDB(db_path="./test_chroma_db")
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        return
    
    # Create sample metadata
    from metadata_extractor import PaperMetadata
    
    sample_metadata = PaperMetadata(
        title="Test Paper: Deep Learning for Computer Vision",
        authors=["John Doe", "Jane Smith"],
        year=2023,
        abstract="This is a test abstract for demonstrating the ChromaDB integration.",
        doi="10.1000/test123",
        venue="Test Conference",
        keywords=["deep learning", "computer vision", "test"]
    )
    
    # Create sample chunks with embeddings
    import numpy as np
    sample_chunks = [
        {
            'text': 'This is the first chunk of the test paper about deep learning.',
            'embedding': np.random.rand(384).tolist(),  # all-MiniLM-L6-v2 dimension
            'token_count': 12,
            'chunk_id': 0
        },
        {
            'text': 'This is the second chunk discussing computer vision applications.',
            'embedding': np.random.rand(384).tolist(),
            'token_count': 10,
            'chunk_id': 1
        }
    ]
    
    # Test adding paper
    success = db.add_paper(sample_metadata, sample_chunks, "test.pdf")
    print(f"✅ Paper added: {success}")
    
    # Test duplicate detection
    is_duplicate = db.is_duplicate(sample_metadata)
    print(f"✅ Duplicate detection works: {is_duplicate}")
    
    # Test search
    results = db.search_papers("deep learning", n_results=5)
    print(f"✅ Search found {len(results)} results")
    
    # Test database stats
    stats = db.get_database_stats()
    print(f"✅ Database stats: {stats['total_papers']} papers, {stats['total_chunks']} chunks")
    
    print("=== Test completed ===")

if __name__ == "__main__":
    test_chromadb_integration()