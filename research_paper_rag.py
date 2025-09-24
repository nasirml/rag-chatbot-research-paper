"""
Enhanced RAG System for Research Papers

This module provides a complete RAG (Retrieval-Augmented Generation) system
specifically designed for research papers with:

- PDF processing and metadata extraction
- ChromaDB vector storage
- Duplicate detection
- Citation management
- Semantic search and retrieval

Author: Generated with GitHub Copilot
Date: September 2025
"""

import os
from typing import List, Dict, Optional, Any
from pathlib import Path
import logging
import json

from pdf_embedding_processor import PDFProcessor
from metadata_extractor import MetadataExtractor, PaperMetadata
from vector_database import ResearchPaperVectorDB
from llm_integration import LLMManager, generate_fallback_answer

logger = logging.getLogger(__name__)

class ResearchPaperRAG:
    """Complete RAG system for research papers."""
    
    def __init__(
        self, 
        db_path: str = "./research_papers_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_provider: str = "anthropic",
        **llm_kwargs
    ):
        """
        Initialize the RAG system.
        
        Args:
            db_path: Path for ChromaDB storage
            embedding_model: Sentence transformer model name
            llm_provider: LLM provider ('anthropic', 'openai', 'fallback', 'none')
            **llm_kwargs: Additional arguments for LLM provider
        """
        self.db_path = Path(db_path)
        self.embedding_model = embedding_model
        
        # Initialize components
        logger.info("Initializing RAG system components...")
        self.pdf_processor = PDFProcessor(model_name=embedding_model)
        self.metadata_extractor = MetadataExtractor()
        self.vector_db = ResearchPaperVectorDB(db_path=str(self.db_path))
        

        # Initialize LLM provider
        self.llm_provider = None
        if llm_provider != "none":
            try:
                self.llm_provider = LLMManager(provider=llm_provider, **llm_kwargs)
                if self.llm_provider.is_available():
                    logger.info(f"✅ LLM provider '{llm_provider}' initialized successfully")
                else:
                    logger.warning(f"⚠️ LLM provider '{llm_provider}' not available, using fallback")
                    self.llm_provider = None
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize LLM provider: {e}, using fallback")
                self.llm_provider = None
        else:
            logger.info("LLM provider disabled, using rule-based generation")
        
        logger.info("✅ RAG system initialized successfully")
    
    def add_paper_from_pdf(
        self, 
        pdf_path: str,
        chunk_size: int = 512,
        overlap: int = 50,
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Add a research paper from PDF to the RAG system.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            force_reprocess: Force reprocessing even if duplicate
            
        Returns:
            Processing result dictionary
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return {'success': False, 'error': f'PDF file not found: {pdf_path}'}
        
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        try:
            # Extract text from PDF
            raw_text = self.pdf_processor.extract_text_from_pdf(str(pdf_path))
            cleaned_text = self.pdf_processor.clean_text(raw_text)
            
            # Extract metadata
            logger.info("Extracting metadata...")
            metadata = self.metadata_extractor.extract_metadata(
                cleaned_text, 
                filename=pdf_path.stem
            )
            
            # Check for duplicates
            if not force_reprocess and self.vector_db.is_duplicate(metadata):
                return {
                    'success': False, 
                    'error': 'Duplicate paper detected',
                    'title': metadata.title,
                    'existing': True
                }
            
            # Process text into chunks
            logger.info("Creating text chunks...")
            chunks = self.pdf_processor.chunk_text(
                cleaned_text, 
                chunk_size=chunk_size, 
                overlap=overlap
            )
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            chunks_with_embeddings = self.pdf_processor.generate_embeddings(chunks)
            
            # Add to vector database
            logger.info("Adding to vector database...")
            success = self.vector_db.add_paper(
                metadata, 
                chunks_with_embeddings, 
                str(pdf_path)
            )
            
            if success:
                result = {
                    'success': True,
                    'title': metadata.title,
                    'authors': metadata.authors,
                    'year': metadata.year,
                    'chunks': len(chunks_with_embeddings),
                    'confidence': metadata.extraction_confidence,
                    'citation': metadata.to_citation_apa()
                }
                logger.info(f"✅ Successfully processed: {metadata.title}")
                return result
            else:
                return {'success': False, 'error': 'Failed to add to database'}
                
        except Exception as e:
            logger.error(f"Failed to process PDF: {e}")
            return {'success': False, 'error': str(e)}
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms for better retrieval.
        """
        # Basic query expansion - can be enhanced with word2vec, WordNet, etc.
        expansions = [query]
        
        # Add variations and synonyms for common academic terms
        academic_synonyms = {
            'neural networks': ['deep learning', 'artificial neural networks', 'neural nets', 'deep neural networks'],
            'machine learning': ['ML', 'artificial intelligence', 'AI', 'statistical learning'],
            'computer vision': ['CV', 'image processing', 'visual recognition', 'image analysis'],
            'natural language processing': ['NLP', 'text processing', 'language modeling', 'text analysis'],
            'deep learning': ['neural networks', 'deep neural networks', 'DNN'],
            'CNNs': ['convolutional neural networks', 'convnets', 'convolutional networks'],
            'transformers': ['attention mechanism', 'self-attention', 'BERT', 'GPT'],
            'performance': ['accuracy', 'results', 'evaluation', 'effectiveness'],
            'method': ['approach', 'technique', 'algorithm', 'methodology'],
            'analysis': ['study', 'investigation', 'examination', 'evaluation']
        }
        
        query_lower = query.lower()
        for term, synonyms in academic_synonyms.items():
            if term in query_lower:
                expansions.extend([query.lower().replace(term, syn) for syn in synonyms])
        
        return list(set(expansions))  # Remove duplicates
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-rank results based on additional relevance signals.
        """
        query_terms = set(query.lower().split())
        
        for result in results:
            text = result.get('text', '').lower()
            title = result.get('paper_title', '').lower()
            abstract = result.get('abstract', '').lower()
            
            # Calculate additional relevance scores
            title_match = sum(1 for term in query_terms if term in title)
            abstract_match = sum(1 for term in query_terms if term in abstract)
            text_match = sum(1 for term in query_terms if term in text)
            
            # Boost score based on where matches occur (title > abstract > content)
            boost_score = (title_match * 3 + abstract_match * 2 + text_match * 1) / len(query_terms)
            
            # Combine with original similarity
            original_similarity = result.get('similarity', 0.0)
            result['combined_score'] = original_similarity + (boost_score * 0.1)
            result['title_relevance'] = title_match / len(query_terms)
            result['abstract_relevance'] = abstract_match / len(query_terms)
        
        # Sort by combined score
        results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        return results

    def search(
        self, 
        query: str, 
        n_results: int = 3,
        year_filter: Optional[int] = None,
        author_filter: Optional[str] = None,
        use_query_expansion: bool = True,
        use_reranking: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant papers and chunks with enhanced quality.
        
        Args:
            query: Search query
            n_results: Number of results to return
            year_filter: Filter by publication year
            author_filter: Filter by author name
            use_query_expansion: Whether to expand query with synonyms
            use_reranking: Whether to re-rank results
            
        Returns:
            List of relevant chunks with metadata
        """

        # Query expansion
        search_queries = [query]
        if use_query_expansion:
            search_queries = self._expand_query(query)
        
        all_results = []
        
        # Search with multiple query variations
        for search_query in search_queries[:3]:  # Limit to top 3 variations
            # Build metadata filter
            metadata_filter = {}
            if year_filter:
                metadata_filter['year'] = year_filter
            
            # Search in vector database
            results = self.vector_db.search_papers(
                search_query, 
                n_results=n_results * 2,  # Get more results for reranking
                filter_metadata=metadata_filter if metadata_filter else None
            )
            all_results.extend(results)
        
        # Remove duplicates based on chunk content
        unique_results = []
        seen_texts = set()
        for result in all_results:
            text_key = result.get('text', '')[:100]  # Use first 100 chars as key
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_results.append(result)
        
        # Filter by author if specified
        if author_filter:
            filtered_results = []
            for result in unique_results:
                authors = result.get('authors', [])
                if any(author_filter.lower() in author.lower() for author in authors):
                    filtered_results.append(result)
            unique_results = filtered_results
        
        # Re-rank results
        if use_reranking:
            unique_results = self._rerank_results(query, unique_results)
        
        # Limit to requested number of results
        final_results = unique_results[:n_results]
        
        # Enhance results with citation information
        for result in final_results:
            # Create citation from metadata
            authors = result.get('authors', [])
            title = result.get('paper_title', '')
            year = result.get('year', 0)
            
            if authors and title:
                author_str = authors[0] if len(authors) == 1 else f"{authors[0]} et al."
                citation = f"{author_str} ({year if year > 0 else 'Unknown'}). {title}"
                result['citation'] = citation
            else:
                result['citation'] = 'Citation information incomplete'
            
            # Add relevance explanation
            relevance_factors = []
            if result.get('title_relevance', 0) > 0:
                relevance_factors.append(f"title match ({result['title_relevance']:.1%})")
            if result.get('abstract_relevance', 0) > 0:
                relevance_factors.append(f"abstract match ({result['abstract_relevance']:.1%})")
            
            result['relevance_explanation'] = ", ".join(relevance_factors) if relevance_factors else "content similarity"
        
        return final_results
    
    def answer_question(
        self,
        question: str,
        n_results: int = 5,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive answer to a question using retrieved context.
        
        Args:
            question: The question to answer
            n_results: Number of context chunks to retrieve
            include_sources: Whether to include source citations
            
        Returns:
            Dictionary with answer and sources
        """
        # Search for relevant context
        search_results = self.search(
            question, 
            n_results=n_results,
            use_query_expansion=True,
            use_reranking=True
        )
        
        if not search_results:
            return {
                'answer': "I couldn't find any relevant information in the research papers database to answer your question.",
                'sources': [],
                'confidence': 0.0
            }
        
        # Prepare context from search results
        context_pieces = []
        sources = []
        
        for i, result in enumerate(search_results):
            context_text = result.get('text', '')
            paper_title = result.get('paper_title', '')
            authors = result.get('authors', [])
            year = result.get('year', 'Unknown')
            similarity = result.get('similarity', 0.0)
            
            context_pieces.append(f"[Source {i+1}] {context_text}")
            
            if include_sources:
                source_info = {
                    'id': i + 1,
                    'title': paper_title,
                    'authors': authors,
                    'year': year,
                    'relevance_score': similarity,
                    'text_excerpt': context_text[:200] + "..." if len(context_text) > 200 else context_text,
                    'citation': result.get('citation', ''),
                    'relevance_explanation': result.get('relevance_explanation', '')
                }
                sources.append(source_info)
        
        # Combine context
        combined_context = "\n\n".join(context_pieces)
        

        # Generate answer using LLM or fallback
        if self.llm_provider and self.llm_provider.is_available():
            logger.info("Using LLM provider for answer generation")
            llm_response = self.llm_provider.generate_answer(question, combined_context, search_results)
            answer = llm_response.get('answer', 'Failed to generate answer')
            llm_metadata = {
                'model': llm_response.get('model', 'unknown'),
                'provider': llm_response.get('provider', 'unknown'),
                'usage': llm_response.get('usage', {}),
                'success': llm_response.get('success', False)
            }
        else:
            logger.info("Using fallback answer generation")
            answer = generate_fallback_answer(question, combined_context, search_results)
            llm_metadata = {
                'model': 'rule-based',
                'provider': 'fallback',
                'usage': {},
                'success': True
            }
        
        # Calculate confidence based on similarity scores
        avg_similarity = sum(r.get('similarity', 0.0) for r in search_results) / len(search_results)
        confidence = min(avg_similarity * 1.2, 1.0)  # Scale up slightly, cap at 1.0
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': confidence,
            'context_used': len(search_results),
            'question': question,
            'llm_metadata': llm_metadata
        }
    
    def _generate_contextual_answer(self, question: str, context: str, search_results: List[Dict]) -> str:
        """
        Generate an answer based on the question and context.
        This is a placeholder - in a real implementation, you'd use an LLM here.
        """
        # Extract key information from context
        key_findings = []
        methodologies = []
        papers_mentioned = set()
        
        for result in search_results:
            text = result.get('text', '').lower()
            title = result.get('paper_title', '')
            authors = result.get('authors', [])
            
            papers_mentioned.add(f"{authors[0]} et al." if authors else "Unknown authors")
            
            # Look for key patterns
            if any(word in text for word in ['found', 'showed', 'demonstrated', 'results', 'concluded']):
                # Extract sentences with findings
                sentences = text.split('.')
                for sentence in sentences:
                    if any(word in sentence for word in ['found', 'showed', 'demonstrated', 'results', 'concluded']):
                        key_findings.append(sentence.strip().capitalize())
                        break
            
            if any(word in text for word in ['method', 'approach', 'algorithm', 'technique']):
                sentences = text.split('.')
                for sentence in sentences:
                    if any(word in sentence for word in ['method', 'approach', 'algorithm', 'technique']):
                        methodologies.append(sentence.strip().capitalize())
                        break
        
        # Construct answer
        answer_parts = []
        
        # Introduction
        if papers_mentioned:
            papers_list = list(papers_mentioned)[:3]  # Limit to top 3
            answer_parts.append(f"Based on research from {', '.join(papers_list)}, here's what I found:")
        
        # Key findings
        if key_findings:
            answer_parts.append("\n**Key Findings:**")
            for i, finding in enumerate(key_findings[:3], 1):  # Top 3 findings
                answer_parts.append(f"{i}. {finding}")
        
        # Methodologies
        if methodologies:
            answer_parts.append("\n**Approaches/Methods:**")
            for i, method in enumerate(methodologies[:2], 1):  # Top 2 methods
                answer_parts.append(f"{i}. {method}")
        
        # Fallback if no structured information found
        if not key_findings and not methodologies:
            # Provide a summary of the most relevant chunk
            most_relevant = search_results[0] if search_results else {}
            text = most_relevant.get('text', '')
            if text:
                answer_parts.append(f"The most relevant information found suggests: {text[:300]}...")
        
        # Add note about sources
        answer_parts.append(f"\nThis answer is based on {len(search_results)} relevant research paper excerpts. Please refer to the sources below for complete details.")
        
        return "\n".join(answer_parts) if answer_parts else "I could not generate a comprehensive answer from the available research papers."

    def get_paper_details(self, title: str) -> Optional[Dict[str, Any]]:
        """Get complete details for a specific paper."""
        return self.vector_db.get_paper_by_title(title)
    
    def list_papers(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all papers in the database."""
        papers = self.vector_db.list_papers(limit)
        
        # Add citation formatting
        for paper in papers:
            authors = paper.get('authors', [])
            title = paper.get('title', '')
            year = paper.get('year', 0)
            
            if authors and title:
                author_str = authors[0] if len(authors) == 1 else f"{authors[0]} et al."
                citation = f"{author_str} ({year if year > 0 else 'Unknown'}). {title}"
                paper['citation'] = citation
            else:
                paper['citation'] = 'Citation information incomplete'
        
        return papers
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.vector_db.get_database_stats()
    
    def delete_paper(self, title: str) -> bool:
        """Delete a paper from the database."""
        return self.vector_db.delete_paper(title)
    
    def batch_process_directory(
        self, 
        directory_path: str,
        file_pattern: str = "*.pdf",
        max_files: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process multiple PDF files from a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            file_pattern: File pattern to match (default: "*.pdf")
            max_files: Maximum number of files to process
            
        Returns:
            Batch processing results
        """
        directory = Path(directory_path)
        if not directory.exists():
            return {'success': False, 'error': f'Directory not found: {directory}'}
        
        # Find PDF files
        pdf_files = list(directory.glob(file_pattern))
        if max_files:
            pdf_files = pdf_files[:max_files]
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        results = {
            'total_files': len(pdf_files),
            'successful': 0,
            'failed': 0,
            'duplicates': 0,
            'processed_papers': [],
            'errors': []
        }
        
        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name}...")
            
            result = self.add_paper_from_pdf(str(pdf_file))
            
            if result['success']:
                results['successful'] += 1
                results['processed_papers'].append({
                    'file': pdf_file.name,
                    'title': result['title'],
                    'citation': result['citation']
                })
            elif result.get('existing'):
                results['duplicates'] += 1
                results['errors'].append({
                    'file': pdf_file.name,
                    'error': 'Duplicate paper',
                    'title': result.get('title', 'Unknown')
                })
            else:
                results['failed'] += 1
                results['errors'].append({
                    'file': pdf_file.name,
                    'error': result['error']
                })
        
        logger.info(f"Batch processing completed: {results['successful']} successful, "
                   f"{results['duplicates']} duplicates, {results['failed']} failed")
        
        return results

def main():
    """Example usage of the RAG system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Research Paper RAG System")
    parser.add_argument('--add', type=str, help='Add PDF file to the system')
    parser.add_argument('--search', type=str, help='Search for papers')
    parser.add_argument('--answer', type=str, help='Answer a question using RAG')
    parser.add_argument('--results', type=int, default=3, help='Number of results to return')
    parser.add_argument('--llm', type=str, default='anthropic', choices=['anthropic', 'openai', 'fallback', 'none'], 
                       help='LLM provider for answer generation')
    parser.add_argument('--list', action='store_true', help='List all papers')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--batch', type=str, help='Process all PDFs in directory')
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = ResearchPaperRAG(llm_provider=args.llm)
    
    if args.add:
        print(f"Adding paper: {args.add}")
        result = rag.add_paper_from_pdf(args.add)
        if result['success']:
            print(f"✅ Added: {result['title']}")
            print(f"📄 Citation: {result['citation']}")
            print(f"📊 Chunks: {result['chunks']}")
        else:
            print(f"❌ Failed: {result['error']}")
    
    elif args.search:
        print(f"Searching for: {args.search}")
        results = rag.search(args.search, n_results=args.results)
        
        if results:
            print(f"\n📚 Found {len(results)} relevant chunks:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['paper_title']}")
                print(f"   Authors: {', '.join(result['authors'])}")
                print(f"   Year: {result['year']}")
                print(f"   Similarity: {result['similarity']:.3f}")
                if result.get('combined_score'):
                    print(f"   Combined Score: {result['combined_score']:.3f}")
                if result.get('relevance_explanation'):
                    print(f"   Relevance: {result['relevance_explanation']}")
                print(f"   Text: {result['text'][:200]}...")
                print(f"   Citation: {result['citation']}")
        else:
            print("No results found.")
    
    elif args.answer:
        print(f"Answering question: {args.answer}")
        result = rag.answer_question(args.answer, n_results=args.results)
        

        llm_info = result.get('llm_metadata', {})
        model_info = f"{llm_info.get('provider', 'unknown')} ({llm_info.get('model', 'unknown')})"
        
        print(f"\n🤖 **Answer** (Confidence: {result['confidence']:.1%}, Model: {model_info}):")
        print(result['answer'])
        
        if result['sources']:
            print(f"\n📚 **Sources** ({len(result['sources'])} papers used):")
            for source in result['sources']:
                print(f"\n[{source['id']}] {source['title']}")
                print(f"    Authors: {', '.join(source['authors'])}")
                print(f"    Year: {source['year']}")
                print(f"    Relevance: {source['relevance_score']:.3f} ({source['relevance_explanation']})")
                print(f"    Excerpt: {source['text_excerpt']}")
                print(f"    Citation: {source['citation']}")
        
        # Show LLM usage info if available
        usage = llm_info.get('usage', {})
        if usage and llm_info.get('success'):
            print(f"\n🔧 **LLM Usage:**")
            if 'input_tokens' in usage:  # Anthropic format
                print(f"    Input tokens: {usage['input_tokens']}")
                print(f"    Output tokens: {usage['output_tokens']}")
            elif 'prompt_tokens' in usage:  # OpenAI format
                print(f"    Prompt tokens: {usage['prompt_tokens']}")
                print(f"    Completion tokens: {usage['completion_tokens']}")
                print(f"    Total tokens: {usage['total_tokens']}")
        
        print(f"\n📊 Used {result['context_used']} context chunks for this answer.")
    
    elif args.list:
        papers = rag.list_papers()
        print(f"\n📚 Database contains {len(papers)} papers:")
        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper['title']}")
            print(f"   Authors: {', '.join(paper['authors'])}")
            print(f"   Year: {paper['year']}")
            print(f"   Chunks: {paper['chunk_count']}")
            print(f"   Citation: {paper['citation']}")
    
    elif args.stats:
        stats = rag.get_stats()
        print(f"\n📊 Database Statistics:")
        print(f"   Total papers: {stats['total_papers']}")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Average chunks per paper: {stats['average_chunks_per_paper']:.1f}")
        print(f"   Database path: {stats['database_path']}")
        
        if stats.get('papers_by_year'):
            print(f"\n📅 Papers by year:")
            for year, count in sorted(stats['papers_by_year'].items()):
                print(f"   {year}: {count} papers")
    
    elif args.batch:
        print(f"Batch processing directory: {args.batch}")
        results = rag.batch_process_directory(args.batch)
        
        print(f"\n📊 Batch Results:")
        print(f"   Total files: {results['total_files']}")
        print(f"   Successful: {results['successful']}")
        print(f"   Duplicates: {results['duplicates']}")
        print(f"   Failed: {results['failed']}")
        
        if results['processed_papers']:
            print(f"\n✅ Successfully processed papers:")
            for paper in results['processed_papers']:
                print(f"   - {paper['title']}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()