#!/usr/bin/env python3
"""

LLM Integration for Enhanced RAG Answer Generation

This module integrates various LLM APIs (Anthropic Claude, OpenAI) 
for generating high-quality answers from retrieved research paper context.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import json

try:
    import anthropic
except ImportError:
    anthropic = None
    
try:
    import openai
except ImportError:
    openai = None

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_answer(
        self, 
        question: str, 
        context: str, 
        search_results: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate an answer based on question and context."""
        pass

class AnthropicProvider(LLMProvider):
    """Anthropic Claude integration for answer generation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (if None, reads from ANTHROPIC_API_KEY env var)
            model: Claude model to use
        """
        if not anthropic:
            raise ImportError("anthropic library not installed. Run: pip install anthropic")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
        
        self.model = model
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def generate_answer(
        self, 
        question: str, 
        context: str, 
        search_results: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Generate answer using Claude API.
        
        Args:
            question: User's question
            context: Retrieved context from papers
            search_results: List of search results with metadata
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0)
            
        Returns:
            Dictionary with generated answer and metadata
        """
        # Create structured context with paper information
        structured_context = self._create_structured_context(search_results)
        
        # Create system prompt
        system_prompt = self._create_system_prompt()
        
        # Create user prompt with question and context
        user_prompt = self._create_user_prompt(question, structured_context)
        
        try:
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            )
            
            answer_text = response.content[0].text
            
            return {
                'answer': answer_text,
                'model': self.model,
                'provider': 'anthropic',
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens
                },
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return {
                'answer': f"Sorry, I encountered an error generating the answer: {str(e)}",
                'model': self.model,
                'provider': 'anthropic',
                'success': False,
                'error': str(e)
            }
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for Claude."""
        return """You are an expert research assistant specializing in academic paper analysis. Your task is to provide comprehensive, accurate answers to questions based on research paper excerpts.

Guidelines:
1. Base your answers ONLY on the provided research paper excerpts
2. Synthesize information from multiple sources when relevant
3. Cite specific papers when making claims (use [Source X] format)
4. If the context doesn't contain enough information, clearly state this
5. Structure your response with clear sections when appropriate
6. Use academic language while remaining accessible
7. Highlight key findings, methodologies, and results
8. Point out any contradictions or different perspectives between papers
9. Include quantitative results when available
10. Conclude with a brief summary if the answer is complex

Format your response as:
**Answer:**
[Your comprehensive answer here]

**Key Points:**
- Point 1
- Point 2
- etc.

**Sources Referenced:** [List the papers you cited]
"""
    
    def _create_user_prompt(self, question: str, context: str) -> str:
        """Create user prompt combining question and context."""
        return f"""Question: {question}

Research Paper Context:
{context}

Please provide a comprehensive answer based on the research paper excerpts above."""
    
    def _create_structured_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Create well-structured context from search results."""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            paper_title = result.get('paper_title', 'Unknown Title')
            authors = result.get('authors', ['Unknown Authors'])
            year = result.get('year', 'Unknown Year')
            text = result.get('text', '')
            section = result.get('section', 'Content')
            
            # Format author list
            if len(authors) == 1:
                author_str = authors[0]
            elif len(authors) <= 3:
                author_str = ', '.join(authors[:-1]) + f" and {authors[-1]}"
            else:
                author_str = f"{authors[0]} et al."
            
            context_part = f"""[Source {i}] {paper_title}
Authors: {author_str} ({year})
Section: {section}
Content: {text}

"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)

class OpenAIProvider(LLMProvider):
    """OpenAI integration for answer generation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: GPT model to use
        """
        if not openai:
            raise ImportError("openai library not installed. Run: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def generate_answer(
        self, 
        question: str, 
        context: str, 
        search_results: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Generate answer using OpenAI API."""
        # Create structured context
        structured_context = self._create_structured_context(search_results)
        
        # Create system message
        system_message = self._create_system_message()
        
        # Create user message
        user_message = f"""Question: {question}

Research Paper Context:
{structured_context}

Please provide a comprehensive answer based on the research paper excerpts above."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answer_text = response.choices[0].message.content
            
            return {
                'answer': answer_text,
                'model': self.model,
                'provider': 'openai',
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'success': True
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {
                'answer': f"Sorry, I encountered an error generating the answer: {str(e)}",
                'model': self.model,
                'provider': 'openai',
                'success': False,
                'error': str(e)
            }
    
    def _create_system_message(self) -> str:
        """Create system message for OpenAI."""
        return """You are an expert research assistant specializing in academic paper analysis. Provide comprehensive, accurate answers based on research paper excerpts. Always cite sources using [Source X] format and base answers only on provided context."""
    
    def _create_structured_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Create structured context from search results."""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            paper_title = result.get('paper_title', 'Unknown Title')
            authors = result.get('authors', ['Unknown Authors'])
            year = result.get('year', 'Unknown Year')
            text = result.get('text', '')
            
            author_str = authors[0] if len(authors) == 1 else f"{authors[0]} et al."
            
            context_part = f"[Source {i}] {paper_title} ({author_str}, {year})\n{text}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)

class LLMManager:
    """Manager class for different LLM providers."""
    
    def __init__(self, provider: str = "anthropic", **kwargs):
        """
        Initialize LLM manager with specified provider.
        
        Args:
            provider: LLM provider ('anthropic', 'openai', 'fallback')
            **kwargs: Provider-specific arguments
        """
        self.provider_name = provider
        self.provider = self._initialize_provider(provider, **kwargs)
    
    def _initialize_provider(self, provider: str, **kwargs) -> Optional[LLMProvider]:
        """Initialize the specified LLM provider."""
        if provider == "anthropic":
            try:
                return AnthropicProvider(**kwargs)
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic provider: {e}")
                return None
        
        elif provider == "openai":
            try:
                return OpenAIProvider(**kwargs)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI provider: {e}")
                return None
        
        elif provider == "fallback":
            # Try Anthropic first, then OpenAI
            for prov in ["anthropic", "openai"]:
                try:
                    return self._initialize_provider(prov, **kwargs)
                except Exception:
                    continue
            return None
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def generate_answer(
        self, 
        question: str, 
        context: str, 
        search_results: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate answer using the configured provider."""
        if self.provider is None:
            return {
                'answer': "LLM provider not available. Please check your API keys and configuration.",
                'provider': self.provider_name,
                'success': False,
                'error': "Provider not initialized"
            }
        
        return self.provider.generate_answer(question, context, search_results, **kwargs)
    
    def is_available(self) -> bool:
        """Check if LLM provider is available."""
        return self.provider is not None

# Fallback answer generator (uses the existing logic)
def generate_fallback_answer(question: str, context: str, search_results: List[Dict]) -> str:
    """
    Fallback answer generation when LLM APIs are not available.
    Uses the existing rule-based approach.
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
    
    if papers_mentioned:
        papers_list = list(papers_mentioned)[:3]
        answer_parts.append(f"Based on research from {', '.join(papers_list)}, here's what I found:")
    
    if key_findings:
        answer_parts.append("\n**Key Findings:**")
        for i, finding in enumerate(key_findings[:3], 1):
            answer_parts.append(f"{i}. {finding}")
    
    if methodologies:
        answer_parts.append("\n**Approaches/Methods:**")
        for i, method in enumerate(methodologies[:2], 1):
            answer_parts.append(f"{i}. {method}")
    
    if not key_findings and not methodologies:
        most_relevant = search_results[0] if search_results else {}
        text = most_relevant.get('text', '')
        if text:
            answer_parts.append(f"The most relevant information found suggests: {text[:300]}...")
    
    answer_parts.append(f"\nThis answer is based on {len(search_results)} relevant research paper excerpts.")
    
    return "\n".join(answer_parts) if answer_parts else "I could not generate a comprehensive answer from the available research papers."