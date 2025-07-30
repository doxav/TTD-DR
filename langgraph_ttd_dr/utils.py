import re
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime



# Text Processing Utilities
def clean_reasoning_tags(text: str) -> str:
    """Remove reasoning tags from model responses for clean final output"""
    if not text:
        return text
    
    reasoning_patterns = [
        r'<think>.*?</think>',
        r'<thinking>.*?</thinking>',
        r'<reasoning>.*?</reasoning>',
        r'<thought>.*?</thought>',
        r'<reflect>.*?</reflect>',
        r'<reflection>.*?</reflection>',
    ]
    
    cleaned_text = text
    for pattern in reasoning_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up whitespace
    cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'  +', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text


def cleanup_placeholder_tags(text: str) -> str:
    """Remove research placeholder tags from final report"""
    if not text:
        return text
    
    placeholder_patterns = [
        r'\[NEEDS RESEARCH[^\]]*\]',
        r'\[SOURCE NEEDED[^\]]*\]', 
        r'\[RESEARCH NEEDED[^\]]*\]',
        r'\[CITATION NEEDED[^\]]*\]',
        r'\[MORE RESEARCH NEEDED[^\]]*\]',
        r'\[REQUIRES INVESTIGATION[^\]]*\]',
        r'\[TO BE RESEARCHED[^\]]*\]',
        r'\[VERIFY[^\]]*\]',
        r'\[CHECK[^\]]*\]',
        r'\[Placeholder for[^\]]+\]',
        r'\[\d+\]\s*\[Placeholder[^\]]+\]',
        r'\[Insert citation[^\]]*\]',  
        r'\[Add reference[^\]]*\]',
        r'\[Reference needed[^\]]*\]',
    ]
    
    cleaned_text = text
    for pattern in placeholder_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text


def remove_references_section(text: str) -> str:
    """Remove any existing References section from text"""
    # Remove References sections
    text = re.sub(r'##\s*References.*?(?=##|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'(?m)^References\s*\n\s*(?:\[\d+\]\s*\n)+', '', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text


# Search Result Formatting Utilities
class SearchResultFormatter:
    """Utility class for formatting search results"""
    
    @staticmethod
    def format_results_for_llm(results: List[Dict[str, Any]], query: str) -> str:
        """Format search results for LLM consumption"""
        if not results:
            return f"No search results found for query: {query}"
        
        formatted = f"Web Search Results for: {query}\n\n"
        
        for i, result in enumerate(results, 1):
            formatted += f"{i}. **{result['title']}**\n"
            formatted += f"   URL: {result['url']}\n"
            formatted += f"   Source: {result['source']}\n"
            if result['published_date']:
                formatted += f"   Date: {result['published_date']}\n"
            formatted += f"   Content: {result['content'][:300]}...\n\n"
        
        return formatted
    
    @staticmethod
    def create_source_metadata(result: Dict[str, Any], gap_id: str = None) -> Dict[str, Any]:
        """Create standardized source metadata for research state"""
        return {
            'id': str(uuid.uuid4()),
            'title': result['title'],
            'url': result['url'],
            'content': result['content'],
            'raw_content': result.get('raw_content', ''),
            'access_date': datetime.now().strftime('%Y-%m-%d'),
            'published_date': result.get('published_date', ''),
            'relevance_score': result['score'],
            'search_engine': result['source'],
            'gap_addressed': gap_id,
            'content_length': len(result['content']),
            'extraction_timestamp': datetime.now().isoformat()
        }


# Gap Analysis Utilities
def parse_gap_analysis(content: str) -> List[Dict[str, Any]]:
    """Parse gap analysis response into structured gaps"""
    gaps = []
    current_gap = {}
    
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('GAP_ID:'):
            if current_gap:
                gaps.append(current_gap)
            current_gap = {'id': line.split(':', 1)[1].strip()}
        elif line.startswith('SECTION:'):
            current_gap['section'] = line.split(':', 1)[1].strip()
        elif line.startswith('GAP_TYPE:'):
            current_gap['gap_type'] = line.split(':', 1)[1].strip()
        elif line.startswith('SPECIFIC_NEED:'):
            current_gap['specific_need'] = line.split(':', 1)[1].strip()
        elif line.startswith('SEARCH_QUERY:'):
            current_gap['search_query'] = line.split(':', 1)[1].strip()
        elif line.startswith('PRIORITY:'):
            current_gap['priority'] = line.split(':', 1)[1].strip()
    
    if current_gap:
        gaps.append(current_gap)
    
    return gaps


def create_fallback_gap(query: str) -> List[Dict[str, Any]]:
    """Create a fallback gap when parsing fails"""
    return [{
        'id': '1',
        'section': 'General',
        'gap_type': 'MISSING_INFO',
        'specific_need': 'More detailed information needed',
        'search_query': query,
        'priority': 'HIGH'
    }]


# Quality Evaluation Utilities
def parse_quality_scores(content: str) -> Dict[str, float]:
    """Parse quality evaluation scores from LLM response"""
    scores = {}
    for line in content.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            try:
                scores[key] = float(value.strip())
            except ValueError:
                scores[key] = 0.5  # Default score
    
    return scores


def update_component_fitness(
    current_fitness: Dict[str, float], 
    improvement: float
) -> Dict[str, float]:
    """
    Update component fitness based on improvement score
    
    Args:
        current_fitness: Current fitness scores
        improvement: Improvement score from quality evaluation
        
    Returns:
        Updated fitness scores
    """
    updated_fitness = dict(current_fitness)
    
    if improvement > 0.1:  # Significant improvement
        updated_fitness['search_strategy'] *= 1.1
        updated_fitness['synthesis_quality'] *= 1.1
        updated_fitness['integration_ability'] *= 1.1
    elif improvement < 0.05:  # Poor improvement
        updated_fitness['search_strategy'] *= 0.95
        updated_fitness['synthesis_quality'] *= 0.95
    
    # Cap fitness values
    for key in updated_fitness:
        updated_fitness[key] = max(0.1, min(2.0, updated_fitness[key]))
    
    return updated_fitness


# Citation Utilities
def build_references_section(citations: Dict[int, Dict[str, Any]]) -> str:
    """
    Build a properly formatted references section
    
    Args:
        citations: Dictionary of citation number to source mapping
        
    Returns:
        Formatted references section
    """
    if not citations:
        return ""
    
    references = "\n\n## References\n\n"
    for num, source in sorted(citations.items()):
        title = source.get('title', 'Untitled')
        url = source.get('url', '')
        access_date = source.get('access_date', datetime.now().strftime('%Y-%m-%d'))
        references += f"[{num}] {title}. Available at: <{url}> [Accessed: {access_date}]\n\n"
    
    return references


def build_metadata_section(state: Dict[str, Any]) -> str:
    """
    Build TTD-DR metadata section for final report
    
    Args:
        state: Research state dictionary
        
    Returns:
        Formatted metadata section
    """
    metadata = "\n---\n\n**TTD-DR Research Metadata:**\n"
    metadata += f"- Algorithm: Test-Time Diffusion Deep Researcher\n"
    metadata += f"- Denoising iterations: {state.get('iteration', 0)}\n"
    metadata += f"- Total gaps addressed: {sum(len(gaps) for gaps in state.get('gap_history', []))}\n"
    metadata += f"- Component fitness: {state.get('component_fitness', {})}\n"
    metadata += f"- Total sources consulted: {len(state.get('citations', {}))}\n"
    metadata += f"- Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    metadata += f"- Total tokens used: {state.get('total_tokens', 0)}\n"
    
    return metadata


# Termination Logic Utilities
def should_terminate_research(state: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Determine if research should terminate based on TTD-DR criteria
    
    Args:
        state: Current research state
        
    Returns:
        Tuple of (should_terminate, termination_reason)
    """
    # Check if termination reason was already set
    if state.get("termination_reason"):
        return True, state["termination_reason"]
    
    # Check iteration count
    if state.get("iteration", 0) >= state.get("max_iterations", 5):
        return True, "max_iterations_reached"
    
    # Check quality metrics
    quality_metrics = state.get("quality_metrics", {})
    completeness = quality_metrics.get("completeness", 0.0)
    improvement = quality_metrics.get("improvement", 0.0)
    
    # TTD-DR termination conditions
    if completeness > 0.9:
        return True, "high_quality_achieved"
    elif improvement < 0.03 and completeness > 0.7:
        return True, "minimal_improvement"
    
    return False, None


# Error Handling Utilities
def create_error_metadata(
    session_id: str, 
    start_time: float, 
    error: Exception
) -> Dict[str, Any]:
    """
    Create standardized error metadata
    
    Args:
        session_id: Research session ID
        start_time: Start time timestamp
        error: Exception that occurred
        
    Returns:
        Error metadata dictionary
    """
    import time
    
    return {
        "session_id": session_id,
        "execution_time": time.time() - start_time,
        "total_tokens": 0,
        "iterations_completed": 0,
        "sources_consulted": 0,
        "citations_count": 0,
        "termination_reason": "error",
        "quality_metrics": {},
        "component_fitness": {},
        "status": "failed",
        "error": str(error),
        "error_type": type(error).__name__,
        "error_messages": [str(error)]
    }


# Validation Utilities
def validate_search_engines(search_engines: List[str]) -> List[str]:
    """
    Validate and filter search engines list
    
    Args:
        search_engines: List of search engine names
        
    Returns:
        Validated list of search engines
    """
    valid_engines = ['tavily', 'duckduckgo', 'naver']
    validated = [engine for engine in search_engines if engine in valid_engines]
    
    if not validated:
        # Return default if none are valid
        return ['tavily', 'duckduckgo', 'naver']
    
    return validated


def validate_search_results_per_gap(results_per_gap: int) -> int:
    """
    Validate search results per gap parameter
    
    Args:
        results_per_gap: Number of results per gap
        
    Returns:
        Validated number (between 1 and 10)
    """
    return max(1, min(10, results_per_gap)) 