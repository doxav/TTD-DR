import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Tavily Search
try:
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

# DuckDuckGo Search
try:
    from duckduckgo_search import DDGS
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False

# Naver Search (Optional, only for Korean)
try:
    from langchain_naver_community.utils import NaverSearchAPIWrapper
    NAVER_AVAILABLE = True
except ImportError:
    NAVER_AVAILABLE = False

# Import SearchResultFormatter from utils
from .utils import SearchResultFormatter


class WebSearchTool:
    """Web search tool that combines multiple search engines"""
    
    def __init__(self):
        """Initialize search clients"""
        self.tavily_client = None
        self.ddgs_client = None
        self.naver_client = None
        self.search_cache = {}  # Search result caching
        self._initialize_search_clients()
    
    def _initialize_search_clients(self):
        """Initialize available search clients"""
        # Initialize Tavily if API key is available
        tavily_api_key = os.getenv('TAVILY_API_KEY')
        if TAVILY_AVAILABLE and tavily_api_key and not tavily_api_key.startswith('tvly-your-'):
            try:
                self.tavily_client = TavilySearch(api_key=tavily_api_key)
                print("Tavily search tool initialized")
            except Exception as e:
                print(f"Tavily initialization failed: {e}")
        else:
            print("Tavily API key not found or is placeholder")
        
        # Initialize DuckDuckGo (always available)
        if DUCKDUCKGO_AVAILABLE:
            try:
                self.ddgs_client = DDGS()
                print("DuckDuckGo search tool initialized")
            except Exception as e:
                print(f"DuckDuckGo initialization failed: {e}")
        else:
            print("DuckDuckGo search tool not available.")
        
        # Initialize Naver (if client credentials are available)
        naver_client_id = os.getenv('NAVER_CLIENT_ID')
        naver_client_secret = os.getenv('NAVER_CLIENT_SECRET')
        if NAVER_AVAILABLE and naver_client_id and naver_client_secret:
            try:
                self.naver_client = NaverSearchAPIWrapper()
                print("Naver search tool initialized")
            except Exception as e:
                print(f"Naver initialization failed: {e}")
        else:
            print("Naver client credentials not found")
    
    def search_with_tavily(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search using LangChain TavilySearch"""
        if not self.tavily_client:
            return []
        
        try:
            # Configure LangChain TavilySearch max_results
            self.tavily_client.max_results = max_results
            
            # Execute search
            response = self.tavily_client.invoke(query)
            
            # Extract results
            search_results = response.get('results', [])
            
            results = []
            for result in search_results:
                results.append({
                    'title': result.get('title', 'No Title'),
                    'url': result.get('url', ''),
                    'content': result.get('content', ''),
                    'score': result.get('score', 0.7),  # Tavily provides scores
                    'source': 'tavily',
                    'published_date': result.get('published_date', ''),
                    'raw_content': result.get('raw_content', result.get('content', ''))
                })
            
            return results
            
        except Exception as e:
            print(f"Tavily search error: {e}")
            return []
    
    def search_with_duckduckgo(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo"""
        if not self.ddgs_client:
            return []
        
        try:
            results = []
            ddg_results = self.ddgs_client.text(
                keywords=query,
                max_results=max_results,
                safesearch='moderate',
                region='kr-ko',  # Korean region for better local results
                timelimit='y'  # Last year for more recent results
            )
            
            for result in ddg_results:
                results.append({
                    'title': result.get('title', 'No Title'),
                    'url': result.get('href', ''),
                    'content': result.get('body', ''),
                    'score': 0.7,  # Default score for DDG
                    'source': 'duckduckgo',
                    'published_date': '',
                    'raw_content': result.get('body', '')
                })
            
            return results
            
        except Exception as e:
            error_msg = str(e)
            print(f"DuckDuckGo search error: {error_msg}")
            
            # Reinitialize client on rate limit or specific errors
            if 'ratelimit' in error_msg.lower() or 'rate' in error_msg.lower() or 'previous call' in error_msg.lower():
                print("DuckDuckGo client reinitializing...")
                try:
                    self.ddgs_client = DDGS()
                    print("DuckDuckGo client reinitialized successfully")
                except Exception as reinit_error:
                    print(f"DuckDuckGo client reinitialization failed: {reinit_error}")
                    self.ddgs_client = None
            
            return []
    
    def search_with_naver(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search using Naver Search API"""
        if not self.naver_client:
            return []
        
        try:
            results = []
            naver_results = self.naver_client.results(query)
            
            for result in naver_results:
                results.append({
                    'title': result.get('title', 'No Title'),
                    'url': result.get('link', ''),
                    'content': result.get('description', ''),
                    'score': 0.8,  # Default score for Naver
                    'source': 'naver',
                    'published_date': '',
                    'raw_content': result.get('description', '')
                })
            
            return results
            
        except Exception as e:
            error_msg = str(e)
            print(f"Naver search error: {error_msg}")
            
            # Reinitialize client on rate limit or specific errors
            if 'ratelimit' in error_msg.lower() or 'rate' in error_msg.lower() or 'previous call' in error_msg.lower():
                print("Naver client reinitializing...")
                try:
                    self.naver_client = NaverSearchAPIWrapper()
                    print("Naver client reinitialized successfully")
                except Exception as reinit_error:
                    print(f"Naver client reinitialization failed: {reinit_error}")
                    self.naver_client = None
            
            return []
    
    def search(self, query: str, max_results: int = 5, enabled_engines: List[str] = None) -> List[Dict[str, Any]]:
        """Perform comprehensive web search using available search engines"""
        # Default to all engines if not specified
        if enabled_engines is None:
            enabled_engines = ['tavily', 'duckduckgo', 'naver']
        
        # Check cache (including engine settings)
        cache_key = f"{query}:{max_results}:{':'.join(sorted(enabled_engines))}"
        if cache_key in self.search_cache:
            print(f"Using cached search results: {query}")
            return self.search_cache[cache_key]
        
        all_results = []
        
        print(f"Searching for: {query}")
        print(f"Enabled engines: {enabled_engines}")
        
        # Try Tavily first (if enabled and available)
        if 'tavily' in enabled_engines and self.tavily_client:
            tavily_results = self.search_with_tavily(query, max_results=max(2, max_results // 2))
            all_results.extend(tavily_results)
            print(f"  Tavily: {len(tavily_results)} results")
        elif 'tavily' in enabled_engines:
            print(f"  Tavily: disabled (not available)")
        
        # Use DuckDuckGo as backup or primary (if enabled and available)
        if 'duckduckgo' in enabled_engines and self.ddgs_client:
            remaining_results = max(1, max_results - len(all_results))
            ddg_results = self.search_with_duckduckgo(query, max_results=remaining_results)
            all_results.extend(ddg_results)
            print(f"  DuckDuckGo: {len(ddg_results)} results")
        elif 'duckduckgo' in enabled_engines:
            print(f"  DuckDuckGo: disabled (not available)")
        
        # Use Naver as a secondary option (if enabled and available)
        if 'naver' in enabled_engines and self.naver_client:
            remaining_results = max(1, max_results - len(all_results))
            naver_results = self.search_with_naver(query, max_results=remaining_results)
            all_results.extend(naver_results)
            print(f"  Naver: {len(naver_results)} results")
        elif 'naver' in enabled_engines:
            print(f"  Naver: disabled (not available)")
        
        # Fallback to mock if no search engines available or enabled
        if not all_results:
            print("No search engines available or enabled, using mock results")
            all_results = [{
                'title': f"Mock result for: {query}",
                'url': f"https://example.com/mock/{uuid.uuid4().hex[:8]}",
                'content': f"This is a mock search result for query: {query}. Please configure search engines or check availability.",
                'score': 0.5,
                'source': 'mock',
                'published_date': '',
                'raw_content': f"Mock content for {query}"
            }]
        
        # Sort by relevance score and return top results
        sorted_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
        final_results = sorted_results[:max_results]
        
        # Cache results
        self.search_cache[cache_key] = final_results
        
        print(f"  Total: {len(final_results)} results")
        return final_results
    
    def is_available(self) -> bool:
        """Check if any search engine is available"""
        return self.tavily_client is not None or self.ddgs_client is not None or self.naver_client is not None
    
    def get_status(self) -> Dict[str, bool]:
        """Get status of all search engines"""
        return {
            'tavily': self.tavily_client is not None,
            'duckduckgo': self.ddgs_client is not None,
            'naver': self.naver_client is not None,
            'any_available': self.is_available()
        }


# Global web search tool instance (lazy initialized)
_web_search_tool = None

def get_web_search_tool() -> WebSearchTool:
    """Get or create the global web search tool instance"""
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = WebSearchTool()
    return _web_search_tool


def search_web(query: str, max_results: int = 5, enabled_engines: List[str] = None) -> List[Dict[str, Any]]:
    """Convenience function for web search"""
    return get_web_search_tool().search(query, max_results, enabled_engines)


def format_search_results(results: List[Dict[str, Any]], query: str) -> str:
    """Convenience function for formatting search results"""
    return SearchResultFormatter.format_results_for_llm(results, query)


def get_search_tool_status() -> Dict[str, bool]:
    """Get the status of available search tools"""
    return get_web_search_tool().get_status() 