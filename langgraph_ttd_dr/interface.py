import asyncio
import logging
import time
import uuid
from typing import Optional, Dict, Any, Tuple, AsyncGenerator, List
from contextlib import asynccontextmanager
import os

from .state import create_initial_state, TTDResearchState
from .workflow import create_ttd_dr_workflow, TTDResearchWorkflow
from .utils import create_error_metadata, validate_search_engines, validate_search_results_per_gap
from .client_factory import create_openai_client, get_model_name, detect_config


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TTDResearcher:
    """Main interface for the Test-Time Diffusion Deep Researcher"""
    
    def __init__(
        self,
        client: Any = None,
        model_name: str = None,
        max_iterations: int = None,
        max_sources: int = None,
        temperature: float = 0.7,
        enable_checkpointing: bool = False,
        search_engines: List[str] = None,
        search_results_per_gap: int = 3,
        use_azure: Optional[bool] = None
    ):
        """Initialize TTD Researcher with configuration parameters"""
        # Create client if not provided
        if client is None:
            try:
                client = create_openai_client(use_azure=use_azure)
            except Exception as e:
                config = detect_config()
                logger.error(f"Failed to create client: {str(e)}")
                logger.info("Available configuration:")
                logger.info(f"  Azure OpenAI: {'Available' if config['has_azure'] else 'Not available'}")
                logger.info(f"  OpenAI: {'Available' if config['has_openai'] else 'Not available'}")
                raise
        
        self.client = client
        self.model_name = model_name or get_model_name(use_azure)
        self.max_iterations = max_iterations if max_iterations is not None else int(os.getenv('MAX_ITERATIONS', '5'))
        self.max_sources = max_sources if max_sources is not None else int(os.getenv('MAX_SOURCES', '20'))
        self.temperature = temperature
        self.enable_checkpointing = enable_checkpointing
        
        # Validate and set search configurations using utility functions
        default_engines = ['tavily', 'duckduckgo', 'naver']
        self.search_engines = validate_search_engines(search_engines if search_engines is not None else default_engines)
        self.search_results_per_gap = validate_search_results_per_gap(search_results_per_gap)
        
        self.workflow = None
        
        # Initialize checkpointer for human feedback sessions
        from langgraph.checkpoint.memory import MemorySaver
        self.checkpointer = MemorySaver()
        
        logger.info(f"TTD Researcher initialized with model: {model_name}")
        logger.info(f"Max iterations: {self.max_iterations}, Max sources: {self.max_sources}")
        logger.info(f"Search engines: {self.search_engines}")
        logger.info(f"Search results per gap: {self.search_results_per_gap}")
        logger.info(f"Recursion limit will be: {os.getenv('RECURSION_LIMIT', '50')}")
    
    def _get_workflow(self) -> TTDResearchWorkflow:
        """Get or create workflow instance"""
        if self.workflow is None:
            # Create workflow with client injection
            self.workflow = create_ttd_dr_workflow(client=self.client)
            # Compile the workflow to create the app
            self.workflow.compile()
            logger.info("TTD-DR workflow created and compiled")
        return self.workflow
    
    def research(
        self,
        query: str,
        system_prompt: str = "You are an expert research assistant.",
        config: Optional[Dict[str, Any]] = None,
        user_feedback: Optional[str] = None  # Kept for backward compatibility but not used
    ) -> Tuple[str, Dict[str, Any]]:
        """Conduct research using TTD-DR algorithm"""
        start_time = time.time()
        session_id = str(uuid.uuid4())
        
        try:
            # Create initial state for research
            initial_state = create_initial_state(
                original_query=query,
                system_prompt=system_prompt,
                model_name=self.model_name,
                max_iterations=self.max_iterations,
                max_sources=self.max_sources,
                temperature=self.temperature,
                client=self.client,
                search_engines=self.search_engines,
                search_results_per_gap=self.search_results_per_gap
            )
            
            # Apply config overrides
            if config:
                initial_state.update(config)
            
            # Get workflow
            workflow = self._get_workflow()
            
            # Prepare invocation config - disable checkpointing for serialization issues
            invoke_config = {} if not self.enable_checkpointing else {"configurable": {"thread_id": session_id}}
            
            # Set recursion limit for workflow execution from environment variable
            recursion_limit = int(os.getenv('RECURSION_LIMIT', '50'))
            invoke_config["recursion_limit"] = recursion_limit
            
            # Execute workflow
            logger.info("Executing TTD-DR workflow...")
            try:
                final_state = workflow.app.invoke(initial_state, config=invoke_config)
                logger.debug(f"Workflow execution completed. Final state keys: {list(final_state.keys())}")
                logger.debug(f"Final status: {final_state.get('status')}")
                logger.debug(f"Current step: {final_state.get('current_step')}")
                if final_state.get('error_messages'):
                    logger.warning(f"Workflow errors: {final_state.get('error_messages')}")
            except Exception as workflow_error:
                logger.error(f"Workflow execution failed: {str(workflow_error)}")
                raise
            
            # Extract results
            final_report = final_state.get("current_draft", "")
            
            # Prepare metadata
            research_metadata = {
                "session_id": session_id,
                "execution_time": time.time() - start_time,
                "total_tokens": final_state.get("total_tokens", 0),
                "iterations_completed": final_state.get("iteration", 0),
                "sources_consulted": len(final_state.get("sources", [])),
                "citations_count": len(final_state.get("citations", {})),
                "termination_reason": final_state.get("termination_reason"),
                "quality_metrics": final_state.get("quality_metrics", {}),
                "component_fitness": final_state.get("component_fitness", {}),
                "status": final_state.get("status", "unknown"),
                "error_messages": final_state.get("error_messages", []),
                "original_query": query,
                "clarified_query": final_state.get("clarified_query"),
                "human_feedback_used": False # Clarification logic removed
            }
            
            logger.info(f"Research completed in {research_metadata['execution_time']:.2f}s")
            logger.info(f"Iterations: {research_metadata['iterations_completed']}")
            logger.info(f"Sources: {research_metadata['sources_consulted']}")
            logger.info(f"Tokens: {research_metadata['total_tokens']}")
            
            return final_report, research_metadata
            
        except Exception as e:
            logger.error(f"Research failed: {str(e)}")
            
            # Use utility function to create error metadata
            error_metadata = create_error_metadata(session_id, start_time, e)
            
            return f"Research failed: {str(e)}", error_metadata
    
    async def research_async(
        self,
        query: str,
        system_prompt: str = "You are a research assistant.",
        session_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Async version of research method"""
        # Run synchronous research in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.research, query, system_prompt, config
        )
    
    def stream_research(
        self,
        query: str,
        system_prompt: str = "You are a research assistant.",
        session_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream research progress in real-time"""
        try:
            # Generate session ID if not provided
            if session_id is None:
                session_id = str(uuid.uuid4())
            
            # Create initial state
            initial_state = create_initial_state(
                original_query=query,
                system_prompt=system_prompt,
                model_name=self.model_name,
                max_iterations=self.max_iterations,
                max_sources=self.max_sources,
                temperature=self.temperature,
                client=self.client,
                search_engines=self.search_engines,
                search_results_per_gap=self.search_results_per_gap
            )
            
            # Apply config overrides
            if config:
                initial_state.update(config)
            
            # Get workflow
            workflow = self._get_workflow()
            
            # Prepare invocation config
            invoke_config = {"configurable": {"thread_id": session_id}} if self.enable_checkpointing else {}
            
            # Stream workflow execution
            for step_output in workflow.app.stream(initial_state, config=invoke_config):
                for node_name, node_output in step_output.items():
                    yield {
                        "node": node_name,
                        "output": node_output,
                        "session_id": session_id,
                        "timestamp": time.time()
                    }
            
        except Exception as e:
            logger.error(f"Streaming research failed: {str(e)}")
            yield {
                "error": str(e),
                "error_type": type(e).__name__,
                "session_id": session_id,
                "timestamp": time.time()
            }
    
    def get_session_state(self, session_id: str) -> Optional[TTDResearchState]:
        """Get the current state of a research session"""
        if not self.enable_checkpointing:
            logger.warning("Checkpointing not enabled - cannot retrieve session state")
            return None
        
        try:
            workflow = self._get_workflow()
            config = {"configurable": {"thread_id": session_id}}
            
            # Get state history
            states = list(workflow.app.get_state_history(config))
            if states:
                return states[0].values  # Most recent state
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session state: {str(e)}")
            return None
    
    def list_sessions(self) -> List[str]:
        """List all active research sessions"""
        return list(self._session_configs.keys())
    
    def visualize_workflow(self):
        """Display workflow visualization"""
        workflow = self._get_workflow()
        workflow.visualize()
    
    def get_workflow_diagram(self) -> str:
        """Get Mermaid diagram of workflow"""
        workflow = self._get_workflow()
        return workflow.get_mermaid_diagram()


# Convenience functions for quick usage
def research(
    query: str,
    client: Any = None,
    model_name: str = None,
    max_iterations: int = 5,
    max_sources: int = 30,
    temperature: float = 0.7,
    system_prompt: str = "You are a research assistant.",
    search_engines: List[str] = None,
    search_results_per_gap: int = 3,
    use_azure: Optional[bool] = None
) -> Tuple[str, Dict[str, Any]]:
    """Simple TTD-DR research function"""
    researcher = TTDResearcher(
        client=client,
        model_name=model_name,
        max_iterations=max_iterations,
        max_sources=max_sources,
        temperature=temperature,
        enable_checkpointing=False,
        search_engines=search_engines,
        search_results_per_gap=search_results_per_gap,
        use_azure=use_azure
    )
    
    return researcher.research(query, system_prompt)


async def research_async(
    query: str,
    client: Any = None,
    model_name: str = None,
    max_iterations: int = 5,
    max_sources: int = 30,
    temperature: float = 0.7,
    system_prompt: str = "You are a research assistant.",
    search_engines: List[str] = None,
    search_results_per_gap: int = 3,
    use_azure: Optional[bool] = None
) -> Tuple[str, Dict[str, Any]]:
    """Async TTD-DR research function"""
    researcher = TTDResearcher(
        client=client,
        model_name=model_name,
        max_iterations=max_iterations,
        max_sources=max_sources,
        temperature=temperature,
        enable_checkpointing=False,
        search_engines=search_engines,
        search_results_per_gap=search_results_per_gap,
        use_azure=use_azure
    )
    
    return await researcher.research_async(query, system_prompt)


@asynccontextmanager
async def create_researcher_session(
    client=None,
    model_name: str = None,
    use_azure: Optional[bool] = None,
    **kwargs
):
    """Context manager for TTD Researcher sessions"""
    researcher = TTDResearcher(client=client, model_name=model_name, use_azure=use_azure, **kwargs)
    try:
        yield researcher
    finally:
        # Cleanup if needed
        logger.info("Research session closed")


# Error handling utilities
class TTDResearchError(Exception):
    """Base exception for TTD Research errors"""
    pass


class TTDPlanningError(TTDResearchError):
    """Error in research planning stage"""
    pass


class TTDSearchError(TTDResearchError):
    """Error in search/retrieval stage"""
    pass


class TTDSynthesisError(TTDResearchError):
    """Error in synthesis/denoising stage"""
    pass


class TTDQualityError(TTDResearchError):
    """Error in quality evaluation stage"""
    pass


def handle_research_errors(func):
    """Decorator for handling research-specific errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Research error in {func.__name__}: {str(e)}")
            raise TTDResearchError(f"Research failed: {str(e)}") from e
    return wrapper