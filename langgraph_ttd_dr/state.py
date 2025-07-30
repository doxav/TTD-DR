from typing import Dict, List, Optional, Any, Tuple, TypedDict
from pydantic import BaseModel, Field
from enum import Enum
import uuid
from datetime import datetime


class TaskStatus(str, Enum):
    """Status enumeration for research tasks"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchGap(BaseModel):
    """Represents a gap identified in the current research draft"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    section: str
    gap_type: str  # PLACEHOLDER_TAG, MISSING_INFO, OUTDATED_INFO, etc.
    specific_need: str
    search_query: str
    priority: str  # HIGH, MEDIUM, LOW
    addressed: bool = False


class ResearchSource(BaseModel):
    """Represents a research source with metadata"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    url: str
    content: Optional[str] = None
    access_date: str = Field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d'))
    citation_number: Optional[int] = None
    relevance_score: float = 0.0


class ComponentFitness(BaseModel):
    """Self-evolution fitness tracking for different components"""
    search_strategy: float = 1.0
    synthesis_quality: float = 1.0
    gap_detection: float = 1.0
    integration_ability: float = 1.0
    
    def update_fitness(self, improvement_score: float):
        """Update fitness based on performance improvement"""
        if improvement_score > 0.1:  # Significant improvement
            self.search_strategy *= 1.1
            self.synthesis_quality *= 1.1
            self.integration_ability *= 1.1
        elif improvement_score < 0.05:  # Poor improvement
            self.search_strategy *= 0.95
            self.synthesis_quality *= 0.95
        
        # Cap fitness values
        for attr in ['search_strategy', 'synthesis_quality', 'gap_detection', 'integration_ability']:
            value = getattr(self, attr)
            setattr(self, attr, max(0.1, min(2.0, value)))


class QualityMetrics(BaseModel):
    """Quality evaluation metrics for research drafts"""
    completeness: float = 0.0
    accuracy: float = 0.0
    depth: float = 0.0
    coherence: float = 0.0
    citations: float = 0.0
    improvement: float = 0.0
    
    def is_high_quality(self) -> bool:
        """Check if the research meets high quality thresholds"""
        return (self.completeness > 0.9 or 
                (self.improvement < 0.03 and self.completeness > 0.7))


class TTDResearchState(TypedDict):
    """Complete state definition for Test-Time Diffusion Deep Researcher"""
    
    # Core Research Data
    session_id: str
    original_query: str
    system_prompt: str
    
    # Clarification Process State
    clarified_query: str  # Refined query after clarification
    needs_clarification: bool  # Whether query needs clarification
    clarification_questions: str  # Questions for user
    clarification_history: List[str]  # History of clarification attempts
    user_response: str  # User's response to clarification questions
    clarification_complete: bool  # Whether clarification process is done
    integrated_aspects: List[str]  # Research aspects identified through clarification
    research_scope: str  # Research scope determined through clarification
    
    # Diffusion Process State
    current_draft: str  # The evolving research draft (core of diffusion process)
    preliminary_draft: str  # Initial draft before denoising
    research_plan: str  # Research strategy and plan
    
    # TTD-DR Algorithm State  
    iteration: int  # Current denoising iteration (legacy)
    current_iteration: int  # Current TTD-DR iteration (Algorithm 1)
    max_iterations: int  # Maximum allowed iterations
    exit_loop: bool  # Algorithm 1 loop termination flag
    
    # Algorithm 1 Components
    search_questions: List[str]  # Q_t from M_Q function
    question_rationale: str  # Reasoning for generated questions
    
    identified_gaps: List[ResearchGap]  # Gaps found in current draft
    sources: List[ResearchSource]  # All gathered sources (Q, A history)
    citations: Dict[str, str]  # Citation mappings
    gap_resolutions: List[str]  # History of gap resolution attempts
    
    # LLM Configuration
    model_name: str  # Model identifier
    temperature: float  # LLM temperature setting
    
    # Search Configuration  
    search_engines: List[str]  # Enabled search engines ['tavily', 'duckduckgo', 'naver']
    search_results_per_gap: int  # Number of search results per research gap
    max_sources: int  # Maximum sources to gather
    
    # Workflow Control
    current_step: str  # Current workflow step identifier
    status: str  # Overall research status
    termination_reason: Optional[str]  # Reason for research termination
    
    # Quality Assessment
    quality_metrics: QualityMetrics  # Current quality evaluation
    component_fitness: ComponentFitness  # Component performance metrics
    draft_history: List[str]  # History of draft versions
    
    # Metadata and Logging
    total_tokens: int  # Total tokens used across all API calls
    error_messages: List[str]  # Any errors encountered
    
    # Note: Client is not stored in state for checkpointing compatibility


def create_initial_state(
    original_query: str,
    system_prompt: str = "You are a research assistant.",
    model_name: str = "gpt-4.1-nano",
    max_iterations: int = 5,
    max_sources: int = 20,
    temperature: float = 0.7,
    client: Any = None,  # Client not stored in state
    search_engines: List[str] = None,
    search_results_per_gap: int = 3
) -> TTDResearchState:
    """Create initial state for TTD-DR research"""
    if search_engines is None:
        search_engines = ['tavily', 'duckduckgo', 'naver']
        
    return TTDResearchState(
        # Core Research Metadata
        session_id=str(uuid.uuid4()),
        original_query=original_query,
        system_prompt=system_prompt,
        
        # Clarification Process State
        clarified_query="",
        needs_clarification=False,
        clarification_questions="",
        clarification_history=[],
        user_response="",
        clarification_complete=False,
        integrated_aspects=[],
        research_scope="",
        
        # Diffusion Process State
        current_draft="",
        preliminary_draft="",
        research_plan="",
        
        # TTD-DR Algorithm State
        iteration=0,
        current_iteration=0,
        max_iterations=max_iterations,
        exit_loop=False,
        search_questions=[],
        question_rationale="",
        identified_gaps=[],
        sources=[],
        citations={},
        gap_resolutions=[],
        
        # LLM Configuration
        model_name=model_name,
        temperature=temperature,
        
        # Search Configuration  
        search_engines=search_engines,
        search_results_per_gap=search_results_per_gap,
        max_sources=max_sources,
        
        # Workflow Control
        current_step="initialization",
        status="initialized",
        termination_reason=None,
        
        # Quality Metrics
        quality_metrics=QualityMetrics(),
        component_fitness=ComponentFitness(),
        draft_history=[],
        
        # Metadata and Logging
        total_tokens=0,
        error_messages=[]
        
        # Client removed for serialization compatibility
    )


# State validation functions
def validate_state_transition(old_state: TTDResearchState, new_state: TTDResearchState) -> bool:
    """Validate that state transition is valid"""
    return True  # Simple implementation for now