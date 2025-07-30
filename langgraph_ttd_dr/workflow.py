from typing import Any, Dict, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .state import TTDResearchState
from .nodes import (
    QueryClarificationNode,
    PlannerNode, 
    DraftGeneratorNode, 
    GapAnalyzerNode, 
    SearchAgentNode,
    DenoisingNode, 
    QualityEvaluatorNode, 
    ReportGeneratorNode,
    NoisyDraftGeneratorNode,
    DraftBasedQuestionGeneratorNode,
    DenoisingUpdaterNode,
    IterationControllerNode
)


class TTDResearchWorkflow:
    """Test-Time Diffusion Deep Research (TTD-DR) Workflow"""
    
    def __init__(self, client=None):
        self.client = client
        self.workflow = self._create_workflow()
        self.app = None
    
    def _create_workflow(self) -> StateGraph:
        """Create LangGraph workflow following Draft-Centric approach"""
        workflow = StateGraph(TTDResearchState)
        
        # Stage 1: Research Plan Generation + Query Clarification
        workflow.add_node("query_clarification", QueryClarificationNode(self.client))
        workflow.add_node("planner", PlannerNode(self.client))
        
        # Initial Draft Generation
        workflow.add_node("noisy_draft_generator", NoisyDraftGeneratorNode(self.client))
        
        # Stage 2: Iterative Search and Synthesis
        workflow.add_node("draft_based_question_generator", DraftBasedQuestionGeneratorNode(self.client))
        workflow.add_node("search_agent", SearchAgentNode(self.client))  
        workflow.add_node("denoising_updater", DenoisingUpdaterNode(self.client))
        
        # Control
        workflow.add_node("iteration_controller", IterationControllerNode(self.client))
        
        # Stage 3: Final Report
        workflow.add_node("report_generator", ReportGeneratorNode(self.client))
        
        # Define Flow
        workflow.add_edge(START, "query_clarification")
        workflow.add_edge("query_clarification", "planner")
        workflow.add_edge("planner", "noisy_draft_generator")
        
        # Iterative Loop
        workflow.add_edge("noisy_draft_generator", "draft_based_question_generator")
        workflow.add_edge("draft_based_question_generator", "search_agent")
        workflow.add_edge("search_agent", "denoising_updater")
        workflow.add_edge("denoising_updater", "iteration_controller")
        
        # Conditional routing from iteration controller
        workflow.add_conditional_edges(
            "iteration_controller",
            self._should_continue_iteration,
            {
                "continue": "draft_based_question_generator",
                "finalize": "report_generator"
            }
        )
        
        workflow.add_edge("report_generator", END)
        
        return workflow
    
    def _should_continue_iteration(self, state: TTDResearchState) -> str:
        """Research Loop Control: Check if should continue iterations"""
        current_iteration = state.get("current_iteration", 0)
        max_iterations = state.get("max_iterations", 3)
        exit_loop = state.get("exit_loop", False)
        
        if exit_loop or current_iteration >= max_iterations:
            print(f"Research loop ending: iteration={current_iteration}, exit_loop={exit_loop}")
            return "finalize"
        else:
            print(f"Research continuing: iteration {current_iteration + 1}/{max_iterations}")
            return "continue"
    
    def compile(
        self, 
        checkpointer: Optional[Any] = None, 
        recursion_limit: int = 50
    ) -> Any:
        """Compile the workflow into an executable application"""
        self.app = self.workflow.compile(checkpointer=checkpointer)
        return self.app
    
    def get_mermaid_diagram(self) -> str:
        """Get Mermaid diagram representation of the workflow"""
        try:
            if self.app:
                return self.app.get_graph().draw_mermaid()
            else:
                # Compile first if not already compiled
                self.compile()
                return self.app.get_graph().draw_mermaid()
        except Exception as e:
            return f"Error generating diagram: {str(e)}"


class AdvancedTTDDRWorkflow(TTDResearchWorkflow):
    """Advanced version of TTD-DR workflow with additional features"""
    
    def __init__(self, client: Any = None):
        super().__init__(client)
        
        # Could add additional nodes here for advanced features
        # e.g., source validation, bias detection, fact-checking


def create_ttd_dr_workflow(client: Any = None) -> TTDResearchWorkflow:
    """Factory function to create a TTD-DR workflow"""
    return TTDResearchWorkflow(client=client)