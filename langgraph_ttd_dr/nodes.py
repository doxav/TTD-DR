import json
import re
from typing import Dict, List, Tuple, Any, Optional, Literal
from datetime import datetime
import uuid
import time # Added for search_timestamp
from concurrent.futures import ThreadPoolExecutor

from .state import TTDResearchState
from .tools import search_web, format_search_results, get_search_tool_status
from .utils import (
    clean_reasoning_tags, 
    cleanup_placeholder_tags, 
    parse_gap_analysis,
    create_fallback_gap,
    parse_quality_scores,
    update_component_fitness,
    build_references_section,
    build_metadata_section,
    remove_references_section,
    should_terminate_research,
    SearchResultFormatter
)
from .client_factory import create_openai_client
from .prompts import (
    get_planning_prompt,
    get_draft_generation_prompt,
    get_gap_analysis_prompt,
    get_denoising_prompt,
    get_quality_evaluation_prompt,
    get_report_finalization_prompt,
    get_clarification_prompt,
    get_system_prompts,
    # New prompt functions
    get_draft_update_prompt,
    get_search_question_generation_prompt,
    get_noisy_draft_prompt,
    get_draft_question_generation_prompt,
    get_denoising_update_prompt,
    get_iteration_evaluation_prompt,
    get_simple_search_answer_prompt,
    get_answer_variant_prompt,
    get_variant_evaluation_prompt,
    get_answer_revision_prompt,
    get_answer_merge_prompt
)

from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START, END
from langgraph.types import Command

class AnswerEvolutionState(TypedDict):
    """Self-Evolution SubGraph State for Search Answer Generation"""
    search_query: str
    search_results: List[Dict[str, Any]]
    original_context: str
    
    # Initial States
    answer_variants: List[str]
    
    # Environmental Feedback
    variant_evaluations: List[Dict[str, Any]]
    
    # Revision Step
    revised_variants: List[str]
    revision_count: int
    evolution_iteration: int
    
    # Cross-over
    final_answer: str

def generate_answer_variants(state: AnswerEvolutionState) -> AnswerEvolutionState:
    """Generate multiple answer variants with different approaches"""
    try:
        client = create_openai_client()
        
        search_query = state['search_query']
        search_results = state['search_results']
        context = state['original_context']
        
        # Generate 3 variants with different focuses
        variants = []
        approaches = ["depth and specificity", "breadth and connections", "practical implications"]
        
        for i, approach in enumerate(approaches):
            variant_prompt = get_answer_variant_prompt(
                search_query=search_query,
                search_results=search_results,
                context=context,
                approach=approach,
                temperature_setting=0.3 + (i * 0.3)
            )
            
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": f"You are an expert research assistant focusing on {approach}."},
                    {"role": "user", "content": variant_prompt}
                ],
                temperature=0.3 + (i * 0.3),  # 0.3, 0.6, 0.9
                max_tokens=800
            )
            
            variant = response.choices[0].message.content.strip()
            variants.append(variant)
            print(f"Generated variant {i+1}: {approach}")
        
        # Store variants in state
        return {
            **state,
            "answer_variants": variants
        }
        
    except Exception as e:
        print(f"Error generating variants: {str(e)}")
        return {**state, "answer_variants": []}

def merge_final_answer(state: AnswerEvolutionState) -> AnswerEvolutionState:
    """Cross-over step: Merge multiple revised variants into final answer"""
    try:
        client = create_openai_client()
        
        search_query = state['search_query']
        # Use revised variants if available, otherwise use original variants
        variants_to_merge = state.get('revised_variants', state.get('answer_variants', []))
        
        if not variants_to_merge:
            return {**state, "final_answer": "No variants available for merging"}
        
        if len(variants_to_merge) == 1:
            return {**state, "final_answer": variants_to_merge[0]}
        
        # Get evaluation data for intelligent merging
        evaluations = state.get('variant_evaluations', [])
        evaluation_context = ""
        if evaluations:
            evaluation_context = f"""
            VARIANT QUALITY SCORES:
            {chr(10).join([f"Variant {i+1}: Fitness={eval.get('fitness_score', 'N/A')}/10, Strengths={eval.get('feedback', {}).get('strengths', 'N/A')}" for i, eval in enumerate(evaluations)])}
            """
        
        merge_prompt = get_answer_merge_prompt(
            query=search_query,
            variants_to_merge=variants_to_merge,
            evaluation_context=evaluation_context
        )
        
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are an expert research synthesizer implementing Cross-over from TTD-DR Self-Evolution."},
                {"role": "user", "content": merge_prompt}
            ],
            temperature=0.3,  # Lower temperature for consistent merging
            max_tokens=1500
        )
        
        final_answer = response.choices[0].message.content.strip()
        
        print(f"Cross-over completed: Merged {len(variants_to_merge)} variants into final answer")
        print(f"Final answer length: {len(final_answer)} characters")
        
        return {
            **state,
            "final_answer": final_answer,
            "evolution_iteration": state.get("evolution_iteration", 0) + 1
        }
        
    except Exception as e:
        print(f"Cross-over merge failed: {str(e)}")
        # Fallback: use the first available variant
        fallback = state.get('revised_variants', state.get('answer_variants', []))
        return {
            **state, 
            "final_answer": fallback[0] if fallback else "Error merging variants",
            "evolution_iteration": state.get("evolution_iteration", 0) + 1
        }

def environmental_feedback(state: AnswerEvolutionState) -> AnswerEvolutionState:
    """Environmental Feedback step: LLM-as-a-judge evaluation"""
    try:
        client = create_openai_client()
        
        search_query = state['search_query']
        answer_variants = state.get('answer_variants', [])
        
        if not answer_variants:
            return {**state, "variant_evaluations": []}
        
        variant_evaluations = []
        
        for i, variant in enumerate(answer_variants):
            # LLM-as-a-judge evaluation prompt from paper
            judge_prompt = get_variant_evaluation_prompt(
                search_query=search_query,
                answer=variant
            )
            
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are an expert research quality evaluator implementing LLM-as-a-judge from the TTD-DR paper."},
                    {"role": "user", "content": judge_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=1000
            )
            
            evaluation_result = response.choices[0].message.content.strip()
            
            # Parse evaluation scores
            scores = {}
            feedback = {"strengths": "", "weaknesses": "", "suggestions": ""}
            
            for line in evaluation_result.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().upper()
                    value = value.strip()
                    
                    if 'SCORE' in key:
                        try:
                            score_name = key.replace('_SCORE', '').lower()
                            scores[score_name] = float(value)
                        except:
                            continue
                    elif 'STRENGTHS' in key:
                        feedback["strengths"] = value
                    elif 'WEAKNESSES' in key:
                        feedback["weaknesses"] = value
                    elif 'IMPROVEMENT_SUGGESTIONS' in key:
                        feedback["suggestions"] = value
            
            # Calculate overall fitness score
            fitness_score = scores.get('overall', 
                                     (scores.get('helpfulness', 5) + 
                                      scores.get('comprehensiveness', 5) + 
                                      scores.get('quality', 5)) / 3)
            
            variant_evaluation = {
                "variant_id": i,
                "variant_text": variant,
                "scores": scores,
                "fitness_score": fitness_score,
                "feedback": feedback,
                "needs_revision": fitness_score < 7.0  # Threshold for revision
            }
            
            variant_evaluations.append(variant_evaluation)
            print(f"Variant {i+1} evaluated: Fitness={fitness_score:.1f}/10")
        
        # Store evaluations in state
        return {
            **state,
            "variant_evaluations": variant_evaluations
        }
        
    except Exception as e:
        print(f"Environmental feedback failed: {str(e)}")
        # Fallback: create dummy evaluations
        dummy_evaluations = []
        for i, variant in enumerate(state.get('answer_variants', [])):
            dummy_evaluations.append({
                "variant_id": i,
                "variant_text": variant,
                "scores": {"overall": 5.0},
                "fitness_score": 5.0,
                "feedback": {"strengths": "N/A", "weaknesses": "N/A", "suggestions": "N/A"},
                "needs_revision": True
            })
        
        return {**state, "variant_evaluations": dummy_evaluations}

def revision_step(state: AnswerEvolutionState) -> AnswerEvolutionState:
    """Revision Step: Improve variants based on feedback"""
    try:
        client = create_openai_client()
        
        search_query = state['search_query']
        variant_evaluations = state.get('variant_evaluations', [])
        
        if not variant_evaluations:
            return state
        
        revised_variants = []
        revision_count = 0
        
        for evaluation in variant_evaluations:
            if evaluation['needs_revision']:
                # Revise this variant based on feedback
                revision_prompt = get_answer_revision_prompt(
                    search_query=search_query,
                    variant_text=evaluation['variant_text'],
                    fitness_score=evaluation['fitness_score'],
                    feedback=evaluation['feedback']
                )
                
                response = client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=[
                        {"role": "system", "content": "You are an expert research writer implementing the Revision Step from TTD-DR Self-Evolution."},
                        {"role": "user", "content": revision_prompt}
                    ],
                    temperature=0.4,  # Moderate creativity for revision
                    max_tokens=1200
                )
                
                revised_answer = response.choices[0].message.content.strip()
                revised_variants.append(revised_answer)
                revision_count += 1
                print(f"Variant {evaluation['variant_id']+1} revised (fitness was {evaluation['fitness_score']:.1f})")
                
            else:
                # Keep high-quality variants as-is
                revised_variants.append(evaluation['variant_text'])
                print(f"Variant {evaluation['variant_id']+1} kept (fitness: {evaluation['fitness_score']:.1f})")
        
        print(f"Revision Step completed: {revision_count}/{len(variant_evaluations)} variants revised")
        
        return {
            **state,
            "revised_variants": revised_variants,
            "revision_count": revision_count
        }
        
    except Exception as e:
        print(f"Revision step failed: {str(e)}")
        # Fallback: return original variants
        original_variants = [eval['variant_text'] for eval in state.get('variant_evaluations', [])]
        return {
            **state,
            "revised_variants": original_variants,
            "revision_count": 0
        }

def should_continue_evolution(state: AnswerEvolutionState) -> str:
    """Decide whether to continue evolution or proceed to final merge"""
    try:
        current_iteration = state.get('evolution_iteration', 0) 
        max_evolution_iterations = 2  # Limit from paper
        
        # Check iteration limits first
        if current_iteration >= max_evolution_iterations:
            print(f"Max evolution iterations ({max_evolution_iterations}) reached - proceeding to merge")
            return "merge_final_answer"
        
        # Check if we have good enough variants
        variant_evaluations = state.get('variant_evaluations', [])
        if variant_evaluations:
            # Find variants needing revision
            needs_revision = [eval for eval in variant_evaluations if eval.get('needs_revision', True)]
            
            if not needs_revision:
                # All variants are good quality
                best_fitness = max(eval['fitness_score'] for eval in variant_evaluations)
                print(f"All variants high quality (best: {best_fitness:.1f}) - proceeding to merge")
                return "merge_final_answer"
            else:
                # Some variants need improvement
                revision_needed = len(needs_revision)
                print(f"{revision_needed}/{len(variant_evaluations)} variants need revision - continuing evolution")
                return "revision_step"
        
        # Default: proceed to merge if no evaluation data
        print("No evaluation data - proceeding to merge")
        return "merge_final_answer"
        
    except Exception as e:
        print(f"Evolution control failed: {str(e)}")
        return "merge_final_answer"

def update_variants_after_revision(state: AnswerEvolutionState) -> AnswerEvolutionState:
    """Update answer_variants with revised_variants after revision step"""
    revised_variants = state.get('revised_variants', [])
    evolution_iteration = state.get('evolution_iteration', 0)
    
    return {
        **state,
        "answer_variants": revised_variants,  # Update variants for next feedback cycle
        "evolution_iteration": evolution_iteration + 1  # Increment iteration
    }

def create_answer_evolution_subgraph() -> StateGraph:
    """Create Self-Evolution SubGraph implementing 4-step process"""
    subgraph = StateGraph(AnswerEvolutionState)
    
    # Initial States - Multiple diverse variants
    subgraph.add_node("generate_answer_variants", generate_answer_variants)
    
    # Environmental Feedback - LLM-as-a-judge evaluation
    subgraph.add_node("environmental_feedback", environmental_feedback) 
    
    # Revision Step - Improve based on feedback
    subgraph.add_node("revision_step", revision_step)
    
    # Helper node to update variants after revision
    subgraph.add_node("update_variants", update_variants_after_revision)
    
    # Cross-over - Merge best variants
    subgraph.add_node("merge_final_answer", merge_final_answer)
    
    # Define the evolution workflow
    subgraph.add_edge(START, "generate_answer_variants")
    subgraph.add_edge("generate_answer_variants", "environmental_feedback")
    
    # Conditional edge for evolution loop
    subgraph.add_conditional_edges(
        "environmental_feedback",
        should_continue_evolution,
        {
            "revision_step": "revision_step",
            "merge_final_answer": "merge_final_answer"
        }
    )
    
    # After revision, update variants and go back to feedback for re-evaluation
    subgraph.add_edge("revision_step", "update_variants")
    subgraph.add_edge("update_variants", "environmental_feedback")
    
    # Final step
    subgraph.add_edge("merge_final_answer", END)
    
    return subgraph.compile()

# Create the compiled subgraph
ANSWER_EVOLUTION_SUBGRAPH = create_answer_evolution_subgraph()

class QueryClarificationNode:
    """Simple Query Clarification Node - Automatically improves user query using LLM"""
    
    def __init__(self, client: Any = None):
        self.client = client
    
    def __call__(self, state: TTDResearchState) -> Dict[str, Any]:
        """
        Automatically clarify and improve the user query
        """
        try:
            original_query = state.get("original_query", "")
            client = self.client
            system_prompts = get_system_prompts()
            
            clarification_prompt = f"""
            Analyze and improve this research query to make it more specific and actionable:
            
            ORIGINAL QUERY: {original_query}
            
            Your task:
            1. Identify what the user likely wants to know
            2. Make the query more specific and research-friendly
            3. Add relevant context and scope
            4. Ensure it's actionable for comprehensive research
            
            Provide a single, improved research query that covers the user's intent more clearly.
            
            IMPROVED QUERY:
            """
            
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": system_prompts.get("clarification", "You are a research query improvement expert.")},
                    {"role": "user", "content": clarification_prompt}
                ],
                temperature=0.5,
                max_tokens=400
            )
            
            clarified_query = clean_reasoning_tags(response.choices[0].message.content.strip())
            
            # Remove "IMPROVED QUERY:" prefix if present
            if clarified_query.startswith("IMPROVED QUERY:"):
                clarified_query = clarified_query.replace("IMPROVED QUERY:", "").strip()
            
            print(f"Query clarification:")
            print(f"   Original: {original_query}")
            print(f"   Improved: {clarified_query}")
            
            return {
                "clarified_query": clarified_query,
                "current_step": "query_clarified",
                "total_tokens": state.get("total_tokens", 0) + 200
            }
            
        except Exception as e:
            print(f"Query clarification failed: {str(e)}")
            # Fallback to original query
            return {
                "clarified_query": state.get("original_query", ""),
                "current_step": "query_clarification_failed",
                "total_tokens": state.get("total_tokens", 0)
            }


class PlannerNode:
    """
    Planner Node - Plans research approach for Algorithm 1
    Generates structured research plan that guides noisy draft creation
    """
    
    def __init__(self, client: Any = None):
        self.client = client
    
    def __call__(self, state: TTDResearchState) -> Dict[str, Any]:
        """
        Plan research approach for TTD-DR Algorithm 1
        Creates plan that will guide noisy draft generation and iterative improvement
        """
        try:
            client = self.client
            system_prompts = get_system_prompts()
            research_query = state.get('clarified_query') or state['original_query']
            
            # Create comprehensive research plan for Algorithm 1
            planning_prompt = f"""
            Create a comprehensive research plan for TTD-DR Algorithm 1 implementation.
            This plan will guide the creation of an initial noisy draft and subsequent iterations.
            
            RESEARCH QUERY: {research_query}
            
            PLANNING INSTRUCTIONS:
            1. **Research Scope**: Define the key areas that need to be covered
            2. **Major Topics**: Identify 4-6 main topics/sections for the research
            3. **Information Types**: Specify what types of information are needed (facts, examples, recent data, etc.)
            4. **Research Depth**: Determine the appropriate level of detail required
            5. **Structure**: Outline the logical flow and organization
            
            This plan will be used to:
            - Generate an initial noisy draft (R0)
            - Guide draft-based question generation
            - Focus the iterative search and denoising process
            
            OUTPUT FORMAT:
            RESEARCH_SCOPE: [define the boundaries and focus]
            
            MAJOR_TOPICS:
            1. [Topic 1 - description]
            2. [Topic 2 - description]  
            3. [Topic 3 - description]
            4. [Topic 4 - description]
            (continue as needed)
            
            INFORMATION_REQUIREMENTS:
            - [Type 1: specific information needs]
            - [Type 2: specific information needs]
            - [Type 3: specific information needs]
            
            RESEARCH_STRATEGY: [how to approach this research systematically]
            """
            
            response = client.chat.completions.create(
                model=state["model_name"],
                messages=[
                    {"role": "system", "content": system_prompts.get("planner", "You are a research planning expert.")},
                    {"role": "user", "content": planning_prompt}
                ],
                temperature=0.4,
                max_tokens=1500
            )
            
            research_plan = clean_reasoning_tags(response.choices[0].message.content.strip())
            
            # Parse key components
            topics_count = research_plan.count("MAJOR_TOPICS:") + research_plan.count("Topic")
            
            print(f"Research plan created")
            print(f"Scope: {research_plan.split('RESEARCH_SCOPE:')[1].split('MAJOR_TOPICS:')[0].strip()[:100] if 'RESEARCH_SCOPE:' in research_plan else 'Comprehensive research'}...")
            print(f"Estimated topics: {min(topics_count, 8)}")
            
            return {
                "research_plan": research_plan,
                "total_tokens": state.get("total_tokens", 0) + response.usage.completion_tokens,
                "current_step": "research_planned",
                "status": "in_progress"
            }
            
        except Exception as e:
            print(f"Planning failed: {str(e)}")
            return {
                "error_messages": state.get("error_messages", []) + [f"Planning failed: {str(e)}"],
                "research_plan": "Basic research plan - comprehensive analysis of the topic",
                "current_step": "planning_failed"
            }


class DraftGeneratorNode:
    """
    Draft Generation Node - Implements Algorithm 1: R_t = M_R(q, R_{t-1}, Q, A)
    Creates initial draft or updates existing draft with new search results
    """
    
    def __init__(self, client: Any = None):
        """Initialize draft generator node with client"""
        self.client = client
    
    def __call__(self, state: TTDResearchState) -> Dict[str, Any]:
        """
        Generate or update draft based on Algorithm 1 from the paper
        R_t = M_R(q, R_{t-1}, Q, A) - Update draft with new information
        """
        try:
            client = self.client
            system_prompts = get_system_prompts()
            
            # Get context for Algorithm 1
            research_query = state.get('clarified_query') or state['original_query']
            research_plan = state.get("research_plan", "")
            current_draft = state.get("current_draft", "")  # R_{t-1}
            sources = state.get("sources", [])  # Q, A history
            current_iteration = state.get("current_iteration", 0)
            
            # Implement Algorithm 1 logic
            if current_draft and len(sources) > 0:
                # Update existing draft with new search results (R_t = M_R(q, R_{t-1}, Q, A))
                updated_draft = self._update_draft_with_search_results(
                    research_query, current_draft, sources, research_plan, client, system_prompts
                )
                draft_type = "updated"
            else:
                # Create initial draft (R_0)
                updated_draft = self._create_initial_draft(
                    research_query, research_plan, client, system_prompts
                )
                draft_type = "initial"
            
            print(f"Draft {draft_type} (iteration {current_iteration + 1}): {len(updated_draft)} chars")
            
            return {
                "current_draft": updated_draft,
                "preliminary_draft": state.get("preliminary_draft", updated_draft),
                "total_tokens": state["total_tokens"] + 500,  # Estimated tokens
                "current_step": f"{draft_type}_draft_complete",
                "status": "in_progress",
                "draft_history": state.get("draft_history", []) + [
                    {
                        "iteration": current_iteration,
                        "type": draft_type,
                        "length": len(updated_draft),
                        "timestamp": time.time()
                    }
                ]
            }
            
        except Exception as e:
            print(f"Draft generation error: {str(e)}")
            return {
                "error_messages": state["error_messages"] + [f"Draft generation failed: {str(e)}"],
                "status": "failed"
            }
    
    def _create_initial_draft(self, query: str, plan: str, client: Any, system_prompts: Dict) -> str:
        """Create initial draft R_0 based on query and plan"""
        try:
            draft_prompt = f"""
            Create an initial research draft based on the following:
            
            RESEARCH QUERY: {query}
            RESEARCH PLAN: {plan}
            
            Instructions:
            1. Create a structured outline addressing the research query
            2. Fill in sections with initial analysis and observations
            3. Identify areas that need more information (these will be filled later)
            4. Use clear headings and logical flow
            5. Write 1000-1500 words covering known aspects
            
            This is an initial "noisy" draft that will be iteratively refined with search results.
            """
            
            response = client.chat.completions.create(
                model="gpt-4.1-nano",  # Using consistent model
                messages=[
                    {"role": "system", "content": system_prompts.get("draft_generator", "You are an expert research writer.")},
                    {"role": "user", "content": draft_prompt}
                ],
                temperature=0.6,
                max_tokens=2500
            )
            
            return clean_reasoning_tags(response.choices[0].message.content.strip())
            
        except Exception as e:
            print(f"Initial draft creation failed: {str(e)}")
            return f"Initial draft creation failed: {str(e)}"
    
    def _update_draft_with_search_results(self, query: str, current_draft: str, sources: List[Dict], 
                                        plan: str, client: Any, system_prompts: Dict) -> str:
        """
        Implement Algorithm 1: R_t = M_R(q, R_{t-1}, Q, A)
        Update existing draft with new search results
        """
        try:
            # Get recent search results (last 3-5 sources)
            recent_sources = sources[-5:] if len(sources) > 5 else sources
            
            # Format search Q&A pairs
            search_qa_pairs = []
            for source in recent_sources:
                gap_addressed = source.get('gap_addressed', '')
                content = source.get('content', '')
                if gap_addressed and content:
                    search_qa_pairs.append(f"Q: {gap_addressed}\nA: {content}")
            
            search_context = "\n\n".join(search_qa_pairs[-3:]) if search_qa_pairs else "No recent search results"
            
            # Draft update prompt
            update_prompt = get_draft_update_prompt(
                query=query,
                plan=plan,
                current_draft=current_draft,
                search_context=search_context
            )
            
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": system_prompts.get("draft_generator", "You are an expert research writer for TTD-DR.")},
                    {"role": "user", "content": update_prompt}
                ],
                temperature=0.3,  # Lower temperature for consistent updates
                max_tokens=3000
            )
            
            updated_draft = clean_reasoning_tags(response.choices[0].message.content.strip())
            print(f"Draft updated from {len(current_draft)} to {len(updated_draft)} chars")
            
            return updated_draft
            
        except Exception as e:
            print(f"Draft update failed: {str(e)}")
            # Fallback: return current draft if update fails
            return current_draft


class GapAnalyzerNode:
    """
    Gap Analysis Node - Identifies knowledge gaps in current draft
    
    This implements the Gap Analysis component of TTD-DR Stage 2
    """
    
    def __init__(self, client: Any = None):
        """Initialize gap analyzer node with client"""
        self.client = client
    
    def __call__(self, state: TTDResearchState) -> Dict[str, Any]:
        """Identify gaps and areas needing additional research"""
        try:
            client = self.client  # Use injected client
            system_prompts = get_system_prompts()
            
            gap_prompt = get_gap_analysis_prompt(
                state.get('clarified_query') or state['original_query'],
                state['current_draft']
            )
            
            response = client.chat.completions.create(
                model=state["model_name"],
                messages=[
                    {"role": "system", "content": system_prompts["gap_analyzer"]},
                    {"role": "user", "content": gap_prompt}
                ],
                temperature=state["temperature"],
                max_tokens=1500
            )
            
            gap_analysis = clean_reasoning_tags(response.choices[0].message.content.strip())
            
            # Parse gap analysis
            identified_gaps = parse_gap_analysis(gap_analysis)
            
            # Create fallback gap if parsing fails
            if not identified_gaps:
                fallback_gap = create_fallback_gap(state.get('clarified_query') or state['original_query'])
                identified_gaps = [fallback_gap]
            
            return {
                "identified_gaps": identified_gaps,
                "total_tokens": state["total_tokens"] + response.usage.completion_tokens,
                "current_step": f"gaps_identified_iteration_{state.get('iteration', 0) + 1}"
            }
            
        except Exception as e:
            return {
                "error_messages": state["error_messages"] + [f"Gap analysis failed: {str(e)}"],
                "status": "failed"
            }


class SearchAgentNode:
    """
    Search Agent Node - Generates dynamic search queries based on current draft
    
    Implements Algorithm 1: Q_t = M_Q(q, P, R_t-1, Q, A) from TTD-DR paper
    """
    
    def __init__(self, client: Any = None):
        """Initialize search agent node with client"""
        self.client = client
    
    def _generate_dynamic_search_questions(self, state: TTDResearchState) -> List[str]:
        """
        Generate new search questions based on current draft and search history
        Implements M_Q(q, P, R_t-1, Q, A) from Algorithm 1 - NOW WITH ANSWER HISTORY
        """
        client = self.client
        system_prompts = get_system_prompts()
        
        # Get current context (Algorithm 1 parameters)
        original_query = state.get('clarified_query') or state.get('original_query', '')  # q
        research_plan = state.get('research_plan', '')  # P
        current_draft = state.get('current_draft', '')  # R_{t-1}
        iteration = state.get('iteration', 0)
        
        # Get search history (Q, A pairs from Algorithm 1)
        sources = state.get('sources', [])
        previous_questions = []  # Q
        previous_answers = []    # A
        
        for source in sources[-10:]:  # Last 10 Q&A pairs
            question = source.get('gap_addressed', '')
            answer = source.get('content', '')
            if question and answer:
                if question not in previous_questions:
                    previous_questions.append(question)
                    previous_answers.append(answer[:300] + "..." if len(answer) > 300 else answer)
        
        # Format Q&A history for Algorithm 1
        qa_history = []
        for i, (q, a) in enumerate(zip(previous_questions[-5:], previous_answers[-5:])):
            qa_history.append(f"Q{i+1}: {q}\nA{i+1}: {a}")
        
        qa_context = "\n\n".join(qa_history) if qa_history else "No previous Q&A pairs"
        
        # Generate targeted search questions based on draft gaps
        search_generation_prompt = get_search_question_generation_prompt(
            original_query=original_query,
            plan=research_plan,
            current_draft=current_draft,
            qa_context=qa_context,
            iteration=iteration
        )
        
        try:
            response = client.chat.completions.create(
                model=state["model_name"],
                messages=[
                    {"role": "system", "content": system_prompts.get("search_agent", "You are an expert research assistant for TTD-DR.")},
                    {"role": "user", "content": search_generation_prompt}
                ],
                temperature=0.7,  # Higher creativity for diverse questions
                max_tokens=1000  # Increased for comprehensive analysis
            )
            
            result = clean_reasoning_tags(response.choices[0].message.content.strip())
            
            # Parse search questions
            search_questions = []
            lines = result.split('\n')
            for line in lines:
                if line.startswith('SEARCH_QUESTION_'):
                    question = line.split(':', 1)[1].strip() if ':' in line else ''
                    if question and len(question) > 10:  # Valid question
                        search_questions.append(question)
            
            print(f"Generated {len(search_questions)} M_Q questions (iteration {iteration + 1})")
            print(f"Analyzed {len(previous_questions)} previous Q&A pairs")
            
            # Display questions for verification
            for i, q in enumerate(search_questions[:3], 1):
                print(f"  {i}. {q[:100]}{'...' if len(q) > 100 else ''}")
            
            return search_questions[:3]  # Return top 3 questions
            
        except Exception as e:
            print(f"Dynamic search question generation failed: {str(e)}")
            return []
    
    def _search_single_query(self, query: str) -> List[Dict]:
        """
        Perform a single search query using the search_web function.
        """
        try:
            print(f"Searching: {query}")
            web_results = search_web(
                query,
                max_results=3, # Fixed number of results for simplicity
                enabled_engines=['tavily', 'duckduckgo', 'naver'] # Example engines
            )
            
            sources = []
            for result in web_results:
                source = {
                    "title": result.get('title', 'Unknown'),
                    "url": result.get('url', ''),
                    "content": result.get('content', result.get('snippet', '')),
                    "relevance_score": result.get('score', 0.5),
                    "source_type": "web_search",
                    "gap_addressed": query,
                    "iteration": 1, # This is a placeholder, actual iteration is handled by SearchAgentNode
                    "search_timestamp": str(time.time())
                }
                sources.append(source)
            return sources
        except Exception as e:
            print(f"Search failed for '{query}': {str(e)}")
            return []
    
    def _process_search_results_with_evolution(self, search_query: str, search_results: List[Dict], context: str) -> str:
        """
        
        """
        try:
            print(f"Starting Self-Evolution for: {search_query[:80]}...")
     
            evolution_input = {
                "search_query": search_query,
                "search_results": search_results,
                "original_context": context,
                "final_answer": ""
            }
            
      
            result_state = ANSWER_EVOLUTION_SUBGRAPH.invoke(evolution_input)
            
            final_answer = result_state.get("final_answer", "")
            print(f"Self-Evolution completed: {len(final_answer)} chars generated")
            
            return final_answer if final_answer else "Error in self-evolution process"
            
        except Exception as e:
            print(f"Self-Evolution failed: {str(e)}")
            # Fallback to simple answer generation
            return self._generate_simple_answer(search_query, search_results, context)
    
    def _generate_simple_answer(self, search_query: str, search_results: List[Dict], context: str) -> str:
        """Fallback simple answer generation without self-evolution"""
        try:
            client = self.client
            search_results_text = chr(10).join([f"- {result.get('title', '')}: {result.get('content', '')[:300]}..." for result in search_results[:3]])
            simple_prompt = get_simple_search_answer_prompt(
                search_query=search_query,
                search_results=search_results_text,
                context=context
            )
            
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are a research assistant."},
                    {"role": "user", "content": simple_prompt}
                ],
                temperature=0.5,
                max_tokens=800
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def __call__(self, state: TTDResearchState) -> Dict[str, Any]:
        """
        Search Agent Node - Implements Algorithm 1 Line 4: A_t = M_A(Q_t)
        Uses questions from DraftBasedQuestionGeneratorNode and applies Self-Evolution
        """
        try:
            client = self.client
            current_iteration = state.get("current_iteration", 0)
            max_iterations = state.get("max_iterations", 3)
            
            # Get search questions from draft-based question generator
            search_queries = state.get("search_questions", [])
            
            if not search_queries:
                print("No search questions provided - skipping search")
                return {
                    "search_results": [],
                    "sources": state.get("sources", []),  # Keep existing sources
                    "current_step": "search_skipped_no_questions"
                }
            
            print(f"Search Agent - Processing search queries")
            print(f"Processing {len(search_queries)} draft-based questions")
            
            # Perform parallel searches
            all_search_results = []
            search_answers = []
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                search_futures = []
                for query in search_queries:
                    future = executor.submit(self._search_single_query, query)
                    search_futures.append((query, future))
                
                for query, future in search_futures:
                    try:
                        results = future.result(timeout=60)
                        all_search_results.extend(results)
                        
                        if results:
                            # Use Self-Evolution SubGraph for answer generation
                            context = f"Draft-based search for iteration {current_iteration + 1}"
                            evolved_answer = self._process_search_results_with_evolution(query, results, context)
                            search_answers.append({
                                "query": query,
                                "answer": evolved_answer,
                                "source_count": len(results),
                                "evolution_used": True
                            })
                        else:
                            search_answers.append({
                                "query": query,
                                "answer": "No relevant information found",
                                "source_count": 0,
                                "evolution_used": False
                            })
                            
                    except Exception as e:
                        print(f"Search failed for query: {query[:50]}... - {str(e)}")
                        search_answers.append({
                            "query": query,
                            "answer": f"Search error: {str(e)}",
                            "source_count": 0,
                            "evolution_used": False
                        })
            
            print(f"Found {len(all_search_results)} sources from {len(search_queries)} questions")
            print(f"Self-Evolution used for {sum(1 for ans in search_answers if ans.get('evolution_used'))} answers")
            
            # Convert search results to sources (Q, A pairs)
            current_sources = state.get("sources", [])
            new_sources = []
            
            for answer_data in search_answers:
                if answer_data.get("answer") and "error" not in answer_data.get("answer", "").lower():
                    # Create source entry for Q&A history
                    source_entry = {
                        "gap_addressed": answer_data["query"],  # Q_t
                        "content": answer_data["answer"],       # A_t  
                        "source_count": answer_data["source_count"],
                        "evolution_used": answer_data.get("evolution_used", False),
                        "search_timestamp": time.time(),
                        "iteration": current_iteration + 1,
                        "algorithm_step": "search_agent"
                    }
                    new_sources.append(source_entry)
            
            updated_sources = current_sources + new_sources
            print(f"Added {len(new_sources)} Q&A pairs to search history (Total: {len(updated_sources)})")
            
            return {
                "search_results": all_search_results,
                "search_answers": search_answers,
                "sources": updated_sources,  # Updated Q, A history
                "current_step": f"search_complete_iteration_{current_iteration + 1}",
                "total_tokens": state.get("total_tokens", 0) + 300  # Estimated tokens for evolution
            }
            
        except Exception as e:
            print(f"Search Agent error: {str(e)}")
            return {
                "error_messages": state.get("error_messages", []) + [f"Search agent failed: {str(e)}"],
                "current_step": "search_failed"
            }


class DenoisingNode:
    """
    Denoising Node - Integrates new sources into existing draft
    
    This implements the core Denoising component of TTD-DR Stage 2
    """
    
    def __init__(self, client: Any = None):
        """Initialize denoising node with client"""
        self.client = client
    
    def __call__(self, state: TTDResearchState) -> Dict[str, Any]:
        """Integrate sources and improve draft quality"""
        try:
            client = self.client  # Use injected client
            system_prompts = get_system_prompts()
            
            # Get recent sources (from current iteration)
            recent_sources = state.get("sources", [])[-10:]  # Last 10 sources
            
            denoising_prompt = get_denoising_prompt(
                state['current_draft'],
                recent_sources,
                state.get("identified_gaps", [])
            )
            
            response = client.chat.completions.create(
                model=state["model_name"],
                messages=[
                    {"role": "system", "content": system_prompts["denoising"]},
                    {"role": "user", "content": denoising_prompt}
                ],
                temperature=state["temperature"],
                max_tokens=3000
            )
            
            improved_draft = clean_reasoning_tags(response.choices[0].message.content.strip())
            improved_draft = cleanup_placeholder_tags(improved_draft)
            
            return {
                "current_draft": improved_draft,
                "draft_history": state["draft_history"] + [improved_draft],
                "iteration": state.get("iteration", 0) + 1,
                "total_tokens": state["total_tokens"] + response.usage.completion_tokens,
                "current_step": f"denoising_completed_iteration_{state.get('iteration', 0) + 1}"
            }
            
        except Exception as e:
            return {
                "error_messages": state["error_messages"] + [f"Denoising failed: {str(e)}"],
                "status": "failed"
            }


class QualityEvaluatorNode:
    """
    Quality Evaluation Node - Evaluates research quality and decides next action using Command
    """
    
    def __init__(self, client: Any = None):
        self.client = client
    
    def __call__(self, state: TTDResearchState) -> Command[Literal["search_agent", "denoising", "report_generator"]]:
        """
        Evaluate research quality and route dynamically using Command
        """
        try:
            client = self.client
            system_prompts = get_system_prompts()
            current_draft = state.get("current_draft", "")
            current_iteration = state.get("current_iteration", 0)
            max_iterations = state.get("max_iterations", 3)
            
            if not current_draft:
                print("No draft to evaluate - routing to search_agent")
                return Command(
                    update={"current_step": "no_draft_to_evaluate"},
                    goto="search_agent"
                )
            
            # Evaluate current research quality
            evaluation_prompt = f"""
            Evaluate the quality and completeness of this research draft:
            
            DRAFT: {current_draft[:2000]}...
            
            ITERATION: {current_iteration}/{max_iterations}
            
            Assess:
            1. Content completeness (1-10)
            2. Information depth (1-10) 
            3. Source diversity (1-10)
            4. Gap coverage (1-10)
            5. Overall readiness (1-10)
            
            Based on evaluation and iteration count, recommend:
            - CONTINUE_SEARCH: Need more information gathering
            - DENOISE: Ready for denoising/refinement
            - FINALIZE: Ready for final report generation
            
            Format:
            COMPLETENESS_SCORE: [1-10]
            DEPTH_SCORE: [1-10]
            DIVERSITY_SCORE: [1-10]
            GAP_COVERAGE_SCORE: [1-10]
            OVERALL_SCORE: [1-10]
            RECOMMENDATION: [CONTINUE_SEARCH/DENOISE/FINALIZE]
            REASONING: [explanation]
            """
            
            response = client.chat.completions.create(
                model=state["model_name"],
                messages=[
                    {"role": "system", "content": system_prompts.get("quality_evaluator", "You are a research quality expert.")},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            
            evaluation_result = clean_reasoning_tags(response.choices[0].message.content.strip())
            
            # Parse evaluation results
            lines = evaluation_result.split('\n')
            scores = {}
            recommendation = "CONTINUE_SEARCH"  # default
            reasoning = ""
            
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    if 'SCORE' in key.upper():
                        try:
                            score_name = key.strip().replace('_SCORE', '').lower()
                            scores[score_name] = float(value.strip())
                        except:
                            continue
                    elif 'RECOMMENDATION' in key.upper():
                        recommendation = value.strip().upper()
                    elif 'REASONING' in key.upper():
                        reasoning = value.strip()
            
            overall_score = scores.get('overall', 0)
            
            # Determine next action using Command logic
            if current_iteration >= max_iterations:
                # Force completion when max iterations reached
                next_node = "report_generator"
                action = f"Max iterations ({max_iterations}) reached - Finalizing report"
            elif recommendation == "FINALIZE" or overall_score >= 8.5:
                # High quality threshold met
                next_node = "report_generator"
                action = "High quality achieved - Finalizing report"
            elif recommendation == "DENOISE" or overall_score >= 6.5:
                # Good quality but needs refinement
                next_node = "denoising"
                action = "Good quality - Starting denoising"
            else:
                # Need more information - continue research loop
                next_node = "search_agent" 
                action = "Need more information - Continuing search & draft update cycle"
            
            print(f"Quality evaluation (iteration {current_iteration}/{max_iterations}):")
            print(f"   Overall score: {overall_score:.1f}/10")
            print(f"   Recommendation: {recommendation}")
            print(f"   Action: {action}")
            print(f"   Reasoning: {reasoning}")
            
            # Use Command for dynamic routing with state update
            return Command(
                update={
                    "quality_evaluation": {
                        "scores": scores,
                        "overall_score": overall_score,
                        "recommendation": recommendation,
                        "reasoning": reasoning,
                        "evaluated_at_iteration": current_iteration
                    },
                    "current_step": "quality_evaluated",
                    "total_tokens": state.get("total_tokens", 0) + 200,  # Estimated tokens
                },
                goto=next_node
            )
            
        except Exception as e:
            print(f"Quality evaluation failed: {str(e)}")
            # Default to continuing search on error
            return Command(
                update={
                    "error_messages": state.get("error_messages", []) + [f"Quality evaluation failed: {str(e)}"],
                    "current_step": "quality_evaluation_failed"
                },
                goto="search_agent"
            )


class ReportGeneratorNode:
    """
    Report Generation Node - Creates final polished research report
    
    This implements Stage 3 of TTD-DR: Final Report Generation
    """
    
    def __init__(self, client: Any = None):
        """Initialize report generator node with client"""
        self.client = client
    
    def __call__(self, state: TTDResearchState) -> Dict[str, Any]:
        """Generate final research report with proper formatting and citations"""
        try:
            client = self.client  # Use injected client
            system_prompts = get_system_prompts()
            
            # Remove any existing references section from draft
            clean_draft = remove_references_section(state['current_draft'])
            
            # Build references section
            references_section = build_references_section(state.get("sources", []))
            
            # Build metadata section
            metadata_section = build_metadata_section(state)
            
            report_prompt = get_report_finalization_prompt(
                state.get('clarified_query') or state['original_query'],
                clean_draft,
                references_section
            )
            
            response = client.chat.completions.create(
                model=state["model_name"],
                messages=[
                    {"role": "system", "content": system_prompts["report_generator"]},
                    {"role": "user", "content": report_prompt}
                ],
                temperature=0.4,  # Balanced creativity for final report
                max_tokens=4000
            )
            
            final_report = clean_reasoning_tags(response.choices[0].message.content.strip())
            
            # Ensure report has proper structure
            if not any(section in final_report.lower() for section in ['references', 'bibliography', 'sources']):
                final_report += f"\n\n{references_section}"
            
            # Add metadata section
            final_report += f"\n\n{metadata_section}"
            
            return {
                "current_draft": final_report,
                "total_tokens": state["total_tokens"] + response.usage.completion_tokens,
                "current_step": "report_finalized",
                "status": "completed"
            }
            
        except Exception as e:
            return {
                "error_messages": state["error_messages"] + [f"Report generation failed: {str(e)}"],
                "status": "failed"
            }


class NoisyDraftGeneratorNode:
    """
    Noisy Draft Generator Node - Creates initial draft
    Creates initial "noisy starting point" as mentioned in paper Section 2.3
    """
    
    def __init__(self, client: Any = None):
        self.client = client
    
    def __call__(self, state: TTDResearchState) -> Dict[str, Any]:
        """
        Generate initial noisy draft R0 based on query and plan only
        This serves as "noisy starting point" that will be iteratively improved
        """
        try:
            client = self.client
            system_prompts = get_system_prompts()
            
            # Get basic inputs
            research_query = state.get('clarified_query') or state['original_query']
            research_plan = state.get("research_plan", "")
            
            # Generate deliberately incomplete/noisy draft
            noisy_draft_prompt = get_noisy_draft_prompt(
                query=research_query,
                plan=research_plan
            )
            
            response = client.chat.completions.create(
                model=state["model_name"],
                messages=[
                    {"role": "system", "content": "You are creating an initial noisy draft that will be iteratively improved through search and denoising."},
                    {"role": "user", "content": noisy_draft_prompt}
                ],
                temperature=0.8,  # Higher temperature for more "noise"
                max_tokens=2000
            )
            
            noisy_draft = clean_reasoning_tags(response.choices[0].message.content.strip())
            
            print(f"Generated noisy draft R0: {len(noisy_draft)} chars")
            print(f"Identified gaps will guide search questions...")
            
            return {
                "current_draft": noisy_draft,  # This is R0
                "draft_history": [{"iteration": 0, "type": "noisy_initial", "content": noisy_draft, "timestamp": time.time()}],
                "current_iteration": 0,  # Start at 0
                "sources": [],  # Empty Q, A history
                "total_tokens": state.get("total_tokens", 0) + response.usage.completion_tokens,
                "current_step": "noisy_draft_generated",
                "status": "in_progress"
            }
            
        except Exception as e:
            print(f"Noisy draft generation failed: {str(e)}")
            return {
                "error_messages": state.get("error_messages", []) + [f"Noisy draft generation failed: {str(e)}"],
                "status": "failed"
            }

class DraftBasedQuestionGeneratorNode:
    """
    Draft-Based Question Generator Node - Generates questions from draft gaps
    Q_t = M_Q(q, P, R_{t-1}, Q, A) - generates questions to address gaps in current draft
    """
    
    def __init__(self, client: Any = None):
        self.client = client
    
    def __call__(self, state: TTDResearchState) -> Dict[str, Any]:
        """
        Generate search questions based on current draft gaps
        Generates targeted search questions based on draft analysis
        """
        try:
            client = self.client
            system_prompts = get_system_prompts()
            
            # Research parameters
            original_query = state.get('clarified_query') or state['original_query']  # q
            research_plan = state.get("research_plan", "")  # P
            current_draft = state.get("current_draft", "")  # R_{t-1}
            sources = state.get("sources", [])  # Q, A history
            current_iteration = state.get("current_iteration", 0)
            
            # Analyze Q, A history
            previous_questions = []
            previous_answers = []
            for source in sources[-10:]:
                q = source.get('gap_addressed', '')
                a = source.get('content', '')
                if q and a:
                    previous_questions.append(q)
                    previous_answers.append(a[:200] + "..." if len(a) > 200 else a)
            
            qa_context = ""
            if previous_questions:
                qa_pairs = [f"Q: {q}\nA: {a}" for q, a in zip(previous_questions[-3:], previous_answers[-3:])]
                qa_context = f"PREVIOUS Q&A PAIRS:\n" + "\n\n".join(qa_pairs)
            else:
                qa_context = "PREVIOUS Q&A PAIRS: None (first iteration)"
            
            # M_Q implementation prompt
            question_generation_prompt = get_draft_question_generation_prompt(
                original_query=original_query,
                plan=research_plan,
                current_draft=current_draft,
                qa_context=qa_context,
                current_iteration=current_iteration
            )
            
            response = client.chat.completions.create(
                model=state["model_name"],
                messages=[
                    {"role": "system", "content": "You are a research assistant analyzing draft gaps and generating targeted search questions."},
                    {"role": "user", "content": question_generation_prompt}
                ],
                temperature=0.6,  # Moderate creativity for diverse questions
                max_tokens=1200
            )
            
            result = clean_reasoning_tags(response.choices[0].message.content.strip())
            
            # Parse questions
            search_questions = []
            rationale = ""
            
            lines = result.split('\n')
            for line in lines:
                if line.startswith('SEARCH_QUESTION_'):
                    question = line.split(':', 1)[1].strip() if ':' in line else ''
                    if question and len(question) > 10:
                        search_questions.append(question)
                elif line.startswith('RATIONALE:'):
                    rationale = line.split(':', 1)[1].strip() if ':' in line else ''
            
            print(f"Generated {len(search_questions)} M_Q questions (iteration {current_iteration + 1})")
            print(f"Rationale: {rationale[:100]}..." if rationale else "")
            
            # Display questions for verification
            for i, q in enumerate(search_questions[:3], 1):
                print(f"  Q{i}: {q[:120]}{'...' if len(q) > 120 else ''}")
            
            return {
                "search_questions": search_questions[:3],  # Limit to 3 questions
                "question_rationale": rationale,
                "current_step": "draft_based_questions_generated",
                "total_tokens": state.get("total_tokens", 0) + response.usage.completion_tokens,
            }
            
        except Exception as e:
            print(f"Draft-based question generation failed: {str(e)}")
            return {
                "error_messages": state.get("error_messages", []) + [f"M_Q failed: {str(e)}"],
                "search_questions": [],
                "current_step": "question_generation_failed"
            }

class DenoisingUpdaterNode:
    """
    Denoising Updater Node - Updates draft with new information
    R_t = M_R(q, R_{t-1}, Q, A) - remove "noise" and update draft with new information
    """
    
    def __init__(self, client: Any = None):
        self.client = client
    
    def __call__(self, state: TTDResearchState) -> Dict[str, Any]:
        """
        Update draft by removing noise and integrating new search results
        Updates draft by integrating new search information
        """
        try:
            client = self.client
            system_prompts = get_system_prompts()
            
            # Research parameters
            original_query = state.get('clarified_query') or state['original_query']  # q
            current_draft = state.get("current_draft", "")  # R_{t-1}
            sources = state.get("sources", [])  # Q, A history
            current_iteration = state.get("current_iteration", 0)
            
            # Get the most recent search results (A_t)
            latest_sources = sources[-3:] if len(sources) >= 3 else sources  # Latest answers
            
            if not latest_sources:
                print("No search results to integrate - keeping current draft")
                return {
                    "current_draft": current_draft,
                    "current_step": "no_denoising_needed"
                }
            
            # Format latest Q&A for denoising
            latest_qa = []
            for source in latest_sources:
                question = source.get('gap_addressed', '')
                answer = source.get('content', '')
                if question and answer:
                    latest_qa.append(f"Q: {question}\nA: {answer}")
            
            qa_context = "\n\n".join(latest_qa) if latest_qa else "No recent search results"
            
            # Format all Q&A history for context
            all_qa_history = []
            for source in sources:
                q = source.get('gap_addressed', '')
                a = source.get('content', '')
                if q and a:
                    all_qa_history.append(f"Q: {q}\nA: {a[:150]}..." if len(a) > 150 else f"Q: {q}\nA: {a}")
            
            full_qa_context = "\n\n".join(all_qa_history[-5:]) if all_qa_history else "No Q&A history"
            
            # M_R implementation prompt
            denoising_prompt = get_denoising_update_prompt(
                original_query=original_query,
                current_draft=current_draft,
                qa_context=qa_context,
                full_qa_context=full_qa_context,
                current_iteration=current_iteration
            )
            
            response = client.chat.completions.create(
                model=state["model_name"],
                messages=[
                    {"role": "system", "content": "You are a research assistant removing noise from draft and integrating new search information."},
                    {"role": "user", "content": denoising_prompt}
                ],
                temperature=0.3,  # Lower temperature for consistent denoising
                max_tokens=3500
            )
            
            denoised_draft = clean_reasoning_tags(response.choices[0].message.content.strip())
            
            # Calculate improvement metrics
            draft_improvement = len(denoised_draft) - len(current_draft)
            sources_integrated = len(latest_sources)
            
            print(f"Denoising completed (R{current_iteration} -> R{current_iteration + 1})")
            print(f"Draft change: {draft_improvement:+d} chars, {sources_integrated} sources integrated")
            print(f"Draft denoising applied successfully")
            
            # Update draft history
            draft_history = state.get("draft_history", [])
            draft_history.append({
                "iteration": current_iteration + 1,
                "type": "denoised",
                "content": denoised_draft,
                "sources_integrated": sources_integrated,
                "timestamp": time.time()
            })
            
            return {
                "current_draft": denoised_draft,  # This is R_t
                "draft_history": draft_history,
                "current_iteration": current_iteration + 1,  # Increment iteration
                "total_tokens": state.get("total_tokens", 0) + response.usage.completion_tokens,
                "current_step": f"draft_denoised_R{current_iteration + 1}",
                "status": "in_progress"
            }
            
        except Exception as e:
            print(f"Denoising update failed: {str(e)}")
            return {
                "error_messages": state.get("error_messages", []) + [f"M_R failed: {str(e)}"],
                "current_step": "denoising_failed"
            }

class IterationControllerNode:
    """
    Iteration Controller Node - Controls research loop termination
    Implements Lines 7-9: if exit_loop then break
    """
    
    def __init__(self, client: Any = None):
        self.client = client
    
    def __call__(self, state: TTDResearchState) -> Dict[str, Any]:
        """
        Control research iteration loop - decide whether to continue or exit
        Implements exit_loop condition checking
        """
        try:
            client = self.client
            system_prompts = get_system_prompts()
            
            # Research loop parameters
            current_iteration = state.get("current_iteration", 0)
            max_iterations = state.get("max_iterations", 3)
            current_draft = state.get("current_draft", "")
            sources = state.get("sources", [])
            original_query = state.get('clarified_query') or state['original_query']
            
            # Check hard limits first
            if current_iteration >= max_iterations:
                print(f"Max iterations ({max_iterations}) reached - Research loop terminating")
                return {
                    "exit_loop": True,
                    "termination_reason": "max_iterations_reached",
                    "current_step": "research_loop_complete"
                }
            
            if not current_draft:
                print("No draft available - continuing loop")
                return {
                    "exit_loop": False,
                    "current_step": "continue_loop_no_draft"
                }
            
            # Evaluate draft quality to determine if more iterations needed
            evaluation_prompt = get_iteration_evaluation_prompt(
                original_query=original_query,
                current_draft=current_draft,
                sources_count=len(sources),
                current_iteration=current_iteration,
                max_iterations=max_iterations
            )
            
            response = client.chat.completions.create(
                model=state["model_name"],
                messages=[
                    {"role": "system", "content": "You are evaluating exit_loop condition for TTD-DR research quality."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.2,  # Low temperature for consistent evaluation
                max_tokens=800
            )
            
            evaluation_result = response.choices[0].message.content.strip()
            
            # Parse evaluation
            decision = "CONTINUE"  # Default
            completeness_score = 5.0
            quality_score = 5.0
            remaining_gaps = ""
            reasoning = ""
            
            for line in evaluation_result.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().upper()
                    value = value.strip()
                    
                    if 'DECISION' in key:
                        decision = value.upper()
                    elif 'COMPLETENESS_SCORE' in key:
                        try:
                            completeness_score = float(value)
                        except:
                            pass
                    elif 'QUALITY_SCORE' in key:
                        try:
                            quality_score = float(value)
                        except:
                            pass
                    elif 'REMAINING_GAPS' in key:
                        remaining_gaps = value
                    elif 'REASONING' in key:
                        reasoning = value
            
            # Determine exit_loop based on evaluation
            should_exit = (decision == "EXIT") or (completeness_score >= 8.0 and quality_score >= 7.5)
            
            if should_exit:
                print(f"Research loop terminating: Quality sufficient")
                print(f"Scores - Completeness: {completeness_score:.1f}, Quality: {quality_score:.1f}")
                print(f"Reasoning: {reasoning}")
                
                return {
                    "exit_loop": True,
                    "termination_reason": "quality_sufficient",
                    "completeness_score": completeness_score,
                    "quality_score": quality_score,
                    "evaluation_reasoning": reasoning,
                    "current_step": "research_loop_complete",
                    "total_tokens": state.get("total_tokens", 0) + response.usage.completion_tokens,
                }
            else:
                print(f"Research loop continuing: More iterations needed")
                print(f"Scores - Completeness: {completeness_score:.1f}, Quality: {quality_score:.1f}")
                print(f"Gaps: {remaining_gaps}")
                
                return {
                    "exit_loop": False,
                    "completeness_score": completeness_score,
                    "quality_score": quality_score,
                    "remaining_gaps": remaining_gaps,
                    "evaluation_reasoning": reasoning,
                    "current_step": "continue_research_loop",
                    "total_tokens": state.get("total_tokens", 0) + response.usage.completion_tokens,
                }
            
        except Exception as e:
            print(f"Iteration control failed: {str(e)}")
            # Default: continue if uncertain
            return {
                "exit_loop": False,
                "error_messages": state.get("error_messages", []) + [f"Iteration control failed: {str(e)}"],
                "current_step": "iteration_control_failed"
            }