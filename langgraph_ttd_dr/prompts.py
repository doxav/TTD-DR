from typing import Dict, Any, List


def get_clarification_prompt(query: str) -> str:
    """
    Generate clarification prompt to analyze user query and determine if more information is needed
    
    Args:
        query: The original user query
        
    Returns:
        Formatted clarification prompt
    """
    return f"""
Analyze the following research query and determine if it needs clarification for better research results:

QUERY: {query}

Your task is to:
1. Assess the clarity and specificity of the query
2. Identify any ambiguous terms or concepts
3. Determine if additional context would significantly improve research quality
4. If clarification is needed, generate specific questions to ask the user

Evaluation criteria for needing clarification:
- Query is too broad or vague
- Missing important context (time period, geographic scope, specific industry, etc.)
- Ambiguous terms that could have multiple interpretations
- Unclear research depth or format preferences
- Multiple possible research directions without clear priority

Provide your response in this format:

NEEDS_CLARIFICATION: [YES/NO]

ANALYSIS:
[Your analysis of the query's clarity and completeness]

CLARIFICATION_QUESTIONS:
[If YES above, provide 2-3 specific questions to help clarify the query. Make them concise and focused.]

Remember: Only request clarification if it would significantly improve the research quality. Don't ask for unnecessary details.
"""


def get_planning_prompt(query: str, planning_context: str = "") -> str:
    """
    Generate research planning prompt with support for integrated multi-faceted research
    
    Args:
        query: The research question
        planning_context: Additional context about integrated aspects and scope
        
    Returns:
        Formatted research planning prompt
    """
    base_prompt = f"""
Create a comprehensive research plan for the following query:

{query}

{planning_context if planning_context else ""}

Your research plan should include:

1. **Research Objectives**
   - Primary research goals
   - Key questions to answer
   - Expected deliverables

2. **Research Strategy**
   - Methodology and approach
   - Information sources to explore
   - Search keywords and terms

3. **Content Structure**
   - Main sections to cover
   - Subsections and topics
   - Logical flow and organization

4. **Quality Criteria**
   - Evidence standards
   - Source reliability requirements
   - Comprehensiveness benchmarks

{f'''
5. **Multi-Faceted Integration** (if applicable)
   - How to synthesize multiple research aspects
   - Cross-aspect connections and relationships
   - Balanced coverage strategy
''' if planning_context else ''}

Focus on creating a detailed, actionable plan that ensures comprehensive coverage of all research aspects.
The plan should be specific enough to guide effective information gathering and synthesis.
"""
    return base_prompt.strip()


def get_draft_generation_prompt(query: str, research_plan: str, draft_context: str = "") -> str:
    """
    Generate draft generation prompt with support for multi-faceted integrated research
    
    Args:
        query: The research question
        research_plan: The research strategy and plan
        draft_context: Additional context about integrated aspects and scope
        
    Returns:
        Formatted draft generation prompt
    """
    return f"""
Create a comprehensive research draft for the following query:

{query}

Research Plan:
{research_plan}

{draft_context if draft_context else ""}

Generate a detailed research draft that:

1. **Comprehensive Coverage**
   - Addresses all aspects mentioned in the research context
   - Provides balanced treatment of each dimension
   - Integrates multiple perspectives cohesively

2. **Structure and Organization**
   - Clear introduction outlining all aspects to be covered
   - Well-organized main sections for each research dimension
   - Logical flow connecting different aspects
   - Conclusion synthesizing all perspectives

3. **Content Requirements**
   - Include current information and recent developments
   - Address key questions from the research plan
   - Identify areas where additional research is needed
   - Provide evidence-based analysis

4. **Integration Strategy**
   - Show connections between different research aspects
   - Highlight complementary and contrasting findings
   - Synthesize insights across multiple dimensions
   - Address potential gaps or conflicts

The draft should be substantial and comprehensive, serving as a solid foundation for further research and refinement. Focus on creating content that thoroughly explores all specified research dimensions while maintaining coherent integration.

Write in a clear, professional academic style suitable for a research report.
"""


def get_gap_analysis_prompt(query: str, current_draft: str) -> str:
    """
    Generate gap analysis prompt for GapAnalyzerNode
    
    Args:
        query: The original research question
        current_draft: The current state of the research draft
        
    Returns:
        Formatted gap analysis prompt
    """
    return f"""
Analyze the following research draft to identify specific gaps and areas that need external research.
Be thorough and aggressive in finding areas for improvement - even good drafts can be enhanced.

Original Query: {query}

Current Draft:
{current_draft}

CRITICAL ANALYSIS REQUIRED:
1. MANDATORY: Find ALL [NEEDS RESEARCH], [SOURCE NEEDED], [CITATION NEEDED] tags
2. Identify claims lacking evidence (even if not explicitly marked)
3. Find areas that could benefit from recent data or statistics
4. Spot generalizations that need specific examples
5. Locate outdated information or areas needing current updates
6. Identify missing perspectives or counterarguments

For each gap you identify, provide:
1. SECTION: Which section has the gap
2. GAP_TYPE: [PLACEHOLDER_TAG, MISSING_INFO, OUTDATED_INFO, NEEDS_EVIDENCE, LACKS_DEPTH, NEEDS_EXAMPLES, MISSING_PERSPECTIVE]
3. SPECIFIC_NEED: Exactly what information is needed
4. SEARCH_QUERY: A specific, targeted search query to address this gap
5. PRIORITY: [HIGH, MEDIUM, LOW] - HIGH for placeholder tags and critical missing info

Format each gap as:
GAP_ID: [number]
SECTION: [section name]
GAP_TYPE: [type]
SPECIFIC_NEED: [what's missing]
SEARCH_QUERY: [search query to find this info]
PRIORITY: [priority level]

IMPORTANT: Identify AT LEAST 4-8 gaps. Be critical and thorough.
Even well-written sections can benefit from additional evidence, examples, or perspectives.
Push for depth, accuracy, and comprehensiveness in the research.
"""


def get_denoising_prompt(query: str, current_draft: str, retrieved_content: str) -> str:
    """
    Generate denoising prompt for DenoisingNode
    
    Args:
        query: The original research question
        current_draft: The current state of the research draft
        retrieved_content: New information retrieved from searches
        
    Returns:
        Formatted denoising prompt
    """
    return f"""
You are performing a denoising step in a research diffusion process.

TASK: Integrate new retrieved information with the existing draft to reduce "noise" (gaps, inaccuracies, incompleteness).

Original Query: {query}

Current Draft:
{current_draft}

New Retrieved Information:
{retrieved_content}

DENOISING INSTRUCTIONS:
1. Identify where the new information fills gaps marked with [NEEDS RESEARCH] or [SOURCE NEEDED]
2. Replace placeholder content with specific, detailed information
3. Add proper citations for new information using [1], [2], etc.
4. Resolve any conflicts between new and existing information
5. Maintain the overall structure and coherence of the draft
6. Enhance depth and accuracy without losing existing valuable insights
7. Mark any remaining research needs with [NEEDS RESEARCH]

Return the improved draft with integrated information.
Focus on substantial improvements - don't just make minor changes.
"""


def get_quality_evaluation_prompt(query: str, previous_draft: str, current_draft: str) -> str:
    """
    Generate quality evaluation prompt for QualityEvaluatorNode
    
    Args:
        query: The original research question
        previous_draft: The previous version of the draft
        current_draft: The current version of the draft
        
    Returns:
        Formatted quality evaluation prompt
    """
    return f"""
Evaluate the research draft quality improvement.

Original Query: {query}

Previous Draft:
{previous_draft}

Current Draft:
{current_draft}

Rate the following aspects from 0.0 to 1.0:

COMPLETENESS: How well does the current draft address all aspects of the query?
ACCURACY: How accurate and reliable is the information?
DEPTH: How detailed and comprehensive is the analysis?
COHERENCE: How well-structured and logically organized is the draft?
CITATIONS: How well are sources cited and integrated?
IMPROVEMENT: How much better is this draft compared to the previous version?

Respond ONLY with:
COMPLETENESS: [score]
ACCURACY: [score]
DEPTH: [score]
COHERENCE: [score]
CITATIONS: [score]
IMPROVEMENT: [score]
"""


def get_report_finalization_prompt(query: str, current_draft: str, citation_context: str) -> str:
    """
    Generate final report polishing prompt for ReportGeneratorNode
    
    Args:
        query: The original research question
        current_draft: The current state of the research draft
        citation_context: Available citations context
        
    Returns:
        Formatted finalization prompt
    """
    return f"""
Apply final polishing to this research report. This is the last step in the TTD-DR diffusion process.

Original Query: {query}

Current Draft:
{current_draft}

{citation_context}

FINALIZATION TASKS:
1. Ensure professional academic formatting with clear sections
2. Verify all citations are properly formatted as [1], [2], etc.
3. Add a compelling title and executive summary
4. Ensure smooth transitions between sections
5. Add conclusion that directly addresses the original query
6. **CRITICAL**: Remove ALL [NEEDS RESEARCH], [SOURCE NEEDED], and similar placeholder tags
7. Replace any remaining placeholders with actual content or remove incomplete sections
8. Polish language and style for clarity and impact

**CRITICAL REQUIREMENTS**: 
- The final report must NOT contain ANY placeholder tags
- Remove incomplete sections or complete them with available information
- Ensure all statements are backed by available evidence
- The report must be publication-ready with no incomplete elements
- DO NOT create a References section - it will be added automatically

Return the final polished research report.
"""


def get_system_prompts() -> Dict[str, str]:
    """
    Get system prompts for different node types
    
    Returns:
        Dictionary of system prompts for each node type
    """
    return {
        "clarification": "You are an expert research consultant who helps clarify and refine research questions for optimal results.",
        "planner": "You are a research planning specialist with expertise in academic research methodology.",
        "draft_generator": "You are an expert research writer skilled at creating structured academic drafts.",
        "gap_analyzer": "You are an expert research analyst skilled at identifying gaps and weaknesses in academic writing.",
        "search_agent": "You are a research information specialist focused on finding relevant and accurate sources.",
        "denoising": "You are an expert research synthesizer performing draft denoising and integration.",
        "quality_evaluator": "You are an expert research quality evaluator with expertise in academic standards.",
        "report_generator": "You are an expert academic writer specializing in final report polishing and formatting."
    }


# Advanced Prompt Templates for Specific Use Cases

def get_search_query_optimization_prompt(gap: Dict[str, Any]) -> str:
    """
    Generate a prompt to optimize search queries for better results
    
    Args:
        gap: Gap information dictionary
        
    Returns:
        Optimized search query prompt
    """
    return f"""
Optimize this search query for better web search results:

Original Gap: {gap.get('specific_need', '')}
Current Query: {gap.get('search_query', '')}
Section: {gap.get('section', '')}
Priority: {gap.get('priority', '')}

Generate 3 alternative search queries that would be more effective:
1. A broad query for general coverage
2. A specific query for detailed information
3. A current/recent query for up-to-date information

Format as:
BROAD: [query]
SPECIFIC: [query] 
CURRENT: [query]
"""


def get_source_relevance_prompt(query: str, gap: Dict[str, Any], source_content: str) -> str:
    """
    Generate prompt to evaluate source relevance to a specific gap
    
    Args:
        query: Original research question
        gap: Gap information
        source_content: Content from retrieved source
        
    Returns:
        Source relevance evaluation prompt
    """
    return f"""
Evaluate how well this source addresses the identified research gap:

Research Query: {query}
Gap Need: {gap.get('specific_need', '')}
Gap Section: {gap.get('section', '')}

Source Content:
{source_content}

Rate the relevance from 0.0 to 1.0 and explain:

RELEVANCE: [score]
EXPLANATION: [How well does this source address the gap? What specific information does it provide?]
USEFUL_QUOTES: [Key quotes or data points that could be used]
LIMITATIONS: [What aspects of the gap does this source NOT address?]
"""


def get_citation_integration_prompt(draft_section: str, source_info: Dict[str, Any], citation_number: int) -> str:
    """
    Generate prompt for integrating a specific citation into a draft section
    
    Args:
        draft_section: The section of the draft to update
        source_info: Information about the source to cite
        citation_number: The citation number to use
        
    Returns:
        Citation integration prompt
    """
    return f"""
Integrate this source citation into the draft section naturally:

Current Section:
{draft_section}

Source Information:
Title: {source_info.get('title', '')}
Content: {source_info.get('content', '')}
Citation Number: [{citation_number}]

INTEGRATION TASKS:
1. Find the most appropriate place to integrate this information
2. Add the citation number [{citation_number}] after relevant statements
3. Ensure the integration flows naturally with existing text
4. Remove or replace any [SOURCE NEEDED] or [NEEDS RESEARCH] tags if this source addresses them
5. Maintain the section's coherence and structure

Return the updated section with the citation properly integrated.
"""


def get_counterargument_prompt(query: str, current_position: str) -> str:
    """
    Generate prompt to identify potential counterarguments or alternative perspectives
    
    Args:
        query: Research question
        current_position: Current position or findings in the draft
        
    Returns:
        Counterargument identification prompt
    """
    return f"""
Identify potential counterarguments or alternative perspectives for this research:

Research Question: {query}

Current Position/Findings:
{current_position}

ANALYSIS TASKS:
1. What are the potential weaknesses in this position?
2. What alternative viewpoints exist on this topic?
3. What evidence might contradict these findings?
4. What limitations should be acknowledged?
5. What questions remain unanswered?

Provide a balanced analysis that would strengthen the research by acknowledging different perspectives.
"""


def get_depth_enhancement_prompt(section: str, topic: str) -> str:
    """
    Generate prompt to add more depth to a shallow section
    
    Args:
        section: The section text that needs more depth
        topic: The topic/subject of the section
        
    Returns:
        Depth enhancement prompt
    """
    return f"""
Enhance the depth and detail of this section about {topic}:

Current Section:
{section}

ENHANCEMENT TASKS:
1. Add more specific details and examples
2. Include relevant data, statistics, or metrics where appropriate
3. Provide historical context or background information
4. Explain underlying mechanisms, causes, or relationships
5. Add technical details or implementation specifics
6. Include comparisons or contrasts with related concepts
7. Discuss implications and consequences

Return an enhanced version that provides much greater depth while maintaining clarity and structure.
Mark areas that would benefit from external sources with [SOURCE NEEDED].
""" 

# Additional prompt functions for nodes.py migration

def get_draft_update_prompt(query: str, plan: str, current_draft: str, search_context: str) -> str:
    """
    Generate draft update prompt for DraftGeneratorNode
    
    Args:
        query: Original research query
        plan: Research plan
        current_draft: Current draft content
        search_context: New search results context
        
    Returns:
        Formatted draft update prompt
    """
    return f"""
UPDATE RESEARCH DRAFT WITH NEW INFORMATION

ORIGINAL QUERY: {query}
RESEARCH PLAN: {plan}

CURRENT DRAFT:
{current_draft}

NEW SEARCH RESULTS:
{search_context}

INSTRUCTIONS FOR DRAFT UPDATE:
1. **Preserve Structure**: Maintain the existing draft structure and flow
2. **Integrate New Information**: Add new facts, examples, and insights from search results
3. **Remove Noise**: Fix any inaccuracies or contradictions identified by new information
4. **Enhance Existing Sections**: Expand weak areas with concrete details
5. **Maintain Coherence**: Ensure smooth integration without abrupt changes
6. **Track Improvements**: Note what specific improvements were made

OUTPUT REQUIREMENTS:
- Return the UPDATED DRAFT (R_t) with new information integrated
- Preserve good existing content while improving weak areas
- Ensure logical flow and consistency
- Target 1500-2500 words for comprehensive coverage

This is the core of TTD-DR: progressive refinement through retrieval-augmented denoising.
"""


def get_search_question_generation_prompt(original_query: str, plan: str, current_draft: str, qa_context: str, iteration: int = 0) -> str:
    """
    Generate search question generation prompt for GapAnalyzerNode
    
    Args:
        original_query: Original research query
        plan: Research plan
        current_draft: Current draft content
        qa_context: Previous Q&A context
        iteration: Current iteration number
        
    Returns:
        Formatted search question generation prompt
    """
    return f"""
GENERATE SEARCH QUESTIONS FROM DRAFT ANALYSIS
Generate NEW search questions based on current research state:

ORIGINAL QUERY (q): {original_query}
RESEARCH PLAN (P): {plan}
ITERATION: {iteration + 1}

CURRENT DRAFT (R_{{t-1}}):
{current_draft if current_draft else 'No draft yet'}

PREVIOUS Q&A PAIRS (Q, A) - ANALYZE WHAT WE ALREADY KNOW:
{qa_context}

CRITICAL INSTRUCTIONS:
1. Analyze BOTH the current draft AND previous answers to identify gaps
2. Avoid asking questions that are already answered in Q&A history
3. Generate questions that BUILD UPON existing knowledge
4. Focus on information gaps NOT covered by previous answers
5. Consider contradictions or inconsistencies in previous answers

Each question should target:
- Information missing from BOTH draft AND previous answers
- Deeper details on topics briefly covered in previous answers
- Recent developments not captured in previous searches
- Different perspectives to validate/contradict previous answers
- Specific examples or case studies not yet found

Output format:
SEARCH_QUESTION_1: [specific question building on previous knowledge]
SEARCH_QUESTION_2: [specific question targeting new gaps] 
SEARCH_QUESTION_3: [specific question for validation/depth]
"""


def get_noisy_draft_prompt(query: str, plan: str) -> str:
    """
    Generate noisy draft prompt for NoisyDraftGeneratorNode
    
    Args:
        query: Research query
        plan: Research plan
        
    Returns:
        Formatted noisy draft prompt
    """
    return f"""
Create an INITIAL DRAFT research report based ONLY on the query and plan.
This should be a "noisy starting point" that will be iteratively improved.

RESEARCH QUERY: {query}
RESEARCH PLAN: {plan}

IMPORTANT - Create a DELIBERATELY INCOMPLETE draft:
1. Write based only on general knowledge, NO external search yet
2. Include placeholder sections that need more information
3. Use phrases like "This needs more research", "More data required"
4. Make educated guesses that may be incorrect (noise)
5. Leave gaps where specific examples or recent data are needed
6. Structure the report but keep content sparse and incomplete

This draft will guide what information to search for next.
Target 800-1200 words with clear gaps and limitations.

NOISY DRAFT R0:
"""


def get_draft_question_generation_prompt(original_query: str, plan: str, current_draft: str, qa_context: str, current_iteration: int) -> str:
    """
    Generate question generation prompt for DraftBasedQuestionGeneratorNode
    
    Args:
        original_query: Original research query
        plan: Research plan  
        current_draft: Current draft content
        qa_context: Previous Q&A context
        current_iteration: Current iteration number
        
    Returns:
        Formatted question generation prompt
    """
    return f"""
GENERATE SEARCH QUESTIONS FROM DRAFT GAPS
Generate search questions to address gaps in the current draft.

ORIGINAL QUERY (q): {original_query}
RESEARCH PLAN (P): {plan}
ITERATION: {current_iteration + 1}

CURRENT DRAFT (R_{{t-1}}) TO ANALYZE:
{current_draft}

{qa_context}

DRAFT ANALYSIS INSTRUCTIONS:
1. Identify specific gaps, placeholders, and "noise" in the current draft
2. Find statements that need verification or more detail
3. Look for missing examples, data, or recent information
4. Avoid asking questions already answered in Q&A history
5. Focus on information that would most improve draft quality

Generate 2-3 specific search questions that target:
- Missing factual information mentioned in draft
- Verification of uncertain statements
- Recent developments or current data
- Specific examples or case studies needed
- Contradictions or gaps in reasoning

Output format:
SEARCH_QUESTION_1: [specific question targeting draft gap]
SEARCH_QUESTION_2: [specific question targeting draft gap]
SEARCH_QUESTION_3: [specific question targeting draft gap]

RATIONALE: [explain which gaps these questions address]
"""


def get_denoising_update_prompt(original_query: str, current_draft: str, qa_context: str, full_qa_context: str, current_iteration: int) -> str:
    """
    Generate denoising update prompt for DenoisingUpdaterNode
    
    Args:
        original_query: Original research query
        current_draft: Current draft content
        qa_context: Recent search results context
        full_qa_context: Full Q&A history context
        current_iteration: Current iteration number
        
    Returns:
        Formatted denoising update prompt
    """
    return f"""
UPDATE DRAFT WITH NEW SEARCH INFORMATION
Remove "noise" (imprecision, incompleteness) from draft and integrate new information.

ORIGINAL QUERY (q): {original_query}
ITERATION: {current_iteration + 1} → {current_iteration + 2}

CURRENT DRAFT (R_{{t-1}}) TO DENOISE:
{current_draft}

NEW SEARCH RESULTS (A_t) TO INTEGRATE:
{qa_context}

FULL Q&A HISTORY (Q, A) FOR CONTEXT:
{full_qa_context}

DENOISING INSTRUCTIONS:
1. **Remove Noise**: Fix imprecise, incomplete, or incorrect information
2. **Integrate New Information**: Add concrete facts, examples, and details from search results
3. **Verify & Correct**: Use search results to verify or correct existing statements
4. **Fill Gaps**: Replace placeholders and "needs research" sections with actual information
5. **Maintain Structure**: Keep the overall organization while improving content
6. **Preserve Quality**: Keep good existing content while enhancing weak areas

OUTPUT REQUIREMENTS:
- Return UPDATED DRAFT (R_t) with noise removed and new information integrated
- Maintain logical flow and coherent structure
- Ensure all new information is properly integrated, not just appended
- Target improvement in completeness, accuracy, and specificity
- Keep the tone and style consistent

This is the core denoising step from TTD-DR.

DENOISED DRAFT (R_t):
"""


def get_iteration_evaluation_prompt(original_query: str, current_draft: str, sources_count: int, current_iteration: int, max_iterations: int) -> str:
    """
    Generate iteration evaluation prompt for IterationControllerNode
    
    Args:
        original_query: Original research query
        current_draft: Current draft content
        sources_count: Number of sources gathered
        current_iteration: Current iteration number
        max_iterations: Maximum iterations allowed
        
    Returns:
        Formatted evaluation prompt
    """
    return f"""
Evaluate if this research draft needs more iterations or is ready for finalization.
This is the exit_loop condition evaluation.

ORIGINAL QUERY: {original_query}
CURRENT ITERATION: {current_iteration}/{max_iterations}
SEARCH ITERATIONS COMPLETED: {sources_count}

CURRENT DRAFT TO EVALUATE:
{current_draft[:2000]}{'...' if len(current_draft) > 2000 else ''}

EVALUATION CRITERIA:
1. **Completeness**: Are major topics adequately covered?
2. **Information Quality**: Is the information sufficient and accurate?
3. **Gap Analysis**: Are there obvious missing pieces that need more search?
4. **Research Depth**: Does it meet the requirements of the original query?
5. **Diminishing Returns**: Would more iterations significantly improve it?

Based on these criteria, decide:
- CONTINUE: If significant gaps remain and more search would help
- EXIT: If draft is sufficiently complete or more iterations won't help much

Consider that we have {max_iterations - current_iteration} iterations remaining.

Output format:
DECISION: [CONTINUE/EXIT]
COMPLETENESS_SCORE: [1-10]
QUALITY_SCORE: [1-10]
REMAINING_GAPS: [brief description of major gaps, if any]
REASONING: [explain the decision]
"""


def get_simple_search_answer_prompt(search_query: str, search_results: str, context: str) -> str:
    """
    Generate simple search answer prompt for fallback answer generation
    
    Args:
        search_query: The search query
        search_results: Search results content
        context: Additional context
        
    Returns:
        Formatted simple answer prompt
    """
    return f"""
Generate a comprehensive answer based on the search results.

SEARCH QUERY: {search_query}

CONTEXT: {context}

SEARCH RESULTS:
{search_results}

INSTRUCTIONS:
1. **Synthesize Information**: Combine relevant information from multiple search results
2. **Direct Relevance**: Focus on information directly relevant to the search query
3. **Structured Response**: Organize the answer clearly and logically
4. **Cite Sources**: Reference specific sources when appropriate
5. **Comprehensive Coverage**: Provide thorough information while staying focused

Generate a detailed, informative answer that addresses the search query comprehensively.
"""


# Self-Evolution related prompts

def get_answer_variant_prompt(search_query: str, search_results: str, context: str, approach: str, temperature_setting: float) -> str:
    """
    Generate answer variant prompt for self-evolution
    
    Args:
        search_query: Search query
        search_results: Search results content
        context: Additional context
        approach: Approach type (depth, breadth, practical)
        temperature_setting: Temperature setting for variance
        
    Returns:
        Formatted variant generation prompt
    """
    return f"""
Based on search results, provide a comprehensive answer focusing on {approach}:

QUESTION: {search_query}
CONTEXT: {context}

SEARCH RESULTS:
{"\n".join([f"- {result.get('title', '')}: {result.get('content', '')[:400]}..." for result in search_results[:4] if isinstance(search_results, list)])}

Focus on {approach}. Provide a detailed, informative answer.
"""


def get_variant_evaluation_prompt(search_query: str, answer: str) -> str:
    """
    Generate variant evaluation prompt for self-evolution
    
    Args:
        search_query: Search query
        answer: Answer to evaluate
        
    Returns:
        Formatted evaluation prompt
    """
    return f"""
You are an expert evaluator assessing research answer quality. 
Evaluate this answer using the criteria from TTD-DR paper:

QUESTION: {search_query}

ANSWER TO EVALUATE:
{answer}

EVALUATION CRITERIA:
1. Helpfulness (1-10):
   - Satisfies user intent
   - Ease of understanding (fluency and coherence)  
   - Accuracy of information
   - Appropriate language and depth

2. Comprehensiveness (1-10):
   - Absence of missing key information
   - Coverage of important aspects
   - Depth of analysis
   - Balanced perspective

3. Information Quality (1-10):
   - Factual accuracy
   - Recency of information
   - Source reliability indicators
   - Specificity and detail

PROVIDE SCORES AND DETAILED FEEDBACK:
HELPFULNESS_SCORE: [1-10]
COMPREHENSIVENESS_SCORE: [1-10]
QUALITY_SCORE: [1-10]
OVERALL_SCORE: [1-10]

STRENGTHS: [What this variant does well]
WEAKNESSES: [Areas needing improvement]
IMPROVEMENT_SUGGESTIONS: [Specific actionable feedback]
"""


def get_answer_revision_prompt(search_query: str, variant_text: str, fitness_score: float, feedback: dict) -> str:
    """
    Generate answer revision prompt for self-evolution
    
    Args:
        search_query: Original search query
        variant_text: Original answer to revise
        fitness_score: Fitness score from evaluation
        feedback: Detailed feedback dictionary with strengths, weaknesses, suggestions
        
    Returns:
        Formatted revision prompt
    """
    return f"""
Revise this research answer based on expert feedback to improve quality.

ORIGINAL QUESTION: {search_query}

CURRENT ANSWER:
{variant_text}

EVALUATION FEEDBACK:
Fitness Score: {fitness_score:.1f}/10
Strengths: {feedback.get('strengths', '')}
Weaknesses: {feedback.get('weaknesses', '')}
Improvement Suggestions: {feedback.get('suggestions', '')}

REVISION INSTRUCTIONS:
1. Address the specific weaknesses mentioned in the feedback
2. Enhance the strengths while fixing the problems
3. Follow the improvement suggestions precisely
4. Maintain the core valuable information
5. Improve structure, clarity, and comprehensiveness
6. Add missing information or correct inaccuracies

Provide a REVISED answer that addresses all feedback points:
"""


def get_answer_merge_prompt(query: str, variants_to_merge: list, evaluation_context: str = "") -> str:
    """
    Generate answer merging prompt for self-evolution crossover
    
    Args:
        query: Original query
        variants_to_merge: List of answer variants to merge
        evaluation_context: Context about variant quality scores
        
    Returns:
        Formatted merge prompt
    """
    return f"""
Merge multiple research answer variants into a single, superior final answer.
This is the Cross-over step from TTD-DR Self-Evolution (Figure 5).

QUESTION: {query}

{evaluation_context}

VARIANTS TO MERGE:
{chr(10).join([f"=== VARIANT {i+1} ===\n{variant}\n" for i, variant in enumerate(variants_to_merge)])}

CROSS-OVER INSTRUCTIONS:
1. Identify the best information from each variant
2. Combine strengths while eliminating weaknesses
3. Resolve any contradictions between variants
4. Create a coherent, comprehensive final answer
5. Ensure the final answer is better than any individual variant
6. Maintain factual accuracy and proper structure

Produce a SUPERIOR MERGED ANSWER that consolidates the best evolutionary paths:
""" 