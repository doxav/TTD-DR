# 🔬 TTD-DR: Test-Time Diffusion Deep Researcher

> Advanced AI-powered research system using diffusion-based iterative refinement

## 📋 Overview

**TTD-DR (Test-Time Diffusion Deep Researcher)** is an innovative research system that applies diffusion-based algorithms to generate comprehensive, high-quality research reports. Unlike traditional retrieval-augmented generation (RAG) systems, TTD-DR uses a **draft-centric approach** where an evolving draft dynamically guides the research process through multiple iterations.

### 🎯 Key Features

- **🔄 Iterative Draft Refinement**: Starts with a "noisy" initial draft and progressively refines it
- **🎯 Draft-Centric Search**: The evolving draft guides what information to search for next
- **🔍 Multi-Engine Search**: Integrates Tavily, DuckDuckGo, and Naver search engines
- **🧠 Gap Analysis**: Automatically identifies knowledge gaps and fills them systematically
- **⚖️ Quality Evaluation**: Continuous assessment of research completeness and quality
- **🌐 Multi-Language Support**: Works with English, Korean, and other languages
- **🚀 Async Support**: Built with modern async/await patterns for optimal performance

### 🧮 Algorithm Highlights

The system implements the **Denoising with Retrieval (Draft-Centric Approach)** algorithm:

1. **Initialize**: Generate a noisy initial draft R₀
2. **Analyze**: Identify gaps in the current draft
3. **Search**: Query multiple search engines to fill identified gaps  
4. **Denoise**: Update the draft with new information
5. **Evaluate**: Assess quality and determine if more iterations are needed
6. **Iterate**: Repeat until quality threshold is met or max iterations reached

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ttd-dr

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Copy the example environment file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
# Required: Choose one
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini

# OR
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional but recommended
TAVILY_API_KEY=tvly-your-tavily-api-key
NAVER_CLIENT_ID=your-naver-client-id
NAVER_CLIENT_SECRET=your-naver-client-secret
```

### 3. Run the Simple Chatbot

The easiest way to get started is with our simple chatbot interface:

```bash
python chatbot.py
```

**Example interaction:**
```
🤖 TTD-DR Deep Research Chatbot
============================================================
Welcome to the Test-Time Diffusion Deep Researcher!
This chatbot can conduct in-depth research on any topic.

🤔 Your research question: What are the latest developments in artificial intelligence in 2024?

🔍 Research Query: What are the latest developments in artificial intelligence in 2024?
⏳ Starting deep research... (this may take a few minutes)
📊 The system will show progress updates during research
------------------------------------------------------------

[Research process with real-time updates...]

============================================================
📋 RESEARCH REPORT COMPLETED
============================================================
📄 Report Length: 3,247 characters
🔄 Iterations: 3
📚 Sources Used: 12
⏱️  Execution Time: 87.3 seconds
🎯 Status: completed

📖 RESEARCH REPORT:
------------------------------
# Latest Developments in Artificial Intelligence (2024)

## Executive Summary
The year 2024 has marked significant advances in artificial intelligence...

[Comprehensive research report continues...]
```

### 4. Quick Example Mode

For a quick demonstration:

```bash
python chatbot.py --example
```

## 📚 Usage Examples

### Interactive Chatbot Commands

| Command | Description |
|---------|-------------|
| `help` | Show welcome message and instructions |
| `status` | Check system status and API configuration |
| `examples` | Display example research queries |
| `quit` / `exit` | Exit the chatbot |

### Example Research Queries

- **Technology**: "What are the latest developments in artificial intelligence in 2024?"
- **Science**: "How does climate change affect global food security?"
- **Comparison**: "What are the key differences between quantum and classical computing?"
- **Analysis**: "What are the ethical implications of genetic engineering?"
- **Current Events**: "Describe recent advances in space exploration technology"

### API Usage

For programmatic access:

```python
import asyncio
from langgraph_ttd_dr.interface import TTDResearcher
from langgraph_ttd_dr.client_factory import create_openai_client

async def research_example():
    # Create client and researcher
    client = create_openai_client()
    researcher = TTDResearcher(
        client=client,
        max_iterations=5,
        max_sources=15
    )
    
    # Conduct research
    report, metadata = await researcher.research(
        "What is the current state of renewable energy technology?"
    )
    
    print(f"Research completed with {len(metadata['all_sources'])} sources")
    print(f"Iterations: {metadata['iterations']}")
    print(f"Report: {report}")

# Run the example
asyncio.run(research_example())
```

## 🏗️ Architecture

### Core Components

```
📦 langgraph_ttd_dr/
├── 🎛️ interface.py          # Main TTDResearcher class
├── 🔗 client_factory.py     # OpenAI/Azure client management
├── 📊 state.py              # Research state management
├── 🔄 workflow.py           # LangGraph workflow definition
├── 🧩 nodes.py              # Individual workflow nodes
├── 💬 prompts.py            # Centralized prompt management
├── 🔍 tools.py              # Web search tools
└── 🛠️ utils.py              # Utility functions

📄 chatbot.py                # Simple usage example
📄 interactive_chatbot.py    # Advanced interactive interface
```

### Workflow Nodes

1. **QueryClarificationNode**: Improves and clarifies the research question
2. **PlannerNode**: Creates a structured research plan
3. **NoisyDraftGeneratorNode**: Generates the initial draft R₀
4. **DraftBasedQuestionGeneratorNode**: Identifies gaps and generates search queries
5. **SearchAgentNode**: Executes multi-engine web searches
6. **DenoisingUpdaterNode**: Updates the draft with new information
7. **IterationControllerNode**: Decides whether to continue or finalize
8. **ReportGeneratorNode**: Produces the final research report

### Search Engines

- **🔍 Tavily**: High-quality, research-focused search results
- **🦆 DuckDuckGo**: Privacy-focused web search (no API key required)
- **🔍 Naver**: Korean and Asian content specialist

## 📊 Configuration Options

### Research Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 5 | Maximum research iterations |
| `max_sources` | 15 | Maximum sources to collect |
| `search_results_per_gap` | 3 | Results per knowledge gap |
| `recursion_limit` | 50 | LangGraph recursion limit |

### Quality Metrics

TTD-DR tracks multiple quality dimensions:

- **Completeness**: How thoroughly the topic is covered
- **Accuracy**: Factual correctness of information
- **Relevance**: How well content matches the query
- **Coherence**: Logical flow and organization
- **Citation Quality**: Source reliability and diversity

## 🔧 System Requirements

- **Python**: 3.8 or higher
- **Dependencies**: See `requirements.txt`
- **APIs**: OpenAI or Azure OpenAI (required), search APIs (optional)
- **Memory**: 2GB+ RAM recommended for complex research

## ⚙️ Advanced Configuration

### Custom Search Engines

```python
researcher = TTDResearcher(
    client=client,
    search_engines=['tavily', 'duckduckgo'],  # Customize search engines
    search_results_per_gap=5,                 # More results per gap
    max_iterations=10                         # Longer research
)
```

### Custom Prompts

You can customize the research behavior by modifying prompts in `langgraph_ttd_dr/prompts.py`.

## 🚨 Troubleshooting

### Common Issues

1. **API Key Errors**
   ```bash
   ❌ Failed to create client: No API key found
   ```
   **Solution**: Check your `.env` file and ensure API keys are correctly set.

2. **Search Failures**
   ```bash
   ❌ All search engines failed
   ```
   **Solution**: Verify search API keys or rely on DuckDuckGo (no key required).

3. **Long Processing Times**
   - Reduce `max_iterations` or `max_sources`
   - Use faster models (e.g., `gpt-4o-mini` instead of `gpt-4o`)

### Debug Mode

Enable debug logging:
```bash
export DEBUG=true
python chatbot.py
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on:

- Code style and standards
- Testing requirements
- Pull request process
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Original Research**: Based on ["Deep Researcher with Test-Time Diffusion"](https://arxiv.org/abs/2507.16075) by Han et al. (2025)
- **LangGraph**: For the excellent workflow framework
- **OpenAI/Azure**: For powerful language models
- **Search Providers**: Tavily, DuckDuckGo, and Naver for search capabilities
- **Research Community**: For insights into diffusion-based approaches
- **OptILLM Project**: Referenced the [deep research plugin](https://github.com/codelion/optillm/blob/main/optillm/plugins/deep_research/research_engine.py) for research engine architecture insights

## 📞 Support

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Join community discussions in GitHub Discussions
- **Documentation**: See the `/docs` folder for detailed documentation

---

**🚀 Ready to conduct deep research? Start with `python chatbot.py` and explore the power of TTD-DR!**