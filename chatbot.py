#!/usr/bin/env python3
"""
TTD-DR Simple Chatbot Example

This file demonstrates how to use the Test-Time Diffusion Deep Researcher (TTD-DR) 
system for conducting deep research through a simple chatbot interface.

Usage Examples:
    python chatbot.py
    
Requirements:
    - API keys configured in .env file
    - Required dependencies installed (pip install -r requirements.txt)
"""

import asyncio
import os
from typing import Optional
from dotenv import load_dotenv

from langgraph_ttd_dr.interface import TTDResearcher
from langgraph_ttd_dr.client_factory import create_openai_client, detect_config


class SimpleTTDChatbot:
    """Simple chatbot interface for TTD-DR research system"""
    
    def __init__(self):
        """Initialize the chatbot with TTD-DR researcher"""
        # Load environment variables
        load_dotenv()
        
        # Create OpenAI client
        try:
            self.client = create_openai_client()
            print("✅ Successfully created OpenAI client")
        except Exception as e:
            print(f"❌ Failed to create client: {e}")
            print("💡 Make sure to configure your API keys in .env file")
            self.client = None
            return
            
        # Initialize TTD Researcher
        try:
            self.researcher = TTDResearcher(
                client=self.client,
                max_iterations=5,     # Maximum research iterations
                max_sources=15,       # Maximum sources per research
                search_engines=['tavily', 'duckduckgo', 'naver'],
                search_results_per_gap=3
            )
            print("✅ TTD Researcher initialized successfully")
            print(f"🔍 Search engines: {self.researcher.search_engines}")
            print(f"🔄 Max iterations: {self.researcher.max_iterations}")
            print(f"📚 Max sources: {self.researcher.max_sources}")
        except Exception as e:
            print(f"❌ Failed to initialize researcher: {e}")
            self.researcher = None
    
    def display_welcome(self):
        """Display welcome message and instructions"""
        print("\n" + "="*60)
        print("🤖 TTD-DR Deep Research Chatbot")
        print("="*60)
        print("Welcome to the Test-Time Diffusion Deep Researcher!")
        print("This chatbot can conduct in-depth research on any topic.")
        print("\n📝 How it works:")
        print("  1. You provide a research question")
        print("  2. The system generates an initial draft")
        print("  3. It identifies knowledge gaps iteratively")
        print("  4. Searches for information to fill those gaps")
        print("  5. Refines the draft through multiple iterations")
        print("  6. Delivers a comprehensive research report")
        print("\n💡 Tips:")
        print("  • Ask complex, open-ended questions for best results")
        print("  • Be specific about what you want to know")
        print("  • The system works best with research-oriented queries")
        print("\n🛑 Commands:")
        print("  • Type 'quit' or 'exit' to stop")
        print("  • Type 'help' for this message")
        print("  • Type 'status' to check system status")
        print("="*60)
    
    def check_system_status(self):
        """Check and display system status"""
        print("\n🔍 System Status Check:")
        print("-" * 40)
        
        # Check API configuration
        config = detect_config()
        if config['has_azure']:
            print("✅ Azure OpenAI: Configured")
            print(f"   Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT', 'Not set')[:30]}...")
            print(f"   Model: {os.getenv('AZURE_OPENAI_DEPLOYMENT', 'Not set')}")
        elif config['has_openai']:
            print("✅ OpenAI API: Configured")
        else:
            print("❌ No API keys found")
            
        # Check search APIs
        tavily_key = os.getenv('TAVILY_API_KEY')
        naver_id = os.getenv('NAVER_CLIENT_ID')
        
        print(f"🔍 Tavily Search: {'✅ Available' if tavily_key else '❌ Not configured'}")
        print(f"🔍 Naver Search: {'✅ Available' if naver_id else '❌ Not configured'}")
        print(f"🔍 DuckDuckGo: ✅ Available (no API key needed)")
        
        # Check researcher status
        if self.researcher:
            print("🤖 TTD Researcher: ✅ Ready")
        else:
            print("🤖 TTD Researcher: ❌ Not available")
            
        print("-" * 40)
    
    async def conduct_research(self, query: str) -> Optional[str]:
        """Conduct research on the given query"""
        if not self.researcher:
            print("❌ Researcher not available. Check your configuration.")
            return None
            
        try:
            print(f"\n🔍 Research Query: {query}")
            print("⏳ Starting deep research... (this may take a few minutes)")
            print("📊 The system will show progress updates during research")
            print("-" * 60)
            
            # Conduct research
            report, metadata = await self.researcher.research(query)
            
            # Display results
            print("\n" + "="*60)
            print("📋 RESEARCH REPORT COMPLETED")
            print("="*60)
            print(f"📄 Report Length: {len(report)} characters")
            print(f"🔄 Iterations: {metadata.get('iterations', 'N/A')}")
            print(f"📚 Sources Used: {len(metadata.get('all_sources', []))}")
            print(f"⏱️  Execution Time: {metadata.get('execution_time', 'N/A'):.1f} seconds")
            print(f"🎯 Status: {metadata.get('status', 'N/A')}")
            print("-" * 60)
            print("\n📖 RESEARCH REPORT:")
            print("-" * 30)
            print(report)
            print("-" * 60)
            
            # Display metadata summary
            if 'draft_evolution' in metadata:
                print(f"📝 Draft Evolution: {len(metadata['draft_evolution'])} stages")
            if 'search_queries' in metadata:
                print(f"🔎 Search Queries Used: {len(metadata['search_queries'])}")
            if 'termination_reason' in metadata:
                print(f"🏁 Completion Reason: {metadata['termination_reason']}")
                
            return report
            
        except Exception as e:
            print(f"❌ Research failed: {e}")
            print("💡 Try a simpler query or check your API configuration")
            return None
    
    def display_examples(self):
        """Display example queries"""
        print("\n💡 Example Research Queries:")
        print("-" * 40)
        examples = [
            "What are the latest developments in artificial intelligence in 2024?",
            "How does climate change affect global food security?",
            "What are the key differences between quantum and classical computing?",
            "Explain the current state of renewable energy technology",
            "What are the ethical implications of genetic engineering?",
            "How do electric vehicles compare to traditional cars in 2024?",
            "What is the impact of social media on mental health?",
            "Describe recent advances in space exploration technology"
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")
        print("-" * 40)
    
    async def start_chat_loop(self):
        """Start the interactive chat loop"""
        if not self.researcher:
            print("❌ Cannot start chatbot - researcher not initialized")
            return
            
        self.display_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("\n🤔 Your research question: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using TTD-DR! Goodbye!")
                    break
                elif user_input.lower() in ['help', 'h']:
                    self.display_welcome()
                    continue
                elif user_input.lower() == 'status':
                    self.check_system_status()
                    continue
                elif user_input.lower() == 'examples':
                    self.display_examples()
                    continue
                elif not user_input:
                    print("❓ Please enter a research question or type 'help' for instructions")
                    continue
                
                # Conduct research
                await self.conduct_research(user_input)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Chat interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                print("💡 Try again or type 'quit' to exit")


async def main():
    """Main function to run the chatbot"""
    print("🚀 Initializing TTD-DR Simple Chatbot...")
    
    # Create and start chatbot
    chatbot = SimpleTTDChatbot()
    await chatbot.start_chat_loop()


def run_quick_example():
    """Run a quick example without interactive mode"""
    print("🚀 TTD-DR Quick Example")
    print("="*50)
    
    async def quick_research():
        chatbot = SimpleTTDChatbot()
        if not chatbot.researcher:
            print("❌ Cannot run example - check configuration")
            return
            
        # Example research question
        example_query = "What is artificial intelligence and how is it being used today?"
        print(f"📝 Example Query: {example_query}")
        
        report = await chatbot.conduct_research(example_query)
        if report:
            print("\n✅ Quick example completed successfully!")
        else:
            print("\n❌ Quick example failed")
    
    # Run the example
    asyncio.run(quick_research())


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        # Run quick example
        run_quick_example()
    else:
        # Run interactive chatbot
        asyncio.run(main()) 