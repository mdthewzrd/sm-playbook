#!/usr/bin/env python3
"""
SM Playbook BMad Interface
Main entry point for the BMAT trading system

This script provides the command-line interface for interacting with
the BMad agent system and trading functionality.
"""

import sys
import os
import argparse
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading-logs/bmad_interface.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class BMadInterface:
    """
    Main interface for the BMad trading system.
    
    Handles agent activation, command processing, and system coordination.
    """
    
    def __init__(self):
        """Initialize the BMad interface."""
        self.current_agent = None
        self.current_workflow = None
        self.session_active = False
        self.command_history = []
        
        # Load agent configurations
        self.bmad_core_path = project_root / '.bmad-core'
        self.agents_path = self.bmad_core_path / 'agents'
        self.workflows_path = self.bmad_core_path / 'workflows'
        self.data_path = self.bmad_core_path / 'data'
        
        self.available_agents = self._load_available_agents()
        self.available_workflows = self._load_available_workflows()
        
        logger.info("BMad interface initialized")
    
    def _load_available_agents(self) -> Dict[str, str]:
        """Load available agent configurations."""
        agents = {}
        
        if not self.agents_path.exists():
            logger.warning(f"Agents path not found: {self.agents_path}")
            return agents
        
        for agent_file in self.agents_path.glob('*.md'):
            agent_name = agent_file.stem
            agents[agent_name] = str(agent_file)
        
        logger.info(f"Loaded {len(agents)} agents: {list(agents.keys())}")
        return agents
    
    def _load_available_workflows(self) -> Dict[str, str]:
        """Load available workflow configurations."""
        workflows = {}
        
        if not self.workflows_path.exists():
            logger.warning(f"Workflows path not found: {self.workflows_path}")
            return workflows
        
        for workflow_file in self.workflows_path.glob('*.yaml'):
            workflow_name = workflow_file.stem
            workflows[workflow_name] = str(workflow_file)
        
        logger.info(f"Loaded {len(workflows)} workflows: {list(workflows.keys())}")
        return workflows
    
    def start_session(self):
        """Start a BMad session."""
        if self.session_active:
            print("BMad session is already active.")
            return
        
        self.session_active = True
        print("="*60)
        print("    🎭 SM PLAYBOOK - BMAD TRADING SYSTEM")
        print("="*60)
        print("Welcome to the Breakthrough Method of Agile AI-driven Development")
        print("for Trading Systems (BMAT)")
        print("")
        print("Available commands:")
        print("  *help          - Show available commands")
        print("  *agent [name]  - Activate trading agent")
        print("  *workflow [name] - Start trading workflow") 
        print("  *status        - Show system status")
        print("  *exit          - Exit BMad system")
        print("")
        print("Type *help for detailed command list")
        print("All commands must start with * (asterisk)")
        print("="*60)
        
        # Start command loop
        self._command_loop()
    
    def _command_loop(self):
        """Main command processing loop."""
        while self.session_active:
            try:
                # Show prompt based on current agent
                if self.current_agent:
                    prompt = f"BMad[{self.current_agent}]> "
                else:
                    prompt = "BMad> "
                
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                
                # Add to command history
                self.command_history.append({
                    'timestamp': datetime.now(),
                    'command': user_input,
                    'agent': self.current_agent
                })
                
                # Process command
                if user_input.startswith('*'):
                    self._process_command(user_input[1:])  # Remove asterisk
                else:
                    print("Commands must start with *. Type *help for available commands.")
                
            except KeyboardInterrupt:
                print("\nUse *exit to quit BMad system")
            except EOFError:
                break
            except Exception as e:
                logger.error(f"Error in command loop: {e}")
                print(f"Error: {e}")
    
    def _process_command(self, command: str):
        """Process a BMad command."""
        parts = command.split()
        if not parts:
            return
        
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        try:
            if cmd == 'help':
                self._show_help()
            elif cmd == 'agent':
                self._handle_agent_command(args)
            elif cmd == 'workflow':
                self._handle_workflow_command(args)
            elif cmd == 'status':
                self._show_status()
            elif cmd == 'exit':
                self._exit_session()
            elif cmd == 'kb-mode':
                self._show_knowledge_base()
            elif cmd == 'chat-mode':
                self._start_chat_mode()
            else:
                # Check if current agent can handle the command
                if self.current_agent:
                    self._delegate_to_agent(command, args)
                else:
                    print(f"Unknown command: *{cmd}")
                    print("Type *help for available commands")
                    
        except Exception as e:
            logger.error(f"Error processing command '{command}': {e}")
            print(f"Error executing command: {e}")
    
    def _show_help(self):
        """Show detailed help information."""
        print("\n" + "="*60)
        print("           BMAD TRADING SYSTEM HELP")
        print("="*60)
        print("\nCORE COMMANDS:")
        print("  *help .................. Show this help message")
        print("  *status ................ Show system status")
        print("  *agent [name] .......... Activate trading agent")
        print("  *workflow [name] ....... Start trading workflow")
        print("  *kb-mode ............... Load knowledge base")
        print("  *chat-mode ............. Start conversational mode")
        print("  *exit .................. Exit BMad system")
        print("\nAVAILABLE AGENTS:")
        for agent_name in self.available_agents:
            description = self._get_agent_description(agent_name)
            print(f"  *agent {agent_name:<20} {description}")
        
        print("\nAVAILABLE WORKFLOWS:")
        for workflow_name in self.available_workflows:
            description = self._get_workflow_description(workflow_name)
            print(f"  *workflow {workflow_name:<15} {description}")
        
        print("\nUSAGE EXAMPLES:")
        print("  *agent trading-orchestrator")
        print("  *workflow strategy-development")
        print("  *status")
        print("\nNOTE: All commands must start with * (asterisk)")
        print("="*60)
    
    def _handle_agent_command(self, args: List[str]):
        """Handle agent activation command."""
        if not args:
            # Show available agents
            print("\nAvailable agents:")
            for i, (agent_name, _) in enumerate(self.available_agents.items(), 1):
                description = self._get_agent_description(agent_name)
                print(f"  {i}. {agent_name} - {description}")
            print("\nUsage: *agent [agent_name]")
            return
        
        agent_name = args[0]
        
        if agent_name not in self.available_agents:
            print(f"Agent '{agent_name}' not found.")
            print("Available agents:", list(self.available_agents.keys()))
            return
        
        # Activate agent
        self._activate_agent(agent_name)
    
    def _handle_workflow_command(self, args: List[str]):
        """Handle workflow command."""
        if not args:
            # Show available workflows
            print("\nAvailable workflows:")
            for i, (workflow_name, _) in enumerate(self.available_workflows.items(), 1):
                description = self._get_workflow_description(workflow_name)
                print(f"  {i}. {workflow_name} - {description}")
            print("\nUsage: *workflow [workflow_name]")
            return
        
        workflow_name = args[0]
        
        if workflow_name not in self.available_workflows:
            print(f"Workflow '{workflow_name}' not found.")
            print("Available workflows:", list(self.available_workflows.keys()))
            return
        
        # Start workflow
        self._start_workflow(workflow_name)
    
    def _activate_agent(self, agent_name: str):
        """Activate a specific agent."""
        try:
            agent_file = self.available_agents[agent_name]
            
            # Load agent configuration
            with open(agent_file, 'r') as f:
                agent_content = f.read()
            
            # Parse agent configuration (simplified)
            self.current_agent = agent_name
            
            print(f"\n🎭 Activated agent: {agent_name}")
            print(f"📋 Loading agent configuration...")
            
            # Show agent-specific help
            self._show_agent_help(agent_name, agent_content)
            
            logger.info(f"Activated agent: {agent_name}")
            
        except Exception as e:
            logger.error(f"Error activating agent {agent_name}: {e}")
            print(f"Error activating agent: {e}")
    
    def _show_agent_help(self, agent_name: str, agent_content: str):
        """Show help for a specific agent."""
        print(f"\n🤖 Agent: {agent_name.upper()}")
        print("-" * 40)
        
        # Extract help template from agent content (simplified)
        if 'help-display-template:' in agent_content:
            help_start = agent_content.find('help-display-template:')
            help_section = agent_content[help_start:help_start+2000]  # Get reasonable chunk
            
            # Find the template content between | markers
            lines = help_section.split('\n')
            in_template = False
            for line in lines:
                if line.strip().startswith('|'):
                    in_template = True
                    print(line[line.find('|')+1:].strip())
                elif in_template and line.strip() and not line.startswith(' '):
                    break
                elif in_template:
                    print(line.strip())
        else:
            print("Agent-specific commands will be available after activation.")
        
        print(f"\n📝 Type *exit to return to main BMad interface")
        print("-" * 40)
    
    def _start_workflow(self, workflow_name: str):
        """Start a trading workflow."""
        try:
            workflow_file = self.available_workflows[workflow_name]
            
            print(f"\n🔄 Starting workflow: {workflow_name}")
            print(f"📋 Loading workflow configuration...")
            
            # Load workflow configuration
            import yaml
            with open(workflow_file, 'r') as f:
                workflow_config = yaml.safe_load(f)
            
            self.current_workflow = workflow_name
            
            # Show workflow stages
            if 'stages' in workflow_config:
                print(f"\n📋 Workflow stages for {workflow_name}:")
                for i, stage in enumerate(workflow_config['stages'], 1):
                    print(f"  {i}. {stage.get('name', 'Unknown')} ({stage.get('agent', 'Unknown')})")
                    print(f"     {stage.get('description', '')}")
            
            print(f"\n🚀 Workflow '{workflow_name}' is ready to execute")
            print("Use workflow-specific commands to proceed")
            
            logger.info(f"Started workflow: {workflow_name}")
            
        except Exception as e:
            logger.error(f"Error starting workflow {workflow_name}: {e}")
            print(f"Error starting workflow: {e}")
    
    def _show_status(self):
        """Show current system status."""
        print("\n" + "="*50)
        print("         BMAD SYSTEM STATUS")
        print("="*50)
        print(f"Session Active:    {'Yes' if self.session_active else 'No'}")
        print(f"Current Agent:     {self.current_agent or 'None'}")
        print(f"Current Workflow:  {self.current_workflow or 'None'}")
        print(f"Commands Executed: {len(self.command_history)}")
        print(f"Available Agents:  {len(self.available_agents)}")
        print(f"Available Workflows: {len(self.available_workflows)}")
        
        # Show recent command history
        if self.command_history:
            print(f"\nRecent Commands:")
            for cmd in self.command_history[-5:]:
                timestamp = cmd['timestamp'].strftime('%H:%M:%S')
                print(f"  [{timestamp}] {cmd['command']}")
        
        print("="*50)
    
    def _show_knowledge_base(self):
        """Show knowledge base information."""
        print("\n📚 Loading BMad Knowledge Base...")
        
        kb_files = [
            'trading-kb.md',
            'bmad-kb.md',
            'playbook-database.md',
            'risk-parameters.md'
        ]
        
        for kb_file in kb_files:
            kb_path = self.data_path / kb_file
            if kb_path.exists():
                print(f"✅ {kb_file} - Available")
            else:
                print(f"❌ {kb_file} - Not found")
        
        print("\nKnowledge base loaded. Agent responses will include relevant information.")
    
    def _start_chat_mode(self):
        """Start conversational chat mode."""
        print("\n💬 Entering chat mode...")
        print("You can now have a natural conversation with the BMad system.")
        print("Type 'exit chat' to return to command mode.")
        print("-" * 40)
        
        while True:
            user_input = input("Chat> ").strip()
            
            if user_input.lower() in ['exit chat', 'quit', 'exit']:
                print("Exiting chat mode...")
                break
            
            if not user_input:
                continue
            
            # Process natural language input
            self._process_chat_input(user_input)
    
    def _process_chat_input(self, user_input: str):
        """Process natural language chat input."""
        # Simplified natural language processing
        lower_input = user_input.lower()
        
        if 'help' in lower_input:
            self._show_help()
        elif 'status' in lower_input:
            self._show_status()
        elif 'agent' in lower_input:
            # Try to extract agent name
            for agent_name in self.available_agents:
                if agent_name in lower_input:
                    self._activate_agent(agent_name)
                    return
            print("Which agent would you like to activate?")
            print("Available:", list(self.available_agents.keys()))
        elif 'workflow' in lower_input:
            # Try to extract workflow name
            for workflow_name in self.available_workflows:
                if workflow_name in lower_input:
                    self._start_workflow(workflow_name)
                    return
            print("Which workflow would you like to start?")
            print("Available:", list(self.available_workflows.keys()))
        else:
            print("I understand you want to:", user_input)
            print("Try being more specific, or use *help for command list")
    
    def _delegate_to_agent(self, command: str, args: List[str]):
        """Delegate command to current active agent."""
        print(f"🤖 {self.current_agent} processing: *{command}")
        
        # This would integrate with actual agent logic
        # For now, provide a placeholder response
        if command == 'playbook-review':
            self._handle_playbook_review()
        elif command == 'signal-generate':
            self._handle_signal_generation()
        elif command == 'workflow-start':
            if args:
                self._start_workflow(args[0])
        else:
            print(f"Command '{command}' forwarded to {self.current_agent} agent")
            print("(Agent-specific logic would be implemented here)")
    
    def _handle_playbook_review(self):
        """Handle playbook review command."""
        print("\n📊 Trading Playbook Review")
        print("-" * 30)
        print("Loading current playbook entries...")
        
        # Check if playbook data exists
        playbook_path = self.data_path / 'playbook-database.md'
        if playbook_path.exists():
            print("✅ Playbook database found")
            print("\nCurrent strategies:")
            print("  1. Mean Reversion Template")
            print("  2. Trend Following Template") 
            print("  3. Breakout Template")
            print("\nUse specific playbook commands for detailed review")
        else:
            print("❌ Playbook database not found")
    
    def _handle_signal_generation(self):
        """Handle signal generation command."""
        print("\n📈 Signal Generation")
        print("-" * 20)
        print("Generating trading signals based on current market conditions...")
        print("(This would connect to live market data and apply strategy rules)")
        print("\nSignal generation requires:")
        print("  - Market data connection")
        print("  - Active strategy configurations")
        print("  - Risk parameter validation")
    
    def _get_agent_description(self, agent_name: str) -> str:
        """Get description for an agent."""
        descriptions = {
            'bmad-orchestrator': 'Master coordinator for all BMad operations',
            'trading-orchestrator': 'Trading system coordinator and playbook manager',
            'strategy-designer': 'Trading strategy development and formalization',
            'backtesting-engineer': 'Strategy validation and performance testing',
            'execution-engineer': 'Trade execution and risk management',
            'indicator-developer': 'Technical indicator development and testing'
        }
        return descriptions.get(agent_name, 'BMad trading agent')
    
    def _get_workflow_description(self, workflow_name: str) -> str:
        """Get description for a workflow."""
        descriptions = {
            'strategy-development': 'Complete strategy development lifecycle',
            'signal-generation': 'Generate and validate trading signals',
            'market-analysis': 'Analyze current market conditions',
            'performance-review': 'Review and optimize strategy performance'
        }
        return descriptions.get(workflow_name, 'BMad trading workflow')
    
    def _exit_session(self):
        """Exit the BMad session."""
        print("\n👋 Exiting BMad system...")
        
        if self.current_agent:
            print(f"Deactivating agent: {self.current_agent}")
            self.current_agent = None
        
        if self.current_workflow:
            print(f"Stopping workflow: {self.current_workflow}")
            self.current_workflow = None
        
        print("Session summary:")
        print(f"  Commands executed: {len(self.command_history)}")
        print(f"  Session duration: Active")
        
        self.session_active = False
        print("BMad system shutdown complete.")


def main():
    """Main entry point for the BMad interface."""
    parser = argparse.ArgumentParser(
        description='SM Playbook BMad Trading System Interface'
    )
    parser.add_argument(
        '--mode',
        choices=['interactive', 'command'],
        default='interactive',
        help='Interface mode (default: interactive)'
    )
    parser.add_argument(
        '--agent',
        help='Directly activate specific agent'
    )
    parser.add_argument(
        '--workflow',
        help='Directly start specific workflow'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and start interface
    interface = BMadInterface()
    
    if args.mode == 'interactive':
        interface.start_session()
    elif args.agent:
        interface.session_active = True
        interface._activate_agent(args.agent)
    elif args.workflow:
        interface.session_active = True
        interface._start_workflow(args.workflow)
    
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nBMad interface interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error in BMad interface: {e}")
        sys.exit(1)