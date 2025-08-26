# BMad Method Knowledge Base

## Overview

The Breakthrough Method of Agile AI-driven Development (BMad) is a framework that combines AI agents with Agile development methodologies. It organizes development through specialized agents, structured workflows, and reusable resources.

## Key Concepts

### Agents

Agents are specialized AI personas with defined roles, responsibilities, and capabilities:

- **Orchestrator**: Coordinates all other agents and workflows
- **Specialized Agents**: Focus on specific aspects of development (PM, Developer, Architect, etc.)
- **Agent Transformation**: Ability to switch between different agent roles as needed

### Workflows

Workflows define prescribed sequences of steps for specific project types:

- **Greenfield**: For new projects
- **Brownfield**: For existing projects
- **Stages**: Sequential steps performed by different agents
- **Artifacts**: Documents and outputs created at each stage

### Commands

BMad uses a command-based interface:

- **Prefix**: All commands start with * (asterisk)
- **Agent Commands**: *agent, *exit, *help, *status
- **Workflow Commands**: *workflow, *workflow-guidance, *workflow-status
- **Task Commands**: *task, *template, *checklist

## Using BMad

### Basic Usage

1. Start with the BMad Orchestrator
2. Use *help to see available commands
3. Transform into specialized agents using *agent
4. Start workflows with *workflow
5. Return to orchestrator with *exit

### Best Practices

- **Start with Orchestrator**: Always begin with the orchestrator agent
- **Follow Workflows**: Use structured workflows for complex tasks
- **Create Artifacts**: Save important outputs at each stage
- **Switch Agents**: Transform into specialized agents for different tasks
- **Use Help Frequently**: Check available commands with *help

## Customization

BMad can be customized for specific domains by:

- Creating specialized agents for domain-specific roles
- Defining custom workflows for domain processes
- Building domain-specific knowledge bases
- Creating reusable templates and tasks

## BMad in Different Environments

### Web Environment

- Uses pre-built bundles from `dist/teams`
- Single text files containing all agent dependencies
- Optimized for interaction via web interfaces

### IDE Environment

- Interacts directly with agent markdown files
- Supports real-time file operations
- Optimized for development workflow execution

## Common Commands

- **General**: *help, *status, *exit
- **Agent Management**: *agent, *chat-mode, *party-mode
- **Workflow Management**: *workflow, *workflow-guidance, *workflow-status
- **Document Management**: *doc-out, *template, *task

## Troubleshooting

- **Commands not recognized**: Ensure you're using * prefix
- **Agent not transforming**: Check if agent ID is correct
- **Workflow not starting**: Verify workflow ID exists
- **Context issues**: May need to reload key information