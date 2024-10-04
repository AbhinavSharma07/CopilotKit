# pylint: disable=line-too-long, unused-import

import os
from typing import cast, TypedDict, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from copilotkit.langchain import copilotkit_customize_config

# Define environment variables for model selection
DEFAULT_MODEL = "openai"
OPENAI_MODEL = "gpt-4o"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"

def get_model() -> ChatOpenAI | ChatAnthropic:
    """
    Retrieve the appropriate model based on the MODEL environment variable.
    
    Returns:
        An instance of ChatOpenAI or ChatAnthropic.
    
    Raises:
        ValueError: If an invalid model is specified.
    """
    model_choice = os.getenv("MODEL", DEFAULT_MODEL).lower()

    if model_choice == "openai":
        return ChatOpenAI(temperature=0, model=OPENAI_MODEL)
    elif model_choice == "anthropic":
        return ChatAnthropic(
            temperature=0,
            model_name=ANTHROPIC_MODEL,
            timeout=None,
            stop=None
        )
    
    raise ValueError(f"Invalid model specified: {model_choice}")

class Translations(TypedDict):
    """Typed dictionary for storing translations in multiple languages."""
    translation_es: str
    translation_fr: str
    translation_de: str

class AgentState(MessagesState):
    """State of the agent, including translations and user input."""
    translations: Optional[Translations]
    input: str

async def translate_node(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Node responsible for translating text into Spanish, French, and German.
    
    Args:
        state (AgentState): Current state of the agent.
        config (RunnableConfig): Configuration for the runnable.
    
    Returns:
        Updated AgentState with translation results.
    """
    # Customize the configuration with CopilotKit
    customized_config = copilotkit_customize_config(
        config,
        emit_messages=True,
        emit_intermediate_state=[
            {
                "state_key": "translations",
                "tool": "Translations"
            }
        ]
    )

    # Bind the Translations tool to the model
    model = get_model().bind_tools(
        tools=[Translations],
        tool_choice=(
            None if state.messages and isinstance(state.messages[-1], HumanMessage)
            else "Translations"
        )
    )

    # Prepare the translation prompt
    translation_prompt = f"""
    You are a helpful assistant that translates text to different languages 
    (Spanish, French, and German).
    Don't ask for confirmation before translating.
    {f'The user is currently working on translating this text: "{state.input}"' if state.get("input") else ""}
    """

    new_message = HumanMessage(content=translation_prompt)

    # Filter out system messages and empty AI messages
    filtered_messages = [
        message for message in state.messages
        if not isinstance(message, SystemMessage) and 
           not (isinstance(message, AIMessage) and not message.content.strip())
    ]

    # Append the new translation request
    filtered_messages.append(new_message)

    # Debugging: Print messages and their types
    for message in filtered_messages:
        print(message)
        print(type(message))
        print("---")

    # Invoke the model with the updated messages and configuration
    response = await model.ainvoke(filtered_messages, customized_config)

    # Process the response and update the state accordingly
    if hasattr(response, "tool_calls") and response.tool_calls:
        ai_message = cast(AIMessage, response)
        translations = cast(Translations, ai_message.tool_calls[0].get("args", {}))
        
        # Update current step with translation results
        state.translations = translations
        state.input = ""  # Clear input after translation
        
        # Append tool confirmation message
        tool_message = ToolMessage(
            content="Translated!",
            tool_call_id=ai_message.tool_calls[0]["id"]
        )
        updated_messages = [new_message, response, tool_message]
    else:
        # If no tool calls, simply append the response
        updated_messages = [new_message, response]

    # Update the state with new messages
    state.messages = updated_messages

    return state

def create_workflow() -> StateGraph:
    """
    Create and compile the workflow graph for the agent.
    
    Returns:
        Compiled StateGraph ready for execution.
    """
    # Initialize the state graph with AgentState
    workflow = StateGraph(AgentState)

    # Add the translation node to the workflow
    workflow.add_node("translate_node", translate_node)

    # Set the entry point of the workflow
    workflow.set_entry_point("translate_node")

    # Define the edge from translate_node to END
    workflow.add_edge("translate_node", END)

    # Initialize memory saver for checkpoints
    memory = MemorySaver()

    # Compile the workflow with the memory checkpointer
    compiled_graph = workflow.compile(checkpointer=memory)

    return compiled_graph

# Entry point: Create and compile the workflow graph
graph = create_workflow()
