import json
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from ai_researcher.state import AgentState
from ai_researcher.model import get_model

async def extract_node(state: AgentState, config: RunnableConfig):
    """
    Extracts and summarizes information from a search step in the given state.
    """
    # Retrieve the next pending search step
    current_step = next((step for step in state["steps"] if step["status"] == "pending"), None)
    
    if not current_step:
        raise ValueError("No pending search step found")
    
    if current_step.get("type") != "search":
        raise ValueError("The current step is not a search step")

    # Generate system message for summarizing the search results
    system_message = f"""
This step was just executed: {json.dumps(current_step)}

This is the result of the search:

Please summarize ONLY the result of the search and include all relevant information from the search and reference links.
DO NOT INCLUDE ANY EXTRA INFORMATION. ALL OF THE INFORMATION YOU ARE LOOKING FOR IS IN THE SEARCH RESULTS.

DO NOT answer the user's query yet. Just summarize the search results.

Use markdown formatting and put the references inline and the links at the end.
Like this:
This is a sentence with a reference to a source [source 1][1] and another reference [source 2][2].
[1]: http://example.com/source1 "Title of Source 1"
[2]: http://example.com/source2 "Title of Source 2"
"""

    # Invoke model to summarize the search results
    response = await get_model().ainvoke([
        state["messages"][0],
        HumanMessage(content=system_message)
    ], config)

    # Update the current step with the search result
    current_step.update({
        "result": response.content,
        "search_result": None,
        "status": "complete",
        "updates": current_step.get("updates", []) + ["Done."]
    })

    # Mark next step as "searching" if available
    next_step = next((step for step in state["steps"] if step["status"] == "pending"), None)
    if next_step:
        next_step["updates"] = ["Searching the web..."]

    return state
