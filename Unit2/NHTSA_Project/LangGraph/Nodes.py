import sys
import os

# Prompts.py and LLM_Tools/ both live one level up in NHTSA_Project/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "LLM_Tools"))

from State import State
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from Prompts import Specify_Task_Type, Retrieve_Data_Prompt, build_schema_summary
from config import mercury_llm, gpt5_4_mini_llm, gpt5_1_llm
from LLM_Tools.NHTSA_Query_Tools import (
    get_rows_by_position,
    filter_rows,
    list_csv_files,
    get_csv_schema,
    filter_csv,
    get_csv_rows_by_position,
    Ask_User,
    User_Answer,
)
# ---------------------------------------------------------------------------
# Load environment variables from a .env file
# ---------------------------------------------------------------------------
load_dotenv()



def classify_task(state: State) -> dict:
    user_request = state["user_request"]

    # Build system + user prompts from Prompts.py
    prompts = Specify_Task_Type(user_request)

    # Construct a messages list so Call_LLM can accept tools and history later
    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user",   "content": prompts["user"]},
    ]

    # llm.invoke accepts a list of role/content dicts and returns an AIMessage;
    # .content extracts the text string from that message.
    content = gpt5_4_mini_llm.invoke(messages).content

    # Strip any stray whitespace; the prompt instructs the model to return only the label
    task_type = content.strip()

    return {"task_type": task_type}

def retrieve_data(state: State) -> dict:
    user_request = state["user_request"]

    # Build the schema string once per call and inject it into the system prompt.
    # This tells the LLM about all 49 parquet columns without requiring a tool call.
    schema_str = build_schema_summary()
    prompts = Retrieve_Data_Prompt(user_request, schema_str)

    # All 8 retrieval tools available in this node.
    # The tool map lets _dispatch_tool look up the callable by name from tool_call dicts.
    tool_list = [
        get_rows_by_position,
        filter_rows,
        list_csv_files,
        get_csv_schema,
        filter_csv,
        get_csv_rows_by_position,
        Ask_User,
        User_Answer,
    ]
    tool_map = {t.name: t for t in tool_list}
    llm = gpt5_1_llm.bind_tools(tool_list)

    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user",   "content": prompts["user"]},
    ]

    # Bounded agentic loop — allows the LLM to call tools for clarification and data
    # fetching, but prevents runaway execution with a hard iteration cap.
    MAX_ITERATIONS = 8
    for _ in range(MAX_ITERATIONS):
        response = llm.invoke(messages)
        messages.append(response)

        # No tool calls means the LLM has produced its final answer — exit the loop
        if not response.tool_calls:
            return {"query_result": response.content}

        # Execute each tool call and feed the results back into the message history
        for tool_call in response.tool_calls:
            tool_fn = tool_map.get(tool_call["name"])
            if tool_fn is None:
                # Unknown tool — return an error message as the tool result
                result = f"Error: unknown tool '{tool_call['name']}'"
            else:
                result = tool_fn.invoke(tool_call["args"])

            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

    # Iteration cap hit without a clean exit — return whatever the last message held
    last = messages[-1]
    fallback = getattr(last, "content", "Retrieval reached max iterations without a final answer.")
    return {"query_result": fallback}

def analyze(state: State) -> dict:
    user_request = state["user_request"]
    
    # imagine LLM call here that reads the request
    # Puts the result of the query, such as a content retrieval, and adds it to the state
    query_result = "query_result"  # placeholder for now
    
    return {"query_result": query_result}

def agentic_loop(state: State) -> dict:
    user_request = state["user_request"]
    
    # imagine LLM call here that reads the request
    # Puts the result of the query, such as a content retrieval, and adds it to the state
    query_result = "query_result"  # placeholder for now
    
    return {"query_result": query_result}


def fetch_from_database(state: State) -> dict:
    user_request = state["user_request"]
    
    # imagine LLM call here that reads the request
    # Puts the result of the query, such as a content retrieval, and adds it to the state
    query_result = "query_result"  # placeholder for now
    
    return {"query_result": query_result}