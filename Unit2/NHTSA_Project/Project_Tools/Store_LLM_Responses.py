import json
import re
import csv
import os


def _coerce_df_index(raw_value):
    if raw_value is None:
        return None

    text_value = str(raw_value).strip()
    if not text_value:
        return None

    try:
        return int(text_value)
    except ValueError:
        try:
            return int(float(text_value))
        except ValueError:
            return None


def ensure_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def get_output_csv_path(output_dir, prompt_func_name):
    return os.path.join(output_dir, f"{prompt_func_name}.csv")


def get_processed_df_indices(output_dir, prompt_func_name):
    """
    Read the df_index column from the exact output CSV associated with the
    current task and output location.
    """
    csv_filename = get_output_csv_path(output_dir, prompt_func_name)
    if not os.path.isfile(csv_filename):
        return set()

    processed_indices = set()

    with open(csv_filename, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'df_index' not in reader.fieldnames:
            return set()

        for row in reader:
            df_index = _coerce_df_index(row.get('df_index'))
            if df_index is not None:
                processed_indices.add(df_index)

    return processed_indices


def parse_llm_json(response_text):
    """
    Strips any markdown formatting (like ```json ... ```) and parses the string as JSON.
    Returns the parsed dictionary, or None if parsing fails.
    """
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL)
    json_str = match.group(1) if match else response_text.strip()
        
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def append_to_csv(df_index, prompt_func_name, parsed_json, output_dir):
    """
    Appends the parsed JSON to a CSV named after the prompt function.
    Explicitly tracks the ground-truth dataframe index (df_index) as its own column.
    Creates the file and header if it doesn't exist.
    """
    ensure_output_dir(output_dir)
    csv_filename = get_output_csv_path(output_dir, prompt_func_name)
    file_exists = os.path.isfile(csv_filename)
    
    if not isinstance(parsed_json, dict):
        print("Error: Parsed JSON is not a dictionary. Cannot save to CSV.")
        return
        
    # Ensure df_index is the first column
    fieldnames = ['df_index'] + list(parsed_json.keys())
    
    with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        row_data = {'df_index': df_index}
        row_data.update(parsed_json)
        
        try:
            writer.writerow(row_data)
        except ValueError as e:
            print(f"Warning: Field name mismatch when writing to CSV: {e}")
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writerow(row_data)

def process_and_store_response(call_llm_func, messages, tools, df_index, prompt_func_name, output_dir):
    """
    Calls the LLM, attempts to parse the JSON. If it fails, modifies the messages to retry.
    Saves to CSV on success, strictly linking the response to df_index.
    """
    max_retries = 3
    for attempt in range(max_retries):
        print(f"Calling LLM for dataframe index {df_index} (Attempt {attempt + 1})...")
        content, finish_reason, prompt_tokens, completion_tok, total_tokens = call_llm_func(messages, tools)
        
        parsed_json = parse_llm_json(content)
        
        if parsed_json is not None:
            print(f"Successfully parsed JSON for index {df_index}. Saving to CSV.")
            append_to_csv(df_index, prompt_func_name, parsed_json, output_dir)
            return parsed_json, content
            
        print(f"Failed to parse JSON for index {df_index}. Retrying...")
        
        # Append the failed output as assistant and ask again
        messages.append({
            "role": "assistant",
            "content": content
        })
        messages.append({
            "role": "user",
            "content": "Your previous response was not valid JSON. Please output ONLY valid JSON without any other text or markdown formatting."
        })
        
    print(f"Failed to get valid JSON for index {df_index} after {max_retries} attempts.")
    return None, None
