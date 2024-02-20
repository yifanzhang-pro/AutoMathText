mport json
import os
import numpy as np
import random
from vllm import LLM, SamplingParams
import gc
import ray

# Free up memory
# ray.shutdown()
gc.collect()

# Constants
MODEL_NAME_OR_PATH = "Qwen/Qwen-72B"
AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
YES_ID = 14080
Yes_ID = 7414
NO_ID = 5664
No_ID = 2308
BATCH_SIZE = 10000
MAX_TEXT_LENGTH = 4096
OUTPUT_DIR = '<OUTPUT_DIR>'
INPUT_DIRECTORY = '<INPUT_DIRECTORY>'
TENSOR_PARALLEL_SIZE = 8

# Initialize LLM
# try to use all available GPUs
try:
    llm = LLM(model=MODEL_NAME_OR_PATH, tensor_parallel_size=TENSOR_PARALLEL_SIZE, trust_remote_code=True)
except Exception as e:
    print(f"Error initializing LLM: {e}")
    print("Trying to use only half GPU")
    ray.shutdown()
    llm = LLM(model=MODEL_NAME_OR_PATH, tensor_parallel_size=int(TENSOR_PARALLEL_SIZE / 2), trust_remote_code=True)

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_jsonl_files(directory, file_list=None):
    """Loads JSON lines from specified files in a directory."""
    data = []
    # If no specific file list is provided, use all .jsonl files in the directory
    if file_list is None:
        file_list = [f for f in os.listdir(directory) if f.endswith('.jsonl')]

    # Iterate through files in the file_list
    for filename in file_list:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r'') as file:
                for line in file:
                    # Parse each line as a JSON object
                    json_obj = json.loads(line)
                    data.append(json_obj)
        else:
            print(f"File not found: {file_path}")
    return data

system_prompt = """<system>
You are ChatGPT, the most capable large language model equipped with extensive expertise in mathematics and coding, particularly skilled in complex reasoning and problem-solving. In the following interaction, I will provide you with a text excerpt from the arXiv website. Your task is to evaluate whether this text contains elements of mathematical intelligence and if it is suitable for educational purposes for YOURSELF in the field of mathematics. Please respond with only YES or NO
<\system>
"""

def construct_prompts(data, system_prompt):
    """Constructs prompts for LLM based on input data and system prompt."""
    prompts = []
    for item in data:
        text = item.get('text', '[No Text]')
        meta = item.get('meta', {})
        title = meta.get('title', '[Not Available]')
        abstract = meta.get('abstract', '[Not Available]')
        subjects = meta.get('subjects', '[Not Available]')
        if len(text) > MAX_TEXT_LENGTH:
            text = text[:MAX_TEXT_LENGTH] + '...'
        prompt = system_prompt + "\nUser: {" + f'''
    "Title": "{title}",
    "Abstract": "{abstract}",
    "Text": "{text}"
''' + '''}
1. Does the text contain elements of mathematical intelligence? Reply with only YES or NO
2. Is the text suitable for educational purposes for YOURSELF in the field of mathematics? Reply with only YES or NO
Assistant: 1.'''
        prompts.append(prompt)
    return prompts

def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed set as {seed}")

def save_batch(data, original_filename, start_index, end_index):
    """Saves a batch of processed data as a JSONL file."""
    base_name = os.path.splitext(original_filename)[0]
    if not os.path.exists(f'{OUTPUT_DIR}/{base_name}'):
        os.makedirs(f'{OUTPUT_DIR}/{base_name}')

    file_name = f'{OUTPUT_DIR}/{base_name}/{base_name}-{start_index}_{end_index}.jsonl'
    with open(file_name, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')



def process_batches(original_filename, prompts, llm, sampling_params, original_data):
    for batch_num in range(0, len(prompts), BATCH_SIZE):
        base_name = os.path.splitext(original_filename)[0]
        start_index = batch_num
        upper_bound = min(batch_num + BATCH_SIZE, len(prompts))
        end_index = upper_bound - 1
        if os.path.exists(f'{OUTPUT_DIR}/{base_name}'):
        # If the line count is the same, skip it
            current_file_name = f'{OUTPUT_DIR}/{base_name}/{base_name}-{start_index}_{end_index}.jsonl'
            if len(load_jsonl(current_file_name)) == BATCH_SIZE:
                print(f"Current Batch File already exists: {current_file_name}")
                continue
        # if batch_num >= 1:
            # return
        # Calculate the upper bound for the current batch
        upper_bound = min(batch_num + BATCH_SIZE, len(prompts))
        batch_prompts = prompts[batch_num:upper_bound]
        outputs = llm.generate(batch_prompts, sampling_params)

        # Append generated text to original data
        for i, output in enumerate(outputs):
            # Calculate the index in the original data
            original_index = batch_num + i
            original_data[original_index]['meta']['lm_name'] = MODEL_NAME_OR_PATH
            original_data[original_index]['meta']['lm_label'] = "1." + output.outputs[0].text
            original_data[original_index]['meta']['logprobs'] = output.outputs[0].logprobs
            if len(output.outputs[0].logprobs) < 5:
                print(f"Error: logprobs is malfunctioning for {original_filename} line {original_index}")
                print(f"URL: {original_data[original_index]['meta'].get('url', '')}")
                print(f"lm_label: {output.outputs[0].text}")
                original_data[original_index]['meta']['q1_logprob_yes'] = -100
                original_data[original_index]['meta']['q1_logprob_no'] = -100
                original_data[original_index]['meta']['q1_relprob_yes'] = 0.5
                original_data[original_index]['meta']['q2_logprob_yes'] = -100
                original_data[original_index]['meta']['q2_logprob_no'] = -100
                original_data[original_index]['meta']['q2_relprob_yes'] = 0.5
                original_data[original_index]['meta']['q1q2_relprob_yes'] = 0.25
                continue
            q1_logprob_yes = output.outputs[0].logprobs[0].get(YES_ID, -100)
            if q1_logprob_yes == -100:
                try: 
                    q1_logprob_yes = output.outputs[0].logprobs[0].get(Yes_ID, -100)
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"URL: {original_data[original_index]['meta'].get('url', '')}")
                    print(f"lm_label: {output.outputs[0].text}")
                    q1_logprob_yes = -100
            q1_logprob_no = output.outputs[0].logprobs[0].get(NO_ID, -100)
            if q1_logprob_no == -100:
                try:
                    q1_logprob_no = output.outputs[0].logprobs[0].get(No_ID, -100)
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"URL: {original_data[original_index]['meta'].get('url', '')}")
                    print(f"lm_label: {output.outputs[0].text}")
                    q1_logprob_no = -100
            q1_relprob_yes = np.exp(q1_logprob_yes) / (np.exp(q1_logprob_yes) + np.exp(q1_logprob_no))
            original_data[original_index]['meta']['q1_logprob_yes'] = q1_logprob_yes
            original_data[original_index]['meta']['q1_logprob_no'] = q1_logprob_no
            original_data[original_index]['meta']['q1_relprob_yes'] = q1_relprob_yes
            q2_idx = 4
            # If the corresponding key of logprobs[4]'s max value is not YES_ID, Yes_ID, NO_ID, or No_ID, use logprobs[5] instead
            if max(output.outputs[0].logprobs[q2_idx], key=output.outputs[0].logprobs[q2_idx].get) not in [YES_ID, Yes_ID, NO_ID, No_ID]:
                q2_idx = min(5, len(output.outputs[0].logprobs) - 1)
            q2_logprob_yes = output.outputs[0].logprobs[q2_idx].get(YES_ID, -100)
            if q2_logprob_yes == -100:
                try: 
                    q2_logprob_yes = output.outputs[0].logprobs[0].get(Yes_ID, -100)
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"URL: {original_data[original_index]['meta']['url']}")
                    print(f"lm_label: {output.outputs[0].text}")
                    q1_logprob_yes = -100
            q2_logprob_no = output.outputs[0].logprobs[0].get(NO_ID, -100)
            if q2_logprob_no == -100:
                try:
                    q2_logprob_no = output.outputs[0].logprobs[0].get(No_ID, -100)
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"URL: {original_data[original_index]['meta']['url']}")
                    print(f"lm_label: {output.outputs[0].text}")
                    q2_logprob_no = -100
            q2_relprob_yes = np.exp(q2_logprob_yes) / (np.exp(q2_logprob_yes) + np.exp(q2_logprob_no))
            original_data[original_index]['meta']['q2_logprob_yes'] = q2_logprob_yes
            original_data[original_index]['meta']['q2_logprob_no'] = q2_logprob_no
            original_data[original_index]['meta']['q2_relprob_yes'] = q2_relprob_yes
            original_data[original_index]['meta']['q1q2_relprob_yes'] = q1_relprob_yes * q2_relprob_yes
            
            # print(f"Prompt: {output.prompt}")
            # print(f"Generated text for {original_filename} line {original_index}: {output.outputs[0].text}")
            # print(f"Q1 relative probability of YES: {q1_relprob_yes}")
            # print(f"Q2 relative probability of YES: {q2_relprob_yes}")

        # Save this batch of data with updated indices
        start_index = batch_num
        end_index = upper_bound - 1
        batch_data = original_data[start_index:end_index + 1]
        save_batch(batch_data, original_filename, start_index, end_index)
        print(f"Batch saved: [{start_index}:{end_index}] for {original_filename}.")


def extract_batch_start(file_name):
    try:
        # Extracts the batch start number from the file name
        return int(file_name.rsplit('-', 1)[1].split('_')[0])
    except (IndexError, ValueError):
        print(f"Error parsing file name: {file_name}")
        return 0

def merge_jsonl_files_in_subdirectory(base_name):
    subdirectory = f'{OUTPUT_DIR}/{base_name}'
    merged_data = []

    # Check if the subdirectory exists
    if not os.path.exists(subdirectory):
        print(f"Subdirectory does not exist: {subdirectory}")
        return

    # List and sort all .jsonl files in the subdirectory
    files = [f for f in os.listdir(subdirectory) if f.endswith('.jsonl')]
    files.sort(key=extract_batch_start)

    # Read and merge data from each file
    for file in files:
        file_path = os.path.join(subdirectory, file)
        with open(file_path, 'r') as f:
            for line in f:
                json_obj = json.loads(line)
                merged_data.append(json_obj)

    # Sort merged data by the product of "q1_relprob_yes" and "q2_relprob_yes"
    merged_data.sort(key=lambda x: x.get('meta', {}).get('q1_relprob_yes', 0) * x.get('meta', {}).get('q2_relprob_yes', 0), reverse=True)

    # Write sorted merged data to a new .jsonl file
    merged_file_path = os.path.join(OUTPUT_DIR, f'{base_name}.jsonl')
    with open(merged_file_path, 'w') as f:
        for item in merged_data:
            json.dump(item, f)
            f.write('\n')

    print(f"Merged file created at: {merged_file_path}")

def load_jsonl(file_path):
    """Load data from a JSONL file."""
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                json_obj = json.loads(line)
                data.append(json_obj)
    else:
        print(f"File not found: {file_path}")
    return data

# Set random seed
set_seed(0)

# Define sampling parameters
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=6, stop=['<|endoftext|>', 'User:', 'Human:', 'Assistant:', '<system>'], logprobs=5)

# Predefined file list, can be modified or left empty (None) to process all .jsonl files
file_list = [f for f in os.listdir(INPUT_DIRECTORY) if f.endswith('.jsonl')]

file_list.sort()
# Sort the file list by reverse alphabetical order
file_list.sort(reverse=True)

for i in range(len(file_list)):
    if not file_list[i].endswith('.jsonl'):
        file_list[i] += '.jsonl'

# Load, construct prompts, and process for each file
for file_name in file_list:
    print(f"Processing file: {file_name}")

    # If the output file already exists
    if os.path.exists(f'{OUTPUT_DIR}/{file_name}'):
        # If the line count is the same, skip it
        if len(load_jsonl(f'{OUTPUT_DIR}/{file_name}')) == len(load_jsonl_files(INPUT_DIRECTORY, [file_name])):
            print(f"File already exists: {file_name}")
            continue
    original_data = load_jsonl_files(INPUT_DIRECTORY, [file_name])
    prompts = construct_prompts(original_data, system_prompt=system_prompt)
    process_batches(file_name, prompts, llm, sampling_params, original_data)
    merge_jsonl_files_in_subdirectory(file_name.split('.jsonl')[0])
    gc.collect()
