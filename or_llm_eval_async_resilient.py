import os
import re
import subprocess
import sys
import tempfile
import json
import asyncio
import argparse
from itertools import zip_longest
from dotenv import load_dotenv

# Local utilities (unchanged)
from utils import (
    is_number_string,
    convert_to_number,
    extract_best_objective,
    eval_model_result
)

# Try to import Google's generative AI client. If not available, provide a helpful error message.
try:
    import google.generativeai as genai  # pip install google-generative-ai
except Exception as e:
    genai = None

# Prompt constants (translated to English)
MATH_MODEL_SYSTEM_PROMPT = (
    "You are an operations research (OR) expert. Based on the user's optimization problem, "
    "construct a mathematical model using a mathematical (linear programming) formulation. "
    "Focus on producing a correct mathematical model expression; detailed explanations are not required. "
    "This model will be used later to guide the generation of Gurobi code. "
)

CODE_GENERATION_SYSTEM_PROMPT = (
    "You are an operations research (OR) expert. Based on the user's optimization problem, construct a mathematical model and "
    "produce complete, reliable Python code that uses Gurobi to solve the optimization problem. "
    "Include model creation, variable definitions, adding constraints, objective function specification, solving, and result output. "
    "Output the code in the following fenced format: ```python\n{code}\n```. Do not provide additional explanations."
)

REQUEST_GUROBI_CODE_WITH_MATH_PROMPT = (
    "Based on the mathematical model above, write complete, reliable Python code using Gurobi to solve the optimization problem. "
    "Include model creation, variable definitions, constraints, objective function, solving, and result output. "
    "Output the code in the following fenced format: ```python\n{code}\n```. Do not provide additional explanations."
)

ERROR_FIX_PROMPT_TEMPLATE = (
    "The code execution produced an error. The error message is:\n{error_msg}\n"
    "Please fix the code and provide a complete, executable code listing again."
)

INFEASIBLE_SOLUTION_PROMPT = (
    "The current model result is *infeasible*. Carefully inspect the mathematical model and the Gurobi code for possible mistakes "
    "that could have caused infeasibility. After checking, output the corrected Gurobi Python code."
)

MAX_ATTEMPT_ERROR_PROMPT = (
    "The model code has been debugged multiple times and still fails. Carefully examine the mathematical model for mistakes. "
    "After checking, rebuild and output the Gurobi Python code."
)

# Load environment variables from .env file
load_dotenv()

# Google Generative AI configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if genai:
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    else:
        # Also accept the common environment variable used by Google SDKs
        google_api_key_alt = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_APIKEY") or os.getenv("GOOGLE_APIKEY")
        if google_api_key_alt:
            genai.configure(api_key=google_api_key_alt)

async def async_query_llm(messages, model_name="models/text-bison-001", temperature=0.2, max_attempts=3):
    """
    Async LLM query using Google Generative AI models.
    Returns (success, result) where result is the model text on success or an error string on failure.
    Retries on transient errors up to max_attempts with exponential backoff.
    """
    if genai is None:
        return False, "Google generative AI client library is not installed. Install with: pip install google-generative-ai"

    import time

    # Convert structured messages to a single prompt string suitable for text generation.
    system_message = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    assistant_messages = [m["content"] for m in messages if m.get("role") == "assistant"]

    # Build a simple conversation-style prompt
    prompt = system_message + "\n\n" if system_message else ""
    for user_msg, asst_msg in zip_longest(user_messages, assistant_messages, fillvalue=None):
        if user_msg:
            prompt += f"User: {user_msg}\n\n"
        if asst_msg:
            prompt += f"Assistant: {asst_msg}\n\n"
    # If there's an extra user message (common), ensure it's included
    if len(user_messages) > len(assistant_messages):
        prompt += f"User: {user_messages[-1]}\n\n"

    for attempt in range(max_attempts):
        try:
            # The google-generative-ai client is synchronous. Run it in a thread to avoid blocking the event loop.
            def call_genai():
                # Use text generation endpoint; for chat-style models you could change to genai.chat.create if available.
                # The API surface may vary across releases; handle common attributes defensively.
                response = genai.generate_text(
                    model=model_name,
                    prompt=prompt,
                    temperature=float(temperature),
                    max_output_tokens=2048
                )
                return response

            response = await asyncio.to_thread(call_genai)

            # Extract text from response with a few compatibility fallbacks
            text = None
            if hasattr(response, "text") and isinstance(response.text, str):
                text = response.text
            elif hasattr(response, "candidates") and isinstance(response.candidates, (list, tuple)) and len(response.candidates) > 0:
                candidate = response.candidates[0]
                # candidate may have .content or .output or .text
                if hasattr(candidate, "content"):
                    text = candidate.content
                elif hasattr(candidate, "output"):
                    text = candidate.output
                elif hasattr(candidate, "text"):
                    text = candidate.text
            elif isinstance(response, dict):
                # Some client versions return dictionaries
                if "candidates" in response and isinstance(response["candidates"], list) and response["candidates"]:
                    text = response["candidates"][0].get("content") or response["candidates"][0].get("output") or response["candidates"][0].get("text")
                else:
                    text = response.get("content") or response.get("output") or response.get("text")

            if text is None:
                return False, "Unable to extract text from Google LLM response (unexpected response structure)."

            return True, text

        except Exception as e:
            # For transient errors, retry with backoff
            print(f"[LLM Error] Attempt {attempt + 1}/{max_attempts} failed for model {model_name}: {str(e)}")
            if attempt < max_attempts - 1:
                # Exponential backoff in seconds: 2, 4, 8, ... (kept short to avoid very long waits)
                wait_time = 2 ** attempt
                print(f"[LLM Error] Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
            else:
                print(f"[LLM Error] Max attempts ({max_attempts}) reached. Returning failure.")
                return False, f"LLM error after {max_attempts} attempts: {str(e)}"

    return False, "Unknown LLM error"

async def async_extract_and_execute_python_code(text_content):
    """
    Async extraction and execution of Python code blocks with a 60-second timeout.
    Returns (success, result) where result is either the best objective (string) or stderr on failure.
    """
    python_code_blocks = re.findall(r'```python\s*([\s\S]*?)```', text_content)

    if not python_code_blocks:
        print("No Python code blocks found.")
        return False, "No Python code blocks found"

    for code_block in python_code_blocks:
        code_block = code_block.strip()
        if not code_block:
            print("Found an empty Python code block; skipping.")
            continue

        print("Found Python code block. Executing...")
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
                tmp_file.write(code_block)
                temp_file_path = tmp_file.name

            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    temp_file_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                # Wait for process completion with 60-second timeout
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)

            except asyncio.TimeoutError:
                print("Python code execution timed out (60 seconds). Possible infinite loop or long-running code.")
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
                return False, "Code execution timeout (60 seconds) - possible infinite loop"

            stdout_str = stdout.decode() if stdout else ""
            stderr_str = stderr.decode() if stderr else ""

            if proc.returncode == 0:
                print("Python code executed successfully. Output:\n")
                print(stdout_str)
                best_obj = extract_best_objective(stdout_str)
                if best_obj is not None:
                    print(f"\nBest objective: {best_obj}")
                    return True, str(best_obj)
                else:
                    print("\nBest objective not found in output.")
                    # Return success but no objective found
                    return True, ""
            else:
                print("Python code execution failed. Stderr:\n")
                print(stderr_str)
                return False, stderr_str

        except Exception as e:
            print(f"Error while executing Python code block: {e}")
            return False, str(e)
        finally:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        print("-" * 30)

    return False, "No valid code blocks executed"

async def async_generate_math_model(user_question, model_name="models/text-bison-001"):
    """
    Generate a mathematical model description from the user's question using the selected Google LLM.
    """
    messages = [
        {"role": "system", "content": MATH_MODEL_SYSTEM_PROMPT},
        {"role": "user", "content": user_question}
    ]

    success, math_model = await async_query_llm(messages, model_name)
    if not success:
        print(f"Failed to generate mathematical model: {math_model}")
        return False, f"MATH_MODEL_FAILED: {math_model}"

    return True, math_model

async def async_generate_and_run_gurobi_code(user_question, model_name="models/text-bison-001", math_model=None, enable_debug=False, max_attempts=3):
    """
    Generate Gurobi Python code (optionally using a math model) and attempt to execute it.
    """
    if math_model:
        messages = [
            {"role": "system", "content": MATH_MODEL_SYSTEM_PROMPT},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": math_model},
            {"role": "user", "content": REQUEST_GUROBI_CODE_WITH_MATH_PROMPT}
        ]
    else:
        messages = [
            {"role": "system", "content": CODE_GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_question}
        ]

    actual_attempts = max_attempts if enable_debug else 1

    is_solve_success, result, executed_content = await async_generate_or_code_solver(messages, model_name, actual_attempts)
    messages.append({"role": "assistant", "content": executed_content})
    return is_solve_success, result, messages, executed_content

async def async_generate_or_code_solver(messages, model_name, max_attempts):
    """
    Generate code from the LLM and try to execute it. On failures, request fixes from the LLM up to max_attempts.
    Returns (success, result, executed_content).
    """
    success, gurobi_code = await async_query_llm(messages, model_name)
    if not success:
        print(f"LLM failed to generate Gurobi code: {gurobi_code}")
        return False, f"CODE_GEN_ERROR: {gurobi_code}", ""

    executed_content = f"{gurobi_code}"
    attempt = 0
    error_msg = None
    while attempt < max_attempts:
        success_exec, error_msg = await async_extract_and_execute_python_code(executed_content)
        if success_exec:
            return True, error_msg, executed_content

        print(f"\nAttempt {attempt + 1} failed. Requesting the LLM to fix the code...\n")

        # Add the failed code and the error message to the conversation and ask for fixes
        messages.append({"role": "assistant", "content": gurobi_code})
        messages.append({"role": "user", "content": ERROR_FIX_PROMPT_TEMPLATE.format(error_msg=error_msg)})

        success, gurobi_code = await async_query_llm(messages, model_name)
        if not success:
            print(f"LLM failed to generate fixed Gurobi code: {gurobi_code}")
            return False, f"CODE_GEN_ERROR: {gurobi_code}", ""

        executed_content = f"{gurobi_code}"
        print("\nReceived a fixed version of the code. Re-executing...\n")
        attempt += 1

    print(f"Reached max attempts ({max_attempts}) without successful execution.")
    return False, error_msg or "UNKNOWN_EXECUTION_ERROR", executed_content

async def async_or_llm_agent(user_question, model_name="models/text-bison-001", use_math_model=False, enable_debug=False, max_attempts=3):
    """
    Orchestrator that optionally generates a mathematical model, then generates and executes Gurobi code.
    Includes extra debugging steps when enabled.
    """
    math_model = None

    # Step 1: Optionally generate a math model
    if use_math_model:
        success, math_model = await async_generate_math_model(user_question, model_name)
        if not success:
            return False, math_model, None, ""

    # Step 2: Generate and execute Gurobi code
    is_solve_success, result, messages, executed_content = await async_generate_and_run_gurobi_code(
        user_question, model_name, math_model, enable_debug, max_attempts
    )

    # Step 3: Additional debugging if requested
    if use_math_model and enable_debug:
        if is_solve_success:
            if not is_number_string(result):
                print('No available solution warning: Output is not a numeric solution.')
                messages.append({"role": "user", "content": INFEASIBLE_SOLUTION_PROMPT})
                is_solve_success, result, executed_content = await async_generate_or_code_solver(messages, model_name, max_attempts=1)
        else:
            if not isinstance(result, str) or not result.startswith("CODE_GEN_ERROR"):
                print('Max attempt debug warning: Code generation/execution errors reached max attempts.')
                messages.append({"role": "user", "content": MAX_ATTEMPT_ERROR_PROMPT})
                is_solve_success, result, executed_content = await async_generate_or_code_solver(messages, model_name, max_attempts=2)

    return is_solve_success, result, math_model, executed_content

async def process_single_case(i, d, args):
    """
    Process a single dataset case with failure tracking.
    """
    user_question, answer = d['question'], d['answer']
    failure_reason = None

    try:
        is_solve_success, llm_result, math_model, executed_content = await async_or_llm_agent(
            user_question,
            args.model,
            use_math_model=args.math,
            enable_debug=args.debug
        )

        if is_solve_success:
            print(f"Code executed successfully. Best objective/result: {llm_result}")
        else:
            print("Code execution failed.")
            if isinstance(llm_result, str) and llm_result.startswith("CODE_GEN_ERROR"):
                failure_reason = "CODE_GEN_ERROR"
            elif isinstance(llm_result, str) and llm_result.startswith("MATH_MODEL_FAILED"):
                failure_reason = "MATH_MODEL_FAILED"
            else:
                failure_reason = "CODE_EXECUTION_FAILED"

        print('------------------')

        pass_flag, correct_flag = eval_model_result(is_solve_success, llm_result, answer)

        print(f"=============== case {i} ==================")
        print(user_question)
        print('math model: ------------------------------------------------------------------------------------------------')
        print(f'{math_model}')
        print('executed content: ------------------------------------------------------------------------------------------------')
        print(f'{executed_content}')
        print('------------------------------------------------------------------------------------------------')
        print(f'solve: {is_solve_success}, llm: {llm_result}, ground truth: {answer}')
        print(f'[Final] run pass: {pass_flag}, solve correct: {correct_flag}')
        print("=================================================================================================")

        return llm_result, pass_flag, correct_flag, i, failure_reason

    except Exception as e:
        print(f"Task {i} failed with unexpected error: {str(e)}")
        failure_reason = f"UNEXPECTED_ERROR: {str(e)}"
        return None, False, False, i, failure_reason

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run optimization problem solving with Google LLMs (resilient async version)')
    parser.add_argument('--math', action='store_true',
                        help='Generate a mathematical model first and use it to guide code generation')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debugging mode with multiple attempts to fix code errors')
    parser.add_argument('--model', type=str, default='models/text-bison-001',
                        help='Model name to use for LLM queries. Examples: "models/text-bison-001" or a chat model if supported.')
    parser.add_argument('--data_path', type=str, default='data/datasets/IndustryOR.json',
                        help='Path to the dataset JSON file (supports both JSONL and regular JSON formats)')
    return parser.parse_args()

def load_dataset(data_path):
    """
    Load dataset from either JSONL format (IndustryOR.json, BWOR.json) or regular JSON format.
    Returns a dict mapping string IDs to items with keys: question, answer, difficulty, id.
    """
    dataset = {}

    with open(data_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        f.seek(0)

        if first_line.startswith('{"en_question"') or first_line.startswith('{"cn_question"'):
            # JSONL format
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        dataset_item = {
                            'question': item.get('en_question', item.get('cn_question', '')),
                            'answer': item.get('en_answer', item.get('cn_answer', '')),
                            'difficulty': item.get('difficulty', 'Unknown'),
                            'id': item.get('id', line_num - 1)
                        }
                        dataset[str(dataset_item['id'])] = dataset_item
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse line {line_num}: {line}")
                        continue
        else:
            # Regular JSON format
            dataset = json.load(f)

    return dataset

async def main():
    args = parse_args()

    dataset = load_dataset(args.data_path)

    dataset_items = list(dataset.items())

    # Process dataset in batches
    batch_size = 50
    all_results = []
    total_batches = (len(dataset_items) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(dataset_items))
        batch_items = dataset_items[start_idx:end_idx]

        print(f"\n{'='*50}")
        print(f"Processing batch {batch_num + 1}/{total_batches} (cases {start_idx + 1}-{end_idx})")
        print(f"{'='*50}\n")

        tasks = []
        for i, d in batch_items:
            task = process_single_case(i, d, args)
            tasks.append(task)

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_batch_results = []
        for idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                task_id = batch_items[idx][0]
                print(f"Exception in task {task_id}: {str(result)}")
                processed_batch_results.append((None, False, False, task_id, f"EXCEPTION: {str(result)}"))
            else:
                processed_batch_results.append(result)

        all_results.extend(processed_batch_results)

        batch_pass_count = sum(1 for _, pass_flag, _, _, _ in processed_batch_results if pass_flag)
        batch_correct_count = sum(1 for _, _, correct_flag, _, _ in processed_batch_results if correct_flag)
        print(f"\nBatch {batch_num + 1} Summary:")
        print(f"  Processed: {len(processed_batch_results)} cases")
        print(f"  Run pass: {batch_pass_count}")
        print(f"  Solve correct: {batch_correct_count}")
        print(f"  Completed: {end_idx}/{len(dataset_items)} total cases")

    # Final aggregation
    pass_count = sum(1 for _, pass_flag, _, _, _ in all_results if pass_flag)
    correct_count = sum(1 for _, _, correct_flag, _, _ in all_results if correct_flag)
    error_datas = [i for _, pass_flag, correct_flag, i, _ in all_results if not pass_flag or not correct_flag]

    # Failure tracking
    failure_summary = {}
    failed_tasks = []
    for llm_result, pass_flag, correct_flag, task_id, failure_reason in all_results:
        if not pass_flag or not correct_flag:
            failed_tasks.append({
                'task_id': task_id,
                'pass_flag': pass_flag,
                'correct_flag': correct_flag,
                'failure_reason': failure_reason or 'UNKNOWN',
                'llm_result': llm_result
            })

            failure_type = failure_reason or 'UNKNOWN'
            if failure_type not in failure_summary:
                failure_summary[failure_type] = []
            failure_summary[failure_type].append(task_id)

    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f'[Total {len(dataset)}] run pass: {pass_count}, solve correct: {correct_count}')
    print(f'[Total fails {len(error_datas)}] error cases: {error_datas}')

    print(f"\n{'='*50}")
    print("FAILURE ANALYSIS")
    print(f"{'='*50}")

    if failure_summary:
        print(f"Total failed tasks: {len(failed_tasks)}")
        print("\nFailure breakdown by type:")
        for failure_type, task_ids in failure_summary.items():
            print(f"  {failure_type}: {len(task_ids)} tasks")
            print(f"    Task IDs: {task_ids}")

        print(f"\nDetailed failure list:")
        for i, failure in enumerate(failed_tasks[:20]):  # Show first 20 failures
            print(f"  {i+1}. Task {failure['task_id']}: {failure['failure_reason']}")
            if failure['llm_result'] and len(str(failure['llm_result'])) < 100:
                print(f"     Result: {failure['llm_result']}")

        if len(failed_tasks) > 20:
            print(f"     ... and {len(failed_tasks) - 20} more failures")
    else:
        print("No failures detected!")

if __name__ == "__main__":
    # Run as script
    asyncio.run(main())
