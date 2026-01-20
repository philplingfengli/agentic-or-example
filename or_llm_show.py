import os
import copy
import json
import shutil
import wcwidth
import io
import time
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from contextlib import redirect_stdout

# Google generative AI client
try:
    import google.generativeai as genai
except Exception:
    genai = None

from utils import (
    is_number_string,
    convert_to_number,
    extract_best_objective,
    extract_and_execute_python_code,
    eval_model_result
)

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI (expects GOOGLE_API_KEY in environment)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY and genai:
    genai.configure(api_key=GOOGLE_API_KEY)


def get_display_width(text):
    """
    Calculate the display width of a string, accounting for wide characters.
    Uses the wcwidth module for accurate width calculation.

    Args:
        text (str): The text to calculate the width for.

    Returns:
        int: The display width of the text.
    """
    return wcwidth.wcswidth(text)


def print_header(text="", add_newline_before=True, add_newline_after=True,
                 border_char="=", side_char="||"):
    """
    Print a header with customizable text in the middle, adjusted to the console window width.
    """
    if add_newline_before:
        print()

    terminal_width = shutil.get_terminal_size().columns
    terminal_width = max(terminal_width, 40)

    side_char_len = len(side_char)

    print(border_char * terminal_width)
    print(side_char + " " * (terminal_width - 2 * side_char_len) + side_char)

    text_display_width = get_display_width(text)
    available_space = terminal_width - 2 * side_char_len

    if text_display_width <= available_space:
        left_padding = (available_space - text_display_width) // 2
        right_padding = available_space - text_display_width - left_padding
        print(side_char + " " * left_padding + text + " " * right_padding + side_char)
    else:
        truncated_text = ""
        truncated_width = 0
        for char in text:
            char_width = get_display_width(char)
            if truncated_width + char_width + 3 > available_space:
                break
            truncated_text += char
            truncated_width += char_width
        truncated_text += "..."
        right_padding = available_space - get_display_width(truncated_text)
        print(side_char + truncated_text + " " * right_padding + side_char)

    print(side_char + " " * (terminal_width - 2 * side_char_len) + side_char)
    print(border_char * terminal_width)

    if add_newline_after:
        print()


def _convert_messages_for_google(messages):
    """
    Convert messages of the form [{"role": "system"/"user"/"assistant", "content": "..."}]
    into the format accepted by Google Generative API: [{"author": "system"/"user"/"assistant", "content": "..."}]
    """
    converted = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        converted.append({"author": role, "content": content})
    return converted


def _extract_text_from_genai_response(resp):
    """
    Extract a sensible text string from the google.generativeai response object.
    Handles a few common response shapes.
    """
    try:
        candidates = getattr(resp, "candidates", None)
        if candidates and len(candidates) > 0:
            cand = candidates[0]
            # cand may be an object or dict
            if isinstance(cand, dict):
                content = cand.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    text = ""
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            text += item["text"]
                    if text:
                        return text
            else:
                content = getattr(cand, "content", None)
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    text = ""
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            text += item["text"]
                    if text:
                        return text
    except Exception:
        pass

    try:
        output = getattr(resp, "output", None)
        if isinstance(output, str):
            return output
        if isinstance(output, list):
            text = ""
            for part in output:
                if isinstance(part, dict):
                    content = part.get("content")
                    if isinstance(content, str):
                        text += content
                    elif isinstance(content, list):
                        for it in content:
                            if isinstance(it, dict) and "text" in it:
                                text += it["text"]
            if text:
                return text
    except Exception:
        pass

    text = getattr(resp, "text", None) or getattr(resp, "response", None)
    if text:
        return str(text)
    return str(resp)


def query_llm(messages, model_name="chat-bison-001", temperature=0.2):
    """
    Call Google LLM and return its text response.

    Args:
        messages (list): List of {"role":..., "content":...}
        model_name (str): Google model name, default "chat-bison-001"
        temperature (float): Sampling temperature.

    Returns:
        str: The model's text output.
    """
    if genai is None:
        raise RuntimeError("google.generativeai is not installed. Install it and set GOOGLE_API_KEY in the environment.")

    gen_messages = _convert_messages_for_google(messages)

    try:
        response = genai.chat.create(model=model_name, messages=gen_messages, temperature=temperature)
    except Exception as e:
        raise RuntimeError(f"Error calling Google Generative API: {e}")

    full_response = _extract_text_from_genai_response(response)

    print("LLM Output: ", end="", flush=True)
    print(full_response)
    return full_response


def generate_or_code_solver(messages_bak, model_name, max_attempts):
    messages = copy.deepcopy(messages_bak)

    print_header("LLM generates Python Gurobi code")

    gurobi_code = query_llm(messages, model_name)

    print_header("Execute generated Python code")
    text = f"{gurobi_code}"
    attempt = 0
    while attempt < max_attempts:
        buffer2 = io.StringIO()
        with redirect_stdout(buffer2):
            success, error_msg = extract_and_execute_python_code(text)
        captured_output2 = buffer2.getvalue()
        for c in captured_output2:
            print(c, end="", flush=True)
            time.sleep(0.005)

        if success:
            messages_bak.append({"role": "assistant", "content": gurobi_code})
            return True, error_msg, messages_bak

        print(f"\nAttempt {attempt + 1} failed. Requesting the model to fix the code...\n")

        # Build a repair request
        messages.append({"role": "assistant", "content": gurobi_code})
        messages.append({"role": "user", "content": f"Execution produced an error:\n{error_msg}\nPlease fix the code and provide the complete, executable code."})

        gurobi_code = query_llm(messages, model_name)
        text = f"{gurobi_code}"

        print("\nReceived repaired code, preparing to re-run...\n")
        attempt += 1

    messages_bak.append({"role": "assistant", "content": gurobi_code})
    print(f"Reached maximum attempts ({max_attempts}) without successful execution.")
    return False, None, messages_bak


def or_llm_agent(user_question, model_name="chat-bison-001", max_attempts=3):
    """
    Request Gurobi code from the LLM and execute it. If execution fails, request fixes.

    Args:
        user_question (str): The user's optimization problem description.
        model_name (str): Google model name to use.
        max_attempts (int): Maximum repair attempts.

    Returns:
        tuple: (success: bool, best_objective: float | None)
    """
    messages = [
        {"role": "system", "content": (
            "You are an operations research expert. Given a user-provided optimization problem, "
            "construct an accurate mathematical model (linear programming if applicable). Focus on providing "
            "a correct mathematical formulation that will be used to generate Gurobi code."
        )},
        {"role": "user", "content": user_question}
    ]

    print_header("LLM: Build a linear programming model")
    math_model = query_llm(messages, model_name)
    validate_math_model = math_model
    messages.append({"role": "assistant", "content": validate_math_model})

    messages.append({"role": "user", "content": (
        "Based on the mathematical model above, write complete and reliable Python code using Gurobi to solve the optimization problem. "
        "Include model construction, variable definitions, constraints, objective, solve call, and result printing. "
        "Return the code only, wrapped in a markdown python code block: ```python\\n{code}\\n``` . Do not include explanations."
    )})

    is_solve_success, result, messages = generate_or_code_solver(messages, model_name, max_attempts)
    print(f"Stage result: {is_solve_success}, {result}")

    if is_solve_success:
        if not is_number_string(result):
            print("!![No feasible solution warning]!!")
            messages.append({"role": "user", "content": (
                "The current model returns *no feasible solution*. Carefully review the mathematical model and Gurobi code "
                "to find potential modeling errors that cause infeasibility. After checking, provide corrected Gurobi Python code "
                "wrapped in a python code block. Do not include explanations."
            )})
            is_solve_success, result, messages = generate_or_code_solver(messages, model_name, max_attempts=1)
    else:
        print("!![Max attempt error warning]!!")
        messages.append({"role": "user", "content": (
            "The model code failed after multiple attempts. Carefully review the mathematical model for errors and reconstruct "
            "the Gurobi Python code. Return the complete code in a python code block without explanations."
        )})
        is_solve_success, result, messages = generate_or_code_solver(messages, model_name, max_attempts=2)

    return is_solve_success, result


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from or_llm_eval_async_resilient import load_dataset

    dataset = load_dataset('data/datasets/IndustryOR.json')
    console = Console()

    # Use Google model by default
    model_name = 'chat-bison-001'

    pass_count = 0
    correct_count = 0
    for i, d in dataset.items():
        print_header("Optimization problem")
        user_question, answer = d['question'], d['answer']
        buffer2 = io.StringIO()
        with redirect_stdout(buffer2):
            md = Markdown(user_question)
            console.print(md)
            print('-------------')

        captured_output2 = buffer2.getvalue()
        for c in captured_output2:
            print(c, end="", flush=True)
            time.sleep(0.005)

        is_solve_success, llm_result = or_llm_agent(user_question, model_name)
        if is_solve_success:
            print(f"Code executed successfully. Best objective: {llm_result}")
        else:
            print("Code execution failed.")
