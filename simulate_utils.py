import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import seaborn as sns
import re
import os
import multiprocessing
import scipy

save_path = './'

brackets = list(np.array([0, 97, 394.75, 842, 1607.25, 2041, 5103])*100/12)
quantiles = [0, 0.25, 0.5, 0.75, 1.0]

from datetime import datetime
world_start_time = datetime.strptime('2001.01', '%Y.%m')

prompt_cost_1k, completion_cost_1k = 0.001, 0.002

# --- Local open-model config (Ollama), configurable via environment ---
# Single source of truth. The entry point may override these at runtime from
# the --ollama_model / --ollama_url CLI flags by reassigning the module globals.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
# How many agents to send concurrently to the local server. Ollama queues
# requests, so a small pool avoids memory pressure on a laptop.
OLLAMA_CONCURRENCY = int(os.environ.get("OLLAMA_CONCURRENCY", "3"))

def prettify_document(document: str) -> str:
    # Remove sequences of whitespace characters (including newlines)
    cleaned = re.sub(r'\s+', ' ', document).strip()
    return cleaned


def get_multiple_completion(dialogs, num_cpus=15, temperature=0, max_tokens=100):
    from functools import partial
    get_completion_partial = partial(get_completion, temperature=temperature, max_tokens=max_tokens)
    with multiprocessing.Pool(processes=num_cpus) as pool:
        results = pool.map(get_completion_partial, dialogs)
    total_cost = sum([cost for _, cost in results])
    return [response for response, _ in results], total_cost

def get_completion(dialogs, temperature=0, max_tokens=100):
    import openai
    openai.api_key = 'Your Key'
    import time
    
    max_retries = 20
    for i in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613", # inaccessible now, try gpt-4o-mini
                messages=dialogs,
                temperature=temperature,
                max_tokens=max_tokens
            )
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            this_cost = prompt_tokens/1000*prompt_cost_1k + completion_tokens/1000*completion_cost_1k
            return response.choices[0].message["content"], this_cost
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(6)
            else:
                print(f"An error of type {type(e).__name__} occurred: {e}")
                return "Error"

# --------------- Local open models (Ollama) helpers ---------------
# Ollama exposes an OpenAI-compatible endpoint, so the same `openai` client
# works for both local and hosted models. Local inference has no dollar cost.

def check_ollama_model():
    """Preflight: raise a clear RuntimeError if Ollama is down or the model
    isn't pulled, so the user learns before a long run starts."""
    import urllib.request
    import json
    tags_url = OLLAMA_BASE_URL.rstrip('/').rsplit('/v1', 1)[0] + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Is it running? ({e})\n"
            f"Start it with: ollama serve"
        )
    available = [m["name"] for m in data.get("models", [])]
    if not any(m == OLLAMA_MODEL or m.startswith(OLLAMA_MODEL + ":") for m in available):
        raise RuntimeError(
            f"Model '{OLLAMA_MODEL}' is not installed in Ollama.\n"
            f"Available models: {available or '(none)'}\n"
            f"Run: ollama pull {OLLAMA_MODEL}"
        )
    print(f"[ollama] Model '{OLLAMA_MODEL}' confirmed available at {OLLAMA_BASE_URL}.")

def get_multiple_completion_llama(dialogs, temperature=0, max_tokens=100):
    """Run completions against the local Ollama server using a thread pool."""
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    fn = partial(_get_completion_llama, temperature=temperature, max_tokens=max_tokens)
    with ThreadPoolExecutor(max_workers=OLLAMA_CONCURRENCY) as pool:
        results = list(pool.map(fn, dialogs))
    # Local inference has no per-token dollar cost.
    return results, 0.0

def _get_completion_llama(dialogs, temperature=0, max_tokens=100):
    """Single completion call to Ollama's OpenAI-compatible endpoint."""
    from openai import OpenAI
    import time

    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    max_retries = 5
    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=list(dialogs),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(3)
            else:
                raise RuntimeError(
                    f"Ollama failed after {max_retries} retries ({type(e).__name__}): {e}"
                )

def format_numbers(numbers):
    return '[' + ', '.join('{:.2f}'.format(num) for num in numbers) + ']'

def format_percentages(numbers):
    return '[' + ', '.join('{:.2%}'.format(num) for num in numbers) + ']'
