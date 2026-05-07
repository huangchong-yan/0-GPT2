"""Automated response evaluation with a local Ollama model."""

import json
import urllib.request


def check_if_running(process_name: str) -> bool:
    import psutil

    for proc in psutil.process_iter(["name"]):
        name = proc.info.get("name") or ""
        if process_name.lower() in name.lower():
            return True
    return False


def query_ollama(
    prompt: str,
    model: str = "phi3",
    url: str = "http://localhost:11434/api/chat",
) -> str:
    """Query a local Ollama chat model and return streamed text."""
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"seed": 123, "temperature": 0, "num_ctx": 2048},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json.get("message", {}).get("content", "")
    return response_data


def score_responses(json_data, format_input_fn, json_key: str, model: str = "phi3"):
    """Score model responses from 0 to 100 using an Ollama judge model."""
    from tqdm import tqdm

    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input_fn(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}` "
            "on a scale from 0 to 100, where 100 is the best score. "
            "Respond with the integer number only."
        )
        score = query_ollama(prompt, model)
        try:
            scores.append(int(score.strip()))
        except ValueError:
            print(f"Could not convert score: {score}")
    return scores

