import json
import os
import re
import csv
from typing import Dict, List, Any, Tuple
import pandas as pd

task_dict: Dict[str, str] = {
    "PIQA": "../CLASSLA-PIQA-Benchmark/evaluation-for-the-paper/results.json",
    "COPA": "../DIALECT-COPA-Benchmark/evaluation-for-the-paper/results.json",
    "Genre": "../Genre-Automatic-Identification-Benchmark/evaluation-for-the-paper/results_combined.json",
    "News Topic": "../IPTC-NewsTopic-Benchmark/evaluation-for-the-paper/results.json",
    "Parliamentary Speech Topic": "../ParlaCAP-Topic-Benchmark/evaluation-for-the-paper/results.json",
    "Sentiment": "../ParlaSent-Benchmark/evaluation-for-the-paper/results.json"
}


# Output CSV file for the dashboard
OUTPUT_CSV = "results.csv"


def load_jsonl(path: str) -> List[dict]:
    """
    Load a JSONL file (one JSON object per line).
    Returns a list of dicts.
    """
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {path} at line {i}: {e}")
            if not isinstance(obj, dict):
                raise ValueError(f"Non-object JSON at line {i} in {path}")
            records.append(obj)
    return records


def parse_key_to_language_metric(key: str) -> Tuple[str, str]:
    """
    Parse keys like 'Macedonian (accuracy)' into ('Macedonian', 'accuracy')
    or 'Macedonian (micro-F1)' into ('Macedonian', 'micro_f1').

    If there is no '(metric)' part, return ('key', 'score').
    """
    pattern = r"^(.*)\s*\(([^()]+)\)\s*$"
    m = re.match(pattern, key)
    if m:
        language = m.group(1).strip()
        metric = m.group(2).strip()
    else:
        language = key.strip()
        metric = "score"

    # Normalize metric: lowercase, spaces and hyphens → underscores
    metric_norm = (
        metric.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
    return language, metric_norm

def rename_model(model: str) -> str:
    """
    Replace model name using model_name_map if present.
    Otherwise return original name.
    """
    model_name_map = {
    "gpt-4o-2024-08-06": "GPT-4o",
    "gpt-3.5-turbo-0125": "GPT-3.5-Turbo",
    "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
    "gemma3:27b": "Gemma 3",
    "gemma2:27b": "Gemma 2",
    "llama3.3:latest": "LLaMA 3.3",
    "Llama 3.3": "LLaMA 3.3",
    "deepseek-r1:14b": "DeepSeek-R1-Distill",
    "DeekSeek-R1": "DeepSeek-R1-Distill",
    "dummy-most_frequent": "Dummy (Most Frequent)",
    "dummy-stratified": "Dummy (Stratified)",
    "SVC": "Support Vector Machine",
    "COMPLEMENTNB": "Naive Bayes Classifier",
    "gpt-5-mini-2025-08-07": "GPT-5-mini",
    "gpt-5": "GPT-5",
    "gpt-5-2025-08-07": "GPT-5",
    "gpt-5-nano-2025-08-07": "GPT-5-Nano",
    "GPT-5-nano": "GPT-5-Nano",
    "llama4:scout": "LLaMA 4 Scout",
    "qwen3:32b": "Qwen 3",
    "Qwen3": "Qwen 3",
    'google/gemini-2.5-flash-lite': "Gemini 2.5 Flash Lite",
    'google/gemini-2.5-flash': "Gemini 2.5 Flash",
    'google/gemini-2.5-pro': "Gemini 2.5 Pro",
    'mistralai/mistral-medium-3.1': "Mistral Medium 3.1",
    'mistralai/mistral-small-3.2-24b-instruct': "Mistral Small 3.2",
    'cohere/command-a': "Command A",
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5",
    "GaMS-27B-quantized": "GaMS-27B-Instruct (quantized)",
    "GaMS-27B": "GaMS-27B-Instruct",
    "GaMS-Instruct 27B": "GaMS-27B-Instruct",
    }

    return model_name_map.get(model, model)

def extract_rows_for_task(task_name: str, jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Given a task name and path to its JSONL file,
    return a list of rows with columns:
    task, model, language, metric, value
    """
    data = load_jsonl(jsonl_path)
    rows: List[Dict[str, Any]] = []

    for record in data:
        # Try both 'Model' and 'model'
        model = record.get("Model") or record.get("model")
        if model is None:
            # Skip records without model info
            continue

        # Apply renaming dictionary
        model_clean = rename_model(model)

        for key, value in record.items():
            if key in ("Model", "model"):
                continue

            # Parse language and metric from the key
            language, metric = parse_key_to_language_metric(key)

            # Convert value to float if possible
            try:
                value_num = float(value)
            except (TypeError, ValueError):
                # Skip non-numeric values (e.g., if something is text)
                continue

            # Correct Serbian (latin) to Serbian
            if language == "Serbian (latin)":
                language = "Serbian"

            # Skip results from Serbian (cyrillic)
            if language != "Serbian (cyrillic)":
                # Skip certain models:
                if model_clean not in ["Command A", "Dummy (Most Frequent)", "Dummy (Stratified)",
                       "Gemini 2.5 Flash Lite", "Llama 4 Scout", "Logistic Regression", "NLI zero-shot model",
                       "Naive Bayes Classifier", "Support Vector Machine", "fastText", "GaMS-27B-Instruct (quantized)", "GaMS-27B-Instruct (quantized)"]:
                    rows.append(
                        {
                            "task": task_name,
                            "model": model_clean,
                            "language": language,
                            "metric": metric,
                            "value": value_num,
                        }
                )
            
    return rows


def main():
    all_rows: List[Dict[str, Any]] = []

    for task_name, path in task_dict.items():
        if not os.path.exists(path):
            print(f"Warning: file not found for task {task_name}: {path}")
            continue

        print(f"Processing task {task_name} from {path} ...")
        rows = extract_rows_for_task(task_name, path)
        print(f"  -> extracted {len(rows)} rows")
        all_rows.extend(rows)

    if not all_rows:
        print("No rows extracted. Check your task_dict and JSONL formats.")
        return

    # Ensure output directory exists
    out_dir = os.path.dirname(OUTPUT_CSV)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = ["task", "model", "language", "metric", "value"]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {OUTPUT_CSV}")

    # print out all different models
    print("List of evaluated models:")
    df = pd.read_csv(OUTPUT_CSV)
    print(sorted(list(df["model"].unique())))

    print("List of evaluated languages:")
    print(list(df["language"].unique()))


if __name__ == "__main__":
    main()
