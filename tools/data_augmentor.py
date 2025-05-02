import json
import time
import requests
import random
from datetime import datetime, timedelta
import sys

if len(sys.argv) > 2:
    INPUT_FILE = sys.argv[1]
    OUTPUT_FILE = sys.argv[2]
else:
    INPUT_FILE = "datasets/ground_truth.json"
    OUTPUT_FILE = "datasets/full_augmented_dataset.json"

SAVE_INTERVAL = 100  # Save every 100 examples
OLLAMA_MODEL = "mistral:latest"  # switched to a smaller, faster model
OLLAMA_URL = "http://localhost:11434/api/generate"

# --- Ollama paraphrasing function ---
def call_ollama_paraphrase(statement, model=OLLAMA_MODEL):
    prompt = f"Paraphrase the following statement while preserving its meaning. Only return the new statement.\n\n'{statement}'"
    data = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=data)
    if response.status_code == 200:
        return response.json()["response"].strip()
    else:
        print(f"Ollama error: {response.text}")
        return None

def map_truth(orig_truth: str):
    if orig_truth in ["true", "mostly-true", "half-true"]:
        return "true"
    elif orig_truth in ["mostly-false", "false", "pants-fire"]:
        return "false"
    return orig_truth

def generate_augmented_merged_set():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    merged = []
    for idx, entry in enumerate(data):
        merged.append(entry)  # Always keep the original
        # Only augment 'true' class (can be changed as needed)
        if map_truth(entry.get('verdict', entry.get('truth_label', ''))) == "true":
            original = entry["statement"]
            paraphrased = call_ollama_paraphrase(original)
            if paraphrased and paraphrased != original:
                new_entry = entry.copy()
                new_entry["augmented_statement"] = paraphrased
                new_entry["generation_mode"] = "ollama_paraphrase"
                merged.append(new_entry)
                print(f"Paraphrased: {original} -> {paraphrased}")
            time.sleep(0.2)  # reduced sleep to speed up
        if len(merged) % SAVE_INTERVAL == 0:
            with open(OUTPUT_FILE, "w") as out_f:
                json.dump(merged, out_f, indent=2)
            print(f"[Progress] Saved {len(merged)} entries.")
    # Final save
    with open(OUTPUT_FILE, "w") as out_f:
        json.dump(merged, out_f, indent=2)
    print(f"[Done] All original and augmented entries saved to {OUTPUT_FILE}.")

if __name__ == "__main__":
    generate_augmented_merged_set()
