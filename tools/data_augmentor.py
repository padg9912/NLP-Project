import json
import time
import requests
import random
from datetime import datetime, timedelta
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

if len(sys.argv) > 2:
    INPUT_FILE = sys.argv[1]
    OUTPUT_FILE = sys.argv[2]
else:
    INPUT_FILE = "datasets/ground_truth.json"
    OUTPUT_FILE = "datasets/full_augmented_dataset.json"

SAVE_INTERVAL = 100  # Save every 100 examples
OLLAMA_MODEL = "mistral:latest"  # switched to a smaller, faster model
OLLAMA_URL = "http://localhost:11434/api/generate"

# Use a T5-based paraphraser model
T5_MODEL_NAME = "Vamsi/T5_Paraphrase_Paws"
tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_NAME)

def paraphrase_with_t5(statement):
    input_text = f"paraphrase: {statement} </s>"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=256, truncation=True)
    outputs = model.generate(
        input_ids, max_length=256, num_beams=5, num_return_sequences=1, temperature=1.5
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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
            paraphrased = paraphrase_with_t5(original)
            if paraphrased and paraphrased != original:
                new_entry = entry.copy()
                new_entry["augmented_statement"] = paraphrased
                new_entry["generation_mode"] = "t5_paraphrase"
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
