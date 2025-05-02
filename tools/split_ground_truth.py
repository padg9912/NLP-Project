import json
import random
from collections import Counter
from sklearn.model_selection import train_test_split
import os

INPUT_FILE = "datasets/ground_truth.json"
TRAIN_FILE = "datasets/train_set.json"
VAL_FILE = "datasets/validate_set.json"
TEST_FILE = "datasets/test_set.json"

# Three-class mapping
def map_verdict(verdict):
    verdict = verdict.lower()
    if verdict in ["true", "mostly-true"]:
        return "true"
    elif verdict in ["false", "pants-fire", "mostly-false"]:
        return "false"
    elif verdict == "half-true":
        return "unknown"
    else:
        return None  # skip if label is missing or unrecognized

def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    # Map labels and filter out unrecognized
    mapped = []
    for ex in data:
        mapped_label = map_verdict(ex.get("verdict", ""))
        if mapped_label is not None:
            ex["verdict"] = mapped_label
            mapped.append(ex)

    print("Label distribution after mapping:", Counter([ex["verdict"] for ex in mapped]))

    # Shuffle and split
    random.seed(42)
    train, temp = train_test_split(mapped, test_size=0.2, stratify=[ex["verdict"] for ex in mapped], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=[ex["verdict"] for ex in temp], random_state=42)

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Save splits
    os.makedirs("datasets", exist_ok=True)
    with open(TRAIN_FILE, "w") as f:
        json.dump(train, f, indent=2)
    with open(VAL_FILE, "w") as f:
        json.dump(val, f, indent=2)
    with open(TEST_FILE, "w") as f:
        json.dump(test, f, indent=2)

if __name__ == "__main__":
    main() 