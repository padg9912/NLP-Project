import json
import glob

output_files = sorted(glob.glob("datasets/augmented_chunk_*.json"))
merged = []
for fname in output_files:
    with open(fname) as f:
        merged.extend(json.load(f))

with open("datasets/full_augmented_dataset.json", "w") as f:
    json.dump(merged, f, indent=2)

print(f"Merged {len(merged)} entries into datasets/full_augmented_dataset.json") 