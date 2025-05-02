import json
import math
import os

INPUT_FILE = "datasets/ground_truth.json"
N_CHUNKS = 4  # Change this to the number of parallel processes you want

with open(INPUT_FILE, "r") as f:
    data = json.load(f)

chunk_size = math.ceil(len(data) / N_CHUNKS)
for i in range(N_CHUNKS):
    chunk = data[i*chunk_size : (i+1)*chunk_size]
    out_path = f"datasets/ground_truth_chunk_{i+1}.json"
    with open(out_path, "w") as out_f:
        json.dump(chunk, out_f, indent=2)
    print(f"Saved {len(chunk)} entries to {out_path}") 