import sys
if len(sys.argv) < 4:
    print("Usage: python RelevancyKCMGS.py <start_line> <max_lines> <GPU_ID>")
    sys.exit(1)

start_line = int(sys.argv[1])
max_lines = int(sys.argv[2])
device_id = int(sys.argv[3])
from icecream import ic
ic(start_line, max_lines, device_id)
import os
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/scratch/rahul.garg/stCache"
os.makedirs(os.environ["SENTENCE_TRANSFORMERS_HOME"], exist_ok=True)
os.environ["TORCH_HOME"] = "/scratch/rahul.garg/torchCache"
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = "/scratch/rahul.garg/hfCache"
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.environ["HF_HOME"] = "/scratch/rahul.garg/hfCache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

import pickle as pkl
from pathlib import Path
import torch
from tqdm import tqdm
import json
from sentence_transformers import SentenceTransformer, util
import time

# Setting up environment paths


# Reading command-line arguments

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

print("Start line:", start_line)
print("Max lines:", max_lines)

# Load the MiniLM model
model = SentenceTransformer('all-MiniLM-L6-v2')
model = model.to(device)

# Load OCRs and Captions
jsonlFilesLoc = "/scratch/rahul.garg/hateful_memes/"
ocrs = {}
labels = {}

for file in Path(jsonlFilesLoc).rglob("*.jsonl"):
    with open(file, "r") as f:
        for line in f:
            line = json.loads(line)
            ocrs[int(line["id"])] = line["text"]
            labels[int(line["id"])] = line["label"]

captionsPath = '/home/rahul.garg/llava/newCaptionsLLAVA.json'
captions = {}
with open(captionsPath, 'r') as f:
    dat = json.load(f)
    for key in dat:
        keyInt = int(key.split('.')[0])
        captions[keyInt] = dat[key]

# Load Llava targets
llavaFile = "/home/rahul.garg/modelRuns/Llava.json"
targets = {}
with open(llavaFile, "r") as f:
    llavaData = json.load(f)
    for key in llavaData.keys():
        id = int(key.split(".")[0])
        if id not in targets:
            targets[id] = llavaData[key]

# Combining OCRs, captions, and targets into context
commonIds = set(ocrs.keys()).intersection(captions.keys()).intersection(targets.keys())
combinedTexts = {
    key: f"{ocrs[key]} . {captions[key]} . The target is {targets[key]}."
    for key in commonIds
}

# Load or generate node embeddings
embedding_pkl_path = '/scratch/rahul.garg/conceptNet/all_node_embeddings.pkl'
nodesEmbCache = None
if not os.path.exists(embedding_pkl_path):
    print("Generating node embeddings...")
    # Regenerate embeddings if the file doesn't exist
    all_nodes = []  # Load all nodes here
    embeddings = {}
    batch_size = 1000
    for i in tqdm(range(0, len(all_nodes), batch_size), desc="Encoding nodes"):
        batch_nodes = all_nodes[i:i + batch_size]
        batch_embeddings = model.encode(batch_nodes, convert_to_tensor=True, device=device)
        for node, embedding in zip(batch_nodes, batch_embeddings):
            embeddings[node] = embedding.cpu()
    
    with open(embedding_pkl_path, 'wb') as f:
        pkl.dump(embeddings, f)
else:
    # Load embeddings from cache
    with open(embedding_pkl_path, 'rb') as f:
        embeddings = pkl.load(f)

nodesEmbCache = {node: embedding.to(device) for node, embedding in embeddings.items()}

# Define get_LM_score using cosine similarity with MiniLM embeddings
def get_LM_score(nodes, context):
    # startTime = time.time()
    nodes = [{"key": "QAcontext", "score": 0}] + nodes[:]  # Prepend a pseudo-node for QA context
    query_embedding = model.encode([context], convert_to_tensor=True, device=device)
    # print("Time to encode query:", time.time() - startTime)
    # startTime = time.time()

    node_keys = [node["key"] for node in nodes if node["key"] != "QAcontext"]
    node_embeddings = torch.stack([nodesEmbCache[node] for node in node_keys]).to(device)

    # Compute cosine similarity between the query and node embeddings
    cosine_scores = util.pytorch_cos_sim(query_embedding, node_embeddings)[0]

    scores = [float(score) for score in cosine_scores]
    # print("Time to compute scores:", time.time() - startTime)
    # startTime = time.time()

    # Assign scores to the nodes
    score_idx = 0
    for node in nodes:
        if node["key"] == "QAcontext":
            node["score"] = 0.0  # For QA context node, score is 0
        else:
            node["score"] = scores[score_idx]
            score_idx += 1

    # print("Time to assign scores:", time.time() - startTime)

    return nodes, query_embedding

# Prepare output path
inputPath = "/scratch/rahul.garg/conceptNet/allCmgs_llava_targets_and_caption_h1.jsonl"
outputName = "allCmgs_llava_Target_and_caption_h1_RelevancyKCMGS_miniLM"
if start_line == 0:
    outputPath = f"/scratch/rahul.garg/{outputName}.jsonl"
else:
    outputPath = f"/scratch/rahul.garg/{outputName}_{start_line}_{max_lines}.jsonl"

# Check if output already exists, and resume
alreadyDone = set()
if os.path.exists(outputPath):
    with open(outputPath, "r") as f:
        for line in f:
            try:
                loadedObj = json.loads(line)
                alreadyDone.add(int(loadedObj["id"]))
            except Exception as e:
                continue

# Processing lines
processed_lines = 0
totalLines = min(start_line + max_lines, 12015) - start_line

with open(inputPath, "r") as f:
    for _ in tqdm(range(start_line), desc="Skipping to start"):
        next(f)

    pbar = tqdm(f, total=totalLines, desc="Processing")
    for line in pbar:
        torch.cuda.empty_cache()

        try:
            if processed_lines >= max_lines:
                break

            loadedObj = json.loads(line)
            id = int(loadedObj["id"])

            if id not in combinedTexts or id in alreadyDone:
                continue

            pbar.set_description(f"Processing {id}: len={len(loadedObj['graph']['nodeDataArray'])}")

            loadedObj["graph"]["nodeDataArray"], _ = get_LM_score(
                loadedObj["graph"]["nodeDataArray"], combinedTexts[id]
            )

            with open(outputPath, "a") as outputf:
                json.dump(loadedObj, outputf)
                outputf.write("\n")

            processed_lines += 1
        except Exception as e:
            print("Error", e)
            continue
