import time
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    MMBTConfig,
    MMBTModel,
    MMBTForClassification,
    get_linear_schedule_with_warmup,
)
import os
os.environ['TRANSFORMERS_CACHE'] = '/staging/users/tpadhi1/rahul/tmp/hfCache'
os.environ['HF_HOME'] = '/staging/users/tpadhi1/rahul/tmp/hfCache'

import transformers
import copy
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle as pkl
import math
import pickle
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Dataset
from PIL import Image
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, RGCNConv, global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
import torch
from sklearn.metrics import classification_report
import numpy as np
import random
from collections import Counter
import json
import warnings
torch.set_num_threads(1)
import multiprocessing

# warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*ANTIALIAS is deprecated.*")



# import torchvision
# import torchvision.transforms as transforms


# triplesObj = {}
# triplesFile = open('./triples_Summaries.jsonl', 'r', encoding='utf-8')
# for line in tqdm(triplesFile, desc="Reading triples"):
#     obj = json.loads(line)
#     triplesObj[int(obj['name'])] = obj['summary']

# triplesFile.close()

# Printing only the first few key-value pairs from the triplesObj
# first_few_pairs = {k: triplesObj[k] for k in list(triplesObj.keys())[:5]}
# print(first_few_pairs)

# captionObj = {}
# captionFile = open('./completeCaptionsFinalAll.jsonl', 'r', encoding='utf-8')
# for line in tqdm(captionFile, desc="Reading captions"):
#     obj = json.loads(line)
#     captionObj[int(obj['name'])] = obj['text']

# captionFile.close()

# print(captionObj)

scenes = {}

graphFile = open('/staging/users/tpadhi1/rahul/llava_Target_and_caption_minilm_h1RelKmgsTop250_noEmbed.jsonl',
                 'r', encoding='utf-8')
for line in tqdm(graphFile, desc="Reading graphs"):
    obj = json.loads(line)
    scenes[int(obj['id'])] = obj['graph']
# name = int(file.parent.name)
# with open(file, 'r', encoding='utf-8') as f:
# obj = json.load(f)
# scenes[name] = obj

allRels = set()
for key in tqdm(scenes, desc="Getting all relations"):
    for node in scenes[key]['linkDataArray']:
        try:
            allRels.add(node['relation'])
        except Exception as e:
            print(node)
            raise e
            exit(0)
allRels = list(allRels)

# create idx2rel and rel2idx
rel2idx = {rel: idx for idx, rel in enumerate(allRels)}
idx2rel = {idx: rel for idx, rel in enumerate(allRels)}


# glove_path = '/scratch/rahul.garg/glove.6B/glove.6B.300d.txt'
# glove_embeddings = load_glove_embeddings(glove_path)

device = torch.device("cpu")
# device = "cpu"
print("Device:", device, flush=True)

# clip_model, preprocess = clip.load(
#     "RN50x4", device=device, jit=False, download_root='/scratch/rahul.garg/hfCache')

# for p in clip_model.parameters():
#     p.requires_grad = False

# num_image_embeds = 4
# num_labels = 1
# gradient_accumulation_steps = 5
# data_dir = '/scratch/rahul.garg/hateful_memes'
# max_seq_length = 500
# max_grad_norm = 0.5
# train_batch_size = 4
# eval_batch_size = 4
# image_encoder_size = 288
# image_features_size = 640
# num_train_epochs = 1

####################################################################################

ent_text = 'concept.txt'

# loaded_array = np.load('concept.nb.npy')
loaded_array = np.load('glove.transe.sgd.ent.npy')


word_list = []
with open(ent_text, 'r') as file:
    for line in file:
        word = line.strip()
        if word:
            word_list.append(word)

known = 0
unknown = 0

# if os.path.exists('/scratch/rahul.garg/gnnCache.pkl'):
#     with open('/scratch/rahul.garg/gnnCache.pkl', 'rb') as f:
#         print("Loading cache", flush=True)
#         graphCache = pkl.load(f)
graphPaths = '/staging/users/tpadhi1/rahul/tmp/gnnCache_llava_target_caption_250_h1/'  # contains id.pkl files
os.makedirs(graphPaths, exist_ok=True)

graphCache = {}
print(f"Loading {len(os.listdir(graphPaths))} graphs", flush=True)

for file in tqdm(os.listdir(graphPaths)):
    with open(os.path.join(graphPaths, file), 'rb') as f:
        obj = pkl.load(f)
        graphCache[int(file.split('.')[0])] = obj


def convert_to_rgcnn_data_with_transe(obj, id=None, rel2idx=rel2idx):
    global known, unknown
    # not none and in cache
    # print("This 2ran")
    # start_time = time.time()
    if id is not None and id in graphCache:
        # print("Cache hit", flush=True)
        return graphCache[id]
    node_data = obj['nodeDataArray']
    edge_data = obj['linkDataArray']

    # Separate entity and relation nodes based on color
    # entity_nodes = {node['key']: i for i, node in enumerate(node_data) if node['color'] == '#ec8c69'}
    entity_nodes = {}
    entity_idx = 0  # Counter for entity nodes to ensure sequential indices

    # Separate entity and relation nodes based on color and assign sequential indices to entity nodes
    for node in node_data:
        if node['key'] in entity_nodes:
            continue

        entity_nodes[node['key']] = entity_idx
        entity_idx += 1

    num_entities = len(entity_nodes)

    # Initialize entity node features with zeros for the case where the GloVe embedding is not found
    # Assuming 300-dimensional GloVe embeddings
    x = torch.zeros((num_entities, 100))

    # Map each entity to its GloVe embedding
    for key, idx in entity_nodes.items():
        # embedding = glove_embeddings.get(key.lower(), np.zeros(300))  # Use zero vector for OOV words
        # print(len(embedding), flush=True)
        try:
            embedding = loaded_array[word_list.index(
                key.lower().replace(' ', '_'))]
            known += 1
        except:
            embedding = np.zeros(100)
            unknown += 1

        if idx >= num_entities:
            print("Index out of range", idx, num_entities,
                  key, entity_nodes, node_data, flush=True)
        x[idx] = torch.tensor(embedding)

    # Initialize edge indices and edge types
    edge_index = [[], []]
    edge_type = []

    # before_edge_time = time.time()

    for edge in edge_data:
        # if edge['from'] in entity_nodes and edge['to'] in relation_nodes:
        # This edge defines a relation from an entity to a relation node
        # for next_edge in edge_data:
        # if next_edge['from'] == edge['to'] and next_edge['to'] in entity_nodes:
        # This edge connects the relation node to another entity
        source_index = entity_nodes[edge['from']]
        target_index = entity_nodes[edge['to']]
        edge_index[0].append(source_index)
        edge_index[1].append(target_index)
        # Use the global rel2idx mapping
        edge_type.append(rel2idx[edge['relation']])

    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)

    # Create a PyTorch Geometric Data object with node features, edge indices, and edge types
    data = Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_type_tensor)
    if id is not None:
        graphCache[id] = data

        with open(os.path.join(graphPaths, f"{id}.pkl"), 'wb') as f:
            pkl.dump(data, f)

    # end_time = time.time()
    # # print all 3 times
    # print(f"Time taken for edge processing: {end_time - before_edge_time}")
    # print(f"Time taken for node processing: {before_edge_time - start_time}")
    # print(f"Time taken for whole graph: {end_time - start_time}")
    return data

####################################################################################


# # Example usage
# obj = {
#     "class": "GraphLinksModel",
#     "nodeDataArray": [
#         {"key": "photo", "color": "#ec8c69"},
#         {"key": "man", "color": "#ec8c69"},
#         {"key": "mustache", "color": "#ec8c69"},
#         {"key": "of", "color": "yellow"},
#         {"key": "with", "color": "yellow"}
#     ],
#     "linkDataArray": [
#         {"from": "photo", "to": "of"},
#         {"from": "of", "to": "man"},
#         {"from": "man", "to": "with"},
#         {"from": "with", "to": "mustache"}
#     ]
# }

# gnn_data = convert_to_rgcnn_data_with_transe(obj)
# print(gnn_data)
def split_data(data, n_parts):
    """Splits data into n_parts roughly equal parts."""
    k, m = divmod(len(data), n_parts)
    return [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_parts)]


num_processes = multiprocessing.cpu_count()
total2beProcessed = len(scenes)
graphParts = split_data(list(scenes.keys()), num_processes)


def processGetGraphs(graphPart, pos, progress_event):
    # for key in tqdm(graphPart, position=pos):
    for key in graphPart:
        convert_to_rgcnn_data_with_transe(scenes[key], key)
        with progress_event.get_lock():
            progress_event.value += 1

progress_events = [multiprocessing.Value('i', 0) for _ in range(num_processes)]
for i in range(num_processes):
    p = multiprocessing.Process(
        target=processGetGraphs, args=(graphParts[i], i, progress_events[i]))
    p.start()
cur = 0
pbar = tqdm(total=total2beProcessed)
while cur < total2beProcessed:
    for i in range(num_processes):
        with progress_events[i].get_lock():
            cur += progress_events[i].value
            progress_events[i].value = 0
    pbar.update(cur - pbar.n)
    time.sleep(1)
pbar.close()
for i in range(num_processes):
    p.join()

exit(0)
