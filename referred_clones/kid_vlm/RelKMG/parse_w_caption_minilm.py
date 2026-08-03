import os
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/scratch/rahul.garg/stCache"
os.makedirs(os.environ["SENTENCE_TRANSFORMERS_HOME"], exist_ok=True)
os.environ["TORCH_HOME"] = "/scratch/rahul.garg/torchCache"
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = "/scratch/rahul.garg/hfCache"
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.environ["HF_HOME"] = "/scratch/rahul.garg/hfCache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
from tqdm import tqdm
import csv
import pandas as pd
import networkx as nx
import pickle as pkl
import matplotlib.pyplot as plt
import multiprocessing
from icecream import ic
import numpy as np
import string
import json
import random
import time
from sentence_transformers import SentenceTransformer, util
import torch
from pathlib import Path

random.seed(time.time())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
inputFile = open('/scratch/rahul.garg/conceptNet/conceptnet.en.csv', 'r', encoding='utf-8')

pdDF = pd.read_csv(inputFile, sep='\t', header=None, names=['relation', 'start', 'end', 'weight'])
pdDF.head()

def drawGraph(GD):
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(GD)
    nx.draw_networkx_nodes(GD, pos, node_size=700)
    nx.draw_networkx_labels(GD, pos, font_size=12, font_color="black")
    nx.draw_networkx_edges(GD, pos, width=2, arrowstyle='->', arrowsize=20)
    edge_labels = nx.get_edge_attributes(GD, 'relation')
    nx.draw_networkx_edge_labels(GD, pos, edge_labels=edge_labels, font_color='red')
    plt.axis('off')  # Turn off the axis
    plt.show()

node_color = '#ec8c69'
edge_color = 'yellow'


def getObj(graph):
    nodeDataArray = []
    linkDataArray = []

    # Define colors for entity and relation nodes
    entity_color = "#ec8c69"
    relation_color = "yellow"

    # Add entity nodes to nodeDataArray
    nodesDone = set()
    for node in graph.nodes:
        if node not in nodesDone:
            nodeDataArray.append({"key": node, "color": entity_color})
            nodesDone.add(node)

    # Convert edges to relation nodes and add connections to linkDataArray
    edgesDone = set()
    for edge_id, (source, target) in enumerate(graph.edges):
        relation_node_key = graph.edges[(source, target)]['relation']

        # Add the relation node to nodeDataArray
        if relation_node_key not in nodesDone:
            nodeDataArray.append({"key": relation_node_key, "color": relation_color})
            nodesDone.add(relation_node_key)

        # Add a link from the source node to the relation node
        if (source, relation_node_key) not in edgesDone:
            linkDataArray.append({"from": source, "to": relation_node_key})
            edgesDone.add((source, relation_node_key))


        # Add a link from the relation node to the target node
        if (relation_node_key, target) not in edgesDone:
            linkDataArray.append({"from": relation_node_key, "to": target})
            edgesDone.add((relation_node_key, target))


    return {"class": "GraphLinksModel", "nodeDataArray": nodeDataArray, "linkDataArray": linkDataArray}

def getGraphObject(graph, node_scores):
    """
    Convert the graph into a format suitable for saving, adding the node similarity scores.
    
    :param graph: The networkx graph.
    :param node_scores: A dictionary where the key is the node and the value is the score.
    :return: A dictionary representing the graph with nodes and links.
    """
    nodeDataArray = []
    linkDataArray = []

    # Define colors for entity and relation nodes
    entity_color = "#ec8c69"

    # Add entity nodes to nodeDataArray, including the node score
    nodesDone = set()
    for node in graph.nodes:
        if node not in nodesDone:
            nodeDataArray.append({
                "key": node, 
                "color": entity_color,
                "score": node_scores.get(node, 0.0)  # Get the score from node_scores, default to 0.0 if not found
            })
            nodesDone.add(node)

    # Add edges with detailed link information
    for source, target in graph.edges:
        relation = graph.edges[(source, target)]['relation']
        linkDataArray.append({"from": source, "to": target, "relation": relation})

    return {"class": "GraphLinksModel", "nodeDataArray": nodeDataArray, "linkDataArray": linkDataArray}


# ic(getObj(subG))
# drawGraph(subG)

# build
if not os.path.exists('/scratch/rahul.garg/conceptNet/conceptnet.en.pkl'):
    G = nx.DiGraph()
    for index, row in tqdm(pdDF.iterrows(), total=len(pdDF)):
        G.add_edge(str(row['start']), str(row['end']), weight=row['weight'], relation=row['relation'])
    with open('/scratch/rahul.garg/conceptNet/conceptnet.en.pkl', 'wb') as f:
        pkl.dump(G, f)
else:
    with open('/scratch/rahul.garg/conceptNet/conceptnet.en.pkl', 'rb') as f:
        G = pkl.load(f)

if not os.path.exists('/scratch/rahul.garg/conceptNet/concNetVocab.pkl'):
    vocab = set()
    for edge in G.edges(data=True):
        # if not nan
        if edge[0] != edge[0]:
            continue
        if edge[1] != edge[1]:
            continue

        vocab.add(str(edge[0]))
        vocab.add(str(edge[1]))
    with open('/scratch/rahul.garg/conceptNet/concNetVocab.pkl', 'wb') as f:
        pkl.dump(vocab, f)
else:
    with open('/scratch/rahul.garg/conceptNet/concNetVocab.pkl', 'rb') as f:
        vocab = pkl.load(f)

# ic(len(vocab))

adjMat = {}
for node in G.nodes():
    # skip if nan
    if node != node:
        continue
    adjMat[node] = {}
    for neighbor in G.neighbors(node):
        adjMat[node][neighbor] = 1

        if neighbor not in adjMat:
            adjMat[neighbor] = {}
        adjMat[neighbor][node] = 1
        

adjMat
writeFreq = 500
iter = 0 

def append_to_jsonl(file_path, data):
    with open(file_path, 'a', encoding='utf-8') as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

# blacklist = set(["uk", "us", "take", "make", "object", "person", "people"])
blacklist = set()

# Step 1: Identify edges to remove
# dont remove nans, gives key error 
edges_to_remove = [(u, v) for u, v, attrs in G.edges(data=True) if attrs.get('relation') == 'hascontext' and u == u and v == v]

# Step 2: Remove the identified edges
# print edges length
# print(len(G.edges))
G.remove_edges_from(edges_to_remove)
# print(len(G.edges))

def get_multihop_neighbors_with_minilm(graph, query_sentence, model, top_k=20, max_hops=2):
    """
    Get multi-hop neighbors based on top-k similar nodes to the query sentence using MiniLM.
    :param graph: The graph from which to retrieve neighbors
    :param query_sentence: A single query sentence to rank nodes
    :param top_k: Number of top-k similar nodes to select
    :param max_hops: Number of hops to explore in the graph
    :return: Dictionary of neighbors and their scores.
    """

    # Rank all nodes based on the query sentence and get both nodes and scores
    ranked_nodes_with_scores = rank_nodes_by_similarity(query_sentence, model, list(graph.nodes), top_k)

    # Initialize the dictionary of neighbors and their scores using the top-k results
    neighbors_with_scores = {node: score for node, score in ranked_nodes_with_scores}
    visited = set(neighbors_with_scores.keys())

    frontier = list(neighbors_with_scores.keys())

    all_neighbors = set()  # To store all neighbors encountered during multi-hop exploration

    # Collect all neighbors during multi-hop exploration
    try:
        for _ in range(max_hops):
            new_frontier = []
            for node in frontier:
                for neighbor in graph.neighbors(node):
                    # Ensure the neighbor hasn't been visited and is not blacklisted
                    if neighbor not in visited and neighbor not in blacklist and neighbor == neighbor:
                        all_neighbors.add(neighbor)
                        visited.add(neighbor)
                        new_frontier.append(neighbor)
            frontier = new_frontier
    except Exception as e:
        ic(e)
        ic(query_sentence)
        ic(node)
        ic(frontier)
        raise e

    # Combine all neighbors from exploration with the top-k nodes
    all_neighbors = list(all_neighbors)
    neighbors_to_score = all_neighbors + list(neighbors_with_scores.keys())

    # Fetch query embedding
    query_embedding = model.encode([query_sentence], convert_to_tensor=True, device=device)

    # Get embeddings for all neighbors (from cache) and compute cosine similarity in a batch
    if neighbors_to_score:
        neighbor_embeddings = torch.stack([nodesEmbCache[neighbor].to(device) for neighbor in neighbors_to_score])
        cosine_scores = util.pytorch_cos_sim(query_embedding, neighbor_embeddings)[0]  # Cosine similarity

        # Update scores for all neighbors based on cosine similarity
        for i, neighbor in enumerate(neighbors_to_score):
            neighbors_with_scores[neighbor] = float(cosine_scores[i].item())

    return neighbors_with_scores  # Dictionary of {neighbor: score}




desc = 'miniLM'
def process_graphs(temp_folder, subset, process_id, blacklist, G, hops, top_k=20):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model = model.to(device)
    output_file = f'{temp_folder}/allCmgs{desc}_h{hops}_{process_id}.jsonl'

    for id, capt in tqdm(subset, desc=f"Processing {process_id}", total=len(subset), position=process_id, leave=False):
        # Get top-k similar nodes and multi-hop neighbors with their scores
        neighbours_with_scores = get_multihop_neighbors_with_minilm(G, capt, model=model, top_k=top_k, max_hops=hops)

        # Extract the nodes and their scores
        all_nodes = list(neighbours_with_scores.keys())
        all_scores = neighbours_with_scores

        # Create a subgraph based on the selected nodes
        final_dgl = G.subgraph(all_nodes)

        # Pass the subgraph and the scores to getGraphObject
        graph_obj = getGraphObject(final_dgl, all_scores)

        # Create the output object
        main_obj = {"id": id, "graph": graph_obj}

        # Save the result to the jsonl file
        append_to_jsonl(output_file, [main_obj])
    
    print(f"Process {process_id} finished", flush=True)



def split_data(data, n_parts):
    """Splits data into n_parts roughly equal parts."""
    k, m = divmod(len(data), n_parts)
    return [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_parts)]

# num_processes = multiprocessing.cpu_count()

num_processes = 10
captionsPath = '/home/rahul.garg/llava/newCaptionsLLAVA.json'
captions = {}
with open(captionsPath, 'r') as f:
    dat = json.load(f)
    for key in dat:
        keyInt = int(key.split('.')[0])
        captions[keyInt] = dat[key]

# for line in open(captionsPath, 'r'):
#     line = json.loads(line)
#     line['name'] = int(line['name'])
#     captions[line['name']] = line['text']


for file in Path('/scratch/rahul.garg/hateful_memes').glob('*.jsonl'):
    for line in open(file, 'r'):
        line = json.loads(line)
        id = int(line['id'])
        if id in captions:
            captions[id] = line['text'] + ' ' + captions[id]
        else:
            print(f"ID {id} not found in captions")
    
nodesEmbCache = None

def rank_nodes_by_similarity(query_sentence, model, all_nodes, top_k=20):
    """
    Rank all nodes in the graph by semantic similarity to a single query sentence using MiniLM.
    :param query_sentence: A single query sentence (could be multiple sentences concatenated)
    :param all_nodes: List of all possible nodes to rank
    :param top_k: Number of top-ranked nodes to return
    :return: List of top_k most similar nodes
    """
    global nodesEmbCache
    query_embedding = model.encode([query_sentence], convert_to_tensor=True, device=device)

    # Initialize the cache if it doesn't exist
    if nodesEmbCache is None:
        # Check if embeddings are saved in a dictionary form
        if not os.path.exists('/scratch/rahul.garg/conceptNet/all_node_embeddings.pkl'):
            embeddings = {}
            batch_size = 1000
            for i in tqdm(range(0, len(all_nodes), batch_size), desc="Encoding nodes"):
                batch_nodes = all_nodes[i:i + batch_size]
                batch_embeddings = model.encode(batch_nodes, convert_to_tensor=True, device=device)
                
                # Store each node's embedding in the dictionary
                for node, embedding in zip(batch_nodes, batch_embeddings):
                    embeddings[node] = embedding.cpu()

            # Save the dictionary to a pickle file
            with open('/scratch/rahul.garg/conceptNet/all_node_embeddings.pkl', 'wb') as f:
                pkl.dump(embeddings, f)
        else:
            # Load the dictionary of embeddings from the pickle file
            with open('/scratch/rahul.garg/conceptNet/all_node_embeddings.pkl', 'rb') as f:
                embeddings = pkl.load(f)

        # Move embeddings to the correct device and cache them
        nodesEmbCache = {node: embedding.to(device) for node, embedding in embeddings.items()}

    # Use the cached dictionary
    all_node_embeddings = torch.stack([nodesEmbCache[node] for node in all_nodes])

    # Compute the cosine similarity between the query and all nodes
    cosine_scores = util.pytorch_cos_sim(query_embedding, all_node_embeddings)

    # Get the top_k highest scoring nodes
    top_results = torch.topk(cosine_scores[0], k=top_k)

    # Move indices back to CPU for easy access (if necessary)
    ranked_nodes = [(all_nodes[i], float(score)) for i, score in zip(top_results.indices.cpu(), top_results.values.cpu())]

    return ranked_nodes 


if __name__ == '__main__':
    
    temp_folder = None
    while temp_folder is None or os.path.exists(temp_folder):
        temp_folder = f'/scratch/rahul.garg/conceptNet/tmp/{"".join(random.choices(string.ascii_letters + string.digits, k=10))}'

    print(f"Using temp folder: {temp_folder}", flush=True)
    os.makedirs(temp_folder)
    try:
        multiprocessing.set_start_method('spawn', force=True)
        captInputs = []
        for id in captions:
            captInputs.append((id, captions[id]))

        # exit(0)
        cmg_parts = split_data(captInputs, num_processes)

        processes = []
        hops = 2
        top_k_init = 700
        for i in range(num_processes):
            p = multiprocessing.Process(target=process_graphs, args=(temp_folder, cmg_parts[i], i, blacklist, G, hops, top_k_init))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        # use cat to concatenate all the files, move it outside temp and delete temp with all files
        time.sleep(10)
        os.system(f'cat {temp_folder}/* > /scratch/rahul.garg/conceptNet/allCmgs_{desc}_h{hops}_k{top_k_init}.jsonl')
        os.system(f'rm -rf {temp_folder}')
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        
        for p in processes:
            p.terminate()
            p.join()

        os.system(f'rm -rf {temp_folder}')
        exit(0)
