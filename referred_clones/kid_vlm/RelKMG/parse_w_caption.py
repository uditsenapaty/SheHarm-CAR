from tqdm import tqdm
import csv
import pandas as pd
import networkx as nx
import pickle as pkl
import os
import matplotlib.pyplot as plt
import multiprocessing
from icecream import ic
import numpy as np
import string
import json
import random
import time

random.seed(time.time())

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

# with open('../cmg_combined_0-12007_latest.pkl', 'rb') as f:
#     cmg = pkl.load(f)
#     print(type(cmg))

# with open('./cmgs/cmg_6000-12000_new.pkl', 'rb') as f:
#     cmg += pkl.load(f)
#     print(type(cmg))

# with open('./cmgs/cmg_12000-18000_new.pkl', 'rb') as f:
#     cmg += pkl.load(f)
#     print(type(cmg))

# subG = cmg[0][1]

# thingsDone = set()
# for id, _ in cmg:
#     thingsDone.add(int(id.split('.')[0]))

# ic(len(thingsDone))


# visualize, it is networkx

node_color = '#ec8c69'
edge_color = 'yellow'

# sampleobj = {
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

def getGraphObject(graph):
    nodeDataArray = []
    linkDataArray = []

    # Define colors for entity and relation nodes
    entity_color = "#ec8c69"

    # Add entity nodes to nodeDataArray
    nodesDone = set()
    for node in graph.nodes:
        if node not in nodesDone:
            nodeDataArray.append({"key": node, "color": entity_color})
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
    for edge in tqdm(G.edges(data=True), total=len(G.edges)):
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

ic(len(vocab))

adjMat = {}
for node in tqdm(G.nodes()):
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
print(len(G.edges))
G.remove_edges_from(edges_to_remove)
print(len(G.edges))

def get_multihop_neighbors(graph, initial_nodes, max_hops):
    visited = set(initial_nodes)
    frontier = list(initial_nodes)
    try:
        for _ in range(max_hops):
            new_frontier = []
            for node in frontier:
                for neighbor in graph.neighbors(node):
                    # check for nan too
                    if neighbor not in visited and neighbor not in blacklist and neighbor == neighbor:
                        visited.add(neighbor)
                        new_frontier.append(neighbor)
            frontier = new_frontier
    except Exception as e:
        ic(e)
        ic(initial_nodes)
        ic(node)
        ic(frontier)
        raise e

    return list(visited)

# for id, subG in tqdm(cmg, desc="Processing", total=len(cmg)):
#     iter += 1
#     nodes = [node.lower() for node in subG.nodes() if node in G and node not in blacklist]
#     neighbours = get_multihop_neighbors(G, nodes, 2)  # Get up to 3-hop neighbors

#     weights = []
#     # add edge where there is edge in cpnet
#     allNodes = nodes + neighbours
#     allNodes = list(set(allNodes))
#     finalDgl = nx.DiGraph(G.subgraph(allNodes)) 
#     # print(len(allNodes), len(finalDgl.nodes()), len(finalDgl.edges()))
            
            

#     graph_obj = getObj(finalDgl)
#     mainObj = {"id": id, "graph": graph_obj}
#     append_to_jsonl('./allCmgs_h2.jsonl', [mainObj])

temp_folder = None
desc = 'llava_targets_and_caption'
def process_graphs(subset, process_id, blacklist, G, hops):
    global temp_folder
    output_file = f'{temp_folder}/allCmgs{desc}_h{hops}_{process_id}.jsonl'
    for id, extNodes in tqdm(subset, desc="Processing", total=len(subset), position=process_id, leave=False):
        nodes = [node.lower() for node in extNodes if node.lower() in G and node.lower() not in blacklist]
        neighbours = get_multihop_neighbors(G, nodes, hops)  # Get up to 3-hop neighbors

        all_nodes = list(set(nodes + neighbours))
        final_dgl = G.subgraph(all_nodes)
                
        graph_obj = getGraphObject(final_dgl)
        main_obj = {"id": id, "graph": graph_obj}
        append_to_jsonl(output_file, [main_obj])
    print("Done", flush=True)


def split_data(data, n_parts):
    """Splits data into n_parts roughly equal parts."""
    k, m = divmod(len(data), n_parts)
    return [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_parts)]

#"".join(random.choices(string.ascii_letters + string.digits, k=delim_k))
# make a new temp folder where to store the files, ensure it doesnt already exist using a while loop

while temp_folder is None or os.path.exists(temp_folder):
    temp_folder = f'/scratch/rahul.garg/conceptNet/tmp/{"".join(random.choices(string.ascii_letters + string.digits, k=10))}'

print(f"Using temp folder: {temp_folder}", flush=True)
os.makedirs(temp_folder)

num_processes = multiprocessing.cpu_count()
# num_processes = 1
# captionsPath = '/home/rahul.garg/modelRuns/completeCaptionsFinalAll.jsonl'
# captions = {}
# for line in open(captionsPath, 'r'):
#     line = json.loads(line)
#     line['name'] = int(line['name'])
#     # check dup
#     # if line['name'] in captions:
#     #     print("Duplicate found")
#     #     # new and old
#     #     print(line)
#     #     print(captions[line['name']])
        
#     captions[line['name']] = line['text']

# verify that all captions have a corresponding nodeDict
id2Node = {}
allEntities = open('/home/rahul.garg/entityExtractionQagnn/ground_res_new_target_llava_capt.jsonl', 'r')
for line in allEntities:
    line = json.loads(line)
    id2Node[line['id']] = line['res']['qc'] # + [line['res']['sent']]

# for key in captions:
#     if captions[key] not in nodeDict:
#         print("Missing", captions[key])
#         continue

#     id2Node[key] = nodeDict[captions[key]]

# # convert into a list of tuples (id, nodes)
id2Node = [(key, val) for key, val in id2Node.items()]


ic(len(id2Node))
ic(id2Node[:5])
# exit(0)
cmg_parts = split_data(id2Node, num_processes)

processes = []
hops = 1
for i in range(num_processes):
    p = multiprocessing.Process(target=process_graphs, args=(cmg_parts[i], i, blacklist, G, hops))
    processes.append(p)
    p.start()

for p in processes:
    p.join()

# use cat to concatenate all the files, move it outside temp and delete temp with all files
time.sleep(10)
os.system(f'cat {temp_folder}/* > /scratch/rahul.garg/conceptNet/allCmgs_{desc}_h{hops}.jsonl')
os.system(f'rm -rf {temp_folder}')
