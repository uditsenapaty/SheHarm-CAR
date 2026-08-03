#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
from tqdm import tqdm
from pathlib import Path
from icecream import ic
from collections import OrderedDict
import sys
import pickle as pkl
import matplotlib.pyplot as plt
import networkx as nx
import copy


# In[ ]:




# In[ ]:


jsonlFile = '/scratch/rahul.garg/noTarget_Rel_Cmgs.jsonl'
jsonFile = open(jsonlFile, 'r')
allLines = jsonFile.readlines()
jsonFile.close()


# In[ ]:


for i in tqdm(range(len(allLines))):
    allLines[i] = json.loads(allLines[i])


# In[ ]:


lengths = []

for obj in tqdm(allLines):
    graph = obj['graph']

    nodes = graph['nodeDataArray']
    # print(len(nodes))
    # break
    nodesSet = set()
    # print("________________")
    for node in nodes:
        if node['key'] in nodesSet:
            print(node)
        nodesSet.add(node['key'])
    lengths.append(len(nodesSet))
    # print("________________")
    # verify edge duplicatio
    edges = graph['linkDataArray']
    edgesSet = set()
    for edge in edges:
        tup = (edge['from'], edge['to'])
        if tup in edgesSet:
            print(edge)
        edgesSet.add(tup)

    

plt.hist(lengths, bins=100)
# make a line at 200
plt.axvline(x=200, color='r', linestyle='--')
plt.show()


# In[ ]:


# sum of lengths
sum(lengths) / len(lengths) 


# In[ ]:


scores = []

for obj in tqdm(allLines):
    graph = obj['graph']

    nodes = graph['nodeDataArray']

    for node in nodes:
        score = node['score']
        scores.append(score)

plt.hist(scores, bins=100)
plt.show()

    


# In[ ]:


sum(scores) / len(scores)


# In[ ]:


RelScores = {}
for obj in tqdm(allLines):
    graph = obj['graph']
    nodes = graph['nodeDataArray']
    if obj['id'] not in RelScores:
        RelScores[obj['id']] = {}

    for node in nodes:
        if node['key'] not in RelScores[obj['id']]:
            RelScores[obj['id']][node['key']] = node['score']
        elif abs(RelScores[obj['id']][node['key']] - node['score']) > 1e-3:
            print("Duplicate key")
            print(node)
            print(obj['id'])
            print(RelScores[obj['id']][node['key']])

RelScores


# In[ ]:


# newFile = '/scratch/rahul.garg/conceptNet/allCmgs_fullContextFixed_h1.jsonl'
# # add these scores to its nodes
# jsonFile = open(newFile, 'r')
# allLines2 = jsonFile.readlines()
# jsonFile.close()

# for i in tqdm(range(len(allLines2))):
#     allLines2[i] = json.loads(allLines2[i])


# In[ ]:


# allLines2[0]


# In[ ]:


# for obj in tqdm(allLines2):
#     graph = obj['graph']
#     nodes = graph['nodeDataArray']
#     for node in nodes:
#         node['score'] = RelScores[obj['id']][node['key']]

# # newFile = '/scratch/rahul.garg/conceptNet/allCmgs_fullContextFixed_h1_withScores.jsonl'
# # jsonFile = open(newFile, 'w')
# # for obj in tqdm(allLines2):
# #     json.dump(obj, jsonFile)
# #     jsonFile.write('\n')


# In[ ]:


id2Concepts = {}

with open('../entityExtractionQagnn/ground_res_without_target.jsonl', 'r') as f:
    allCompulsoryNodes = f.readlines()
    for i in range(len(allCompulsoryNodes)):
        obj = json.loads(allCompulsoryNodes[i])
        id2Concepts[obj['id']] = obj['res']['qc']

id2Concepts


# In[ ]:


uniqueAbsents = set()

for obj in tqdm(allLines):
    # make copy
    id = obj['id']
    graph = obj['graph']

    nodes = graph['nodeDataArray']
    nodesPresent = set()
    # verify that all nodes are present
    for node in nodes:
        nodesPresent.add(node['key'])

    for key in id2Concepts[id]:
        if key not in nodesPresent:
            uniqueAbsents.add(key)
            # print("Node not present")
            # print(key)
            # print(id)

uniqueAbsents


# In[ ]:


# make new file keeping only top k nodes
k = 200
lengths = []

for obj in tqdm(allLines):
    # make copy
    id = obj['id']
    graph = obj['graph']

    nodes = graph['nodeDataArray']
    # print(len(nodes))
    # nodes = sorted(nodes, key=lambda x: x['score'], reverse=True)
    # keep all relation nodes
    # relNodes = [node for node in nodes if 'color' not in node or node['color'] == 'yellow']
    # objNodes = [node for node in nodes if 'color' in node and node['color'] != 'yellow']
    # objNodes = sorted(objNodes, key=lambda x: x['score'], reverse=True)
    nodes = sorted(nodes, key=lambda x: x['score'], reverse=True)
    nodesFiltered = nodes[:k]
    # check if any compulsory node is after k and add it
    for i in range(k, len(nodes)):
        if nodes[i]['key'] in id2Concepts[id] or nodes[i]['key'] == 'QAcontext':
            nodesFiltered.append(nodes[i])
    # print(len(objNodes), len(relNodes))
    # nodes = relNodes + objNodes
    # print(len(nodes))

    # assert that all are unique
    # nodesSet = set()
    # for node in nodesFiltered:
    #     if node['key'] in nodesSet:
    #         print(node)
    #     nodesSet.add(node['key'])
    lengths.append(len(nodesFiltered))


    

    # linkDataArray.append({"from": source, "to": target, "relation": relation})
    
    # also update the links
    nodeKeys = set()
    for node in nodesFiltered:
        nodeKeys.add(node['key'])
    links = graph['linkDataArray']
    newLinks = []
    for link in links:
        if (link['from'] in nodeKeys and link['to'] in nodeKeys):
            newLinks.append(link)

    # connect QAcontext to all compulsory nodes
    assert 'QAcontext' in nodeKeys
    for key in id2Concepts[id]:
        if key in nodeKeys:
            newLinks.append({"from": "QAcontext", "to": key, "relation": "QAcontextRel"})
    graph['nodeDataArray'] = nodesFiltered
    graph['linkDataArray'] = newLinks


plt.hist(lengths, bins=100)
plt.show()


# In[ ]:


allLines[0]['graph']['nodeDataArray']


# In[ ]:


newJsonlFile = f'/scratch/rahul.garg/no_target_h1RelKmgsTop{k}.jsonl'
jsonFile = open(newJsonlFile, 'w')
for line in tqdm(allLines, total=len(allLines), desc='Writing'):
    json.dump(line, jsonFile)
    jsonFile.write('\n')
jsonFile.close()

