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


jsonlFile = '/scratch/rahul.garg/allCmgs_llava_Target_h2_RelCombined.jsonl'
jsonFile = open(jsonlFile, 'r')
# allLines = jsonFile.readlines()
# jsonFile.close()


# In[ ]:


# for i in tqdm(range(len(allLines))):
#     allLines[i] = json.loads(allLines[i])


# In[ ]:


k = 750


# In[ ]:


lengths = []
edgeLengths = []
for txtLine in tqdm(jsonFile, total=12126):
    obj = json.loads(txtLine)
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

    edgeLengths.append(len(edgesSet))

    

plt.hist(lengths, bins=100)
# make a line at 200
plt.axvline(x=k, color='r', linestyle='--')
plt.show()

plt.hist(edgeLengths, bins=100)
plt.show()


# In[ ]:


# sum of lengths
sum(lengths) / len(lengths) 


# In[ ]:


sum(edgeLengths) / len(edgeLengths)


# In[ ]:


scores = []

# seek to the beginning of the file
jsonFile.seek(0)

for text in tqdm(jsonFile, total=12126):
    obj = json.loads(text)
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


# RelScores = {}
# for obj in tqdm(allLines):
#     graph = obj['graph']
#     nodes = graph['nodeDataArray']
#     if obj['id'] not in RelScores:
#         RelScores[obj['id']] = {}

#     for node in nodes:
#         if node['key'] not in RelScores[obj['id']]:
#             RelScores[obj['id']][node['key']] = node['score']
#         elif abs(RelScores[obj['id']][node['key']] - node['score']) > 1e-3:
#             print("Duplicate key")
#             print(node)
#             print(obj['id'])
#             print(RelScores[obj['id']][node['key']])

# RelScores


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

with open('../entityExtractionQagnn/ground_res_new_target.jsonl', 'r') as f:
    allCompulsoryNodes = f.readlines()
    for i in range(len(allCompulsoryNodes)):
        obj = json.loads(allCompulsoryNodes[i])
        id2Concepts[obj['id']] = obj['res']['qc']

id2Concepts


# In[ ]:


uniqueAbsents = set()
jsonFile.seek(0)
for text in tqdm(jsonFile, total=12126):
    obj = json.loads(text)
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


jsonFile.seek(0)
# print keys of first graph
obj = json.loads(jsonFile.readline())
print(obj['graph'].keys())


# In[ ]:


# make new file keeping only top k nodes
jsonFile.seek(0)
lengths = []
edgeLengths = []
finalObjs = []
for txt in tqdm(jsonFile, total=12126):
    obj = json.loads(txt)
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
    # add to edgeLengths
    edgeLengths.append(len(newLinks))
    graph['linkDataArray'] = newLinks
    # remove contextEmbed from graph
    if 'contextEmbed' in graph:
        del graph['contextEmbed']

    finalObjs.append(obj)

plt.hist(lengths, bins=100)
plt.show()

plt.hist(edgeLengths, bins=100)
plt.show()


# In[ ]:


jsonFile.seek(0)


# In[ ]:


newJsonlFile = f'/scratch/rahul.garg/llava_Target_h2RelKmgsTop{k}_noEmbed.jsonl'
wjsonFile = open(newJsonlFile, 'w')
for line in tqdm(finalObjs):
    json.dump(line, wjsonFile)
    wjsonFile.write('\n')
wjsonFile.close()

