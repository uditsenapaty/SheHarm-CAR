import os
import csv
import json
from sentence_transformers import SentenceTransformer
from torch import nn
import torch
from tqdm import tqdm
import pickle as pkl

if not os.path.exists('/scratch/rahul.garg/sentence_transformers'):
    os.makedirs('/scratch/rahul.garg/sentence_transformers')
# Set the Sentence Transformers home directory
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/scratch/rahul.garg/sentence_transformers'

# Load the captions
with open('./completeCaptionsFinalAll.jsonl', 'r', encoding='utf-8') as captionsFile:
    captions = [json.loads(line) for line in captionsFile.readlines()]

# Load the triples
triples = []
with open('./conceptnet.en.csv', 'r', encoding='utf-8') as triplesFile:
    triplesObj = csv.reader(triplesFile)
    for row in triplesObj:
        splitted = row[0].split('\t')[:-1]
        splitted[0], splitted[1] = splitted[1], splitted[0]
        triples.append(splitted)

# Load a pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for captions
print("Computing embeddings for captions...")
caption_texts = [caption['text'] for caption in captions]
if not os.path.exists('/scratch/rahul.garg/caption_embeddings.pkl'):
    caption_embeddings = model.encode(caption_texts, convert_to_tensor=True, show_progress_bar=True)
    with open('/scratch/rahul.garg/caption_embeddings.pkl', 'wb') as f:
        pkl.dump(caption_embeddings, f)
else:
    print("Loading caption embeddings from file...")
    with open('/scratch/rahul.garg/caption_embeddings.pkl', 'rb') as f:
        caption_embeddings = pkl.load(f)


# Concatenate elements of each triple and compute embeddings
print("Computing embeddings for triples...")
triple_texts = [' '.join(triple) for triple in triples]
if not os.path.exists('/scratch/rahul.garg/triple_embeddings.pkl'):
    triple_embeddings = model.encode(triple_texts, convert_to_tensor=True, show_progress_bar=True)
    with open('/scratch/rahul.garg/triple_embeddings.pkl', 'wb') as f:
        pkl.dump(triple_embeddings, f)
else:
    print("Loading triple embeddings from file...")
    with open('/scratch/rahul.garg/triple_embeddings.pkl', 'rb') as f:
        triple_embeddings = pkl.load(f)

# Initialize the cosine similarity function
cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

# Assume k, model, caption_embeddings, and triple_embeddings are already defined and computed

k = 1000  # Number of closest triples to find
output_file = './closest_triples_with_names.jsonl'

# Load processed captions to skip if already done
processed_captions = set()
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            processed_caption = json.loads(line)
            processed_captions.add(processed_caption['name'])

# Append results to a JSONL file
with open(output_file, 'a', encoding='utf-8') as f:
    for idx, caption_embedding in tqdm(enumerate(caption_embeddings), total=len(caption_embeddings)):
        caption_name = captions[idx]['name']
        
        # Skip if this caption has already been processed
        if caption_name in processed_captions:
            print(f"Skipping {caption_name}...", flush=True)
            continue
        
        # Calculate similarities
        similarities = cos_sim(caption_embedding.unsqueeze(0), triple_embeddings)
        
        # Get top k indices
        top_k_values, top_k_indices = torch.topk(similarities, k)
        
        # Extract the corresponding triples for the top k indices
        closest_triples_for_caption = [triples[i.item()] for i in top_k_indices]
        
        # Construct result and append to the file
        result = {
            "name": caption_name,
            "caption": captions[idx]['text'],
            "closest_triples": closest_triples_for_caption
        }
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
        f.flush()
