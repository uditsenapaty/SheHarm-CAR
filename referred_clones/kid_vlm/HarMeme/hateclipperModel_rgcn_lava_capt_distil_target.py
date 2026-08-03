device_id = 0

print(f"Device ID: {device_id}", flush=True)
import os
os.environ['TRANSFORMERS_CACHE'] = '/scratch/rahul.garg/hfCache'
os.environ['HF_HOME'] = '/scratch/rahul.garg/hfCache'

os.environ['TORCH_HOME'] = '/scratch/rahul.garg/torchCache'
# visible cuda devices
os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, RGCNConv, global_mean_pool
import json
from collections import Counter
import random
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from madgrad import MADGRAD
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score,precision_score, recall_score
import clip
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
import copy
import optuna
import pandas as pd
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    MMBTConfig,
    MMBTModel,
    MMBTForClassification,
    get_linear_schedule_with_warmup,
    pipeline
)
import warnings
import pickle as pkl
import wandb
import copy

import torchmetrics
from transformers import CLIPModel, AutoConfig, AutoModel, CLIPProcessor, CLIPTokenizer
from PIL import Image, ImageDraw, ImageFont, ImageOps
import matplotlib.pyplot as plt
from icecream import ic
import torch_geometric
import pandas as pd
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
print(device, flush=True)
def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed_value)
    torch_geometric.seed_everything(seed_value)
set_seed(42)
top_acc_target = []
top_model_target = []
captions = {}
with open('../LLaVa Generation/newCaptions_Harmeme_no_target.json', 'r') as file:
    fileObj = json.load(file)
    for key in fileObj:
        captions[key.split('.')[0]] = fileObj[key]

print(f"Loaded {len(captions)} captions", flush=True)
# print some random 5
for key in random.sample(list(captions.keys()), 5):
    print(key, captions[key])
graphPaths = '../RelKMG/gnnCache_harmeme_llava_target_fixed_750_h2/'  # contains id.pkl files
os.makedirs(graphPaths, exist_ok=True)

graphCache = {}
print(f"Loading {len(os.listdir(graphPaths))} graphs", flush=True)

for file in tqdm(os.listdir(graphPaths), desc="Loading graphs from cache"):
    with open(os.path.join(graphPaths, file), 'rb') as f:
        obj = pkl.load(f)
        graphCache[file.split('.')[0]] = obj

ent_text = '../Entity Extraction/concept.txt'

# loaded_array = np.load('concept.nb.npy')
loaded_array = np.load('../Entity Extraction/glove.transe.sgd.ent.npy')


word_list = []
with open(ent_text, 'r') as file:
    for line in file:
        word = line.strip()
        if word:
            word_list.append(word)

scenes = {}

graphFile = open('../Entity Extraction/harmeme_llava_Target_and_caption_minilm_h2RelKmgsTop750_noEmbed.jsonl',
                 'r', encoding='utf-8')
for line in tqdm(graphFile, desc="Reading graphs from jsonl"):
    obj = json.loads(line)
    scenes[obj['id']] = obj['graph']

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
known = 0
unknown = 0
class GatedFusion(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        super(GatedFusion, self).__init__()
        self.gate_fc1 = nn.Linear(input_dim1, input_dim1)
        self.gate_fc2 = nn.Linear(input_dim2, input_dim2)
        self.fc = nn.Linear(input_dim1 + input_dim2, output_dim)

    def forward(self, e1, e2):
        gate1 = torch.sigmoid(self.gate_fc1(e1))
        gate2 = torch.sigmoid(self.gate_fc2(e2))
        gated_e1 = gate1 * e1
        gated_e2 = gate2 * e2
        combined = torch.cat((gated_e1, gated_e2), dim=-1)
        output = self.fc(combined)
        return output


class BilinearPoolingFusion(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        super(BilinearPoolingFusion, self).__init__()
        self.fc1 = nn.Linear(input_dim1, output_dim)
        self.fc2 = nn.Linear(input_dim2, output_dim)
        self.fc_out = nn.Linear(output_dim * output_dim, output_dim)

    def forward(self, e1, e2):
        e1_transformed = self.fc1(e1)
        e2_transformed = self.fc2(e2)
        bilinear = torch.bmm(e1_transformed.unsqueeze(2), e2_transformed.unsqueeze(1))
        bilinear_flat = bilinear.view(bilinear.size(0), -1)
        output = self.fc_out(bilinear_flat)
        return output

class HANFusion(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, output_dim):
        super(HANFusion, self).__init__()
        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(input_dim2, hidden_dim)
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, e1, e2):
        e1_transformed = torch.tanh(self.fc1(e1))
        e2_transformed = torch.tanh(self.fc2(e2))
        combined = torch.cat((e1_transformed, e2_transformed), dim=-1)
        attention_weights = torch.softmax(self.attention(combined), dim=-1)
        attended = attention_weights * combined
        output = self.fc_out(attended)
        return output

class MulFusion(nn.Module):
    def __init__(self, input_dim, input_dim2, intermediate_dim, output_dim):
        super(MulFusion, self).__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc2 = nn.Linear(input_dim2, intermediate_dim)
        self.fc3 = nn.Linear(intermediate_dim, output_dim)

    def forward(self, e1, e2):
        e1_transformed = torch.relu(self.fc1(e1))
        e2_transformed = torch.relu(self.fc2(e2))
        
        e1_e2_fused = torch.mul(e1_transformed, e2_transformed)
        
        output = self.fc3(e1_e2_fused)
        return output
class RGCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_bases=None):
        super(RGCNEncoder, self).__init__()
        # Initialize the first RGCN layer
        self.conv1 = RGCNConv(in_channels, hidden_channels,
                              num_relations=num_relations, num_bases=num_bases)
        # Initialize the second RGCN layer
        self.conv2 = RGCNConv(hidden_channels, out_channels,
                              num_relations=num_relations, num_bases=num_bases)

    def forward(self, x, edge_index, edge_type, batch):
        # x: Node features [num_nodes, in_channels]
        # edge_index: Graph edges [2, num_edges]
        # edge_type: Edge types [num_edges], indicating the type of relation for each edge
        # batch: Batch vector [num_nodes], indicating to which graph a node belongs

        # Apply the first RGCN layer with a ReLU activation
        x = F.relu(self.conv1(x, edge_index, edge_type))
        
        # Apply the second RGCN layer
        x = self.conv2(x, edge_index, edge_type)
        # Use global mean pooling to aggregate node features for each graph in the batch
        # Resulting size: [num_graphs, out_channels]
        x = global_mean_pool(x, batch)

        return x
class CLIPClassifier(nn.Module):
    def __init__(self, gnnHidden=256,weight=None, gnnOutput=1024, ptFusion='align', graphFusion='bilinear', fusionOutput=1024,
                hanDim=1024, mulDim=1024, map_dim=1024, lr=1e-4, drop_probs=[0.1, 0.4, 0.2, 0.2], num_mapping_layers=1, num_pre_output_layers=1, weight_decay=1e-4,
                rel2idx=rel2idx, lossAlpha1=0.5   , lossBeta1 = 5 , trainEncoder=False, consistStrat='post'):
        super().__init__()
        self.num_mapping_layers = num_mapping_layers
        self.map_dim = map_dim
        self.fusion = ptFusion
        self.graphFusion = graphFusion
        self.fusionOutput = fusionOutput
        self.hanDim = hanDim
        self.mulDim = mulDim
        self.num_pre_output_layers = num_pre_output_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.weight_image_loss = 0
        self.weight_text_loss = 0
        self.weight_fine_grained_loss = 0
        self.weight_super_loss = 0
        self.drop_probs = drop_probs
        self.lossAlpha1 = lossAlpha1
        self.lossBeta1 = lossBeta1
        self.consistStrat = consistStrat

        self.text_encoder_name = 'clip'

        self.acc = torchmetrics.Accuracy('binary')
        self.auroc = torchmetrics.AUROC('binary')
        self.precision_score = torchmetrics.Precision('binary')
        self.recall = torchmetrics.Recall('binary')
        self.f1 = torchmetrics.F1Score('binary')
    
        self.clip = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')

        self.image_encoder = copy.deepcopy(self.clip.vision_model)

        self.text_encoder = copy.deepcopy(self.clip.text_model)
        self.text_encoder_capt = copy.deepcopy(self.clip.text_model)
    
        image_map_layers = [nn.Linear(self.image_encoder.config.hidden_size, self.map_dim), nn.Dropout(p=self.drop_probs[0])]
        text_map_layers = [nn.Linear(self.text_encoder.config.hidden_size, self.map_dim), nn.Dropout(p=self.drop_probs[0])]
        for _ in range(1, self.num_mapping_layers):
            image_map_layers.extend([nn.ReLU(), nn.Linear(self.map_dim, self.map_dim), nn.Dropout(p=self.drop_probs[0])])
            text_map_layers.extend([nn.ReLU(), nn.Linear(self.map_dim, self.map_dim), nn.Dropout(p=self.drop_probs[0])])

        self.image_map = nn.Sequential(*image_map_layers)
        self.text_map = nn.Sequential(*text_map_layers)
        
        if self.fusion in ['align', 'align_shuffle']:
            pre_output_input_dim = self.map_dim
        elif self.fusion == 'concat':
            pre_output_input_dim = self.map_dim*2
        elif self.fusion.startswith('cross'):
            pre_output_input_dim = self.map_dim**2
        elif self.fusion == 'align_concat':
            pre_output_input_dim = self.map_dim*3
        elif self.fusion == 'attention_m':
            self.gen_query = nn.Linear(self.map_dim, self.map_dim//4) 
            self.gen_key = nn.Linear(self.map_dim, self.map_dim//4) 
            self.soft = nn.Softmax(dim=1)
            pre_output_input_dim = self.map_dim*2
        else:
            raise ValueError("Invalid fusion type.")

        pre_output_layers = [nn.Dropout(p=self.drop_probs[1])]
        output_input_dim = pre_output_input_dim
        if self.num_pre_output_layers >= 1: # first pre-output layer
            pre_output_layers.extend([nn.Linear(pre_output_input_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=self.drop_probs[2])])
            output_input_dim = self.map_dim
        for _ in range(1, self.num_pre_output_layers): # next pre-output layers
            pre_output_layers.extend([nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=self.drop_probs[2])])

        self.pre_output = nn.Sequential(*pre_output_layers)
        self.gnnHidden = gnnHidden
        self.gnnOutput = gnnOutput

        self.rgcn = RGCNEncoder(100, self.gnnHidden, self.gnnOutput, len(rel2idx))

        if self.graphFusion == 'gated':
            self.fusionLayer = GatedFusion(output_input_dim, self.gnnOutput, self.fusionOutput)
        elif self.graphFusion == 'bilinear':
            self.fusionLayer = BilinearPoolingFusion(output_input_dim, self.gnnOutput, self.fusionOutput)
        elif self.graphFusion == 'han':
            self.fusionLayer = HANFusion(output_input_dim, self.gnnOutput, self.hanDim, self.fusionOutput)
        elif self.graphFusion == 'mul':
            self.fusionLayer = MulFusion(output_input_dim, self.gnnOutput, self.mulDim, self.fusionOutput)
        else:
            raise ValueError("Invalid graph fusion type.")
        
        self.post_fusion = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.fusionOutput, self.map_dim)
        )
        self.output_target = nn.Linear(output_input_dim, 4)
        if weight is not None:
            self.target_cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean', weight=weight)
        else:   
            print("Weight is none")
            self.target_cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')

        for _, p in self.image_encoder.named_parameters():
            p.requires_grad_(trainEncoder)

        for _, p in self.text_encoder.named_parameters():
            p.requires_grad_(trainEncoder)

        for _, p in self.text_encoder_capt.named_parameters():
            p.requires_grad_(False)

        del self.clip

    def forward(self, batch, batch_idx):
        image_features = self.image_encoder(pixel_values=batch['pixel_values']).pooler_output
        image_features = self.image_map(image_features)

        text_features = self.text_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).pooler_output
        text_features = self.text_map(text_features)

        text_features_capt = self.text_encoder_capt(input_ids=batch['caption_input_ids'], attention_mask=batch['caption_attention_mask']).pooler_output
        text_features_capt = self.text_map(text_features_capt)
        # text_features_capt = self.post_text2(text_features_capt)
        # text_features_capt = F.normalize(text_features_capt, p=2, dim=1)

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        text_features_capt = F.normalize(text_features_capt, p=2, dim=1)

        graph_data = batch['graph_data']
        graph_features = self.rgcn(graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch)
        # relu and normalize
        graph_features = F.relu(graph_features)
        # ic(image_features.shape, text_features.shape, graph_features.shape) # all are [batch_size, d]
        graph_features = F.normalize(graph_features, p=2, dim=1)


        output = {}

        if self.fusion in ['align', 'align_shuffle']:
            features = torch.mul(image_features, text_features)
        elif self.fusion == 'concat':
            features = torch.cat([image_features, text_features], dim=1)
        elif self.fusion.startswith('cross'):
            features = torch.bmm(image_features.unsqueeze(2), text_features.unsqueeze(1)) # [16, d, d]
            if self.fusion == 'cross_nd':
                mask = torch.eye(self.map_dim).repeat(features.shape[0], 1, 1).bool()
                features[mask] = torch.zeros(features.shape[0]*self.map_dim, device=features.device)
                del mask
            features = features.reshape(features.shape[0], -1)  # [batch_size, d*d]
        elif self.fusion == 'align_concat':
                features = torch.cat([torch.mul(image_features, text_features), image_features, text_features], dim=1)  # [batch_size, 3*d]
        elif self.fusion == 'attention_m':
            q1 = F.relu(self.gen_query(image_features))
            k1 = F.relu(self.gen_key(image_features))
            q2 = F.relu(self.gen_query(text_features))
            k2 = F.relu(self.gen_key(text_features))
            score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
            score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
            wt_score1_score2_mat = torch.cat((score1, score2), 1)
            wt_i1_i2 = self.soft(wt_score1_score2_mat.float()) #prob
            prob_1 = wt_i1_i2[:,0]
            prob_2 = wt_i1_i2[:,1]
            wtd_i1 = image_features * prob_1[:, None]
            wtd_i2 = text_features * prob_2[:, None]
            features = torch.cat((wtd_i1,wtd_i2), 1) # [batch_size, 2*d]
        else:
            raise ValueError()

        features_pre_output = self.pre_output(features) # POINT 1
        # ic(features_pre_output.shape, graph_features.shape)
        finalCombined = self.fusionLayer(features_pre_output, graph_features)

        combinedEmbedding = self.post_fusion(finalCombined) # POINT 2

        # consistency loss
        # ic(combinedEmbedding.shape, text_features_capt.shape)
        # consistency_loss = F.mse_loss(combinedEmbedding, text_features_capt)
        if self.consistStrat == 'post':
            consistency_loss = F.mse_loss(combinedEmbedding, text_features_capt)
        elif self.consistStrat == 'pre':
            consistency_loss = F.mse_loss(features_pre_output, text_features_capt)
        else:
            raise ValueError("Invalid consistency strategy")

        

        logits_target = self.output_target(features_pre_output)  
        preds_target_proxy = torch.softmax(logits_target, dim=1)
        preds_target = torch.argmax(preds_target_proxy, dim=1)
        # ic(preds)
        # print(logits.shape, batch['labels'].float().squeeze().shape)
        # mainLoss1 = self.intensity_cross_entropy_loss(logits_intensity, batch['intensity_labels'].long())
        mainLoss2 = self.target_cross_entropy_loss(logits_target, batch['target_labels'].long())
        # normalize
        output['loss'] =  mainLoss2 * (self.lossAlpha1) + consistency_loss * self.lossBeta1  
        output['mainLoss1'] = mainLoss2
        # output['mainLoss2'] = mainloss2
        output['consistencyLoss'] = consistency_loss
        output['preds_target'] = preds_target
        output['probs_target'] = preds_target_proxy
        return output
        
    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if p.requires_grad]}
            ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer
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

    return data

class HatefulMemesDataset(Dataset):
    def __init__(self, root_folder, image_folder, split='train', image_size=224):
        super(HatefulMemesDataset, self).__init__()
        self.root_folder = root_folder
        self.image_folder = image_folder
        self.split = split
        self.image_size = image_size

        self.df = pd.read_json(f"{root_folder}/{split}.jsonl", lines=True)
        # keep only those rows where the id exists in scenes
        self.df = self.df[self.df['id'].isin(scenes.keys())]
        self.df = self.df[self.df['id'].isin(captions.keys())]
        print(self.df.head())
        print(len(self.df))
        self.df['Intensity'] = self.df['labels'].apply(lambda x: x[0])
        self.df['Target'] = self.df['labels'].apply(lambda x: x[1])
        
        self.intensity_mapping = {'not harmful': 0, 'somewhat harmful': 1, 'very harmful': 2}
        self.target_mapping = { 'individual': 0, 'society': 1, 'organization': 2, 'community': 3}
        
        self.df['Intensity'] = self.df['Intensity'].map(self.intensity_mapping)
        self.df['Target'] = self.df['Target'].map(self.target_mapping)

    def __len__(self):
        return len(self.df)

        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {}
        image_fn = row['image']
        item['image'] = Image.open(f"{self.image_folder}/{image_fn}").convert('RGB').resize((self.image_size, self.image_size))
        item['text'] = row['text']
        item['caption'] = captions[row['id']]

        item['intensity_label'] = row['Intensity']
        item['target_label'] = row['Target']
        item['graph'] = scenes[row['id']]
        item['id'] = row['id']

        return item
class CustomCollator(object):

    def __init__(self):
        self.image_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.text_processor = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

    def __call__(self, batch):
        pixel_values = self.image_processor(images=[item['image'] for item in batch], return_tensors="pt")['pixel_values']
        text_output = self.text_processor([item['text'] for item in batch], padding=True, return_tensors="pt", truncation=True)
        caption_output = self.text_processor([item['caption'] for item in batch], padding=True, return_tensors="pt", truncation=True)
        target_labels = torch.LongTensor([item['target_label'] for item in batch])

        batch_new = {}
        batch_new['pixel_values'] = pixel_values
        batch_new['input_ids'] = text_output['input_ids']
        batch_new['attention_mask'] = text_output['attention_mask']
        # batch_new['intensity_labels'] = intensity_labels
        batch_new['target_labels'] = target_labels
        graphList = [convert_to_rgcnn_data_with_transe(row["graph"], row["id"]) for row in batch]
        # print(graphList)
        batch_new['graph_data'] = Batch.from_data_list(graphList)

        batch_new['caption_input_ids'] = caption_output['input_ids']
        batch_new['caption_attention_mask'] = caption_output['attention_mask']

        return batch_new
    
    
    
def train(model, optimizer, train_loader, val_loader, n_epochs, model_name=f'hateclipper_hop2_KD.pth', trial=None):
    best_acc_target = 0
    bestStats_target = None
    for epoch in range(n_epochs):
        model.train()
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"):
            # move to device
            for key in batch:
                batch[key] = batch[key].to(device)

            optimizer.zero_grad()
            output = model(batch, batch_idx)
            output['loss'].backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            all_preds_target = []
            all_labels_target = []
            all_probs_target = []
            for batch_idx, batch in enumerate(val_loader):
                for key in batch:
                    batch[key] = batch[key].to(device)

                output = model(batch, batch_idx)
                all_preds_target.extend(output['preds_target'].cpu().numpy())
                all_labels_target.extend(batch['target_labels'].cpu().numpy())
                all_probs_target.extend(output['probs_target'].cpu().numpy())
                
            acc_target = accuracy_score(all_labels_target, all_preds_target)
            f1_target = f1_score(all_labels_target, all_preds_target, average='weighted')
            if acc_target >= best_acc_target:
                best_acc_target = acc_target
                bestStats_target = (acc_target, f1_target)
                torch.save(model.state_dict(), f"target_{model_name}")
            print(f"Epoch {epoch}, Val Acc Target: {acc_target}, Val F1 Target: {f1_target}", flush=True)
            wandb.log({"Val Acc Target": acc_target, "Val F1 Target": f1_target})
        if trial is not None:
            trial.report( best_acc_target, epoch)
            if trial.should_prune():
                wandb.log({"Pruned": True})
                wandb.log({"Best Val Accuracy Target": bestStats_target[0], "Best Val F1 Target": bestStats_target[1]})
                wandb.finish()
                raise optuna.TrialPruned()
            
    wandb.log({"Best Val Accuracy Target": bestStats_target[0], "Best Val F1 Target": bestStats_target[1]})

    return best_acc_target


def evaluate(model, test_loader, mainLog=True):
    all_preds_target = []
    all_labels_target = []
    all_probs_target = []
    model.load_state_dict(torch.load("target_hateclipper_hop2_KD.pth"))
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Test"):
            for key in batch:
                batch[key] = batch[key].to(device)
            output = model(batch, batch_idx)
            all_labels_target.extend(batch['target_labels'].cpu().numpy())
            all_preds_target.extend(output['preds_target'].cpu().numpy())
            all_probs_target.extend(output['probs_target'].cpu().numpy())
        
        acc_target = accuracy_score(all_labels_target, all_preds_target)
        f1_target = f1_score(all_labels_target, all_preds_target, average='weighted')
        precision_target = precision_score(all_labels_target, all_preds_target, average='weighted')
        recall_target = recall_score(all_labels_target, all_preds_target, average='weighted')

        wandb.log({"Test Acc Target": acc_target, "Test F1 Target": f1_target})
        wandb.log({"Test Precision Target": precision_target, "Test Recall Target": recall_target})
        
    return acc_target

def calculate_class_weights(train_loader):
    class_counts = Counter()

    for batch in train_loader:
        labels = batch['target_labels'].numpy()  
        class_counts.update(labels)

    total_samples = float(sum(class_counts.values()))

    class_weights = {cls: total_samples / float(count) for cls, count in class_counts.items()}

    class_weights_tensor = torch.tensor([class_weights[i] for i in range(len(class_counts))], dtype=torch.float32)

    return class_weights_tensor       
     
def objective(trial):
    torch.cuda.empty_cache()
    
    set_seed(42)
    wandb.init(project="no_weight_target_harmeme_hateClipperOptuna_new_Target_distil_hop2", mode="online", name=f"{device_id}_{trial.number}", 
    dir='/scratch/rahul.garg')
    
    gnnHidden = trial.suggest_int("gnnHidden", 2, 512)
    gnnOutput = trial.suggest_int("gnnOutput", 2, 1024)
    graphFusion = trial.suggest_categorical("graphFusion", ['gated', 'bilinear', 'han', 'mul'])
    fusionOutput = trial.suggest_int("fusionOutput", 2, 1024)
    ptFusion = 'align'
    hanDim = None
    mulDim = None
    
    if graphFusion == 'han':
        hanDim = trial.suggest_int("hanDim", 2, 1024)
    elif graphFusion == 'mul':
        mulDim = trial.suggest_int("mulDim", 2, 1024)
        
    map_dim = trial.suggest_int("map_dim", 2, 2048)
    lr = trial.suggest_float("lr", 1e-10, 1e-2)
    drop_probs = [trial.suggest_float("drop1", 0.0, 0.9), trial.suggest_float("drop2", 0.0, 0.9), trial.suggest_float("drop3", 0.0, 0.9)]
    num_mapping_layers = trial.suggest_int("num_mapping_layers", 1, 5)
    num_pre_output_layers = trial.suggest_int("num_pre_output_layers", 1, 5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1)
    lossAlpha1 = trial.suggest_float("lossAlpha1", 0, 100)
    lossBeta1 = trial.suggest_float("lossBeta1", 0, 1000)
    learnEncoders = trial.suggest_categorical("learnEncoders", [True, False])
    consistStrat = trial.suggest_categorical("consistStrat", ['pre', 'post'])
    
    wandb.config.update(
        {
            "gnnHidden": gnnHidden,
            "gnnOutput": gnnOutput,
            "ptFusion": ptFusion,
            "graphFusion": graphFusion,
            "fusionOutput": fusionOutput,
            "hanDim": hanDim,
            "mulDim": mulDim,
            "graphSize": 750,
            "map_dim": map_dim,
            "lr": lr,
            "drop_probs": drop_probs,
            "num_mapping_layers": num_mapping_layers,
            "num_pre_output_layers": num_pre_output_layers,
            "weight_decay": weight_decay,
            "lossAlpha1": lossAlpha1,
            "lossBeta1": lossBeta1,
            "learnEncoders": learnEncoders,
            "consistStrat": consistStrat
        }
    )


    rootData = '../datasets/memes_tgt/defaults/annotations'
    imFolder = '../datasets/memes/defaults/images'

    train_dataset = HatefulMemesDataset(rootData, imFolder, split='target_train')
    val_dataset = HatefulMemesDataset(rootData, imFolder, split='target_val')
    test_dataset = HatefulMemesDataset(rootData, imFolder, split='target_test')

    collator=CustomCollator()
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collator)
    
    class_weights_target = calculate_class_weights(train_loader)
    n_epochs = 30
    model = CLIPClassifier(gnnHidden=gnnHidden,weight=None,  gnnOutput=gnnOutput, ptFusion=ptFusion, graphFusion=graphFusion, fusionOutput=fusionOutput,
                hanDim=hanDim, mulDim=mulDim, map_dim=map_dim, lr=lr, drop_probs=drop_probs, num_mapping_layers=num_mapping_layers, num_pre_output_layers=num_pre_output_layers, weight_decay=weight_decay, lossAlpha1=lossAlpha1 , lossBeta1=lossBeta1 , trainEncoder=learnEncoders, consistStrat=consistStrat
                ).to(device)

    optimizer = model.configure_optimizers()
    toReturn = train(model, optimizer, train_loader, val_loader, n_epochs, trial=trial)

    save_parameters = evaluate(model, test_loader)

    wandb.finish()
    return toReturn

study_name = 'no_weight_target_harmeme_hateClipperOptuna_new_Target_distil_hop2'
storage_url = f"sqlite:///./{study_name}.db"

pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10, interval_steps=1)
study = optuna.create_study(direction="maximize" , study_name=study_name, storage=storage_url, load_if_exists=True, pruner=pruner)
study.optimize(objective, n_trials=1000, show_progress_bar=True, n_jobs=1)

print("Number of finished trials: ", len(study.trials))

# objective(None)
