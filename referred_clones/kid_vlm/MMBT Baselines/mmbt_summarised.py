
import os
os.environ['TRANSFORMERS_CACHE'] = '/scratch/rahul.garg/hfCache'
os.environ['HF_HOME'] = '/scratch/rahul.garg/hfCache'

os.environ['TORCH_HOME'] = '/scratch/rahul.garg/torchCache'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, RGCNConv, global_mean_pool
import json
from collections import Counter
import random
import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from madgrad import MADGRAD
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
import clip
import pickle
from matplotlib import pyplot as plt
from tqdm import tqdm
import copy
import optuna
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
import sys
# print all args
print(sys.argv)
# get device id
device_id = int(sys.argv[1])
# ensure its a valid device
if device_id not in [0, 1]:
    print("Invalid device id", flush=True)
    exit(0)

print(f"Device ID: {device_id}", flush=True)
# import torchvision
# import torchvision.transforms as transforms

study_name = "diffFusions_with_dropout_sample"
# Ignore all DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    transformers.set_seed(seed_value)


set_seed(42)

class customMMBT(MMBTForClassification):
    def __init__(self, config, transformer, encoder, clfDropout=0.1):
        super().__init__(config, transformer, encoder)
       
        self.relu = nn.ReLU()
        
        self.classifier = nn.Linear(768, 1)
        self.dropout = nn.Dropout(clfDropout)


    def forward(
        self,
        input_modal,
        input_ids=None,
        modal_start_tokens=None,
        modal_end_tokens=None,
        attention_mask=None,
        token_type_ids=None,
        modal_token_type_ids=None,
        position_ids=None,
        modal_position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mmbt(
            input_modal=input_modal,
            input_ids=input_ids,
            modal_start_tokens=modal_start_tokens,
            modal_end_tokens=modal_end_tokens,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            modal_token_type_ids=modal_token_type_ids,
            position_ids=position_ids,
            modal_position_ids=modal_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
        )

        pooled_output = outputs[1] 
        pooled_output = self.relu(pooled_output) 

        logits = self.classifier(pooled_output)

        output = (logits,) + outputs[2:]
        return output



triplesObj = {}
triplesFile = open('/home/rahul.garg/modelRuns/triples_Summaries.jsonl', 'r', encoding='utf-8')
for line in triplesFile:
    obj = json.loads(line)
    triplesObj[int(obj['name'])] = obj['summary']

triplesFile.close()

# Printing only the first few key-value pairs from the triplesObj
first_few_pairs = {k: triplesObj[k] for k in list(triplesObj.keys())[:5]}
first_few_pairs

captionObj = {}
captionFile = open('/home/rahul.garg/modelRuns/completeCaptionsFinalAll.jsonl', 'r', encoding='utf-8')
for line in captionFile:
    obj = json.loads(line)
    captionObj[int(obj['name'])] = obj['text']

captionFile.close()

captionObj


def slice_image(im, desired_size):
    '''
    Resize and slice image
    '''
    old_size = im.size

    ratio = float(desired_size)/min(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    ar = np.array(im)
    images = []
    if ar.shape[0] < ar.shape[1]:
        middle = ar.shape[1] // 2
        half = desired_size // 2

        images.append(Image.fromarray(ar[:, :desired_size]))
        images.append(Image.fromarray(ar[:, middle-half:middle+half]))
        images.append(Image.fromarray(
            ar[:, ar.shape[1]-desired_size:ar.shape[1]]))
    else:
        middle = ar.shape[0] // 2
        half = desired_size // 2

        images.append(Image.fromarray(ar[:desired_size, :]))
        images.append(Image.fromarray(ar[middle-half:middle+half, :]))
        images.append(Image.fromarray(
            ar[ar.shape[0]-desired_size:ar.shape[0], :]))

    return images


def resize_pad_image(im, desired_size):
    '''
    Resize and pad image to a desired size
    '''
    old_size = im.size

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                      (desired_size-new_size[1])//2))

    return new_im


class ClipEncoderMulti(nn.Module):
    def __init__(self, num_embeds, num_features, clip_model):
        super().__init__()
        self.model = clip_model
        self.num_embeds = num_embeds
        self.num_features = num_features

    def forward(self, x):
        # 4x3x288x288 -> 1x4x640
        out = self.model.encode_image(x.view(-1, 3, 288, 288))
        out = out.view(-1, self.num_embeds, self.num_features).float()
        return out  # Bx4x640


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, max_seq_length, image_encoder_size, device):
        self.data = [json.loads(l) for l in open(
            data_path) if int(json.loads(l)["id"]) in captionObj]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.image_encoder_size = image_encoder_size
        self.device = device

        self.combined_texts = []

        self.transforms = transforms

        for index in range(len(self.data)):
            item_text = self.data[index]["text"]
            item_id = int(self.data[index]["id"])

            # caption_with_tags = f"<caption_start> {captionObj[item_id]} <caption_end>"

            combined_text = f"{item_text} <triple_start> {triplesObj[item_id]} <triple_end"
            self.combined_texts.append(combined_text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        combined_text = self.combined_texts[index]

        sentence = torch.LongTensor(self.tokenizer.encode(
            combined_text, add_special_tokens=True))

        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        sentence = sentence[:self.max_seq_length]

        label = torch.FloatTensor([self.data[index]["label"]])

        image = Image.open(os.path.join(
            self.data_dir, self.data[index]["img"])).convert("RGB")
        sliced_images = slice_image(image, 288)
        sliced_images = [np.array(self.transforms(im)) for im in sliced_images]
        image = resize_pad_image(image, self.image_encoder_size)
        image = np.array(self.transforms(image))

        sliced_images = [image] + sliced_images
        sliced_images = torch.from_numpy(
            np.array(sliced_images)).to(self.device)

        return {
            "image_start_token": start_token,
            "image_end_token": end_token,
            "sentence": sentence,
            "image": sliced_images,
            "label": label,
            "id": int(self.data[index]["id"])
        }

    def get_label_frequencies(self):
        label_freqs = Counter()
        for row in self.data:
            label_freqs.update([row["label"]])
        return label_freqs

    def get_labels(self):
        labels = []
        for row in self.data:
            labels.append(row["label"])
        return labels


def collate_fn(batch):
    lens = [len(row["sentence"]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = input_row["sentence"]
        mask_tensor[i_batch, :length] = 1

    img_tensor = torch.stack([row["image"] for row in batch])
    tgt_tensor = torch.stack([row["label"] for row in batch])
    img_start_token = torch.stack([row["image_start_token"] for row in batch])
    img_end_token = torch.stack([row["image_end_token"] for row in batch])

    return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, tgt_tensor


def load_examples(tokenizer, preprocess, max_seq_length, num_image_embeds, image_encoder_size, device, evaluate=0):

    if evaluate == 0:
        path = '/scratch/rahul.garg/hateful_memes/train.jsonl'
    elif evaluate == 1:
        path = '/scratch/rahul.garg/hateful_memes/dev_seen.jsonl'
    else:
        path = '/scratch/rahul.garg/hateful_memes/test_seen.jsonl'

    transforms = preprocess
    dataset = JsonlDataset(path, tokenizer, transforms, max_seq_length -
                           num_image_embeds - 2, image_encoder_size=image_encoder_size, device=device)
    return dataset


def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, device):

    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def evaluate(model, tokenizer, criterion, dataloader, device, tres=0.5):

    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    proba = None
    out_label_ids = None
    for batch in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            labels = batch[5]
            inputs = {
                "input_ids": batch[0],
                "input_modal": batch[2],
                "attention_mask": batch[1],
                "modal_start_tokens": batch[3],
                "modal_end_tokens": batch[4],
                # "graph_data": batch[6],
                "return_dict": False
            }
            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            logits = outputs[0]
            tmp_eval_loss = criterion(logits, labels)
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = torch.sigmoid(logits).detach().cpu().numpy() > tres
            proba = torch.sigmoid(logits).detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, torch.sigmoid(
                logits).detach().cpu().numpy() > tres, axis=0)
            proba = np.append(proba, torch.sigmoid(
                logits).detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    result = {
        "loss": eval_loss,
        "accuracy": accuracy_score(out_label_ids, preds),
        "AUC": roc_auc_score(out_label_ids, proba),
        "micro_f1": f1_score(out_label_ids, preds, average="micro"),
        "prediction": preds,
        "labels": out_label_ids,
        "proba": proba
    }

    return result


def train(model, tokenizer, criterion, optimizer, scheduler, train_dataloader, eval_dataloader, num_train_epochs, max_grad_norm, gradient_accumulation_steps, device, trial=None):
    optimizer_step = 0
    global_step = 0
    train_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_valid_auc = 0.0
    global_steps_list = []
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    val_auc_list = []
    eval_every = len(train_dataloader) // 5
    running_loss = 0
    file_path = "/scratch/rahul.garg/models/"
    model_best = "mmbt_summary_best.pt"

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    model.zero_grad()

    bestModel = None
    optunaMarker = 0
    loggingInterval = 50
    bestStats = {"loss": 0, "acc": 0, "auc": 0}
    for i in range(num_train_epochs):
        print("Epoch", i+1, f"from {num_train_epochs}")
        whole_y_pred = np.array([])
        whole_y_t = np.array([])
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {i+1}")):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            labels = batch[5]
            inputs = {
                "input_ids": batch[0],
                "input_modal": batch[2],
                "attention_mask": batch[1],
                "modal_start_tokens": batch[3],
                "modal_end_tokens": batch[4],
                "return_dict": False
            }
            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            logits = outputs[0]
            loss = criterion(logits, labels)
            tr_loss += loss.item()

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            
            running_loss += loss.item()
            global_step += 1

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule

                optimizer_step += 1
                optimizer.zero_grad()

            if global_step % loggingInterval == 0:
                wandb.log({
                    "Train Loss": tr_loss / loggingInterval
                })

                tr_loss = 0

            if (step + 1) % eval_every == 0:

                average_train_loss = running_loss / eval_every
                train_loss_list.append(average_train_loss)
                global_steps_list.append(global_step)
                running_loss = 0.0

                val_result = evaluate(
                    model, tokenizer, criterion, eval_dataloader, device)

                
                val_loss_list.append(val_result['loss'])
                val_acc_list.append(val_result['accuracy'])
                val_auc_list.append(val_result['AUC'])

                wandb.log({
                    "Valid Loss": val_result['loss'],
                    "Valid Accuracy": val_result['accuracy'],
                    "Valid AUC": val_result['AUC']
                })
                # checkpoint
                if val_result['AUC'] > best_valid_auc:
                    best_valid_auc = val_result['AUC']
                    val_loss = val_result['loss']
                    val_acc = val_result['accuracy']

                    bestStats = {"loss": val_loss, "acc": val_acc, "auc": best_valid_auc}

                    model_path = f'{file_path}/mmbt-model-embs-auc{best_valid_auc:.3f}-loss{val_loss:.3f}-acc{val_acc:.3f}.pt'
                    bestModel = model_path
                    torch.save(model.state_dict(), 'best_model_summary_nocap.pth') 
                    print(f"AUC improved, so saving this model")
                    # save_checkpoint(model_path, model, val_result['loss'])

                print("Train loss:", f"{average_train_loss:.4f}",
                      "Val loss:", f"{val_result['loss']:.4f}",
                      "Val acc:", f"{val_result['accuracy']:.4f}",
                      "AUC:", f"{val_result['AUC']:.4f}", flush=True)

        print('\n')

    plt.plot(global_steps_list, val_auc_list)
    plt.grid()
    plt.xlabel('Global Steps')
    plt.ylabel('AUC')
    plt.title('MMBT Area Under the Curve')
    plt.show()

    # load_checkpoint(bestModel, model, device)
    return bestStats

def evaluate2(model, tokenizer, criterion, dataloader, device, tres=0.5):
    
    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    proba = None
    out_label_ids = None
    model.load_state_dict(torch.load('best_model_summary_nocap.pth'))
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            labels = batch[5]
            inputs = {
                "input_ids": batch[0],
                "input_modal": batch[2],
                "attention_mask": batch[1],
                "modal_start_tokens": batch[3],
                "modal_end_tokens": batch[4],
                # "graph_data": batch[6],
                "return_dict": False
            }
            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            logits = outputs[0]
            tmp_eval_loss = criterion(logits, labels)
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = torch.sigmoid(logits).detach().cpu().numpy() > tres
            proba = torch.sigmoid(logits).detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, torch.sigmoid(logits).detach().cpu().numpy() > tres, axis=0)
            proba = np.append(proba, torch.sigmoid(logits).detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    result = {
        # "loss": eval_loss,
        "Test accuracy": accuracy_score(out_label_ids, preds),
        "Test AUC": roc_auc_score(out_label_ids, proba),
        "Test micro_f1": f1_score(out_label_ids, preds, average="macro"),
        "Test precision": precision_score(out_label_ids, preds, average="macro"),
        "Test recall": recall_score(out_label_ids, preds, average="macro"),
        # "prediction": preds,
        # "labels": out_label_ids,
        # "proba": proba
    }
    wandb.log(result)

    return result
def objective(trial):

    wandb.init(project=f"mmbt_runs", mode="online", name=f"mmbt_summary_3")

    set_seed(42)
    print("Starting new trial")
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)#, download_root='/scratch/rahul.garg/clipCache')

    for p in clip_model.parameters():
        p.requires_grad = False

    num_image_embeds = 4
    num_labels = 1

    data_dir = '/scratch/rahul.garg/hateful_memes'
    max_seq_length = 500

    train_batch_size = 4
    eval_batch_size = 4
    image_features_size = 640
    num_train_epochs = 5

    clfDropout = 0.6275770708531836
    max_grad_norm = 0.5
    weight_decay = 0.0005
    learning_rate = 2e-4
    image_encoder_size = 288
    gradient_accumulation_steps = 20
    warmup_denom = 10

    model_name = 'Hate-speech-CNERG/bert-base-uncased-hatexplain'
    transformer_config = AutoConfig.from_pretrained(model_name)
    transformer = AutoModel.from_pretrained(
        model_name, config=transformer_config)
    img_encoder = ClipEncoderMulti(
        num_image_embeds, image_features_size, clip_model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

    # Define new special tokens
    new_special_tokens = {
        "additional_special_tokens": ["<caption_start>", "<caption_end>", "<triple_start>", "<triple_end>", "<sep>"]
    }

    # Add the new special tokens to the tokenizer
    tokenizer.add_special_tokens(new_special_tokens)

    # Resize the token embeddings in your model to account for the new tokens
    # This is important because the model's embedding size needs to be adjusted to accommodate the new tokens
    transformer.resize_token_embeddings(len(tokenizer))

    config = MMBTConfig(transformer_config, num_labels=num_labels,
                        modal_hidden_size=image_features_size)
    model = customMMBT(config, transformer, img_encoder
                          , clfDropout=clfDropout)
                           
    # model = MMBTForClassification(config, transformer, img_encoder)

    model.to(device)

    train_dataset = load_examples(tokenizer, evaluate=0, preprocess=preprocess, max_seq_length=max_seq_length,
                                  num_image_embeds=num_image_embeds, image_encoder_size=image_encoder_size, device=device)
    eval_dataset = load_examples(tokenizer, evaluate=1, preprocess=preprocess, max_seq_length=max_seq_length,
                                 num_image_embeds=num_image_embeds, image_encoder_size=image_encoder_size, device=device)
    test_dataset = load_examples(tokenizer, evaluate=2, preprocess=preprocess, max_seq_length=max_seq_length,
                                 num_image_embeds=num_image_embeds, image_encoder_size=image_encoder_size, device=device)

    train_sampler = RandomSampler(train_dataset)
    eval_sampler = SequentialSampler(eval_dataset)
    test_sampler = SequentialSampler(test_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=train_batch_size,
        collate_fn=collate_fn
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=eval_batch_size,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=eval_batch_size,
        collate_fn=collate_fn
    )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias",
                "LayerNorm.weight"
                ]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    t_total = (len(train_dataloader) //
               gradient_accumulation_steps) * num_train_epochs
    warmup_steps = t_total // warmup_denom

    optimizer = MADGRAD(optimizer_grouped_parameters, lr=learning_rate)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, t_total
    )

    criterion = nn.BCEWithLogitsLoss()
    # load_checkpoint(
    #     "/scratch/rahul.garg/models//model-embs-auc0.756-loss0.708-acc0.694.pt", model, device)
    res = train(model, tokenizer, criterion, optimizer, scheduler, train_dataloader, eval_dataloader, num_train_epochs, max_grad_norm, gradient_accumulation_steps, device)

    wandb.log({
        "Best AUC": res['auc'],
        "Corresponding Loss": res['loss'],
        "Corresponding Accuracy": res['acc']
    })
    evaluate2(model, tokenizer, criterion, test_dataloader, device)
    wandb.finish()

    return res['auc']


objective('trial')

# storage_url = f"sqlite:///./{study_name}.db"

# pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
# study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage_url, load_if_exists=True, pruner=pruner)
# study.optimize(objective, n_trials=1, show_progress_bar=True, n_jobs=1)

# print("Number of finished trials: ", len(study.trials))
# print("Best trial:")
# trial = study.best_trial
# print("  Value: ", trial.value)
# print("  Params: ")
# for key, value in trial.params.items():
#     print("    {}: {}".format(key, value))
