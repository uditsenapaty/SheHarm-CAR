import pickle as pkl
import sys
from collections import OrderedDict
from icecream import ic
from pathlib import Path
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
from tqdm import tqdm
import json
import os

# set hf home and transformers cache
os.environ["HF_HOME"] = "/scratch/rahul.garg/cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/rahul.garg/cache"


if len(sys.argv) < 4:
    print("Usage: python RelevancyKCMGS.py <start_line> <max_lines> <GPU_ID>")
    sys.exit(1)

start_line = int(sys.argv[1])
max_lines = int(sys.argv[2])
gpu_id = int(sys.argv[3])

print("Start line:", start_line)
print("Max lines:", max_lines)


class RobertaForMaskedLMwithLoss(RobertaForMaskedLM):
    #
    def __init__(self, config):
        super().__init__(config)

    #
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        masked_lm_labels=None,
    ):
        #
        assert attention_mask is not None
        embed2return = self.roberta.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        sequence_output = outputs[
            0
        ]  # hidden_states of final layer (batch_size, sequence_length, hidden_size)
        prediction_scores = self.lm_head(sequence_output)
        outputs = (prediction_scores, sequence_output) + outputs[2:]
        if masked_lm_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            bsize, seqlen = input_ids.size()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
            ).view(bsize, seqlen)
            masked_lm_loss = (masked_lm_loss * attention_mask).sum(dim=1)
            outputs = (masked_lm_loss,) + outputs
            # (masked_lm_loss), prediction_scores, sequence_output, (hidden_states), (attentions)
        return outputs, embed2return


TOKENIZER = RobertaTokenizer.from_pretrained("roberta-large")
LM_MODEL = RobertaForMaskedLMwithLoss.from_pretrained("roberta-large")
LM_MODEL.cuda(gpu_id)
# LM_MODEL = DataParallel(LM_MODEL)
LM_MODEL.eval()

labels = {}

jsonlFilesLoc = "/scratch/rahul.garg/hateful_memes/"
ocrs = {}
for file in Path(jsonlFilesLoc).rglob("*.jsonl"):
    with open(file, "r") as f:
        for line in f:
            line = json.loads(line)
            # print(line.keys())
            # dict_keys(['id', 'img', 'label', 'text'])
            ocrs[int(line["id"])] = line["text"]
            labels[int(line["id"])] = line["label"]
ic(len(ocrs))

captionPath = (
    "/home/rahul.garg/modelRuns/completeCaptionsFinalAll.jsonl"
)
captions = {}
with open(captionPath, "r") as f:
    for line in f:
        line = json.loads(line)
        # print(line.keys())
        # dict_keys(['name', 'text'])

        captions[int(line["name"])] = line["text"]

ic(len(captions))

hatredFile = "/scratch/rahul.garg/HatReD/datasets/hatred/annotations"

targets = {}
lengthFreqs = {}
for file in Path(hatredFile).glob("*.jsonl"):
    with open(file, "r") as f:
        for line in f:
            line = json.loads(line)
            # print(line.keys())
            # dict_keys(['id', 'img', 'target', 'reasonings'])
            targets[int(line["id"])] = line["target"]
            length = len(line["target"])
            if length in lengthFreqs:
                lengthFreqs[length] += 1
            else:
                lengthFreqs[length] = 1

ic(len(targets))

llavaFile = "/home/rahul.garg/modelRuns/Llava.json"
# it also has targets only

with open(llavaFile, "r") as f:
    llavaData = json.load(f)
    for key in llavaData.keys():
        id = int(key.split(".")[0])
        # skip if label is 0, skip
        # if id not in labels:
        #     continue

        # if labels[id] == 1:
        #     continue

        if id in targets:
            continue

        targets[id] = llavaData[key]

ic(len(targets))

# print unique type of values in targets
uniqueTargetsType = set()
for key in targets:
    uniqueTargetsType.add(type(targets[key]))

ic(uniqueTargetsType)
# redo targets but like wherever its list, make it string with comma
for key in targets:
    if type(targets[key]) == list:
        # print(targets[key])
        # removeNonStrings
        newList = []
        for item in targets[key]:
            if type(item) == str:
                newList.append(item)

        # print(targets[key], newList)
        targets[key] = ", ".join(newList)

uniqueTargetsType = set()
for key in targets:
    uniqueTargetsType.add(type(targets[key]))

commonIds = set()
# do int parse
for key in ocrs.keys():
    if key in captions and key in targets:
        commonIds.add(key)
ic(len(commonIds))

ic(uniqueTargetsType)
combinedTexts = {}
for key in commonIds:
    combinedTexts[key] = ocrs[key] + ". " + captions[key] #+ \
        # ". The target is " + targets[key] + "."

inputPath = "/scratch/rahul.garg/conceptNet/allCmgs_noTargetFixed_h1.jsonl"
outputName = "allCmgs_noTargetFixed_h1_RelevancyKCMGS"
if start_line == 0:
    outputPath = f"/scratch/rahul.garg/{outputName}.jsonl"
else:
    outputPath = f"/scratch/rahul.garg/{outputName}_{start_line}_{max_lines}.jsonl"

# def get_LM_score(nodes, context):
#     nodes = nodes[:]
#     nodes.insert(0, -1) #QAcontext node
#     sents, scores = [], []
#     for node in nodes:
#         if node==-1:
#             sent = context.lower()
#         else:
#             sent = '{} {}.'.format(context.lower(), ' '.join(node['key'].split('_')))
#         sent = TOKENIZER.encode(sent, add_special_tokens=True)
#         sents.append(sent)
#     n_cids = len(nodes)
#     cur_idx = 0
#     batch_size = 50
#     while cur_idx < n_cids:
#         #Prepare batch
#         input_ids = sents[cur_idx: cur_idx+batch_size]
#         max_len = max([len(seq) for seq in input_ids])
#         for j, seq in enumerate(input_ids):
#             seq += [TOKENIZER.pad_token_id] * (max_len-len(seq))
#             input_ids[j] = seq
#         input_ids = torch.tensor(input_ids).cuda() #[B, seqlen]
#         mask = (input_ids!=1).long() #[B, seq_len]
#         #Get LM score
#         with torch.no_grad():
#             outputs = LM_MODEL(input_ids, attention_mask=mask, masked_lm_labels=input_ids)
#             loss = outputs[0] #[B, ]
#             _scores = list(-loss.detach().cpu().numpy()) #list of float
#         scores += _scores
#         cur_idx += batch_size
#     assert len(sents) == len(scores) == len(nodes)
#     cid2score = OrderedDict(sorted(list(zip(nodes, scores)), key=lambda x: -x[1])) #score: from high to low
# return cid2score


def get_LM_score(nodes, context):
    nodes = [{"key": "QAcontext", "score": None}] + nodes[
        :
    ]  # Prepends a pseudo-node for QA context
    sents = []

    for node in nodes:
        if node["key"] == "QAcontext":
            sent = context.lower()
        else:
            # print(node["key"])
            sent = "{} {}.".format(
                context.lower(), " ".join(str(node["key"]).split("_")))
        sent = TOKENIZER.encode(sent, add_special_tokens=True)
        sents.append(sent)

    # get mainEmbed
    singleton = torch.tensor(sents[0]).cuda(gpu_id).unsqueeze(0)
    mask = (singleton != TOKENIZER.pad_token_id).long()
    with torch.no_grad():
        outputs, embed = LM_MODEL(
            singleton, attention_mask=mask, masked_lm_labels=singleton
        )
        mainEmbed = embed

    scores = []
    cur_idx = 0
    batch_size = 100
    

    while cur_idx < len(nodes):
        # Prepare batch
        input_ids = [
            sents[i] for i in range(cur_idx, min(cur_idx + batch_size, len(nodes)))
        ]
        max_len = max(len(seq) for seq in input_ids)
        input_ids = [
            seq + [TOKENIZER.pad_token_id] * (max_len - len(seq)) for seq in input_ids
        ]
        input_ids = torch.tensor(input_ids).cuda(gpu_id)  # [B, seqlen]
        mask = (input_ids != TOKENIZER.pad_token_id).long()  # [B, seq_len]

        # Get LM score
        with torch.no_grad():
            outputs, _ = LM_MODEL(
                input_ids, attention_mask=mask, masked_lm_labels=input_ids
            )

            loss = outputs[0]
            _scores = -loss.detach().cpu().numpy()  # list of float
        scores.extend(_scores)
        cur_idx += batch_size

    # Assign scores to nodes
    for node, score in zip(nodes, scores):
        node["score"] = float(score)

    return nodes, mainEmbed


# check if the output file already exists, and resume
alreadyDone = set()
once = False
if os.path.exists(outputPath):
    with open(outputPath, "r") as f:
        for line in f:
            try:
                loadedObj = json.loads(line)
                id = int(loadedObj["id"])
                alreadyDone.add(id)
            except Exception as e:
                print("Error", e)
                # print('Error', line)
                once = True
                continue

ic(alreadyDone, len(alreadyDone))

processed_lines = 0  # Counter to keep track of how many lines have been processed

totalLines = min(start_line + max_lines, 12015) - start_line

with open(inputPath, "r") as f:
    for _ in range(start_line):
        next(f)

    pbar = tqdm(f, total=totalLines, desc="Processing")
    for line in pbar:
        torch.cuda.empty_cache()

        try:
            if processed_lines >= max_lines:
                break
            loadedObj = json.loads(line)

            id = int(loadedObj["id"])
            if id not in combinedTexts:
                print("Not in combinedTexts", id)
                continue

            # if id not in graphLeft:
            #     # print("Not in graphLeft", id)
            #     processed_lines += 1
            #     continue

            if id in alreadyDone:
                print("Already done", id, flush=True)
                processed_lines += 1
                continue

            loadedObj["graph"]["nodeDataArray"], contextEmbed = get_LM_score(
                loadedObj["graph"]["nodeDataArray"], combinedTexts[id]
            )

            # save embed
            print(contextEmbed.cpu().numpy()[0].shape)
            # exit(1)
            loadedObj["graph"]["contextEmbed"] = contextEmbed.cpu().numpy().tolist()

            with open(outputPath, "a") as outputf:
                json.dump(loadedObj, outputf)
                outputf.write("\n")

            processed_lines += 1
        except Exception as e:
            print("Error", e)
            raise e
            # print('Error', line)
            continue
