import multiprocessing
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm
import nltk
import json
import string
import os
from pprint import pprint
# icecream
from icecream import ic
from pathlib import Path
import time
from queue import Empty
import builtins

def custom_print(*args, **kwargs):
    # Ensure that the original print is used with an explicit flush=True
    kwargs['flush'] = True
    original_print(*args, **kwargs)

# Capture the original print function before any modification
original_print = builtins.print

# Override the print function in the builtins module
builtins.print = custom_print

__all__ = ['create_matcher_patterns', 'ground']


# the lemma of it/them/mine/.. is -PRON-

blacklist = set(["-PRON-", "actually", "likely", "possibly", "want",
                 "make", "my", "someone", "sometimes_people", "sometimes", "would", "want_to",
                 "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"
                 ])


nltk.download('stopwords', quiet=True)
nltk_stopwords = nltk.corpus.stopwords.words('english')

# CHUNK_SIZE = 1

CPNET_VOCAB = './concept.txt'
PATTERN_PATH = './matcher_patterns.json'
nlp = None
matcher = None


def load_cpnet_vocab(cpnet_vocab_path):
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]
    cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]
    return cpnet_vocab


def create_pattern(nlp, doc, debug=False):
    pronoun_list = set(["my", "you", "it", "its", "your", "i", "he",
                       "she", "his", "her", "they", "them", "their", "our", "we"])
    # Filtering concepts consisting of all stop words and longer than four words.
    if len(doc) >= 5 or doc[0].text in pronoun_list or doc[-1].text in pronoun_list or \
            all([(token.text in nltk_stopwords or token.lemma_ in nltk_stopwords or token.lemma_ in blacklist) for token in doc]):
        if debug:
            return False, doc.text
        return None  # ignore this concept as pattern

    pattern = []
    for token in doc:  # a doc is a concept
        pattern.append({"LEMMA": token.lemma_})
    if debug:
        return True, doc.text
    return pattern


def create_matcher_patterns(cpnet_vocab_path, output_path, debug=False):
    cpnet_vocab = load_cpnet_vocab(cpnet_vocab_path)
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
    docs = nlp.pipe(cpnet_vocab)
    all_patterns = {}

    if debug:
        f = open("filtered_concept.txt", "w")

    for doc in tqdm(docs, total=len(cpnet_vocab)):

        pattern = create_pattern(nlp, doc, debug)
        if debug:
            if not pattern[0]:
                f.write(pattern[1] + '\n')

        if pattern is None:
            continue
        all_patterns["_".join(doc.text.split(" "))] = pattern

    print("Created " + str(len(all_patterns)) + " patterns.")
    with open(output_path, "w", encoding="utf8") as fout:
        json.dump(all_patterns, fout)
    if debug:
        f.close()


def lemmatize(nlp, concept):

    doc = nlp(concept.replace("_", " "))
    lcs = set()
    # for i in range(len(doc)):
    #     lemmas = []
    #     for j, token in enumerate(doc):
    #         if j == i:
    #             lemmas.append(token.lemma_)
    #         else:
    #             lemmas.append(token.text)
    #     lc = "_".join(lemmas)
    #     lcs.add(lc)
    lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
    return lcs


def load_matcher(nlp, pattern_path):
    with open(pattern_path, "r", encoding="utf8") as fin:
        all_patterns = json.load(fin)

    matcher = Matcher(nlp.vocab)
    for concept, pattern in all_patterns.items():
        matcher.add(concept, [pattern])
    return matcher


def ground_qa_pair(s):
    all_concepts = ground_mentioned_concepts(nlp, matcher, s)
    # answer_concepts = ground_mentioned_concepts(nlp, matcher, a)
    if len(all_concepts) == 0:
        all_concepts = hard_ground(nlp, s, CPNET_VOCAB)  # not very possible

    # if len(answer_concepts) == 0:
    #     answer_concepts = hard_ground(nlp, a, CPNET_VOCAB)  # some case

    # question_concepts = question_concepts -  answer_concepts
    all_concepts = sorted(list(all_concepts))

    return {"sent": s,  "qc": all_concepts}


def ground_mentioned_concepts(nlp, matcher, s, ans=None):

    s = s.lower()
    doc = nlp(s)
    matches = matcher(doc)

    mentioned_concepts = set()
    span_to_concepts = {}

    if ans is not None:
        ans_matcher = Matcher(nlp.vocab)
        ans_words = nlp(ans)
        # print(ans_words)
        ans_matcher.add(
            ans, None, [{'TEXT': token.text.lower()} for token in ans_words])

        ans_match = ans_matcher(doc)
        ans_mentions = set()
        for _, ans_start, ans_end in ans_match:
            ans_mentions.add((ans_start, ans_end))

    for match_id, start, end in matches:
        if ans is not None:
            if (start, end) in ans_mentions:
                continue

        span = doc[start:end].text  # the matched span

        # a word that appears in answer is not considered as a mention in the question
        # if len(set(span.split(" ")).intersection(set(ans.split(" ")))) > 0:
        #     continue
        original_concept = nlp.vocab.strings[match_id]
        original_concept_set = set()
        original_concept_set.add(original_concept)

        # print("span", span)
        # print("concept", original_concept)
        # print("Matched '" + span + "' to the rule '" + string_id)

        # why do you lemmatize a mention whose len == 1?

        if len(original_concept.split("_")) == 1:
            # tag = doc[start].tag_
            # if tag in ['VBN', 'VBG']:

            original_concept_set.update(
                lemmatize(nlp, nlp.vocab.strings[match_id]))

        if span not in span_to_concepts:
            span_to_concepts[span] = set()

        span_to_concepts[span].update(original_concept_set)

    for span, concepts in span_to_concepts.items():
        concepts_sorted = list(concepts)
        # print("span:")
        # print(span)
        # print("concept_sorted:")
        # print(concepts_sorted)
        concepts_sorted.sort(key=len)

        # mentioned_concepts.update(concepts_sorted[0:2])

        shortest = concepts_sorted[0:3]

        for c in shortest:
            if c in blacklist:
                continue

            # a set with one string like: set("like_apples")
            lcs = lemmatize(nlp, c)
            intersect = lcs.intersection(shortest)
            if len(intersect) > 0:
                mentioned_concepts.add(list(intersect)[0])
            else:
                mentioned_concepts.add(c)

        # if a mention exactly matches with a concept

        exact_match = set([concept for concept in concepts_sorted if concept.replace(
            "_", " ").lower() == span.lower()])
        # print("exact match:")
        # print(exact_match)
        assert len(exact_match) < 2
        mentioned_concepts.update(exact_match)

    return mentioned_concepts


def hard_ground(nlp, sent, cpnet_vocab):
    sent = sent.lower()
    doc = nlp(sent)
    res = set()
    for t in doc:
        if t.lemma_ in cpnet_vocab:
            res.add(t.lemma_)
    sent = " ".join([t.text for t in doc])
    if sent in cpnet_vocab:
        res.add(sent)
    try:
        assert len(res) > 0
    except Exception:
        print(f"for {sent}, concept not found in hard grounding.")
    return res


def split_data(data, n_parts):
    """Splits data into n_parts roughly equal parts."""
    k, m = divmod(len(data), n_parts)
    return [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_parts)]


OUTPUTCPU = '/scratch/rahul.garg/grounding_output/'
os.makedirs(OUTPUTCPU, exist_ok=True)


def multiGround(sents, pos, queue, progress_event):
    # with open( f'./outputs/ground_res_{pos}.jsonl', 'w') as fout:
    # for id, s in tqdm(sents, position=pos):
    for id, s in sents:
        res = ground_qa_pair(s)
        obj = {"id": id, "res": res}
        queue.put(obj)
        with progress_event.get_lock():
            progress_event.value += 1
        
        # fout.write(json.dumps(obj) + '\n')
        # fout.flush()

    queue.put(None)
        

def match_mentioned_concepts(sents, num_processes):
    res = []
    queue = multiprocessing.Queue()
    progress_events = [multiprocessing.Value('i', 0) for _ in range(num_processes)]

    parts = split_data(sents, num_processes)
    total = len(sents)

    # verify sum of lengths of parts equals to total
    print(sum([len(p) for p in parts]), total)

    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(
            target=multiGround, args=(parts[i], i, queue, progress_events[i]))
        processes.append(p)
        p.start()

    pbar = tqdm(total=total)
    cur = 0
    while True:
        time.sleep(0.1)
        for i in range(num_processes):
            with progress_events[i].get_lock():
                cur += progress_events[i].value
                progress_events[i].value = 0
        pbar.update(cur - pbar.n)

        if cur == total:
            break

    pbar.close()
    print(f'all processes finished, fetching results {queue.qsize()}')

    pbar = tqdm(total=total)
    sentinels = 0
    while sentinels < num_processes: # dont use queue empty as it is not reliable
        obj = queue.get()
        if obj is None:
            sentinels += 1
            continue
        res.append(obj)
        pbar.update(1)
    
    
    pbar.close()

    print(f'all results fetched: {len(res)} | {queue.qsize()}')

    for p in processes:
        p.join()

    print('all processes joined')
    print(f'grounded {len(res)} concepts')

    return res


# To-do: examine prune
def prune(data, cpnet_vocab_path):
    # reload cpnet_vocab
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]

    prune_data = []
    for mainObj in tqdm(data):
        item = mainObj["res"]
        qc = item["qc"]
        prune_qc = []
        for c in qc:
            if c[-2:] == "er" and c[:-2] in qc:
                continue
            if c[-1:] == "e" and c[:-1] in qc:
                continue
            have_stop = False
            # remove all concepts having stopwords, including hard-grounded ones
            for t in c.split("_"):
                if t in nltk_stopwords:
                    have_stop = True
            if not have_stop and c in cpnet_vocab:
                prune_qc.append(c)

        try:
            assert len(prune_qc) > 0
        except Exception as e:
            pass

        item["qc"] = prune_qc
        mainObj["res"] = item
        prune_data.append(mainObj)
    return prune_data


def ground(sents, cpnet_vocab_path, pattern_path, output_path, num_processes=1, debug=False):
    global PATTERN_PATH, CPNET_VOCAB, nlp, matcher

    if PATTERN_PATH is None:
        PATTERN_PATH = pattern_path
        CPNET_VOCAB = load_cpnet_vocab(cpnet_vocab_path)

    if nlp is None or matcher is None:
        nlp = spacy.load('en_core_web_sm', disable=[
                         'ner', 'parser', 'textcat'])
        nlp.add_pipe('sentencizer')
        matcher = load_matcher(nlp, PATTERN_PATH)

    res = match_mentioned_concepts(sents, num_processes)
    res = prune(res, cpnet_vocab_path)

    # check_path(output_path)
    with open(output_path, 'w') as fout:
        for dic in res:
            fout.write(json.dumps(dic) + '\n')

    print(f'grounded concepts saved to {output_path}')
    print()
    return res


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
    "/home/rahul.garg/llava/newCaptionsLLAVA.json"
)
captions = {}
with open(captionPath, "r") as f:
    capObj = json.load(f)
    for key in capObj.keys():
        captInt = int(key.split(".")[0])
        captions[captInt] = capObj[key]

    # for line in f:
    #     line = json.loads(line)
    #     # print(line.keys())
    #     # dict_keys(['name', 'text'])

    #     captions[int(line["name"])] = line["text"]

ic(len(captions))

# hatredFile = "/scratch/rahul.garg/HatReD/datasets/hatred/annotations"

targets = {}
# lengthFreqs = {}
# for file in Path(hatredFile).glob("*.jsonl"):
#     with open(file, "r") as f:
#         for line in f:
#             line = json.loads(line)
#             # print(line.keys())
#             # dict_keys(['id', 'img', 'target', 'reasonings'])
#             targets[int(line["id"])] = line["target"]
#             length = len(line["target"])
#             if length in lengthFreqs:
#                 lengthFreqs[length] += 1
#             else:
#                 lengthFreqs[length] = 1

# ic(len(targets))

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
    combinedTexts[key] = ocrs[key] + " . " + captions[key] + \
        " . The target is " + targets[key] + " ."
    # lower case
    combinedTexts[key] = combinedTexts[key].lower()
    # remove non alpha numeric
    combinedTexts[key] = ''.join(
        e for e in combinedTexts[key] if e.isalnum() or e.isspace())
    

# choose some random 100 keys


# sample = list(combinedTexts.keys())
# sample = sample[:100]
# combinedTexts2 = {}
# for key in sample:
#     combinedTexts2[key] = combinedTexts[key]

# combinedTexts = combinedTexts2

ic(combinedTexts[list(combinedTexts.keys())[0]])
ic(len(combinedTexts))


tups = []
for key in combinedTexts:
    tups.append((key, combinedTexts[key]))

ic(tups[:5])
ic(len(tups))
# tups = tups[:200]
cpus = os.cpu_count()
ic(cpus)
ground(cpnet_vocab_path="./concept.txt", pattern_path="./matcher_patterns.json",
       sents=tups, output_path="./ground_res_new_target_llava_capt.jsonl", num_processes=cpus)
