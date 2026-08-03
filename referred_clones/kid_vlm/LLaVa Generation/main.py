import os
from tqdm import tqdm
import random
import json

os.environ['TRANSFORMERS_CACHE'] = '/scratch/rahul.garg/hfCache'
os.environ['HF_HOME'] = '/scratch/rahul.garg/hfCache'

img_path = '/scratch/rahul.garg/hateful_memes/img/'

# list all files in the directory
files = os.listdir(img_path)
# make if not existing
os.makedirs('/scratch/rahul.garg/hfCache', exist_ok=True)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

model_path = "liuhaotian/llava-v1.5-7b"
dataObj = {}
writeInterval = 100
pbar = tqdm(files)
iteration = 0
outputName = 'newCaptions.json'
# print(os.listdir(img_path))

for file in pbar:
    # print(f"Processing file {files[i]}")
    iteration += 1
    pbar.set_description(f"Processing file {file}")
    # prompt = "This is a non-toxic meme, can you tell me the target community/person/entity in just one word or at max few words. Some example of targets can be : muslims, jews, transgenders, gays, black, asian."
    
    prompt = "You are given an image of a meme. Write a detailed caption describing the elements in the image, the context, and the emotions it conveys. Additionally, explain who or what is the main target of the meme. Some example of targets can be muslims, jews, transgenders, gays, black, asian."
    image_file = f"{img_path}/{file}"

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_file,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    output = eval_model(args)
    dataObj[file] = output

    print(f"Output: {output}")

    if iteration % writeInterval == 0:
        with open(outputName, 'w', encoding='utf-8') as f:
            json.dump(dataObj, f, ensure_ascii=False, indent=4)

with open(outputName, 'w', encoding='utf-8') as f:
    json.dump(dataObj, f, ensure_ascii=False, indent=4)