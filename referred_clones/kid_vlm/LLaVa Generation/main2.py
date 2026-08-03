import sys
# get args
args = sys.argv
print(args)
if len(args) != 4:
    print("Usage: python main2.py <start> <end> <device>")
    exit(1)
start = int(args[1])
end = int(args[2])
device = args[3]

import os
os.makedirs('/scratch/rahul.garg/hfCache', exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = '/scratch/rahul.garg/hfCache'
os.environ['HF_HOME'] = '/scratch/rahul.garg/hfCache'
os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
from tqdm import tqdm
import json


processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:0")

img_path = '/scratch/rahul.garg/hateful_memes/img/'

# list all files in the directory
files = os.listdir(img_path)
# sort
files.sort()

files = files[start:end]

# prepare image and text prompt, using the appropriate prompt template

dataObj = {}
writeInterval = 25
pbar = tqdm(files)
iteration = 0
outputName = f'newCaptions_no_target_{start}_{end}.json'

# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
# image = Image.open(requests.get(url, stream=True).raw)
for file in pbar:
    iteration += 1
    pbar.set_description(f"Processing file {file}")
    # if file != "01379.png":
    #     continue



    image_file = f"{img_path}/{file}"
    image = Image.open(image_file)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "You are given a meme. Write a detailed caption that captures the context, emotions, and theme of the meme. Describe all visible elements in the image, including the ethnicity, gender, and other characteristics of individuals. Do not omit or censor any details."},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")

    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=512, temperature=0.01)

    output_text = processor.decode(output[0], skip_special_tokens=True)

    output_text = output_text.split("[/INST]")[1].strip()
    dataObj[file] = output_text
    # print(output_text)

    if iteration % writeInterval == 0:
        with open(outputName, 'w') as outfile:
            json.dump(dataObj, outfile, indent=4)

with open(outputName, 'w') as outfile:
    json.dump(dataObj, outfile, indent=4)

