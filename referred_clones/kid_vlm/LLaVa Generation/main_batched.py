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

import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import json
from icecream import ic

# Initialize processor and model
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True
) 
model.to("cuda:0")

img_path = '/scratch/rahul.garg/hateful_memes/img/'

# List and sort all files in the directory
files = os.listdir(img_path)
files.sort()
files = files[start:end]

dataObj = {}
writeInterval = 2
batch_size = 8  # Adjust batch size based on your GPU memory
outputName = f'newCaptions_no_target_{start}_{end}.json'

if os.path.exists(outputName):
    with open(outputName) as json_file:
        dataObj = json.load(json_file)

# filter out already processed files
ic(len(files))
files = [file for file in files if file not in dataObj]
ic(len(files))

pbar = tqdm(range(0, len(files), batch_size))
iteration = 0

for i in pbar:
    batch_files = files[i:i+batch_size]
    images = []
    prompts = []
    
    # Prepare batch images and prompts
    for file in batch_files:
        image_file = f"{img_path}/{file}"
        image = Image.open(image_file)
        images.append(image)
        
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
        prompts.append(prompt)

    # Prepare inputs for the model
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to("cuda:0")
    
    # Generate output
    output = model.generate(**inputs, max_new_tokens=512, temperature=0.01)
    output_texts = processor.batch_decode(output, skip_special_tokens=True)
    
    # Process and store outputs
    for j, file in enumerate(batch_files):
        output_text = output_texts[j].split("[/INST]")[1].strip()
        dataObj[file] = output_text
        iteration += 1

    if iteration % writeInterval == 0:
        with open(outputName, 'w') as outfile:
            json.dump(dataObj, outfile, indent=4)

# Final save
with open(outputName, 'w') as outfile:
    json.dump(dataObj, outfile, indent=4)
