import os
import json
from transformers import pipeline
from tqdm import tqdm

# Set the cache directory for transformers
cache_dir = '/scratch/rahul.garg/sentence_transformers/transformers_cache'
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HF_HOME'] = cache_dir
# Ensure the cache directory exists
os.makedirs(cache_dir, exist_ok=True)

# Initialize the summarizer pipeline with the cache_dir specified in the environment variable
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def getTripleSentence(triples, tripleK=50):
    sentences = [" ".join(triple) for triple in triples[:tripleK]]
    sentence = ". ".join(sentences)
    # Note: Removed the cache_dir argument from here as it's not valid for the generate method
    obj = summarizer(sentence, max_length=250, min_length=30, do_sample=False)
    return triples[:tripleK], obj[0]['summary_text']

# Function to count the total number of lines in the input file
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

# Count total lines for tqdm
total_lines = file_len('./closest_triples_with_names.jsonl')

# Open the output file
outputFile = open('./triples_Summaries.jsonl', 'w')

# Process the input file with tqdm progress bar
with open('./closest_triples_with_names.jsonl', 'r') as f:
    for line in tqdm(f, total=total_lines, desc="Processing"):
        jsonObj = json.loads(line)
        name = int(jsonObj['name'])
        triples = jsonObj['closest_triples']
        cut_off_triples, summary = getTripleSentence(triples)
        newJson = {
            'name': name,
            'triples': cut_off_triples,
            'summary': summary
        }
        outputFile.write(json.dumps(newJson) + '\n')
        outputFile.flush()

outputFile.close()
