import os
import re, json
from codify import codify
from datasets import load_dataset
from tqdm import tqdm

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

import warnings
from transformers import logging

logging.set_verbosity_error()

import openai
from constant import openai_key

import argparse

parser = argparse.ArgumentParser(description="Run character evaluation with specific settings.")
parser.add_argument("--eval_engine", type=str, default="gpt-4.1", help="Evaluation engine to use")
parser.add_argument("--llm_path", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Path to LLM")
parser.add_argument("--classifier_path", type=str, default="KomeijiForce/deberta-v3-base-check-scene", help="Path to classifier model")
parser.add_argument("--character", type=str, default="Haruhi", help="Character name")
parser.add_argument("--method", type=str, default="vanilla", help="Method to use")
parser.add_argument("--device_id", type=str, default="7", help="GPU device ID")

args = parser.parse_args()

eval_engine = args.eval_engine
llm_path = args.llm_path
classifier_path = args.classifier_path
character = args.character
method = args.method
device_id = args.device_id

device = torch.device(f"cuda:{device_id}")

openai.api_key = openai_key

tokenizer = AutoTokenizer.from_pretrained(llm_path)
model = AutoModelForCausalLM.from_pretrained(llm_path)
model = model.to(device)

if llm_path != classifier_path:

    classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
    classifier = AutoModelForSequenceClassification.from_pretrained(classifier_path)
    
    classifier = classifier.to(device)

def string_to_function(code_str, func_name):
    local_vars = {}
    exec(code_str, globals(), local_vars)
    return local_vars[func_name]

if method == "vanilla":
    grounding = ""
elif method == "textual profile":
    grounding = "\n\n".join([data["segment"] for data in load_dataset("KomeijiForce/Fandom_Codified_Profiles")["train"] if data["character"] == character])
elif method == "codified rag":
    questions = [data["question"] for data in load_dataset("KomeijiForce/Fandom_Codified_RAG")["train"] if data["character"] == character]
    segments = [data["segment"] for data in load_dataset("KomeijiForce/Fandom_Codified_RAG")["train"] if data["character"] == character]
elif method == "codified profile":
    functions = [string_to_function(data["code"], "parse_by_scene") for data in load_dataset("KomeijiForce/Fandom_Codified_Profiles")["train"]
                    if data["character"] == character]

def check_scene(text, question):

    if llm_path != classifier_path:
    
        prompt = f"Scene: {text}\n\nQuestion: {question}\n\nDirectly answer only yes/no/unknown."
    
        with torch.no_grad():
            choice = classifier(**classifier_tokenizer(prompt, return_tensors="pt").to(device)).logits[0].argmax(-1).item()
        
        answer = [False, None, True][choice]

    else:
        
        prompt = f"Scene: {text}\n\nQuestion: {question}\n\nDirectly answer only yes/no/unknown.\n\nAnswer:"
        ids = [tokenizer.encode(token)[1] for token in [' yes', ' no', ' unknown']]
    
        with torch.no_grad():
            choice = model(**tokenizer(prompt, return_tensors="pt").to(device)).logits[0, -1, ids].argmax(-1).item()
    
        answer = [True, False, None][choice]

    return answer

def generate(prompt):

    prompt = prompt+"\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=tokenizer.encode("assistant")[1]
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_text = generated_text[len(prompt):].strip().split("\n")[0].rstrip("assistant")
    
    return generated_text

def scoring(scene, prediction, reference):
    
    score_instruction = f'''# Scene
{scene}

Your Response: {prediction}
Ground Truth: {reference}

Score the action of {character} in the response based on the ground truth.
A: The ground truth entails the action of {character} in the response. (Following the same character logic.)
B: The ground truth is neutral to the action of {character} in the response. (Reflecting a different facet.)
C: The ground truth contradicts the action of {character} in the response. (Following a contradicted character logic.)

''' + '''Output in json: 
```json
{
"reasoning": "...",
"score": "A/B/C"
}
```'''

    score = openai.ChatCompletion.create(
    model=eval_engine,
    temperature=0.0,
    messages=[
        {"role": "user", "content": score_instruction},
    ],
    ).choices[0]['message']["content"]

    score = json.loads(re.findall(r"```json\n(.*)?\n```", score, re.DOTALL)[0].strip())["score"]
    score = {"A": 100, "B": 50, "C": 0}[score]

    return score

benchmark = [data for data in load_dataset("KomeijiForce/Fandom_Benchmark")["train"] if data["character"] == character]

scores = []

bar = tqdm(benchmark)
for data in bar:
    scene = data['scene']
    question = data['question']
    reference = data['reference']

    prompt = f'''# Scene
{scene}

# Question
{question} Answer a concise narration in one sentence.'''

    if method == "codified profile":
        grounding = "\n".join([statement for function in functions for statement in function(scene.split("\n\n")[-1])])
    elif method == "codified rag":
        grounding = "\n\n".join([segment for segment, question in zip(segments, questions) if check_scene(scene.split("\n\n")[-1], question)])

    prompt = f"# Background Knowledge\n{grounding}\n\n{prompt}"
    
    prediction = generate(prompt)
    score = scoring(scene, prediction, reference)
    scores.append(score)

    bar.set_description(f"Character '{character}': Score={np.mean(scores):.2f}")
