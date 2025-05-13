import os
import re, json
from codify import codify

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

character = "Koishi"

if not os.path.exists(f"codes/{character}.code.json"):

    textual_profile = open(f"profiles/{character}.profile.txt").read()
    
    codes = []
    
    for segment in textual_profile.split("\n\n"):
        code = codify(character, segment)
        codes.append(code)

    json.dump(codes, open(f"codes/{character}.code.json", "w"))

device = torch.device("cuda:1")

path = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path)
model = model.to(device)

classifier_path = "KomeijiForce/deberta-v3-base-check-scene"
classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
classifier = AutoModelForSequenceClassification.from_pretrained(classifier_path)

classifier = classifier.to(device)

def string_to_function(code_str, func_name):
    local_vars = {}
    exec(code_str, globals(), local_vars)
    return local_vars[func_name]

def check_scene(text, question):
    
    prompt = f"Scene: {text}\n\nQuestion: {question}\n\nDirectly answer only yes/no/unknown."

    with torch.no_grad():
        logits = classifier(**classifier_tokenizer(prompt, return_tensors="pt").to(device)).logits[0]
        choice = logits.argmax(-1).item()

    return [False, None, True][choice]

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
    
    # Decode and print
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_text = generated_text[len(prompt):].strip().split("\n")[0].rstrip("assistant")
    
    return generated_text

scene = "Koishi spots Cirno giggling mischievously as she freezes a line of unsuspecting frogs along the riverbank."
question = "What's the next action of Koishi to response to the scene?"

prompt = f'''# Scene
{scene}

# Question
{question} Answer a concise narration in one sentence.'''

codes = json.load(open(f"codes/{character}.code.json"))
functions = [string_to_function(code, "parse_by_scene") for code in codes]
grounding = "\n".join([statement for function in functions for statement in function(scene.split("\n\n")[-1])])

prompt = f"# Background Knowledge\n{grounding}\n\n{prompt}"

prediction = generate(prompt)

print(scene)
print("-"*100)
print(question)
print("-"*100)
print(grounding)
print("-"*100)
print(prediction)