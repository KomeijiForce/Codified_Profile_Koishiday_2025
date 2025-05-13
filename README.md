# [Koishi's Day 2025](https://danbooru.donmai.us/wiki_pages/koishi_day): Codifying Character Logic in Role-Playing

- Let there be fantasy

## Fandom Benchmark

The Fandom Benchmark used in our experiments can be accessed via [this link](https://huggingface.co/datasets/KomeijiForce/Fandom_Benchmark).

## Distilled Condition Checkers

We also release the DeBERTa-V3 condition checkers distilled from GPT-4.1:

Base model (0.1B, 69% consistency with GPT-4.1): [this link](https://huggingface.co/KomeijiForce/deberta-v3-base-check-scene)

Large model (0.3B, 71% consistency with GPT-4.1): [this link](https://huggingface.co/KomeijiForce/deberta-v3-large-check-scene)

The dataset for distillation can be accessed via [this link](https://huggingface.co/datasets/KomeijiForce/Check_Scenes)

### How to use?

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load Condition Checker
classifier_path = "KomeijiForce/deberta-v3-base-check-scene"
classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
classifier = AutoModelForSequenceClassification.from_pretrained(classifier_path)

# Formalize Condition Checking into Prompt
scene = "Koishi quietly emerged from behind the dense foliage, her sudden appearance catching the corner of Satori's eye as she stepped into the direct line of vision, smiling mischievously."
question = "Does Koishi enter someone's direct field of vision?"

prompt = f'''Scene: {scene}

Question: {question}

Directly answer only yes/no/unknown.'''

# Scoring and Decoding
with torch.no_grad():
    logits = classifier(**classifier_tokenizer(prompt, return_tensors="pt")).logits[0]
    choice = logits.argmax(-1).item()

answer = [False, None, True][choice]

print(answer)
# True
```
