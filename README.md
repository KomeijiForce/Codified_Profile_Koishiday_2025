# [Koishi's Day 2025](https://danbooru.donmai.us/wiki_pages/koishi_day): Codifying Character Logic in Role-Playing

- Let there be fantasy

## Introduction [\[Link to Paper\]](https://arxiv.org/abs/2505.07705)

**Want your fictional characters to act logically? Why not codify their behavior?**

Codified Profiles turn character reasoning into executable functions, using structured control flow like `if-then-else` and condition queries (`check_condition`) to ensure consistent, interpretable, and updatable behavior across diverse scenes. Unlike prompt-only approaches that rely on the LLM's fragile implicit reasoning, codified profiles enable stable decision-making, controlled randomness, and strong performance - even with small models. Whether you're building story agents, simulation games, or interactive narratives, Codified Profiles offer a scalable and modular foundation for bringing characters to life with precision.

## Fandom Benchmark

We construct Fandom Benchmark, a large-scale, behavior-centric evaluation suite for role-playing agents, consisting of 5,141 scenes and 83 characters from six popular fictional universes (e.g., AGOT, JOJO, FMA). Each scene pairs a grounded narrative context with a reference character action, enabling precise evaluation using NLI-based scoring. The benchmark focuses on complex situational reasoning beyond dialogue, supporting the assessment of character logic.

![image](https://github.com/user-attachments/assets/8fd4a3af-94ee-4899-9440-53b43372ccf2)


The Fandom Benchmark used in our experiments can be accessed via [this link](https://huggingface.co/datasets/KomeijiForce/Fandom_Benchmark).

## Distilled Condition Checkers

We also release the DeBERTa-V3 condition checkers distilled from GPT-4.1:

Base model (0.1B, 71% consistency with GPT-4.1): [this link](https://huggingface.co/KomeijiForce/deberta-v3-base-check-scene)

Large model (0.3B, 72% consistency with GPT-4.1): [this link](https://huggingface.co/KomeijiForce/deberta-v3-large-check-scene)

The dataset (20,759 cases) for distillation can be accessed via [this link](https://huggingface.co/datasets/KomeijiForce/Check_Scenes)

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

## Citation
```bibtex
@article{codified_profile,
  title={Codifying Character Logic in Role-Playing},
  author={Peng, Letian and Shang, Jingbo},
  journal={arXiv preprint arXiv:2505.07705},
  year={2025}
}
```
