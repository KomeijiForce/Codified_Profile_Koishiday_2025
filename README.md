# [Koishi's Day 2025](https://danbooru.donmai.us/wiki_pages/koishi_day): Codifying Character Logic in Role-Playing

- Let there be fantasy

## Introduction [\[Link to Paper\]](https://arxiv.org/abs/2505.07705)

**Want your fictional characters to act logically? Why not codify their behavior?**

Codified Profiles turn character reasoning into executable functions, using structured control flow like `if-then-else` and condition queries (`check_condition`) to ensure consistent, interpretable, and updatable behavior across diverse scenes. Unlike prompt-only approaches that rely on the LLM's fragile implicit reasoning, codified profiles enable stable decision-making, controlled randomness, and strong performance - even with small models. Whether you're building story agents, simulation games, or interactive narratives, Codified Profiles offer a scalable and modular foundation for bringing characters to life with precision.

For example, text profile
```text
**Taking actions unconsciously**
As her actions are controlled entirely by her unconscious, Koishi's presence cannot be noticed by other beings unless she allows it,
even while standing in plain view because she's closed her mind, or her Third Eye.
Likewise, her older sister's ability to read minds has no effect on her, as she does not technically have a mind to read,
which is why she said "my sister can't win against me."
Even someone who manages to communicate with Koishi will forget about her after she leaves,
unless they already knew of her as Satori's sister.
```

will be codified as

```python
def parse_by_scene(scene):
    scene_statements = []

    # 1. Is Koishi present in the scene?
    if check_scene(scene, "Is Koishi present in the scene?"):
        # 2. Does Koishi allow herself to be noticed?
        if check_scene(scene, "Does Koishi allow herself to be noticed?"):
            scene_statements.append("Koishi is noticed by others in the scene.")
        else:
            scene_statements.append("Koishi is not noticed by anyone, even if she is in plain sight.")

        # 3. Does someone try to read Koishi's mind?
        if check_scene(scene, "Does someone try to read Koishi's mind?"):
            scene_statements.append("No one can read Koishi's mind; her mind is closed off.")

        # 4. Does Koishi interact or communicate with someone?
        if check_scene(scene, "Does Koishi interact or communicate with someone?"):
            # 5. Does the person already know Koishi as Satori's sister?
            if check_scene(scene, "Does the person already know Koishi as Satori's sister?"):
                scene_statements.append("After Koishi leaves, the person remembers her because they know her as Satori's sister.")
            else:
                scene_statements.append("After Koishi leaves, the person forgets about her.")

        # 6. Does Koishi take any actions?
        if check_scene(scene, "Does Koishi take any actions?"):
            scene_statements.append("Koishi's actions are unconscious and not planned.")

    return scene_statements
```

### Why codified profile?

Codified profiles provide persistent logic in role-playing, no longer relying on the model's implicit or explicit reasoning. This is good news for smaller role-playing models with weaker reasoning abilities, enabling 1B model to rival 8B model.

![1747126387253](https://github.com/user-attachments/assets/4be4b8f3-1d4e-4562-9f0e-de9a8a7a2fa1)

## Quick Start

You can quickly start with the demo in `quick_start.py`. You first need to create a `constant.py` file and place `openai_key="your key"` into it. Then you can upload the textual profile `{character}.profile.txt` to `profiles/` and then modify the `character`, `scene`, and `question` inside `quick_start.py` to enact role-playing with codified profiles.

The default setup is `llama-3.2-1b-instruct` for role-playing and the distilled `deberta-v3-base-check-scene` for condition checking, which enables a high-quality role-playing system on a **6G** GPU.

The example of `Koishi` in `quick_start.py` will provide you output with

```
Koishi spots Cirno giggling mischievously as she freezes a line of unsuspecting frogs along the riverbank.
----------------------------------------------------------------------------------------------------
What's the next action of Koishi to response to the scene?
----------------------------------------------------------------------------------------------------
Koishi acts immediately on her impulse, regardless of social norms or consequences.
Koishi behaves in a playful, childlike, and unpredictable manner.
Koishi performs her urban legend in a playful, not truly menacing, manner.
Koishi's presence goes unnoticed by others, even if she is in plain sight.
Koishi's actions are guided entirely by her unconscious.
Koishi is perceived as an 'imaginary friend' by the children, who will forget her as they grow up.
The act of Koishi's action is difficult for others to perceive.
Koishi does not read minds and may act oblivious to others' thoughts or intentions.
Koishi is undetectable and forgotten by everyone present.
Koishi makes them see illusions or confront their hidden weaknesses.
----------------------------------------------------------------------------------------------------
Koishi will then begin to dance around the frozen frogs, her movements playful and carefree, as if she's enjoying the absurdity of the situation.
```

## Fandom Benchmark

We construct Fandom Benchmark, a large-scale, behavior-centric evaluation suite for role-playing agents, consisting of 5,141 scenes and 83 characters from six popular fictional universes (e.g., AGOT, JOJO, FMA). Each scene pairs a grounded narrative context with a reference character action, enabling precise evaluation using NLI-based scoring. The benchmark focuses on complex situational reasoning beyond dialogue, supporting the assessment of character logic.

![image](https://github.com/user-attachments/assets/8fd4a3af-94ee-4899-9440-53b43372ccf2)

The Fandom Benchmark used in our experiments can be accessed via [this link](https://huggingface.co/datasets/KomeijiForce/Fandom_Benchmark).

The test cases from the full 8 seasons of AGOT for code evolving experiments can be accessed via [this link](https://huggingface.co/datasets/KomeijiForce/Fandom_Benchmark_Full_AGOT).

### Codified Profiles

The textual and codified profiles of characters in the Fandom benchmark can be accessed via [this link](https://huggingface.co/datasets/KomeijiForce/Fandom_Codified_Profiles).

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
