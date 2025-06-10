# [Koishi's Day 2025](https://danbooru.donmai.us/wiki_pages/koishi_day): Codifying Character Logic in Role-Playing

- Let there be fantasy

💚 Koishi's Day Project 2024: [[Active Passive Constraint for Fine-gained Role-playing]](https://github.com/KomeijiForce/Active_Passive_Constraint_Koishiday_2024)

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

| Artifact              | Haruhi | K-On! | JOJO  | FMA   | AGOT  | ATLA  | Average |
|-----------------------|--------|-------|-------|-------|-------|--------|---------|
| **Main**              |        |       |       |       |       |        |         |
| 1B + Text             | 51.60  | 59.73 | 52.76 | 59.11 | 54.25 | 58.54  | 56.00   |
| 1B + Code             | 55.83  | 60.80 | 54.28 | 60.32 | 58.96 | 60.37  | 58.43   |
| 1B + Code + Distill   | 62.25  | 61.40 | 55.01 | 59.96 | 60.09 | 61.97  | **60.21** |
| 8B + Text             | 70.53  | 68.47 | 60.64 | 68.02 | 61.52 | 66.70  | 65.98   |
| **Minor**             |        |       |       |       |       |        |         |
| 1B + Text              | N/A | 63.17  | 61.98 | 52.46 | 58.04 | 61.36 | 59.40   |
| 1B + Code              | N/A | 64.98  | 60.21 | 60.68 | 60.13 | 66.97 | 62.59   |
| 1B + Code + Distill    | N/A | 66.29  | 61.15 | 61.29 | 62.74 | 69.56 | **64.25**   |
| 8B + Text              | N/A | 65.88  | 64.70 | 65.83 | 66.21 | 65.88 | 65.70   |


## Quick Start

You can quickly start with the demo in `quick_start.py`. You first need to create a `constant.py` file and place `openai_key="your key"` into it. Then you can upload the textual profile `{character}.profile.txt` to `profiles/` and then modify the `character`, `scene`, and `question` inside `quick_start.py` to enact role-playing with codified profiles. You can also pass `random=True` to `codify` to enable controllable randomness in codified profiles.

**(OpenAI API is only required for codification and benchmarking, you can use pre-codified profiles for role-playing in [[Huggingface](https://huggingface.co/KomeijiForce)])**

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

### We are extending Fandom Benchmark to more artifacts

[[Extended Benchmark]](https://huggingface.co/datasets/KomeijiForce/Fandom_Benchmark_Extended) [[Codified Profiles]](KomeijiForce/Fandom_Extended_Codified_Profiles_Claude_4.0)


| Artifact              | Spy x Family | Jujutsu Kaisen | Little Witch Academia |
|-----------------------|----------|-----------|-----------|
| **Main**              |          |           |           |
| 1B + Text             | 60.62    |  60.34    |  64.35    |
| 1B + Code + Distill   | **63.20**|**63.92**  |**66.81**  |
| 8B + Text             | 69.78    |  64.86    |  68.19    |
| 8B + Code + Distill   | **70.69**|**65.69**  |**70.31**  |
| **Minor**             |          |           |           |
| 1B + Text             | 65.93    |  58.19    |  64.71    |
| 1B + Code + Distill   | **67.99**|**60.42**  |**68.28**  |
| 8B + Text             | 70.16    |  66.68    |  69.03    |
| 8B + Code + Distill   | **72.33**|**68.45**  |**71.29**  |

### Bandori (BanG Dream) Conversational Benchmark

[[Bandori Benchmark]](KomeijiForce/Fandom_Benchmark_Bandori) [[Codified Bandori Profiles]](KomeijiForce/Fandom_Benchmark_Bandori)

The scene in the **Bandori** benchmark is different from previous ones because the scene and response are in conversational formats like:

```
Arisa: I know you're on a roll right now Kasumi, but just calm down...
Kasumi: I can't! I can't calm down!
Arisa: I'm saying you're losing it, Kasumi.
Saaya: We know exactly what you mean, Kasumi.
Kasumi: Right?!
Rimi: Of course. We all felt it, singing together like that♪
Kasumi: Yay! I'm so glad you understand, Rimi-rin♪
Tae: Sharing moments like this is what's really important.
Kasumi: Sharing...! I wanna share this feeling with everybody! This sparkling, heart-pounding feeling!
```

| Band           | 🎸 Poppin'Party | 🌹 Roselia | 🎀 Pastel*Palettes | 🔥 Afterglow |
|----------------|----------------|------------|--------------------|--------------|
| 8B + Text      | 70.10          | 70.95      | 70.62              | 71.00        | 
| 8B + Code + Distill| **70.17**      |**71.90**  |**73.67**        | **71.72**   |
|                | 🎪 **Hello, Happy World!** | 🦋 **Morfonica** | ⚡ **RAISE A SUILEN** | 🧭 **MyGO!!!!!** |
| 8B + Text      | 74.10                  | 72.27        | 68.16            | 64.34        |
| 8B + Code + Distill| **75.94**         | **73.18**    | **70.67**     | **66.26**        |

Full results per character: [[Bandori Character Performance]](https://github.com/KomeijiForce/Codified_Profile_Koishiday_2025/blob/main/appendix/bandori_result.md)
The paired scene checker for conversational scenes: [[DeBERTa-V3-Base-Conversational-Scene-Checker]](https://huggingface.co/KomeijiForce/deberta-v3-base-check-conversational-scene)

### Benchmarking
The following script can be used to benchmark characters from the Fandom Benchmark, character names can be found in `all_characters.json`. The method name should be inside `["vanilla", "textual profile", "codified rag", "codified profile"]`. When `llm_path` and `classifier_path` are set to the same, the role-playing LLM will also do the classification.

```
python benchmark.py \
  --eval_engine "gpt-4.1" \
  --llm_path "meta-llama/Llama-3.2-1B-Instruct" \
  --classifier_path "KomeijiForce/deberta-v3-base-check-scene" \
  --character "Nagato" \
  --method "codified profile" \
  --device_id "7"
```

Other sets of codified:

[Codification w/ Claude 3.7](https://huggingface.co/datasets/KomeijiForce/Fandom_Codified_Profiles_Claude_3.7)

[Codification w/ Claude 4.0_sonnet](https://huggingface.co/datasets/KomeijiForce/Fandom_Codified_Profiles_Claude_4.0_sonnet) (Best Performance on Average)

[Codification w/ Randomness](https://huggingface.co/datasets/KomeijiForce/Fandom_Codified_Profiles_with_Randomness) (With Explicit Randomness in control flow)

| Artifact            | Haruhi | K-On! | JOJO  | FMA   | AGOT  | ATLA  | Average |
|---------------------|--------|-------|-------|-------|-------|--------|---------|
| **Main**            |        |       |       |       |       |        |         |
| gpt-4.1-mini        | 73.33  | 68.58 | 62.71 | 68.07 | 67.03 | 68.54  | 68.04   |
| gpt-4.1             | 72.14  | 69.23 | 63.92 | 69.60 | 67.51 | 67.90  | 68.38   |
| claude-3.7-sonnet    | 76.86  | 68.53 | 63.35 | 70.82 | 68.69 | 68.73  | 69.50 |
| claude-4.0-sonnet    | 75.07  | 68.99 | 63.85 | 74.26 | 67.22 | 70.82  | **70.04** |
| **Minor**           |        |       |       |       |       |        |         |
| gpt-4.1-mini        | N/A    | 71.65 | 64.66 | 70.35 | 69.77 | 70.43  | 69.37   |
| gpt-4.1             | N/A    | 68.59 | 70.06 | 68.21 | 70.51 | 71.95  | 69.87   |
| claude-3.7-sonnet    | N/A    | 75.05 | 67.05 | 67.04 | 73.79 | 69.25  | 70.44 |
| claude-4.0-sonnet    | N/A    | 71.79 | 67.66 | 70.94 | 74.63 | 73.72  | **71.75** |

To reproduce the results in the table, set both `llm_path` and `classifier_path` to `meta-llama/Llama-3.1-8B-Instruct`.

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
