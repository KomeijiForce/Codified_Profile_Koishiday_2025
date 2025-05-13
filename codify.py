import openai
import json
import re
from constant import openai_key
from tqdm.notebook import tqdm
import numpy as np
model_engine = "gpt-4.1"
openai.api_key = openai_key

def codify(character, profile, random=False):

    prompt = f'''Character: {character}

Profile: {profile}

Complete the following tasks one by one:

1. Explain the character logic indicated by the given profile. What character behavior can be triggered by what kind of conditions?

2. Compile the character logic in this profile statement into a function parse_by_scene(scene), which takes a string scene as the input and outputs triggered character behavior (a list of string scene_statement) scene_statements. You can use a pre-defined function check_scene(scene, question) that outputs True/False/None (Unknown) to build the logic. Don't use check_condition too many times to ensure efficiency.'''
    
    if random:
        prompt = f'''{prompt} When facing randomness, you shall use `random.random() < p` or `random.choice`.'''


    icl_input = '''Character: Haruhi

Profile: Haruhi has a dominant personality and is skilled at persuading others to follow her lead, using charm, intimidation, or sheer charisma. She is straightforward and assertive, often expecting others to comply with her demands.

Complete the following tasks one by one:

1. Explain the character logic indicated by the given profile. What character behavior can be triggered by what kind of conditions?

2. Compile the character logic in this profile statement into a function parse_by_scene(scene), which takes a string scene as the input and outputs triggered character behavior (a list of string scene_statement) scene_statements. You can use a pre-defined function check_scene(scene, question) that outputs True/False/None (Unknown) to build the logic. Don't use check_condition too many times to ensure efficiency.'''

    icl_code = '''Certainly! Let's proceed step by step.

---

**Task 1: Explain the character logic indicated by the given profile.**

**Profile Summary:**  
Haruhi is dominant, persuasive, and expects compliance. She uses charm, intimidation, or charisma to get her way. She is straightforward and assertive.

**Character Logic Explanation:**

- **Trigger: When Haruhi wants something or has a goal.**  
  - **Behavior:** She will assertively state her desire or demand, expecting others to comply.

- **Trigger: When others resist or question her.**  
  - **Behavior:** She will use persuasion, either by charming them, intimidating them, or leveraging her charisma to win them over.

- **Trigger: When someone is indecisive or hesitant.**  
  - **Behavior:** She will take charge, make decisions for the group, and expect others to follow.

- **Trigger: When someone tries to lead or dominate her.**  
  - **Behavior:** She will push back, reassert her authority, and make it clear she is in charge.

- **Trigger: When things are not going her way.**  
  - **Behavior:** She will become more forceful, possibly impatient, and double down on her demands.

**Summary Table:**

| Condition                                      | Triggered Behavior                                      |
|------------------------------------------------|---------------------------------------------------------|
| Haruhi wants something                         | Assertive demand, expects compliance                    |
| Others resist/question                         | Persuasion via charm/intimidation/charisma              |
| Others are indecisive/hesitant                 | Takes charge, makes decisions                           |
| Someone tries to dominate/lead her             | Pushes back, reasserts authority                        |
| Things not going her way                       | Becomes more forceful/impatient                         |

---

**Task 2: Compile the character logic into a function `parse_by_scene(scene)`**

Assume `check_scene(scene, question)` is available and returns True/False/None.

```python
def parse_by_scene(scene):
    scene_statements = []

    # 1. Does Haruhi want something or have a goal in this scene?
    if check_scene(scene, "Does Haruhi want something or have a goal?"):
        scene_statements.append("Haruhi assertively states her desire and expects others to comply.")

        # 2. Do others resist or question her?
        resists = check_scene(scene, "Do others resist or question Haruhi?")
        if resists:
            scene_statements.append("Haruhi uses charm, intimidation, or charisma to persuade them to follow her lead.")

        # 3. Are others indecisive or hesitant?
        if check_scene(scene, "Are others indecisive or hesitant?"):
            scene_statements.append("Haruhi takes charge and makes decisions for the group, expecting others to follow.")

    # 4. Is someone trying to lead or dominate Haruhi?
    if check_scene(scene, "Is someone trying to lead or dominate Haruhi?"):
        scene_statements.append("Haruhi pushes back and reasserts her authority, making it clear she is in charge.")

    # 5. Are things not going Haruhi's way?
    if check_scene(scene, "Are things not going Haruhi's way?"):
        scene_statements.append("Haruhi becomes more forceful and impatient, doubling down on her demands.")

    return scene_statements
```

**Notes:**
- The function checks the most likely/important triggers first for efficiency.
- It avoids redundant checks and only uses `check_scene` as needed.
- The output is a list of scene statements describing Haruhi's triggered behavior in the given scene.'''

    response = openai.ChatCompletion.create(
model=model_engine,
temperature=1e-8,
messages=[
    {"role": "user", "content": icl_input},
    {"role": "assistant", "content": icl_code},
    {"role": "user", "content": prompt},
],
).choices[0]['message']["content"]
    
    code = re.findall(r"```python\n(.*?)\n```", response, re.DOTALL)[0]

    return code