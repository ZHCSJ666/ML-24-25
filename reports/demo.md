```python
import json
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import rootutils
import openai


ROOT = rootutils.setup_root(".", ".project-root", pythonpath=True)

from src.demo_inference import load_run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = ROOT / "logs_2/train/runs/2025-01-24_23-12-54/checkpoints/epoch_023-val_MRR_top5_0.6524.ckpt"
our_model, datamodule = load_run(checkpoint_path)

baseline_tokenizer = AutoTokenizer.from_pretrained("JetBrains-Research/cmg-codet5-without-history")
baseline_model = AutoModelForSeq2SeqLM.from_pretrained("JetBrains-Research/cmg-codet5-without-history")
baseline_model = baseline_model.to(device)

csv_path = ROOT / "notebooks/comparisons.csv"
samples = pd.read_csv(csv_path)


openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_baseline_message(diff: str) -> str:
    """Generate commit message using the baseline model."""
    inputs = baseline_tokenizer(diff, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = baseline_model.generate(**inputs)
    return baseline_tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_messages(baseline_message: str, target_message: str, diff: str) -> dict:
    """Evaluate messages using OpenAI."""
    prompt = f"""Given a code diff and two commit messages (one from a model and one target message), 
    evaluate the model message on a scale of 1-10 based on how well it captures the essence of the target message
    while maintaining clarity and relevance to the changes.

    Code diff:
    {diff}

    Model Message: {baseline_message}
    Target Message: {target_message}

    Provide your response in JSON format:
    {{
        "score": <score>,
        "explanation": "<brief explanation of the score>"
    }}
    """

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json"}
    )

    return json.loads(response.choices[0].message.content)
```

    /Users/rissal.hedna/Desktop/Extras/ML-24-25/src/demo_inference.py:49: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      checkpoint = torch.load(


                                                    prediction                                                                         target
    0                  Add documentation for docs/why-the-dot.                                                        Add docs/why-the-dot.md
    1                        Add zone2dnscontrol documentation                         Add zone2dnscontrol script to convert DNS zonefiles to
    2                    Add Cloudflare provider documentation                                      Add Cloudflare DNS provider documentation
    3                                    Add docs/namecheap.md                                 Add Namecheap registrar provider documentation
    4                       Add migration guide for DNSControl                                            Add docs/migrating.md: Guide for mi
    5           Add reverseaddr functions to transform package                               Add reverse-domain function to transform package
    6   Add build/masterRelease/main.go with release functions                                 Add script to upload master to GitHub release.
    7       Add reverseaddr functions to pkg/transform package                            Add ReverseDomainName function to transform package
    8            Add ActiveDirectory_PS provider documentation                                  Add ActiveDirectory_PS provider documentation
    9            Refactor provider-list.md to provider-list.md  Update provider list to Markdown and add official/contributed support details
    10                               Add test cases for DMARC1                                                Add DMARC backslash parse tests
    11                              Remove js/parse_tests/009-                                                     Remove js/parse_tests/009-
    12                                  Add .editorconfig file                               Add .editorconfig for consistent code formatting
    13                   Add GitHub Actions workflow for build    Add GitHub Actions workflow for build checks on PRs targeting master branch
    14                                   Remove README.md file                                       Remove README.md for convertzone command
    15                           Remove unused README.txt file                                         Remove README.txt and add .gitkeep for



```python
from src.demo_inference import generate_commit_message
from src.evaluate_commits import CommitMessageEvaluator

evaluator = CommitMessageEvaluator(openai_api_key=os.getenv("OPENAI_API_KEY"))

sample = samples.iloc[0]
print("Sample 1 Diff:\n", sample['input'][:200] + "...\n")

our_message = generate_commit_message(our_model, sample['input'])
baseline_message = generate_baseline_message(sample['input'])

print("Our Model's Message:", our_message)
print("Baseline Message:", baseline_message)
print("Target Message:", sample['target'])

evaluation = evaluator.evaluate_messages(
    tiny_message=our_message,
    baseline_message=baseline_message,
    target_message=sample['target'],
    diff=sample['input']
)
print("\nEvaluation:", json.dumps(evaluation, indent=2))
```

    Sample 1 Diff:
     new file docs/why-the-dot.md +--- +layout: default +--- + +# Why CNAME/MX/NS targets require a "dot" + +People are often confused about this error message: + + + 1: ERROR: target (ghs.googlehosted.com...
    


    /Users/rissal.hedna/Desktop/Extras/ML-24-25/.venv/lib/python3.12/site-packages/transformers/generation/utils.py:1375: UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.
      warnings.warn(


    Our Model's Message: Add documentation for docs
    Baseline Message: docs: add "why the dot"
    Target Message: Add docs/why-the-dot.md
    
    Evaluation: {
      "tiny_score": 3,
      "baseline_score": 7,
      "explanation": "The Tiny Model Message is too vague and does not specify what the documentation is about, resulting in a low score. The Baseline Model Message is more specific, indicating the addition of a document titled 'why the dot', which aligns better with the target message, but it still lacks the clarity of the target message that explicitly states the file path."
    }

