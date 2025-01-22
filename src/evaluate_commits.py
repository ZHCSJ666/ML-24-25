import json
import os
from pathlib import Path
from typing import Dict

import openai
import rootutils

ROOT = rootutils.setup_root(".", ".project-root", pythonpath=True)

from src.demo_inference import generate_commit_message, load_run

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class CommitMessageEvaluator:
    def __init__(self, openai_api_key: str, model_checkpoint_path: Path):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model, self.datamodule = load_run(model_checkpoint_path)

    def get_gpt4_commit_message(self, diff: str) -> str:
        """Generate a commit message using GPT-4."""
        prompt = f"""Given the following code diff, create a simplified commit message following these strict rules:
1. Maximum 8 words
2. Use only these verbs: added, updated, removed, fixed, refactored
3. Reference only code elements visible in the diff
4. Format: "<verb> <code_element> [brief_detail]"
5. Exclude all contextual information not visible in the code changes
6. Focus on the technical change, not the purpose or impact
7. Use consistent terminology for similar changes
Output only the simplified message without any explanation or formatting.
Code diff:
{diff}

Commit message:"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()

    def evaluate_messages(self, model_message: str, gpt_message: str, diff: str) -> Dict:
        """Compare and evaluate both commit messages."""
        prompt = f"""Given a code diff and two commit messages, evaluate them on a scale of 1-10 based on clarity,
        conciseness, and relevance to the changes. Provide a brief explanation.

        Code diff:
        {diff}

        Message 1: {model_message}
        Message 2: {gpt_message}

        Provide your response in JSON format:
        {{
            "model_score": <score>,
            "gpt_score": <score>,
            "explanation": "<explanation>"
        }}
        """

        response = self.client.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return json.loads(response.choices[0].message.content)

    def evaluate_sample(self, diff: str) -> Dict:
        """Evaluate a single diff sample using both models."""
        model_message = generate_commit_message(self.model, diff)
        gpt_message = self.get_gpt4_commit_message(diff)

        evaluation = self.evaluate_messages(model_message, gpt_message, diff)

        return {
            "diff": diff,
            "model_message": model_message,
            "gpt_message": gpt_message,
            "evaluation": evaluation,
        }


def main():
    evaluator = CommitMessageEvaluator(
        openai_api_key=OPENAI_API_KEY,
        model_checkpoint_path=ROOT
        / "logs/train/runs/2025-01-20_02-45-28/checkpoints/epoch_036-val_MRR_top5_0.6450.ckpt",
    )

    # Example usage with a test sample
    datamodule = evaluator.datamodule
    datamodule.setup()
    test_loader = datamodule.test_dataloader()

    # Evaluate first sample
    batch = next(iter(test_loader))

    # Get the tokenizer from the model and decode the input_ids
    tokenizer = evaluator.model.diff_tokenizer
    diff = tokenizer.decode(batch.input_ids[0], skip_special_tokens=True)

    result = evaluator.evaluate_sample(diff)
    print("Diff:", result["diff"][:200] + "...")
    print("\nModel message:", result["model_message"])
    print("GPT message:", result["gpt_message"])
    print("\nEvaluation:", json.dumps(result["evaluation"], indent=2))


if __name__ == "__main__":
    main()
