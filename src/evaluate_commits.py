import json
import os
import time
from typing import Dict

import openai
import pandas as pd
import rootutils

ROOT = rootutils.setup_root(".", ".project-root", pythonpath=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class CommitMessageEvaluator:
    """A class that evaluates commit messages from different models against target messages."""

    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)

    def evaluate_messages(
        self, tiny_message: str, baseline_message: str, target_message: str, diff: str
    ) -> Dict:
        """Compare and evaluate both model messages against the target message."""
        prompt = f"""Given a code diff and three commit messages (two from different models and one target message),
        evaluate the model messages on a scale of 1-10 based on how well they capture the essence of the target message
        while maintaining clarity and relevance to the changes.

        Code diff:
        {diff}

        Tiny Model Message: {tiny_message}
        Baseline Model Message: {baseline_message}
        Target Message: {target_message}

        Provide your response in JSON format:
        {{
            "tiny_score": <score>,
            "baseline_score": <score>,
            "explanation": "<brief explanation of the scores>"
        }}
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)

    def evaluate_sample(self, row: pd.Series) -> Dict:
        """Evaluate a single sample from the CSV data."""
        evaluation = self.evaluate_messages(
            tiny_message=row["t5-efficient-extra-tiny"],
            baseline_message=row["baseline-cmg-codet5-without-history"],
            target_message=row["target"],
            diff=row["input"],
        )

        return {
            "diff": row["input"],
            "tiny_message": row["t5-efficient-extra-tiny"],
            "baseline_message": row["baseline-cmg-codet5-without-history"],
            "target_message": row["target"],
            "evaluation": evaluation,
        }

    def create_batch_prompt(
        self, tiny_message: str, baseline_message: str, target_message: str, diff: str, index: int
    ) -> Dict:
        """Create a prompt for batch processing without explanation field."""
        timestamp = int(time.time())
        return {
            "custom_id": f"commit_eval_{timestamp}_{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Given a code diff and three commit messages (two from different models and one target message),
evaluate the model messages on a scale of 1-10 based on how well they capture the essence of the target message
while maintaining clarity and relevance to the changes.
Code diff:
{diff.strip()}
Tiny Model Message: {tiny_message}
Baseline Model Message: {baseline_message}
Target Message: {target_message}

Provide your response in JSON format:
{{
    "tiny_score": <score>,
    "baseline_score": <score>
}}""",
                    }
                ],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
        }

    def prepare_batch_data(self, df: pd.DataFrame, batch_size: int, output_file: str) -> str:
        """Prepare batch data from DataFrame and save to JSONL file."""
        batch_prompts = []
        for idx, (_, row) in enumerate(df.head(batch_size).iterrows()):
            prompt = self.create_batch_prompt(
                tiny_message=row["t5-efficient-extra-tiny"],
                baseline_message=row["baseline-cmg-codet5-without-history"],
                target_message=row["target"],
                diff=row["input"],
                index=idx,
            )
            batch_prompts.append(prompt)

        # Save to JSONL file
        with open(output_file, "w") as f:
            for prompt in batch_prompts:
                f.write(json.dumps(prompt) + "\n")

        return output_file

    def process_batch_results(self, results_file: str) -> Dict:
        """Process batch results and calculate averages."""
        tiny_scores = []
        baseline_scores = []

        with open(results_file) as f:
            for line in f:
                result = json.loads(line)
                response = json.loads(result["choices"][0]["message"]["content"])
                tiny_scores.append(response["tiny_score"])
                baseline_scores.append(response["baseline_score"])

        return {
            "tiny_average": sum(tiny_scores) / len(tiny_scores),
            "baseline_average": sum(baseline_scores) / len(baseline_scores),
            "sample_count": len(tiny_scores),
        }


def batch_evaluate(batch_size: int = 100):
    """Run batch evaluation on a specified number of samples."""
    evaluator = CommitMessageEvaluator(openai_api_key=OPENAI_API_KEY)

    # Load CSV file
    csv_path = ROOT / "notebooks/comparisons.csv"
    df = pd.read_csv(csv_path)

    # Create batches directory if it doesn't exist
    batch_dir = ROOT / "notebooks/batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Prepare batch data
    batch_file = batch_dir / "evaluation_batch.jsonl"
    evaluator.prepare_batch_data(df, batch_size, batch_file)

    # Initialize OpenAI client for batch processing
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    # Submit batch request
    with open(batch_file, "rb") as f:
        file_response = client.files.create(file=f, purpose="batch")
        file_id = file_response.id

    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    batch_id = batch_response.id
    print(f"Batch submitted with ID: {batch_id}")

    # Check batch status until complete
    while True:
        status = client.batches.retrieve(batch_id)
        print(f"Batch status: {status.status}")
        if status.status == "completed":
            break
        elif status.status == "failed":
            print("Batch processing failed!")
            return
        time.sleep(60)

    # Download and process results
    output_file = batch_dir / "evaluation_results.jsonl"
    output_content = client.files.content(status.output_file_id).text
    with open(output_file, "w") as f:
        f.write(output_content)

    # Calculate and display averages
    results = evaluator.process_batch_results(output_file)
    print("\nBatch Evaluation Results:")
    print(f"Number of samples processed: {results['sample_count']}")
    print(f"Average score for tiny model: {results['tiny_average']:.2f}")
    print(f"Average score for baseline model: {results['baseline_average']:.2f}")


if __name__ == "__main__":
    batch_evaluate(batch_size=500)
