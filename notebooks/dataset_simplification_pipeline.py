import json
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from huggingface_hub import login
from openai import OpenAI

LANGUAGES = ["Go"]
CHANGE_TYPES = ["ADD"]
ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = ROOT / "data"
INPUT_JSONL_DIR = DATA_DIR / "jsonl"
BATCHES_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"
MERGED_DIR = DATA_DIR / "merged"
HUGGINGFACE_REPO = None
FORMATTED_BATCHES_DIR = DATA_DIR / "formatted_batches"


def load_arrow_dataset(arrow_dir):
    """Load and filter an Arrow dataset based on specified languages and change types.

    Args:
        arrow_dir (str): Path to the Arrow dataset directory.

    Returns:
        pandas.DataFrame: Filtered dataset converted to pandas DataFrame.

    Raises:
        ValueError: If dataset is empty after filtering.
    """
    try:
        dataset = load_from_disk(arrow_dir)
        print(f"Initial dataset info: {dataset}")

        dataset = dataset.filter(
            lambda example: example["language"] in LANGUAGES, num_proc=mp.cpu_count()
        ).filter(
            lambda example: all(mod["change_type"] in CHANGE_TYPES for mod in example["mods"]),
            num_proc=mp.cpu_count(),
        )

        print(f"Filtered dataset info: {dataset}")

        if len(dataset) == 0:
            raise ValueError("Dataset is empty after filtering")

        return dataset.to_pandas()
    except Exception as e:
        print(f"Error loading/filtering Arrow dataset from {arrow_dir}: {str(e)}")
        raise


def assign_custom_id(df):
    """Assign a unique UUID to each row in the DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with added 'custom_id' column.
    """
    df = df.copy()
    df["custom_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    return df


def convert_to_jsonl(df, output_path):
    """Convert DataFrame to JSONL format and save to specified path.

    Args:
        df (pandas.DataFrame): DataFrame to convert
        output_path (str): Path where JSONL file will be saved
    """
    df.to_json(output_path, orient="records", lines=True)
    print(f"Saved JSONL to {output_path}")


def load_jsonl(file_path):
    """Load data from a JSONL file.

    Args:
        file_path (str): Path to JSONL file

    Returns:
        list: List of dictionaries containing the loaded JSON data
    """
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def split_batches(input_file_path, output_dir, max_tokens_per_batch=5000):
    """Split JSONL file into smaller batches based on token count.

    Args:
        input_file_path (str): Path to input JSONL file
        output_dir (str): Directory to save batch files
        max_tokens_per_batch (int, optional): Maximum tokens per batch. Defaults to 5000.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file_path) as f:
        lines = f.readlines()

    current_batch = []
    current_entries = 0
    current_tokens = 0
    batch_index = 0

    for line in lines:
        payload = json.loads(line)
        num_tokens = len(payload.get("message", "").split())

        if current_tokens + num_tokens > max_tokens_per_batch or current_entries >= 500:
            batch_file_path = os.path.join(output_dir, f"batch_{batch_index}.jsonl")
            with open(batch_file_path, "w") as batch_file:
                batch_file.writelines(current_batch)
            print(
                f"Saved batch {batch_index} with {current_entries} entries and {current_tokens} tokens to {batch_file_path}"
            )

            current_batch = []
            current_tokens = 0
            current_entries = 0
            batch_index += 1

        current_batch.append(line)
        current_tokens += num_tokens
        current_entries += 1

    if current_batch:
        batch_file_path = os.path.join(output_dir, f"batch_{batch_index}.jsonl")
        with open(batch_file_path, "w") as batch_file:
            batch_file.writelines(current_batch)
        print(
            f"Saved batch {batch_index} with {current_entries} entries and {current_tokens} tokens to {batch_file_path}"
        )


def upload_to_openai(batch_file_path, output_file_path):
    """Upload batch file to OpenAI API and save responses.

    Args:
        batch_file_path (str): Path to batch file to process
        output_file_path (str): Path where API responses will be saved
    """
    print(f"Processing {batch_file_path}...")

    client = OpenAI()
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    os.makedirs(FORMATTED_BATCHES_DIR, exist_ok=True)

    try:
        batch_data = []
        with open(batch_file_path) as f:
            for line in f:
                entry = json.loads(line)
                batch_data.append(prepare_batch_entry(entry))

        batch_name = Path(batch_file_path).stem
        formatted_batch_path = FORMATTED_BATCHES_DIR / f"{batch_name}_formatted.jsonl"
        with open(formatted_batch_path, "w") as f:
            for entry in batch_data:
                f.write(json.dumps(entry) + "\n")

        print(f"Saved formatted batch to {formatted_batch_path}")

        with open(formatted_batch_path, "rb") as f:
            file_response = client.files.create(file=f, purpose="batch")
            file_id = file_response.id

        os.remove(formatted_batch_path)

        batch_response = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"Batch for {batch_file_path}"},
        )
        batch_id = batch_response.id
        print(f"Submitted batch {batch_id} for {batch_file_path}")

        while True:
            status = client.batches.retrieve(batch_id)
            print(f"Batch status: {status.status}")

            if status.status == "completed":
                output_file_id = status.output_file_id
                output_content = client.files.content(output_file_id).text
                with open(output_file_path, "w") as f:
                    f.write(output_content)
                print(f"Results saved to {output_file_path}")
                break
            elif status.status == "failed":
                print(f"Batch {batch_file_path} failed. Attempting to split...")
                split_and_process_batch(batch_file_path, output_file_path, client)
                break

            time.sleep(60)

    except Exception as e:
        print(f"Error processing batch {batch_file_path}: {str(e)}")
        raise


def split_and_process_batch(batch_file_path, output_file_path, client):
    """Split failed batch into smaller chunks and retry processing.

    Args:
        batch_file_path (str): Path to original batch file
        output_file_path (str): Path where final output will be saved
        client (OpenAI): OpenAI client instance
    """
    base_dir = os.path.dirname(batch_file_path)
    base_name = os.path.basename(batch_file_path)
    name_without_ext = os.path.splitext(base_name)[0]

    entries = []
    with open(batch_file_path) as f:
        for line in f:
            entry = json.loads(line)
            entries.append(entry)

    mid_point = len(entries) // 2
    chunks = [entries[:mid_point], entries[mid_point:]]

    all_responses = []
    for i, chunk in enumerate(chunks):
        chunk_file_path = os.path.join(base_dir, f"{name_without_ext}_split_{i}.jsonl")

        formatted_entries = [prepare_batch_entry(entry) for entry in chunk]
        with open(chunk_file_path, "w") as f:
            for entry in formatted_entries:
                f.write(json.dumps(entry) + "\n")

        try:
            with open(chunk_file_path, "rb") as f:
                file_response = client.files.create(file=f, purpose="batch")
                file_id = file_response.id

            batch_response = client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": f"Split batch for {chunk_file_path}"},
            )

            while True:
                status = client.batches.retrieve(batch_response.id)
                if status.status == "completed":
                    output_file_id = status.output_file_id
                    chunk_responses = client.files.content(output_file_id).text
                    all_responses.extend(json.loads(line) for line in chunk_responses.splitlines())
                    break
                elif status.status == "failed":
                    print(f"Split chunk {chunk_file_path} failed")
                    split_and_process_batch(chunk_file_path, output_file_path, client)
                    break
                time.sleep(60)

        finally:
            if os.path.exists(chunk_file_path):
                os.remove(chunk_file_path)

    if all_responses:
        with open(output_file_path, "w") as f:
            for response in all_responses:
                f.write(json.dumps(response) + "\n")


def load_responses(output_dir):
    """Load API responses from output directory.

    Args:
        output_dir (str): Directory containing response files

    Returns:
        dict: Dictionary mapping custom_ids to API responses
    """
    responses = {}
    for file in Path(output_dir).glob("*.jsonl"):
        with open(file) as f:
            for line in f:
                entry = json.loads(line)
                responses[entry["custom_id"]] = entry["response"]["body"]["choices"][0]["message"][
                    "content"
                ]
    return responses


def merge_with_custom_id(input_jsonl, responses, output_jsonl):
    """Merge original data with API responses using custom_id.

    Args:
        input_jsonl (str): Path to input JSONL file
        responses (dict): Dictionary of API responses
        output_jsonl (str): Path where merged data will be saved
    """
    merged_data = []
    with open(input_jsonl) as f_in, open(output_jsonl, "w") as f_out:
        for line in f_in:
            entry = json.loads(line)
            custom_id = entry["custom_id"]
            if custom_id in responses:
                entry["message"] = responses[custom_id]
                merged_data.append(entry)
                f_out.write(json.dumps(entry) + "\n")
    print(f"Merged data saved to {output_jsonl}")


def load_arrow_dataset_as_jsonl():
    """Load Arrow dataset and convert to JSONL format for each split."""
    os.makedirs(INPUT_JSONL_DIR, exist_ok=True)

    splits = ["train", "test", "validation"]
    for split in splits:
        arrow_dir = DATA_DIR / "playground" / f"01-filtered-{split}"
        df = load_arrow_dataset(arrow_dir)
        df = assign_custom_id(df)
        output_jsonl = INPUT_JSONL_DIR / f"{split}.jsonl"
        convert_to_jsonl(df, output_jsonl)


def create_and_upload_batches():
    """Create batches from JSONL files and upload them to OpenAI API."""
    os.makedirs(BATCHES_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    splits = ["train", "test", "validation"]
    for split in splits:
        input_jsonl = INPUT_JSONL_DIR / f"{split}.jsonl"
        batch_output_dir = BATCHES_DIR / split
        split_output_dir = OUTPUTS_DIR / split

        os.makedirs(batch_output_dir, exist_ok=True)
        os.makedirs(split_output_dir, exist_ok=True)

        print(f"Creating batches for {split} split...")
        split_batches(input_jsonl, batch_output_dir)

    for split in splits:
        batch_dir = BATCHES_DIR / split
        batch_files = list(Path(batch_dir).glob("*.jsonl"))
        if not batch_files:
            raise ValueError(f"No batches found for {split} split in {batch_dir}")
        print(f"Found {len(batch_files)} batches for {split} split")

    for split in splits:
        print(f"\nProcessing {split} split batches...")
        batch_output_dir = BATCHES_DIR / split
        split_output_dir = OUTPUTS_DIR / split

        batch_files = sorted(Path(batch_output_dir).glob("*.jsonl"))
        processed_batches = set()

        for output_file in Path(split_output_dir).glob("*_output.jsonl"):
            batch_name = output_file.stem.replace("_output", "")
            processed_batches.add(batch_name)

        for batch_file in batch_files:
            if batch_file.name.endswith("_output.jsonl") or batch_file.stem in processed_batches:
                print(f"Skipping already processed batch: {batch_file.name}")
                continue

            output_file = split_output_dir / f"{batch_file.stem}_output.jsonl"

            if output_file.exists() and output_file.stat().st_size > 0:
                print(f"Skipping batch {batch_file.name} - output already exists")
                continue

            print(f"Processing batch: {batch_file.name}")
            upload_to_openai(str(batch_file), str(output_file))


def merge_and_save_final_dataset():
    """Merge API responses with original data and save final dataset."""
    os.makedirs(MERGED_DIR, exist_ok=True)

    splits = ["train", "test", "validation"]
    for split in splits:
        input_jsonl = INPUT_JSONL_DIR / f"{split}.jsonl"
        output_file = MERGED_DIR / f"merged_{split}.jsonl"

        split_output_dir = OUTPUTS_DIR / split
        responses = load_responses(split_output_dir)

        merge_with_custom_id(input_jsonl, responses, output_file)


def upload_to_huggingface():
    """Upload merged dataset to Hugging Face Hub."""
    splits = ["train", "test", "validation"]
    dataset_dict = {}

    for split in splits:
        merged_jsonl = MERGED_DIR / f"merged_{split}.jsonl"
        data = load_jsonl(merged_jsonl)
        dataset = Dataset.from_list(data)
        dataset_dict[split] = dataset
        print(f"Loaded {len(data)} examples for {split} split.")

    dataset_hf = DatasetDict(dataset_dict)

    print("\nDataset info:")
    for split, dataset in dataset_hf.items():
        print(f"\n{split} split:")
        print(dataset)

    login()

    dataset_hf.push_to_hub(HUGGINGFACE_REPO, private=False)

    print("\nDataset successfully uploaded to Hugging Face Hub!")


def prepare_batch_entry(entry):
    """Prepare an entry for batch processing with OpenAI API.

    Args:
        entry (dict): Input entry containing commit data.

    Returns:
        dict: Formatted entry for OpenAI batch processing.
    """
    return {
        "custom_id": entry["custom_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that simplifies commit messages.",
                },
                {
                    "role": "user",
                    "content": f"Code changes:\n{entry['mods']}\n\nOriginal message: {entry['message']}\n\nCreate a simplified commit message following these strict rules:\n1. Maximum 8 words\n2. Use only these verbs: added, updated, removed, fixed, refactored\n3. Reference only code elements visible in the diff\n4. Format: \"<verb> <code_element> [brief_detail]\"\n5. Exclude all contextual information not visible in the code changes\n6. Focus on the technical change, not the purpose or impact\n7. Use consistent terminology for similar changes\n\nOutput only the simplified message without any explanation or formatting.",
                },
            ],
            "max_tokens": 300,
            "temperature": 0.7,
        },
    }


def validate_config():
    """Validate configuration settings and required paths."""
    required_paths = {
        "DATA_DIR": DATA_DIR,
        "INPUT_JSONL_DIR": INPUT_JSONL_DIR,
        "BATCHES_DIR": BATCHES_DIR,
        "OUTPUTS_DIR": OUTPUTS_DIR,
        "MERGED_DIR": MERGED_DIR,
        "FORMATTED_BATCHES_DIR": FORMATTED_BATCHES_DIR,
    }

    if not LANGUAGES:
        raise ValueError("LANGUAGES must not be empty")

    if not CHANGE_TYPES:
        raise ValueError("CHANGE_TYPES must not be empty")

    if HUGGINGFACE_REPO is None:
        raise ValueError("HUGGINGFACE_REPO must be configured")

    for name, path in required_paths.items():
        if path is None:
            raise ValueError(f"{name} path must not be None")


def main():
    """Execute the complete dataset simplification pipeline."""
    validate_config()
    load_arrow_dataset_as_jsonl()
    create_and_upload_batches()
    merge_and_save_final_dataset()
    upload_to_huggingface()


if __name__ == "__main__":
    main()
