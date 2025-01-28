import json
import os
import re
import statistics
from math import ceil
from pathlib import Path
from typing import Sequence
import random

from tqdm import tqdm
import hydra
import matplotlib.pyplot as plt
import polars as pl
import rootutils
import seaborn as sns
import wandb
from omegaconf import OmegaConf, DictConfig
from wandb.errors import CommError

from src.utils.chat_completion import LLMChatCompleter

# https://stackoverflow.com/a/53014308/7121776
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
rootutils.setup_root(__file__, ".project-root", pythonpath=True)


def download_wandb_tables(
    run_id: str,
    output_dir: Path,
    split_names: Sequence[str] = ("test",),
    api_key: str | None = None,
    project: str = "commit-message-generation",
    entity: str = "iasmlproject2024-university-of-hamburg",
) -> dict[str, Path]:
    api_key = api_key or os.environ.get("WANDB_API_KEY")
    assert api_key is not None, "WANDB_API_KEY must be set"
    api = wandb.Api(api_key=api_key)

    tables: dict[str, Path] = dict()
    for split in split_names:
        try:
            wandb_artifact_file = output_dir / run_id / f"{split}.table.json"
            df_file = output_dir / run_id / f"{split}.csv"
            if not wandb_artifact_file.exists():
                artifact = api.artifact(f"{entity}/{project}/run-{run_id}-{split}:latest")
                artifact.download(output_dir / run_id)
            if not df_file.exists():
                with open(wandb_artifact_file, "r") as f:
                    data = json.load(f)
                df = pl.DataFrame(data["data"], schema=data["columns"], orient="row")
                df.write_csv(df_file)
            tables[split] = df_file
        # thrown when artifact is not found
        except CommError as e:
            print(e)
    return tables


def download_text_predictions_on_test_split(
    experiments: list[dict[str, str]], output_dir: Path, wandb_api_key: str
) -> Path:
    csv_path = output_dir / "predictions.csv"
    if csv_path.exists():
        return csv_path
    df_combined = None
    for experiment in experiments:
        path = download_wandb_tables(
            experiment["wandb_run_id"], output_dir, api_key=wandb_api_key
        )["test"]
        df = pl.read_csv(path)
        if df_combined is None:
            df_combined = df.drop("step").rename({"prediction": experiment["name"]})
        else:
            df = df.rename({"prediction": experiment["name"]})
            df_combined = df_combined.with_columns(df[experiment["name"]])
    # order columns
    df_combined = df_combined[
        ["input"] + [experiment["name"] for experiment in experiments] + ["target"]
    ]
    df_combined.write_csv(csv_path)
    return csv_path


def plot_first_word_classification_accuracy(
    result: pl.DataFrame,
    target_first_word: str,
    accuracy_col: str,
    image_path: Path,
    display: bool = False,
) -> None:
    order = sorted(result[target_first_word])
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("colorblind", n_colors=len(order))
    sns.barplot(
        data=result.to_pandas(),
        x=target_first_word,
        y=accuracy_col,
        palette=palette,
        # hue=target_first_word,
        order=order,
        errorbar=None,
    )

    # Add annotations for each bar
    for index, row in enumerate(sorted(result.to_dicts(), key=lambda x: x[target_first_word])):
        plt.text(
            index, row[accuracy_col] + 0.01, f"{row[accuracy_col]:.2f}", ha="center", fontsize=10
        )

    # Add labels and title
    plt.title("Classification Accuracy on Ground-Truth First Word", fontsize=16)
    plt.xlabel("Ground-Truth First Word", fontsize=12)
    plt.ylabel("Classification Accuracy", fontsize=12)

    # Rotate x-axis labels if necessary
    plt.xticks(rotation=45)

    # Display the plot
    plt.tight_layout()
    plt.savefig(image_path)
    if display:
        plt.show()
    plt.close()


def plot_word_frequencies(
    result: pl.DataFrame,
    target_first_word: str,
    predicted_first_word_col: str,
    image_path: Path,
    display: bool = False,
) -> None:
    bar_data = {}
    for index, row in enumerate(result.to_dicts()):
        category = row[target_first_word]
        bar_data[category] = []
        for word, count in list(
            zip(row[predicted_first_word_col], row[f"{predicted_first_word_col} counts"])
        ):
            bar_data[category].append([word, count])

    num_columns = min(3, len(bar_data))
    num_rows = ceil(len(bar_data) / num_columns)
    # fig, axes = plt.subplots(num_rows, num_columns, figsize=(12 * num_rows, 6 * num_columns))
    # axes = axes.flatten()

    # Plotting the most common words for each category
    for i, (category, data) in enumerate(bar_data.items()):
        sorted_words = sorted(data, key=lambda x: x[1], reverse=True)
        words, counts = zip(*sorted_words)

        max_words = 10
        words = words[:max_words]
        counts = counts[:max_words]

        # Create a barplot for each category
        fig, axes = plt.subplots(1, 1, figsize=(12, 6))
        ax = axes
        sns.barplot(x=list(counts), y=list(words), ax=ax)
        ax.set_title(f"Top {max_words} Words for '{category}'")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Words")
        ax.tick_params(axis="y", labelsize=10)

        # extent = axes[i].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        plt.tight_layout()
        plt.savefig(image_path.parent / (image_path.name + "_" + category + image_path.suffix))
        plt.close()

    # for i in range(len(bar_data), len(axes)):
    #     axes[i].axis("off")

    # plt.tight_layout()
    # plt.savefig(image_path)
    # if display:
    #     plt.show()


def first_word_classification_evaluation(df: pl.DataFrame, config: dict, output_dir: Path) -> None:
    def first_word_accuracy(target: str, prediction: str) -> int:
        is_same = target == prediction
        return int(is_same)

    def first_word(text: str) -> str:
        match = re.match(r"\b\w+", text)
        word = match.group(0).lower() if match else None
        # Our ground-truth data had some (just a few) commit messages starting with these words.
        # This is just some cleaning up, to make sure our evaluation graph looks nicer.
        if word in ["fixing", "fixup"]:
            word = "fix"
        return word

    accuracies = None
    for experiment in config["experiments"]:  # type: dict[str, str]
        name = experiment["name"]
        target_first_word = "ground-truth"
        predicted_first_word_col = f"prediction-{name}"
        accuracy_col = f"accuracy-{name}"

        df_copy: pl.DataFrame = df.clone().with_columns(
            pl.struct(["target"])
            .map_elements(lambda row: first_word(row["target"]), return_dtype=pl.String)
            .alias(target_first_word),
            pl.struct([name])
            .map_elements(lambda row: first_word(row[name]), return_dtype=pl.String)
            .alias(predicted_first_word_col),
        )

        df_copy: pl.DataFrame = df_copy.with_columns(
            pl.struct([target_first_word, predicted_first_word_col])
            .map_elements(
                lambda row: first_word_accuracy(
                    row[target_first_word], row[predicted_first_word_col]
                ),
                return_dtype=pl.Int32,
            )
            .alias(accuracy_col)
        )

        result = (
            df_copy.group_by(target_first_word)
            .agg(
                [
                    # pl.col(accuracy_col).sum().alias("values_sum"),
                    # pl.col(accuracy_col).count().alias("values_count"),
                    # calculate accuracy per group
                    (pl.col(accuracy_col).sum() / pl.col(accuracy_col).count()).alias(
                        accuracy_col
                    ),
                    # unique first words
                    pl.col(predicted_first_word_col)
                    .unique(maintain_order=True)
                    .alias(predicted_first_word_col),
                    # unique first words counts
                    pl.col(predicted_first_word_col)
                    .unique_counts()
                    .alias(f"{predicted_first_word_col} counts"),
                    # pl.col(predicted_first_word_col).unique().count().alias("unique_count"),
                ]
            )
            .sort(target_first_word)
        )
        if accuracies is None:
            accuracies = result.select([target_first_word, accuracy_col])
        else:
            accuracies = accuracies.with_columns(result[accuracy_col])
        pl.Config.set_tbl_rows(len(result))
        print(result)
        plot_first_word_classification_accuracy(
            result,
            target_first_word,
            accuracy_col,
            output_dir / f"first_word_accuracies_{name}.png",
        )
        plot_word_frequencies(
            result,
            target_first_word,
            predicted_first_word_col,
            output_dir / f"first_word_frequencies_{name}.png",
        )


@hydra.main(version_base="1.3", config_path="../configs", config_name="benchmark.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for benchmarking.

    :param cfg: DictConfig configuration composed by Hydra.
    """

    assert "{model_message}" in cfg.prompt
    assert "{target_message}" in cfg.prompt
    assert "{diff}" in cfg.prompt

    config = OmegaConf.to_container(cfg, resolve=True)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    predictions_csv_path = download_text_predictions_on_test_split(
        config["experiments"], output_dir, config["wandb_api_key"]
    )
    df: pl.DataFrame = pl.read_csv(predictions_csv_path)

    # (1). Perform first word prediction accuracy
    # ###############################################
    first_word_classification_evaluation(df, config, output_dir)

    # (2). perform LLM-based evaluation
    # ######################################
    llm_evaluation_csv_path = output_dir / f"llm_evaluation_{cfg.llm_evaluation_suffix}.csv"
    if not llm_evaluation_csv_path.exists():
        completer: LLMChatCompleter = hydra.utils.instantiate(cfg.completer, _convert_="object")
        df_copy: list[dict] = df.to_dicts()
        # store original index information to enable tracing
        for idx in range(len(df_copy)):
            df_copy[idx]["index"] = idx
        random.shuffle(df_copy)
        df_copy = df_copy[: cfg.num_samples_for_llm_evaluation]
        for experiment in config["experiments"]:  # type: dict[str, str]
            name = experiment["name"]

            def score_predicted_commit_message(input_, target_, prediction):
                messages = [
                    {
                        "role": "user",
                        "content": cfg.prompt.format(
                            diff=input_, model_message=prediction, target_message=target_
                        ),
                    }
                ]
                # uncomment this to debug and estimate token count
                if cfg.llm_evaluation:
                    response = completer.complete_chat(messages)
                    score = json.loads(response.content)["score"]
                else:
                    score = completer.count_tokens(messages)
                return score

            for row in tqdm(df_copy):
                row[name + "_evaluation"] = score_predicted_commit_message(
                    row["input"], row["target"], row[name]
                )
            print(name, sum(row[name + "_evaluation"] for row in df_copy))
            print(name, statistics.mean(row[name + "_evaluation"] for row in df_copy))
        pl.DataFrame(df_copy).write_csv(llm_evaluation_csv_path)

    df = pl.read_csv(llm_evaluation_csv_path)
    result = df.select(
        *[
            pl.col(experiment["name"] + "_evaluation").mean().alias(experiment["name"])
            for experiment in config["experiments"]
        ],
    )

    models = result.columns
    scores = [result[column][0] for column in models]

    # Create a horizontal bar chart
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Blues", len(models))
    plt.barh(models, scores, color=palette)

    # Add labels and title
    plt.xlabel("Mean Score (1-10)")
    plt.ylabel("Models")
    plt.title("GPT4o-Mini Mean Score (1-10) on Model Predictions")

    # Add data labels to the bars
    for index, value in enumerate(scores):
        plt.text(value + 0.1, index, f"{value:.2f}", va="center")

    # Show the plot
    plt.tight_layout()
    plt.savefig(output_dir / f"llm_evaluation_{cfg.llm_evaluation_suffix}.png")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    main()
