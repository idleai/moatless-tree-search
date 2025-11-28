"""
Adapt preference fine-tuning datasets from CSV format to Tinker format.

This script reads the preprocessed preference datasets created by
cs229_create_preference_finetuning_datasets.py and converts them to the
format expected by Tinker's DPO trainer.
"""

import ast
import logging
from pathlib import Path

import datasets
import pandas as pd
import tinker
import torch
from tinker_cookbook import renderers
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
from tinker_cookbook.supervised.types import SupervisedDataset
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


class SimplePreferenceDataset(SupervisedDataset):
    """Simple dataset that holds preference pairs in memory. This is not to deal with HF format errors"""

    def __init__(self, data_rows: list[dict], batch_size: int, renderer: renderers.Renderer, max_length: int | None):
        self.data_rows = data_rows
        self.batch_size = batch_size
        self.renderer = renderer
        self.max_length = max_length
        self.shuffled_data = data_rows.copy()

    def get_batch(self, index: int) -> list[tinker.Datum]:
        """Get a batch of data as list of Datums (interleaved chosen/rejected)."""
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_rows = self.shuffled_data[start_idx:end_idx]

        datums = []
        for row in batch_rows:
            try:
                chosen_datum = conversation_to_datum(row['chosen'], self.renderer, self.max_length)
                rejected_datum = conversation_to_datum(row['rejected'], self.renderer, self.max_length)

                # interleave: [chosen, rejected], this is the main difference
                # with our dataset structure in the .csv
                datums.append(chosen_datum)
                datums.append(rejected_datum)
            except Exception as e:
                logger.error(f"Failed to process row: {e}")
                continue

        return datums

    def __len__(self) -> int:
        """Number of batches."""
        return (len(self.data_rows) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, seed: int = 0):
        """shuffle the data for a new epoch."""
        import random
        rng = random.Random(seed)
        self.shuffled_data = self.data_rows.copy()
        rng.shuffle(self.shuffled_data)


def parse_conversation_string(conv_str: str | list) -> list[dict[str, str]]:
    """Parse conversation from CSV string format to list of message dicts.

    Args:
        conv_str: Either a string representation of a list, or already a list

    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    if isinstance(conv_str, list):
        return conv_str

    try:
        return ast.literal_eval(conv_str)
    except (ValueError, SyntaxError) as e:
        logger.error(f"Failed to parse conversation string: {e}")
        raise


def normalize_message_content(content) -> str:
    """Normalize message content to a simple string."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                # this is mainly to ignore other types (images, audio, etc.) for now
        return "\n".join(text_parts)

    logger.warning(f"Unexpected content type: {type(content)}, converting to string")
    return str(content)


def conversation_to_datum(
    conversation: list[dict[str, str]],
    renderer: renderers.Renderer,
    max_length: int | None = None,
) -> tinker.Datum:
    """Convert a conversation to a Tinker Datum object.

    Args:
        conversation: List of message dicts with 'role' and 'content'
        renderer: Tinker renderer for the chat format
        max_length: Optional max sequence length

    Returns:
        tinker.Datum object ready for DPO training
    """
    messages: list[renderers.Message] = []
    for msg in conversation:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content_raw = msg.get("content", "")

            content = normalize_message_content(content_raw)

            messages.append({"role": role, "content": content})
        else:
            logger.warning(f"Unexpected message format: {type(msg)}")
            continue

    if not messages:
        raise ValueError("No valid messages found in conversation")

    # use the tinker renderer to convert to tokens and weights
    # The renderer handles the chat template and masking automatically
    tokens, weights = renderer.build_supervised_example(messages)

    orig_nonzero_weights = sum(1 for w in weights if w > 0)
    orig_total_tokens = len(tokens)

    datum = datum_from_tokens_weights(tokens, weights, max_length)

    datum_weights = datum.loss_fn_inputs['weights'].data
    final_nonzero_weights = sum(1 for w in datum_weights if w > 0)

    if orig_nonzero_weights > 0 and final_nonzero_weights == 0:
        logger.warning(
            f"Truncation removed all trainable tokens. "
            f"Original: {orig_total_tokens} tokens ({orig_nonzero_weights} trainable), "
            f"After max_length={max_length}: {datum.model_input.length} tokens (0 trainable)"
        )

    return datum


def create_dataset_builder(
    train_csv_path: str,
    test_csv_path: str | None = None,
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    renderer_name: str = "qwen3_instruct",
    batch_size: int = 1,
    max_length: int | None = None,
    max_train_pairs: int | None = None,
    max_test_pairs: int | None = None,
):
    """Create a dataset builder function for Tinker DPO training.

    Args:
        train_csv_path: Path to training preference CSV
        test_csv_path: Optional path to test preference CSV
        model_name: Model name for tokenizer
        renderer_name: Chat template renderer name
        batch_size: Batch size for training
        max_length: Optional max sequence length
        max_train_pairs: Optional limit on training pairs
        max_test_pairs: Optional limit on test pairs

    Returns:
        Function that returns (train_dataset, optional_test_dataset)
    """
    def dataset_builder() -> tuple[SupervisedDataset, SupervisedDataset | None]:
        """Build train and optional test datasets."""
        tokenizer = get_tokenizer(model_name)
        renderer = renderers.get_renderer(renderer_name, tokenizer)

        logger.info(f"Loading preference dataset from {train_csv_path}")
        train_df = pd.read_csv(train_csv_path)

        if max_train_pairs:
            train_df = train_df.head(max_train_pairs)

        train_df['chosen'] = train_df['chosen'].apply(parse_conversation_string)
        train_df['rejected'] = train_df['rejected'].apply(parse_conversation_string)

        train_data = train_df.to_dict('records')

        train_dataset = SimplePreferenceDataset(
            data_rows=train_data,
            batch_size=batch_size,
            renderer=renderer,
            max_length=max_length,
        )

        test_dataset = None
        if test_csv_path and Path(test_csv_path).exists():
            logger.info(f"Loading test dataset from {test_csv_path}")
            test_df = pd.read_csv(test_csv_path)

            if max_test_pairs:
                test_df = test_df.head(max_test_pairs)

            test_df['chosen'] = test_df['chosen'].apply(parse_conversation_string)
            test_df['rejected'] = test_df['rejected'].apply(parse_conversation_string)

            test_data = test_df.to_dict('records')

            test_dataset = SimplePreferenceDataset(
                data_rows=test_data,
                batch_size=len(test_data),  # use full test set as one batch
                renderer=renderer,
                max_length=max_length,
            )

        logger.info(f"Train dataset: {len(train_dataset)} batches")
        if test_dataset:
            logger.info(f"Test dataset: {len(test_dataset)} batches")

        return train_dataset, test_dataset

    return dataset_builder


def main():
    """Example usage of the dataset adapter."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    base_name = "20251109_qwen3_coder_30b_a3b_instruct_0_7_exp_3_n_50_fmt_tool_call_hist_messages_8"
    datasets_dir = Path("./datasets")

    train_csv = datasets_dir / f"{base_name}_trajectories_preference_train.csv"
    test_csv = datasets_dir / f"{base_name}_trajectories_preference_test.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    dataset_builder = create_dataset_builder(
        train_csv_path=str(train_csv),
        test_csv_path=str(test_csv) if test_csv.exists() else None,
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        renderer_name="qwen3_instruct",
        batch_size=2,  # each batch contains 2 preference pairs (4 datums total)
        max_length=24576,  # must be >= 22K to preserve assistant responses avoid truncating them
        max_train_pairs=None,
        max_test_pairs=None,
    )

    logger.info("Building datasets...")
    train_dataset, test_dataset = dataset_builder()

    logger.info(f"\nDataset summary:")
    logger.info(f"- Train: {len(train_dataset)} batches")
    if test_dataset:
        logger.info(f"- Test: {len(test_dataset)} batches")

if __name__ == "__main__":
    main()