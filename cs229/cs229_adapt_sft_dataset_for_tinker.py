"""
Adapt SFT datasets from CSV format to Tinker format.

This script reads the preprocessed SFT datasets created by
cs229_create_sft_datasets.py and converts them to the format expected
by Tinker supervised training loop.
ref: https://tinker-docs.thinkingmachines.ai/supervised-learning
"""

import ast
import logging
from pathlib import Path

import pandas as pd
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.types import SupervisedDataset
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


class SimpleSFTDataset(SupervisedDataset):
    """Simple dataset that holds SFT examples in memory for per-action training.

    Each row in the dataset represents one assistant action to train on.
    The dataset contains:
        - trajectory_conversation: Full conversation history
        - trajectory_subid: Which assistant action to train on
    """

    def __init__(self, data_rows: list[dict], batch_size: int, renderer: renderers.Renderer, max_length: int | None):
        self.data_rows = data_rows
        self.batch_size = batch_size
        self.renderer = renderer
        self.max_length = max_length
        self.shuffled_data = data_rows.copy()

    def get_batch(self, index: int) -> list[tinker.Datum]:
        """Get a batch of data as list of Datums.

        Each row becomes one datum that trains on a specific assistant action.
        """
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_rows = self.shuffled_data[start_idx:end_idx]

        datums = []
        for row in batch_rows:
            try:
                conversation = parse_conversation_string(row['trajectory_conversation'])

                trajectory_subid = row['trajectory_subid']

                datum = conversation_to_datum(
                    conversation=conversation,
                    assistant_action_index=trajectory_subid,
                    renderer=self.renderer,
                    max_length=self.max_length
                )
                datums.append(datum)
            except Exception as e:
                logger.error(f"Failed to process row (trajectory_subid={row.get('trajectory_subid', '?')}): {e}")
                continue

        return datums

    def __len__(self) -> int:
        """Number of batches."""
        return (len(self.data_rows) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, seed: int = 0):
        """Shuffle the data for a new epoch."""
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
        return "\n".join(text_parts)

    logger.warning(f"Unexpected content type: {type(content)}, converting to string")
    return str(content)


class ToolCallWrapper:
    """Simple wrapper to convert dict tool_calls to object format expected by renderer."""
    def __init__(self, tool_call_dict: dict):
        self.id = tool_call_dict.get("id", "")
        self.type = tool_call_dict.get("type", "function")
        self.function = ToolCallFunctionWrapper(tool_call_dict.get("function", {}))


class ToolCallFunctionWrapper:
    """Wrapper for function field in tool_calls."""
    def __init__(self, function_dict: dict):
        self.name = function_dict.get("name", "")
        self.arguments = function_dict.get("arguments", "{}")


def conversation_to_datum(
    conversation: list[dict],
    assistant_action_index: int,
    renderer: renderers.Renderer,
    max_length: int | None = None,
) -> tinker.Datum:
    """Convert a conversation to a Tinker Datum object for per-action SFT with isolation.

    For per-action isolation, we train ONLY on the target assistant action.
    Actually the Qwen3 renderer automatically isolates training to that action 
    (it only sets weight=1 for the last assistant message)

    Args:
        conversation: Full conversation history
        assistant_action_index: Which assistant action to train on (0-indexed among assistant messages)
        renderer: Tinker renderer for the chat format
        max_length: Optional max sequence length

    Returns:
        tinker.Datum object ready for supervised training
    """
    assistant_count = 0
    target_message_idx = None

    for i, msg in enumerate(conversation):
        if msg.get("role") == "assistant":
            if assistant_count == assistant_action_index:
                target_message_idx = i
                break
            assistant_count += 1

    if target_message_idx is None:
        raise ValueError(
            f"Could not find assistant action at index {assistant_action_index}. "
            f"Conversation has {assistant_count} assistant messages."
        )

    # build the conversation until that point and including the target assistant message
    # this makes the target assistant the last message, so the renderer will
    # automatically set weight=1 only for it (per-action isolation)
    messages_for_training = []
    for i, msg in enumerate(conversation[:target_message_idx + 1]):
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content_raw = msg.get("content", "")
            content = normalize_message_content(content_raw)

            message_dict = {"role": role, "content": content}

            # convert tool_calls from dict to object format expected by renderer
            if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
                message_dict["tool_calls"] = [
                    ToolCallWrapper(tc) if isinstance(tc, dict) else tc
                    for tc in msg["tool_calls"]
                ]

            messages_for_training.append(message_dict)
        else:
            logger.warning(f"Unexpected message format: {type(msg)}")
            continue

    if not messages_for_training:
        raise ValueError("No valid messages found in conversation")

    # here is where Qwen renderer by default set weight=1 only for the last
    # assistant message so this is the only "trainable one"
    tokens, weights = renderer.build_supervised_example(messages_for_training)

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
    max_train_examples: int | None = None,
    max_test_examples: int | None = None,
):
    """Create a dataset builder function for Tinker SFT training.

    Args:
        train_csv_path: Path to training SFT CSV
        test_csv_path: Optional path to test SFT CSV
        model_name: Model name for tokenizer
        renderer_name: Chat template renderer name
        batch_size: Batch size for training
        max_length: Optional max sequence length
        max_train_examples: Optional limit on training examples
        max_test_examples: Optional limit on test examples

    Returns:
        Function that returns (train_dataset, optional_test_dataset)
    """
    def dataset_builder() -> tuple[SupervisedDataset, SupervisedDataset | None]:
        """Build train and optional test datasets."""
        tokenizer = get_tokenizer(model_name)
        renderer = renderers.get_renderer(renderer_name, tokenizer)

        logger.info(f"Loading SFT dataset from {train_csv_path}")
        train_df = pd.read_csv(train_csv_path)

        if max_train_examples:
            train_df = train_df.head(max_train_examples)

        train_df['trajectory_conversation'] = train_df['trajectory_conversation'].apply(parse_conversation_string)

        train_data = train_df.to_dict('records')

        train_dataset = SimpleSFTDataset(
            data_rows=train_data,
            batch_size=batch_size,
            renderer=renderer,
            max_length=max_length,
        )

        test_dataset = None
        if test_csv_path and Path(test_csv_path).exists():
            logger.info(f"Loading test dataset from {test_csv_path}")
            test_df = pd.read_csv(test_csv_path)

            if max_test_examples:
                test_df = test_df.head(max_test_examples)

            test_df['trajectory_conversation'] = test_df['trajectory_conversation'].apply(parse_conversation_string)

            test_data = test_df.to_dict('records')

            test_dataset = SimpleSFTDataset(
                data_rows=test_data,
                batch_size=len(test_data),  # use full test set as one batch
                renderer=renderer,
                max_length=max_length,
            )

        logger.info(f"Train dataset: {len(train_dataset)} batches, {len(train_data)} examples")
        if test_dataset:
            logger.info(f"Test dataset: {len(test_dataset)} batches, {len(test_data)} examples")

        return train_dataset, test_dataset

    return dataset_builder


def main():
    """Example usage of the SFT dataset adapter."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # TODO: maybe we can parametrize this to be specified directly in by cli args
    base_name = "20251109_qwen3_coder_30b_a3b_instruct_0_7_exp_3_n_50_fmt_tool_call_hist_messages_8"
    datasets_dir = Path("./datasets")

    train_csv = datasets_dir / f"{base_name}_trajectories_sft_train.csv"
    test_csv = datasets_dir / f"{base_name}_trajectories_sft_test.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    dataset_builder = create_dataset_builder(
        train_csv_path=str(train_csv),
        test_csv_path=str(test_csv) if test_csv.exists() else None,
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        renderer_name="qwen3_instruct",
        batch_size=2,
        max_length=24576,  # must be >= 22K as in DPO to preserve assistant responses
        max_train_examples=None,
        max_test_examples=None,
    )

    logger.info("Building datasets...")
    train_dataset, test_dataset = dataset_builder()

    logger.info(f"\nDataset summary:")
    logger.info(f"- Train: {len(train_dataset)} batches")
    if test_dataset:
        logger.info(f"- Test: {len(test_dataset)} batches")

if __name__ == "__main__":
    main()
