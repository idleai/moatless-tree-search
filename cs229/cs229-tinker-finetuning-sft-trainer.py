"""
Run SFT training using Tinker API.

This script configures and launches supervised fine-tuning using the preprocessed
SFT datasets and Tinker distributed training infrastructure.

Prerequisites:
    - Set TINKER_API_KEY environment variable
    - Ensure preprocessed CSV files exist in ./datasets/
    - Run this BEFORE DPO training to get a base checkpoint

Usage Examples:
    # quick test run (5 examples, 2 GPUs)
    python cs229-tinker-finetuning-sft-trainer.py mode=test

    # small test run (50 examples, 4 GPUs)
    python cs229-tinker-finetuning-sft-trainer.py mode=small

    # full training run (all data, 8 GPUs)
    python cs229-tinker-finetuning-sft-trainer.py mode=full

    # custom configuration
    python cs229-tinker-finetuning-sft-trainer.py \
        mode=full \
        learning_rate=0.0005 \
        num_epochs=3 \
        num_replicas=16

    # with W&B logging
    python cs229-tinker-finetuning-sft-trainer.py \
        mode=full \
        wandb_project=cs229-sft \
        wandb_name=qwen3-4b-sft-run1
"""

import asyncio
import logging
from pathlib import Path

import chz
from tinker_cookbook.supervised.train import Config as SFTConfig
from tinker_cookbook.supervised.train import main as train_sft_main
from tinker_cookbook.hyperparam_utils import get_lr

from cs229_adapt_sft_dataset_for_tinker import create_dataset_builder

logger = logging.getLogger(__name__)


@chz.chz
class RunConfig:
    """Configuration for SFT training run."""

    # run mode: controls dataset size and infrastructure
    mode: str = "test"  # "test", "small", or "full"

    dataset_name: str = "20251109_qwen3_coder_30b_a3b_instruct_0_7_exp_3_n_50_fmt_tool_call_hist_messages_8"

    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str = "qwen3_instruct"  # this is the same for all Qwen 3 models

    # training hyperparameters
    learning_rate: float | None = None  # TODO: if None, this should use recommended LR from get_lr()
    # but this default fallback based on model_type is not working properly so it is recommended to 
    # send learning_rate directly as a param in the cli command 
    num_epochs: int = 1
    lora_rank: int = 16
    lr_schedule: str = "linear"
    warmup_steps: int = 10

    # infrastructure, overridden by mode if not explicitly set
    num_replicas: int | None = None
    batch_size: int = 2

    log_path: str = "./logs/sft_qwen3_4b"
    load_checkpoint: str | None = None
    save_every: int = 50
    eval_every: int = 10

    # W&B logging (optional)
    wandb_project: str | None = None
    wandb_name: str | None = None

    max_length: int = 24576
    base_url: str | None = None


def main(cfg: RunConfig):
    """Configure and run SFT training."""
    mode_configs = {
        "test": {
            "max_train_examples": 5,
            "max_test_examples": 1,
            "num_replicas": 2,
            "save_every": 2,
            "eval_every": 2,
        },
        "small": {
            "max_train_examples": 50,
            "max_test_examples": 10,
            "num_replicas": 4,
            "save_every": 10,
            "eval_every": 5,
        },
        "full": {
            "max_train_examples": None,
            "max_test_examples": None,
            "num_replicas": 8,
            "save_every": cfg.save_every,
            "eval_every": cfg.eval_every,
        },
    }

    if cfg.mode not in mode_configs:
        raise ValueError(f"Invalid mode '{cfg.mode}'. Must be 'test', 'small', or 'full'")

    mode_cfg = mode_configs[cfg.mode]

    num_replicas = cfg.num_replicas if cfg.num_replicas is not None else mode_cfg["num_replicas"]

    # get recommended learning rate if not specified
    if cfg.learning_rate is None:
        recommended_lr = get_lr(cfg.model_name)
        learning_rate = recommended_lr
        logger.info(f"Using recommended learning rate: {learning_rate:.2e}")
    else:
        learning_rate = cfg.learning_rate
        logger.info(f"Using custom learning rate: {learning_rate:.2e}")

    datasets_dir = Path("./datasets")
    train_csv = datasets_dir / f"{cfg.dataset_name}_trajectories_sft_train.csv"
    test_csv = datasets_dir / f"{cfg.dataset_name}_trajectories_sft_test.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    dataset_builder = create_dataset_builder(
        train_csv_path=str(train_csv),
        test_csv_path=str(test_csv) if test_csv.exists() else None,
        model_name=cfg.model_name,
        renderer_name=cfg.renderer_name,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        max_train_examples=mode_cfg["max_train_examples"],
        max_test_examples=mode_cfg["max_test_examples"],
    )

    config = SFTConfig(
        # required parameters
        log_path=cfg.log_path,
        model_name=cfg.model_name,
        dataset_builder=dataset_builder,
        load_checkpoint_path=cfg.load_checkpoint,

        # training hyperparams
        learning_rate=learning_rate,
        lr_schedule=cfg.lr_schedule,
        num_epochs=cfg.num_epochs,
        lora_rank=cfg.lora_rank,

        # optimizer (Adam)
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,

        # checkpoint and evaluation
        save_every=mode_cfg["save_every"],
        eval_every=mode_cfg["eval_every"],
        infrequent_eval_every=100,

        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,

        evaluator_builders=[],
        infrequent_evaluator_builders=[],
    )

    logger.info(f"SFT Training Configuration - Mode: {cfg.mode.upper()}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Dataset: {train_csv.name}")
    logger.info(f"- Train examples: {mode_cfg['max_train_examples'] or 'ALL'}")
    logger.info(f"- Test examples: {mode_cfg['max_test_examples'] or 'ALL'}")
    logger.info(f"Log path: {config.log_path}")
    logger.info(f"Training:")
    logger.info(f"- Learning rate: {config.learning_rate:.2e}")
    logger.info(f"- LoRA rank: {config.lora_rank}")
    logger.info(f"- Num epochs: {config.num_epochs}")
    logger.info(f"- LR schedule: {config.lr_schedule}")
    logger.info(f"- Batch size: {cfg.batch_size}")
    logger.info(f"Checkpointing:")
    logger.info(f"- Save every: {config.save_every} steps")
    logger.info(f"- Eval every: {config.eval_every} steps")
    if cfg.wandb_project:
        logger.info(f"W&B:")
        logger.info(f"- Project: {cfg.wandb_project}")
        logger.info(f"- Name: {cfg.wandb_name or 'auto-generated'}")

    asyncio.run(train_sft_main(config))


if __name__ == "__main__":
    import sys
    sys.exit(chz.nested_entrypoint(main))
