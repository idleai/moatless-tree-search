"""
Run DPO training on RunPod using Tinker API.

This script configures and launches DPO training using the preprocessed
preference datasets and Tinker's distributed training infrastructure.

Prerequisites:
    - Set TINKER_API_KEY environment variable
    - Ensure preprocessed CSV files exist in ./datasets/
    - (optional) Have an SFT checkpoint to load as starting point

Usage Examples:
    # quick test run (5 pairs, 2 GPUs)
    python cs229-tinker-finetuning-dpo-trainer.py --mode test

    # small test run (20 pairs, 4 GPUs)
    python cs229-tinker-finetuning-dpo-trainer.py --mode small

    # full training run (all data, 8 GPUs)
    python cs229-tinker-finetuning-dpo-trainer.py --mode full

    # custom configuration
    python cs229-tinker-finetuning-dpo-trainer.py \
        --mode full \
        --learning_rate 1e-5 \
        --dpo_beta 0.2 \
        --num_epochs 3 \
        --num_replicas 16

    # with W&B logging (first to need to do wandb login)
    python cs229-tinker-finetuning-dpo-trainer.py \
        --mode full \
        --wandb_project cs229-dpo \
        --wandb_name qwen3-4b-run1

    # load from SFT checkpoint TODO: adapt this when we have the SFT running
    python cs229-tinker-finetuning-dpo-trainer.py \
        --mode full \
        --load_checkpoint ./logs/sft_run_001/final_state.bin
"""

import logging
from pathlib import Path

import chz
from tinker_cookbook.preference.train_dpo import Config as DPOConfig
from tinker_cookbook.preference.train_dpo import main as train_dpo_main

from cs229_adapt_preference_finetuning_dataset_for_tinker import create_dataset_builder

logger = logging.getLogger(__name__)


@chz.chz
class RunConfig:
    """configuration for DPO training run."""

    # run mode: controls dataset size and infrastructure
    mode: str = "test"  # "test", "small", or "full"

    # dataset TODO: we can do this dinamically
    dataset_name: str = "20251109_qwen3_coder_30b_a3b_instruct_0_7_exp_3_n_50_fmt_tool_call_hist_messages_8"

    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    renderer_name: str = "qwen3_instruct" # this is the same for all Qwen 3 models

    learning_rate: float = 5e-6
    dpo_beta: float = 0.1
    num_epochs: int = 1
    lora_rank: int = 16
    lr_schedule: str = "linear"

    # infrastructure, overridden by mode if not explicitly set
    num_replicas: int | None = None
    batch_size: int = 2

    log_path: str = "./logs/dpo_qwen3_4b"
    load_checkpoint: str | None = None
    save_every: int = 50
    eval_every: int = 10

    # W&B logging (optional)
    wandb_project: str | None = None
    wandb_name: str | None = None

    max_length: int = 24576
    base_url: str | None = None


def main(cfg: RunConfig):
    """Configure and run DPO training."""
    mode_configs = {
        "test": {
            "max_train_pairs": 5,
            "max_test_pairs": 1,
            "num_replicas": 2,
            "save_every": 2,
            "eval_every": 2,
        },
        "small": {
            "max_train_pairs": 20,
            "max_test_pairs": 5,
            "num_replicas": 4,
            "save_every": 5,
            "eval_every": 5,
        },
        "full": {
            "max_train_pairs": None,
            "max_test_pairs": None,
            "num_replicas": 8,
            "save_every": cfg.save_every,
            "eval_every": cfg.eval_every,
        },
    }

    if cfg.mode not in mode_configs:
        raise ValueError(f"Invalid mode '{cfg.mode}'. Must be 'test', 'small', or 'full'")

    mode_cfg = mode_configs[cfg.mode]

    num_replicas = cfg.num_replicas if cfg.num_replicas is not None else mode_cfg["num_replicas"]

    datasets_dir = Path("./datasets")
    train_csv = datasets_dir / f"{cfg.dataset_name}_trajectories_preference_train.csv"
    test_csv = datasets_dir / f"{cfg.dataset_name}_trajectories_preference_test.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    dataset_builder = create_dataset_builder(
        train_csv_path=str(train_csv),
        test_csv_path=str(test_csv) if test_csv.exists() else None,
        model_name=cfg.model_name,
        renderer_name=cfg.renderer_name,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        max_train_pairs=mode_cfg["max_train_pairs"],
        max_test_pairs=mode_cfg["max_test_pairs"],
    )

    config = DPOConfig(
        # required parameters
        log_path=cfg.log_path,
        model_name=cfg.model_name,
        dataset_builder=dataset_builder,
        load_checkpoint_path=cfg.load_checkpoint,

        # training hyperparams
        learning_rate=cfg.learning_rate,
        lr_schedule=cfg.lr_schedule,
        num_epochs=cfg.num_epochs,
        dpo_beta=cfg.dpo_beta,
        lora_rank=cfg.lora_rank,

        # optimizer (Adam)
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,

        # infrastructure, checkpoint and evaluation
        num_replicas=num_replicas,
        base_url=cfg.base_url,

        save_every=mode_cfg["save_every"],
        eval_every=mode_cfg["eval_every"],
        infrequent_eval_every=100,

        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,

        reference_model_name=None,

        evaluator_builders=[],
        infrequent_evaluator_builders=[],
    )

    logger.info(f"DPO Training Configuration - Mode: {cfg.mode.upper()}")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Dataset: {train_csv.name}")
    logger.info(f"- Train pairs: {mode_cfg['max_train_pairs'] or 'ALL (1,308)'}")
    logger.info(f"- Test pairs: {mode_cfg['max_test_pairs'] or 'ALL'}")
    logger.info(f"Log path: {config.log_path}")
    logger.info(f"Training:")
    logger.info(f"- Learning rate: {config.learning_rate}")
    logger.info(f"- DPO beta: {config.dpo_beta}")
    logger.info(f"- LoRA rank: {config.lora_rank}")
    logger.info(f"- Num epochs: {config.num_epochs}")
    logger.info(f"- LR schedule: {config.lr_schedule}")
    logger.info(f"Infrastructure:")
    logger.info(f"- Num replicas: {config.num_replicas}")
    logger.info(f"- Batch size: {cfg.batch_size}")
    logger.info(f"Checkpointing:")
    logger.info(f"- Save every: {config.save_every} steps")
    logger.info(f"- Eval every: {config.eval_every} steps")
    if cfg.wandb_project:
        logger.info(f"W&B:")
        logger.info(f"- Project: {cfg.wandb_project}")
        logger.info(f"- Name: {cfg.wandb_name or 'auto-generated'}")

    train_dpo_main(config)


if __name__ == "__main__":
    import sys
    sys.exit(chz.nested_entrypoint(main))
