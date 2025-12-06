import argparse
import json
import logging
import os
from datetime import datetime, timezone

from moatless.agent.settings import AgentSettings
from moatless.benchmark.schema import (
    DateTimeEncoder,
    Evaluation,
    EvaluationInstance,
    EvaluationStatus,
    InstanceStatus,
    TreeSearchSettings,
)
from moatless.completion.completion import CompletionModel
from moatless.completion.model import Usage
from moatless.schema import MessageHistoryType


# Default message history type (matches mts_debug.py)
MESSAGE_HISTORY_TYPE = MessageHistoryType.MESSAGES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate predictions.jsonl and evaluation.json from an existing run directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "run_output_dir",
        type=str,
        help="Path to the run output directory (e.g., logs/mts_debug/run_20250101_120000_gh-gpt4o)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset JSON file containing instance_ids (e.g., datasets/verified_not_lite_quarter_dataset.json)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing predictions.jsonl and evaluation.json files"
    )

    return parser.parse_args()


def load_run_summary(run_output_dir: str) -> dict:
    """Load run_summary.json from the run directory."""
    summary_path = os.path.join(run_output_dir, "run_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"run_summary.json not found in {run_output_dir}")

    with open(summary_path, 'r') as f:
        return json.load(f)


def load_dataset_instance_ids(dataset_path: str) -> list:
    """Load instance_ids from a dataset JSON file."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and "instance_ids" in data:
        return data["instance_ids"]
    elif isinstance(data, list):
        # If it's a list, assume it's a list of instance dicts or IDs
        if data and isinstance(data[0], dict):
            return [item.get("instance_id") for item in data if item.get("instance_id")]
        return data
    else:
        raise ValueError(f"Could not find instance_ids in dataset file: {dataset_path}")


def write_predictions_jsonl(results: list, model_name: str, output_dir: str) -> str:
    """
    Write predictions.jsonl file compatible with SWE-bench harness.

    Format per line: {"instance_id": "...", "model_patch": "...", "model_name_or_path": "..."}
    """
    predictions_path = os.path.join(output_dir, "predictions.jsonl")

    with open(predictions_path, 'w') as f:
        for result in results:
            prediction = {
                "instance_id": result["instance_id"],
                "model_patch": result.get("model_patch", ""),
                "model_name_or_path": model_name,
            }
            f.write(json.dumps(prediction) + "\n")

    return predictions_path


def save_evaluation_json(
    results: list,
    model_name: str,
    tree_search_config: dict,
    output_dir: str,
    evaluation_name: str,
    start_time: datetime,
    end_time: datetime,
) -> str:
    """
    Save evaluation.json file compatible with moatless-streamlit.
    """
    # Create completion model for settings
    completion_model = CompletionModel(
        model=model_name,
        response_format="tool_call",
    )

    # Create agent settings
    agent_settings = AgentSettings(
        completion_model=completion_model,
        message_history_type=MESSAGE_HISTORY_TYPE,
        thoughts_in_action=False,
    )

    # Create tree search settings
    settings = TreeSearchSettings(
        max_expansions=tree_search_config.get("max_expansions", 3),
        max_iterations=tree_search_config.get("max_iterations", 50),
        max_cost=tree_search_config.get("max_cost", 5.0),
        max_depth=tree_search_config.get("max_depth", 25),
        min_finished_nodes=tree_search_config.get("min_finished_nodes", 2),
        max_finished_nodes=tree_search_config.get("max_finished_nodes", 3),
        model=completion_model,
        agent_settings=agent_settings,
    )

    # Create evaluation instances from results
    instances = []
    for result in results:
        # Parse start/end times
        started_at = None
        completed_at = None

        if result.get("start_time"):
            try:
                started_at = datetime.fromisoformat(result["start_time"])
                if started_at.tzinfo is None:
                    started_at = started_at.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass

        if result.get("end_time"):
            try:
                completed_at = datetime.fromisoformat(result["end_time"])
                if completed_at.tzinfo is None:
                    completed_at = completed_at.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass

        instance = EvaluationInstance(
            instance_id=result["instance_id"],
            status=InstanceStatus.COMPLETED if result.get("status") == "completed" else InstanceStatus.ERROR,
            started_at=started_at,
            completed_at=completed_at,
            submission=result.get("model_patch", ""),
            error=result.get("error"),
            iterations=result.get("total_nodes"),
            usage=Usage(
                prompt_tokens=result.get("prompt_tokens", 0),
                completion_tokens=result.get("completion_tokens", 0),
            ) if result.get("prompt_tokens") or result.get("completion_tokens") else None,
            duration=result.get("duration_seconds"),
        )
        instances.append(instance)

    # Create evaluation
    evaluation = Evaluation(
        evaluations_dir=output_dir,
        evaluation_name=evaluation_name,
        settings=settings,
        start_time=start_time,
        finish_time=end_time,
        status=EvaluationStatus.COMPLETED,
        instances=instances,
    )

    # Save to output_dir/evaluation.json
    eval_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(evaluation.model_dump(), f, cls=DateTimeEncoder, indent=2)

    return eval_path


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    args = parse_args()
    run_output_dir = args.run_output_dir

    if not os.path.isdir(run_output_dir):
        logger.error(f"Directory does not exist: {run_output_dir}")
        return 1

    # Check for existing files
    predictions_path = os.path.join(run_output_dir, "predictions.jsonl")
    eval_path = os.path.join(run_output_dir, "evaluation.json")

    if not args.force:
        if os.path.exists(predictions_path):
            logger.warning(f"predictions.jsonl already exists. Use --force to overwrite.")
        if os.path.exists(eval_path):
            logger.warning(f"evaluation.json already exists. Use --force to overwrite.")
        if os.path.exists(predictions_path) or os.path.exists(eval_path):
            return 1

    # Load dataset instance_ids
    try:
        dataset_instance_ids = load_dataset_instance_ids(args.dataset)
        logger.info(f"Loaded {len(dataset_instance_ids)} instance_ids from dataset: {args.dataset}")
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    # Load run_summary.json (required)
    try:
        summary = load_run_summary(run_output_dir)
        logger.info(f"Loaded run_summary.json from {run_output_dir}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    results = summary.get("results", [])
    model_name = summary.get("model_name")
    tree_search_config = summary.get("tree_search_config", {})
    run_timestamp = summary.get("run_timestamp", "")

    if not model_name:
        logger.error("No model_name found in run_summary.json")
        return 1

    logger.info(f"Found {len(results)} results in run_summary.json")
    logger.info(f"Model name: {model_name}")

    # Build a map of existing results by instance_id
    results_by_id = {r["instance_id"]: r for r in results}

    # Find missing instances and add empty patches for them
    missing_ids = []
    for instance_id in dataset_instance_ids:
        if instance_id not in results_by_id:
            missing_ids.append(instance_id)
            # Add a placeholder result with empty patch
            results_by_id[instance_id] = {
                "instance_id": instance_id,
                "status": "missing",
                "model_patch": "",
            }

    if missing_ids:
        logger.warning(f"Found {len(missing_ids)} missing instances from dataset, adding empty patches:")
        for mid in missing_ids:
            logger.warning(f"  - {mid}")

    # Build final results list in dataset order
    final_results = []
    for instance_id in dataset_instance_ids:
        if instance_id in results_by_id:
            final_results.append(results_by_id[instance_id])

    logger.info(f"Final results count: {len(final_results)}")

    # Parse timestamps
    start_time_str = summary.get("run_start_time")
    end_time_str = summary.get("run_end_time")

    if start_time_str:
        try:
            start_time = datetime.fromisoformat(start_time_str)
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            start_time = datetime.now(tz=timezone.utc)
    else:
        start_time = datetime.now(tz=timezone.utc)

    if end_time_str:
        try:
            end_time = datetime.fromisoformat(end_time_str)
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            end_time = datetime.now(tz=timezone.utc)
    else:
        end_time = datetime.now(tz=timezone.utc)

    # Generate predictions.jsonl
    predictions_path = write_predictions_jsonl(final_results, model_name, run_output_dir)
    logger.info(f"Generated: {predictions_path}")

    # Generate evaluation.json
    evaluation_name = os.path.basename(run_output_dir)
    eval_path = save_evaluation_json(
        results=final_results,
        model_name=model_name,
        tree_search_config=tree_search_config,
        output_dir=run_output_dir,
        evaluation_name=evaluation_name,
        start_time=start_time,
        end_time=end_time,
    )
    logger.info(f"Generated: {eval_path}")

    # Print summary
    status_counts = {}
    for r in final_results:
        status = r.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info(f"  Total instances: {len(final_results)}")
    for status, count in sorted(status_counts.items()):
        logger.info(f"  {status}: {count}")
    logger.info("=" * 60)

    # Print SWE-bench harness command
    run_id = f"moatless_{model_name}_{run_timestamp}" if run_timestamp else f"moatless_{model_name}"
    logger.info(f"To run SWE-bench harness:\npython -m swebench.harness.run_evaluation \\\n  --dataset_name princeton-nlp/SWE-bench_Verified \\\n  --predictions_path {predictions_path} \\\n  --max_workers 16 \\\n  --timeout 900 \\\n  --cache_level env \\\n  --clean True \\\n  --run_id {run_id}")

    return 0


if __name__ == "__main__":
    exit(main())
