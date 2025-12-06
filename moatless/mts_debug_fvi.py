import argparse
import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from moatless.agent.code_agent import CodingAgent
from moatless.agent.settings import AgentSettings
from moatless.benchmark.schema import (
    DateTimeEncoder,
    Evaluation,
    EvaluationInstance,
    EvaluationStatus,
    InstanceStatus,
    TreeSearchSettings,
)
from moatless.benchmark.swebench import create_repository, create_index
from moatless.benchmark.utils import get_moatless_instance
from moatless.completion.completion import CompletionModel, set_rate_limit_backoff
from moatless.completion.model import Usage
from moatless.discriminator import AgentDiscriminator
from moatless.feedback.reward_feedback import RewardFeedbackGenerator
from moatless.file_context import FileContext
from moatless.search_tree import SearchTree
from moatless.selector import BestFirstSelector
from moatless.schema import MessageHistoryType
from moatless.value_function.base import ValueFunction2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MCTS debug on a SWE-bench instance or dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments - now accepts either instance_id or dataset path
    parser.add_argument(
        "instance_or_dataset",
        type=str,
        help="SWE-bench instance ID (e.g., django__django-16379) or path to dataset JSON file"
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name to use (e.g., gh-gpt4o)"
    )

    # Parallelism control
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=16,
        help="Maximum number of parallel instances to run (only used with dataset file)"
    )

    # Tree search config arguments (optional)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=50,
        help="Maximum iterations (20 for simple, 50-100 for MCTS)"
    )
    parser.add_argument(
        "--max-expansions",
        type=int,
        default=3,
        help="Maximum expansions (1 for simple, 3+ for MCTS with exploration)"
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=5.0,
        help="Maximum cost in dollars"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=25,
        help="Maximum depth per trajectory"
    )
    parser.add_argument(
        "--min-finished-nodes",
        type=int,
        default=2,
        help="For MCTS: wait for N finished nodes"
    )
    parser.add_argument(
        "--max-finished-nodes",
        type=int,
        default=3,
        help="For MCTS: stop after N finished nodes"
    )
    parser.add_argument(
        "--rate-limit-backoff",
        type=int,
        default=600,
        help="Backoff time in seconds when encountering 429 rate limit errors (default: 600 = 10 minutes)"
    )

    # Resume from checkpoint
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_DIR",
        help="Resume from a previous run directory. Skips already-completed instances (those with trajectory.json)"
    )

    # Retry options for resume
    parser.add_argument(
        "--retry-no-result",
        action="store_true",
        help="When resuming, also retry instances that finished with 'no_result' status (no solution found)"
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="When resuming, also retry instances that finished with 'error' status"
    )

    return parser.parse_args()


# Environment paths
INDEX_STORE_DIR = "/home/idlecs229/repos/moatless-tree-search/parts"
REPO_BASE_DIR = "/home/idlecs229/repos/moatless-tree-search/downloads"
OUTPUT_DIR = "/home/idlecs229/repos/moatless-tree-search/logs/mts_debug"

# MCTS components - enable for full MCTS behavior
USE_VALUE_FUNCTION = True
USE_DISCRIMINATOR = True
USE_FEEDBACK = False

# Message history type - matches eval configs
MESSAGE_HISTORY_TYPE = MessageHistoryType.MESSAGES  # or REACT, SUMMARY

# Base completion configs for different models
# Keys are model name prefixes - first matching prefix will be used
MODEL_COMPLETION_CONFIGS = {
    "gh-gpt4o": {
        # Default settings for gh-gpt4o
    },
    "qwen3-coder-30b-a3b-instruct": {
        "max_tokens": 32768,
        "timeout": 600.0,
    },
}


def get_completion_config(model_name: str) -> dict:
    """Get completion config for a model by matching against MODEL_COMPLETION_CONFIGS keys.

    Returns the config dict for the first matching prefix, or empty dict if no match.
    """
    for prefix, config in MODEL_COMPLETION_CONFIGS.items():
        if model_name.startswith(prefix):
            return config
    return {}


def detect_completed_instances(output_dir: str, skip_no_result: bool = True, skip_errors: bool = True) -> dict:
    """
    Detect instances that have already been completed by checking for trajectory.json files
    with run_status="finished" in metadata.

    Args:
        output_dir: Directory to scan for completed instances
        skip_no_result: If True, treat 'no_result' status as completed (skip on resume).
                       If False, these instances will be re-run.
        skip_errors: If True, treat 'error' status as completed (skip on resume).
                    If False, these instances will be re-run.

    Returns a dict mapping instance_id -> result dict (loaded from run_summary.json if available,
    or reconstructed from trajectory.json).
    """
    completed = {}

    if not os.path.exists(output_dir):
        return completed

    # Build the set of statuses to consider as "completed" (skip on resume)
    completed_statuses = {"completed"}
    if skip_no_result:
        completed_statuses.add("no_result")
    if skip_errors:
        completed_statuses.add("error")

    # First, try to load results from run_summary.json if it exists
    summary_path = os.path.join(output_dir, "run_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
            # Extract results indexed by instance_id
            for result in summary_data.get("results", []):
                instance_id = result.get("instance_id")
                if instance_id and result.get("status") in completed_statuses:
                    completed[instance_id] = result
        except (json.JSONDecodeError, KeyError):
            pass

    # Also scan for trajectory.json files to catch any instances not in summary
    for entry in os.listdir(output_dir):
        instance_dir = os.path.join(output_dir, entry)
        if not os.path.isdir(instance_dir):
            continue

        # Skip if already found in summary
        if entry in completed:
            continue

        trajectory_file = os.path.join(instance_dir, "trajectory.json")
        if os.path.exists(trajectory_file):
            try:
                with open(trajectory_file, 'r') as f:
                    traj_data = json.load(f)

                # Check for run_status="finished" in metadata
                metadata = traj_data.get("metadata", {})
                if metadata.get("run_status") == "finished":
                    # This instance completed successfully
                    completed[entry] = {
                        "instance_id": entry,
                        "status": "completed",
                        "persist_path": trajectory_file,
                        "model_patch": "",  # Would need to reconstruct from trajectory
                        "run_completed_at": metadata.get("run_completed_at"),
                    }
            except (OSError, json.JSONDecodeError):
                pass

    return completed


def load_previous_run_config(run_dir: str) -> dict:
    """
    Load configuration from a previous run's run_summary.json.

    Returns dict with tree_search_config, model_name, dataset_path, etc.
    Returns empty dict if not found.
    """
    summary_path = os.path.join(run_dir, "run_summary.json")
    if not os.path.exists(summary_path):
        return {}

    try:
        with open(summary_path, 'r') as f:
            data = json.load(f)
        return {
            "model_name": data.get("model_name"),
            "dataset_path": data.get("dataset_path"),
            "tree_search_config": data.get("tree_search_config", {}),
            "max_parallel": data.get("max_parallel", 16),
        }
    except (json.JSONDecodeError, KeyError):
        return {}


def save_checkpoint(
    output_dir: str,
    run_timestamp: str,
    run_start_time: float,
    model_name: str,
    dataset_path: str,
    tree_search_config: dict,
    max_parallel: int,
    total_instances: int,
    results: list,
    lock: threading.Lock,
):
    """
    Atomically save a checkpoint of the current run state.

    This is called after each instance completes to enable resumption.
    Uses a temp file + rename pattern for atomic writes.
    """
    with lock:
        # Calculate current stats
        status_counts = {}
        for r in results:
            status = r.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in results)
        total_completion_tokens = sum(r.get("completion_tokens", 0) for r in results)
        total_tokens = sum(r.get("total_tokens", 0) for r in results)
        total_nodes = sum(r.get("total_nodes", 0) for r in results)

        checkpoint_data = {
            "run_timestamp": run_timestamp,
            "run_start_time": datetime.fromtimestamp(run_start_time).isoformat(),
            "last_updated": datetime.now().isoformat(),
            "status": "in_progress",
            "model_name": model_name,
            "dataset_path": dataset_path,
            "tree_search_config": tree_search_config,
            "max_parallel": max_parallel,
            "total_instances": total_instances,
            "completed_count": len(results),
            "status_counts": status_counts,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "total_nodes": total_nodes,
            "output_dir": output_dir,
            "results": results,
        }

        # Write atomically using temp file + rename
        summary_path = os.path.join(output_dir, "run_summary.json")
        fd, temp_path = tempfile.mkstemp(suffix=".json", prefix="checkpoint_", dir=output_dir)
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            os.replace(temp_path, summary_path)  # Atomic on POSIX
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise


def run_single_instance(instance_id: str, model_name: str, tree_search_config: dict, run_timestamp: str, output_dir: str, repo_base_dir: str, index_store_dir: str) -> dict:
    """Run MCTS on a single instance. Returns a result dict."""
    logger = logging.getLogger(f"mts.{instance_id}")
    start_time = time.time()
    result = {
        "instance_id": instance_id,
        "status": "unknown",
        "error": None,
        "final_node_id": None,
        "persist_path": None,
        "model_patch": "",  # Git patch for SWE-bench harness
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "duration_seconds": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_nodes": 0,
    }

    try:
        # Generate PERSIST_PATH with instance_id subdirectory for moatless-streamlit compatibility
        # Structure: {output_dir}/{instance_id}/trajectory.json
        instance_dir = f"{output_dir}/{instance_id}"
        os.makedirs(instance_dir, exist_ok=True)
        persist_path = f"{instance_dir}/trajectory.json"
        result["persist_path"] = persist_path

        # Load SWE-bench instance
        instance = get_moatless_instance(instance_id)
        problem_statement = f"<task>\nSolve the following reported issue in the {instance['repo']} repository:\n\n{instance['problem_statement']}\n</task>"

        logger.info(f"Loaded instance: {instance_id}")
        logger.info(f"Repository: {instance['repo']}")

        # Create repository and index (matches evaluation_runner.py lines 377-380)
        repository = create_repository(instance, repo_base_dir=repo_base_dir)
        code_index = create_index(instance, repository=repository, index_store_dir=index_store_dir)

        logger.info(f"Repository created at: {repository.repo_dir}")

        # Create completion model (matches run_evaluation.py lines 431-442)
        # Get model-specific config overrides
        completion_config = get_completion_config(model_name)
        completion_model = CompletionModel(
            model=model_name,
            response_format="tool_call",
            **completion_config
        )

        logger.info(f"Model: {completion_model.model}, Response format: {completion_model.response_format}")
        if completion_config:
            logger.info(f"Model config overrides: {completion_config}")
        selector = None
        value_function = None
        discriminator = None
        feedback_generator = None

        # Only use selector for multi-expansion MCTS
        if tree_search_config["max_expansions"] > 1:
            selector = BestFirstSelector()
            logger.info("Using BestFirstSelector for multi-expansion MCTS")

        if USE_VALUE_FUNCTION:
            value_function = ValueFunction2(completion_model=completion_model, correction_award=0)
            logger.info("Value function 2 enabled")

        if USE_DISCRIMINATOR:
            discriminator = AgentDiscriminator(
                completion=completion_model,
                n_agents=5,
                n_rounds=3,
            )
            logger.info("Discriminator enabled (n_agents=5, n_rounds=3)")

        if USE_FEEDBACK:
            feedback_generator = RewardFeedbackGenerator()
            logger.info("Feedback generator enabled")

        agent = CodingAgent.create(
            repository=repository,
            completion_model=completion_model,
            code_index=code_index,
            runtime=None,  # Set to TestbedEnvironment for test execution
            message_history_type=MESSAGE_HISTORY_TYPE,
            thoughts_in_action=False
        )

        logger.info(f"Agent created with message_history_type={MESSAGE_HISTORY_TYPE.value}")
        file_context = FileContext(repo=repository)
        search_tree = SearchTree.create(
            message=problem_statement,
            repository=repository,
            file_context=file_context,
            agent=agent,
            selector=selector,
            value_function=value_function,
            discriminator=discriminator,
            feedback_generator=feedback_generator,
            max_iterations=tree_search_config["max_iterations"],
            max_expansions=tree_search_config["max_expansions"],
            max_cost=tree_search_config["max_cost"],
            max_depth=tree_search_config["max_depth"],
            min_finished_nodes=tree_search_config["min_finished_nodes"],
            max_finished_nodes=tree_search_config["max_finished_nodes"],
            persist_path=persist_path,
            metadata={
                "instance_id": instance_id,
                "debug": True,
            },
        )

        logger.info(f"Search tree created with max_iterations={tree_search_config['max_iterations']}, max_expansions={tree_search_config['max_expansions']}")

        def tree_event_handler(event):
            """Simple event handler for debugging"""
            event_type = event.get("event_type", "unknown")
            logger.debug(f"Tree event: {event_type}")

        search_tree.add_event_handler(tree_event_handler)

        logger.info("Starting search...")
        node = search_tree.run_search()

        if node:
            logger.info(f"Search completed. Final node: {node.node_id}")
            logger.info(f"Observation: {node.observation}")

            # Extract git patch for SWE-bench harness
            if node.file_context:
                patch = node.file_context.generate_git_patch(ignore_tests=True)
                result["model_patch"] = patch if patch else ""
                logger.info(f"Generated patch ({len(result['model_patch'])} chars)")
            else:
                logger.warning("No file_context on final node")

            # Print some stats
            total_usage = search_tree.total_usage()
            logger.info(f"Total usage: {total_usage}")
            result["prompt_tokens"] = total_usage.prompt_tokens
            result["completion_tokens"] = total_usage.completion_tokens
            result["total_tokens"] = total_usage.prompt_tokens + total_usage.completion_tokens

            # Show the tree structure and count nodes
            from moatless.node import generate_ascii_tree
            logger.info(f"\nTree structure:\n{generate_ascii_tree(search_tree.root)}")

            # Count total nodes in the tree
            node_count = len(search_tree.root.get_all_nodes())
            result["total_nodes"] = node_count
            logger.info(f"Total nodes in tree: {node_count}")

            result["status"] = "completed"
            result["final_node_id"] = node.node_id
        else:
            logger.warning("Search did not return a final node")
            # Still capture usage even without a result
            total_usage = search_tree.total_usage()
            logger.info(f"Total usage: {total_usage}")
            result["prompt_tokens"] = total_usage.prompt_tokens
            result["completion_tokens"] = total_usage.completion_tokens
            result["total_tokens"] = total_usage.prompt_tokens + total_usage.completion_tokens

            # Count total nodes in the tree
            node_count = len(search_tree.root.get_all_nodes())
            result["total_nodes"] = node_count
            logger.info(f"Total nodes in tree: {node_count}")

            # Try to get the best trajectory from finished nodes
            best_node = search_tree.get_best_trajectory()
            if best_node and best_node.file_context:
                patch = best_node.file_context.generate_git_patch(ignore_tests=True)
                result["model_patch"] = patch if patch else ""
                result["final_node_id"] = best_node.node_id
                logger.info(f"Got patch from best trajectory node {best_node.node_id} ({len(result['model_patch'])} chars)")
            result["status"] = "no_result"

        # Mark the run as finished in the search tree metadata and persist
        search_tree.metadata["run_status"] = "finished"
        search_tree.metadata["run_completed_at"] = datetime.now(tz=timezone.utc).isoformat()
        if persist_path:
            search_tree.persist(persist_path)

    except Exception as e:
        logger.error(f"Error running instance {instance_id}: {e}", exc_info=True)
        result["status"] = "error"
        result["error"] = str(e)

    # Record completion time
    end_time = time.time()
    result["end_time"] = datetime.now().isoformat()
    result["duration_seconds"] = round(end_time - start_time, 2)

    return result


def load_dataset(dataset_path: str) -> list:
    """Load instance IDs from a dataset JSON file."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Support both formats: list of instance_ids or dict with instance_ids key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "instance_ids" in data:
        return data["instance_ids"]
    else:
        raise ValueError(f"Invalid dataset format. Expected list or dict with 'instance_ids' key.")


def write_predictions_jsonl(results: list, model_name: str, output_dir: str) -> str:
    """
    Write predictions.jsonl file compatible with SWE-bench harness.

    Format per line: {"instance_id": "...", "model_patch": "...", "model_name_or_path": "..."}

    Returns the path to the predictions file.
    """
    predictions_path = f"{output_dir}/predictions.jsonl"

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

    Creates an Evaluation object with TreeSearchSettings and EvaluationInstance list,
    then saves it using EvaluationFileRepository.

    Returns the path to the evaluation file.
    """
    # Create completion model for settings
    completion_config = get_completion_config(model_name)
    completion_model = CompletionModel(
        model=model_name,
        response_format="tool_call",
        **completion_config
    )

    # Create agent settings
    agent_settings = AgentSettings(
        completion_model=completion_model,
        message_history_type=MESSAGE_HISTORY_TYPE,
        thoughts_in_action=False,
    )

    # Create tree search settings
    settings = TreeSearchSettings(
        max_expansions=tree_search_config["max_expansions"],
        max_iterations=tree_search_config["max_iterations"],
        max_cost=tree_search_config["max_cost"],
        max_depth=tree_search_config["max_depth"],
        min_finished_nodes=tree_search_config["min_finished_nodes"],
        max_finished_nodes=tree_search_config["max_finished_nodes"],
        model=completion_model,
        agent_settings=agent_settings,
    )

    # Create evaluation instances from results
    instances = []
    for result in results:
        instance = EvaluationInstance(
            instance_id=result["instance_id"],
            status=InstanceStatus.COMPLETED if result["status"] == "completed" else InstanceStatus.ERROR,
            started_at=datetime.fromisoformat(result["start_time"]) if result.get("start_time") else None,
            completed_at=datetime.fromisoformat(result["end_time"]) if result.get("end_time") else None,
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

    # Save directly to output_dir/evaluation.json (not using repository which adds subdirectory)
    eval_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(evaluation.model_dump(), f, cls=DateTimeEncoder, indent=2)

    return eval_path


def run_parallel_instances(instance_ids: list, model_name: str, tree_search_config: dict, max_parallel: int, dataset_path: str = None, resume_dir: str = None, retry_no_result: bool = False, retry_errors: bool = False):
    """Run multiple instances in parallel using ThreadPoolExecutor.

    Args:
        instance_ids: List of instance IDs to run
        model_name: Model name to use
        tree_search_config: Tree search configuration
        max_parallel: Maximum parallel instances
        dataset_path: Path to dataset file (for logging)
        resume_dir: If provided, resume from this directory (skip completed instances)
        retry_no_result: If True, retry instances that finished with 'no_result' status
        retry_errors: If True, retry instances that finished with 'error' status
    """
    logger = logging.getLogger("mts.parallel")
    run_start_time = time.time()

    # Handle resume vs new run
    if resume_dir:
        # Resuming from previous run
        run_output_dir = resume_dir
        # Extract timestamp from directory name (format: run_YYYYMMDD_HHMMSS_modelname)
        dir_name = os.path.basename(run_output_dir)
        if dir_name.startswith("run_"):
            parts = dir_name.split("_")
            if len(parts) >= 3:
                run_timestamp = f"{parts[1]}_{parts[2]}"
            else:
                run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"=== RESUMING RUN from {run_output_dir} ===")
        if retry_no_result:
            logger.info("  --retry-no-result: Will re-run instances with 'no_result' status")
        if retry_errors:
            logger.info("  --retry-errors: Will re-run instances with 'error' status")

        # Detect already-completed instances
        # skip_no_result and skip_errors are the inverse of retry flags
        completed_instances = detect_completed_instances(
            run_output_dir,
            skip_no_result=not retry_no_result,
            skip_errors=not retry_errors
        )
        logger.info(f"Found {len(completed_instances)} instances to skip")

        # Filter out completed instances
        pending_instance_ids = [iid for iid in instance_ids if iid not in completed_instances]
        skipped_count = len(instance_ids) - len(pending_instance_ids)

        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} already-completed instances")
            for iid in instance_ids:
                if iid in completed_instances:
                    logger.debug(f"  Skipping: {iid}")

        # Start with results from completed instances
        results = list(completed_instances.values())

        # Use existing repo/index dirs from the previous run
        repo_base_dir = f"{REPO_BASE_DIR}/{dir_name}"
        index_store_dir = f"{INDEX_STORE_DIR}/{dir_name}"
    else:
        # New run
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = f"{OUTPUT_DIR}/run_{run_timestamp}_{model_name}"
        pending_instance_ids = instance_ids
        results = []
        completed_instances = {}

        # Create run-specific repo base dir to avoid conflicts in parallel runs
        repo_base_dir = f"{REPO_BASE_DIR}/run_{run_timestamp}_{model_name}"
        index_store_dir = f"{INDEX_STORE_DIR}/run_{run_timestamp}_{model_name}"

    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(repo_base_dir, exist_ok=True)
    os.makedirs(index_store_dir, exist_ok=True)

    logger.info(f"Output directory: {run_output_dir}")
    logger.info(f"Repository base directory: {repo_base_dir}")
    logger.info(f"Index store directory: {index_store_dir}")

    total = len(instance_ids)
    total_pending = len(pending_instance_ids)
    completed = len(completed_instances)

    # Lock for thread-safe checkpoint saving
    checkpoint_lock = threading.Lock()

    # Save initial checkpoint immediately (so run_summary.json exists from the start)
    save_checkpoint(
        output_dir=run_output_dir,
        run_timestamp=run_timestamp,
        run_start_time=run_start_time,
        model_name=model_name,
        dataset_path=dataset_path,
        tree_search_config=tree_search_config,
        max_parallel=max_parallel,
        total_instances=total,
        results=results,
        lock=checkpoint_lock,
    )
    logger.info(f"Initial checkpoint saved to {run_output_dir}/run_summary.json")

    if total_pending == 0:
        logger.info("All instances already completed! Nothing to do.")
        # Still generate final outputs
    else:
        logger.info(f"Starting parallel run: {total_pending} pending instances (of {total} total) with max_parallel={max_parallel}")

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            # Submit all pending tasks
            future_to_instance = {
                executor.submit(run_single_instance, instance_id, model_name, tree_search_config, run_timestamp, run_output_dir, repo_base_dir, index_store_dir): instance_id
                for instance_id in pending_instance_ids
            }

            # Process completed futures
            for future in as_completed(future_to_instance):
                instance_id = future_to_instance[future]
                completed += 1

                try:
                    result = future.result()
                    results.append(result)
                    duration = result.get("duration_seconds", "N/A")
                    logger.info(f"[{completed}/{total}] Instance {instance_id}: {result['status']} (took {duration}s)")
                except Exception as e:
                    logger.error(f"[{completed}/{total}] Instance {instance_id} raised exception: {e}")
                    results.append({
                        "instance_id": instance_id,
                        "status": "exception",
                        "error": str(e),
                        "duration_seconds": None,
                    })

                # Save checkpoint after each instance completes
                save_checkpoint(
                    output_dir=run_output_dir,
                    run_timestamp=run_timestamp,
                    run_start_time=run_start_time,
                    model_name=model_name,
                    dataset_path=dataset_path,
                    tree_search_config=tree_search_config,
                    max_parallel=max_parallel,
                    total_instances=total,
                    results=results,
                    lock=checkpoint_lock,
                )

    # Calculate total run time
    run_end_time = time.time()
    total_duration_seconds = round(run_end_time - run_start_time, 2)

    # Summary
    logger.info("=" * 60)
    logger.info("Run Summary:")
    logger.info(f"  Total instances: {total}")
    logger.info(f"  Total duration: {total_duration_seconds}s ({total_duration_seconds/60:.1f} minutes)")
    status_counts = {}
    for r in results:
        status = r["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    for status, count in sorted(status_counts.items()):
        logger.info(f"  {status}: {count}")

    # Aggregate token usage
    total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in results)
    total_completion_tokens = sum(r.get("completion_tokens", 0) for r in results)
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    total_nodes = sum(r.get("total_nodes", 0) for r in results)
    logger.info(f"  Total prompt tokens: {total_prompt_tokens:,}")
    logger.info(f"  Total completion tokens: {total_completion_tokens:,}")
    logger.info(f"  Total tokens: {total_tokens:,}")
    logger.info(f"  Total nodes across all instances: {total_nodes:,}")
    logger.info("=" * 60)

    # Save summary to file (in the run's subdirectory)
    summary_path = f"{run_output_dir}/run_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "run_timestamp": run_timestamp,
            "run_start_time": datetime.fromtimestamp(run_start_time).isoformat(),
            "run_end_time": datetime.fromtimestamp(run_end_time).isoformat(),
            "total_duration_seconds": total_duration_seconds,
            "status": "completed",  # Mark as completed (vs "in_progress" during checkpoints)
            "model_name": model_name,
            "dataset_path": dataset_path,
            "tree_search_config": tree_search_config,
            "max_parallel": max_parallel,
            "total_instances": total,
            "completed_count": len(results),
            "status_counts": status_counts,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "total_nodes": total_nodes,
            "output_dir": run_output_dir,
            "results": results,
        }, f, indent=2)
    logger.info(f"Summary saved to: {summary_path}")

    # Generate predictions.jsonl for SWE-bench harness
    predictions_path = write_predictions_jsonl(results, model_name, run_output_dir)
    logger.info(f"SWE-bench predictions saved to: {predictions_path}")

    # Generate evaluation.json for moatless-streamlit
    evaluation_name = f"run_{run_timestamp}_{model_name}"
    eval_path = save_evaluation_json(
        results=results,
        model_name=model_name,
        tree_search_config=tree_search_config,
        output_dir=run_output_dir,
        evaluation_name=evaluation_name,
        start_time=datetime.fromtimestamp(run_start_time, tz=timezone.utc),
        end_time=datetime.fromtimestamp(run_end_time, tz=timezone.utc),
    )
    logger.info(f"Moatless evaluation saved to: {eval_path}")

    run_id = f"moatless_{model_name}_{run_timestamp}"
    logger.info(f"To run SWE-bench harness:\npython -m swebench.harness.run_evaluation \\\n  --dataset_name princeton-nlp/SWE-bench_Lite \\\n  --predictions_path {predictions_path} \\\n  --max_workers 16 \\\n  --timeout 900 \\\n  --cache_level env \\\n  --clean True \\\n  --run_id {run_id}")

    return results


def main():
    # Set up logging to see what's happening
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Parse command-line arguments
    args = parse_args()

    # Set rate limit backoff if specified
    if args.rate_limit_backoff != 600:  # Only log if non-default
        set_rate_limit_backoff(args.rate_limit_backoff)
    else:
        set_rate_limit_backoff(args.rate_limit_backoff)

    # Tree search settings from command-line arguments
    tree_search_config = {
        "max_iterations": args.max_iterations,
        "max_expansions": args.max_expansions,
        "max_cost": args.max_cost,
        "max_depth": args.max_depth,
        "min_finished_nodes": args.min_finished_nodes,
        "max_finished_nodes": args.max_finished_nodes,
    }

    instance_or_dataset = args.instance_or_dataset
    model_name = args.model_name

    # Handle resume mode
    if args.resume:
        resume_dir = args.resume
        if not os.path.isdir(resume_dir):
            logger.error(f"Resume directory does not exist: {resume_dir}")
            return

        # Load previous run config to get dataset path
        prev_config = load_previous_run_config(resume_dir)
        if prev_config.get("dataset_path"):
            dataset_path = prev_config["dataset_path"]
            logger.info(f"Resuming run from: {resume_dir}")
            logger.info(f"Using dataset from previous run: {dataset_path}")

            # Load instance IDs from the dataset
            instance_ids = load_dataset(dataset_path)
            logger.info(f"Found {len(instance_ids)} instances in dataset")

            # Use config from CLI args (allows overriding), but default to previous run
            if not any([
                args.max_iterations != 50,  # Check if user provided non-default values
                args.max_expansions != 3,
                args.max_cost != 5.0,
                args.max_depth != 25,
            ]):
                # Use previous run's config if user didn't override
                if prev_config.get("tree_search_config"):
                    tree_search_config = prev_config["tree_search_config"]
                    logger.info(f"Using tree_search_config from previous run: {tree_search_config}")

            run_parallel_instances(
                instance_ids=instance_ids,
                model_name=model_name,
                tree_search_config=tree_search_config,
                max_parallel=args.max_parallel,
                dataset_path=dataset_path,
                resume_dir=resume_dir,
                retry_no_result=args.retry_no_result,
                retry_errors=args.retry_errors
            )
        else:
            logger.error(f"Could not find dataset_path in previous run config at: {resume_dir}")
            logger.error("Please specify the dataset path as the first argument when resuming.")
            return
    # Check if input is a dataset file or a single instance ID
    elif os.path.isfile(instance_or_dataset) and instance_or_dataset.endswith('.json'):
        # It's a dataset file - run multiple instances in parallel
        logger.info(f"Loading dataset from: {instance_or_dataset}")
        instance_ids = load_dataset(instance_or_dataset)
        logger.info(f"Found {len(instance_ids)} instances in dataset")

        run_parallel_instances(
            instance_ids=instance_ids,
            model_name=model_name,
            tree_search_config=tree_search_config,
            max_parallel=args.max_parallel,
            dataset_path=instance_or_dataset
        )
    else:
        # It's a single instance ID
        logger.info(f"Running single instance: {instance_or_dataset}")
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create a subdirectory for this run
        run_output_dir = f"{OUTPUT_DIR}/run_{run_timestamp}_{model_name}"
        os.makedirs(run_output_dir, exist_ok=True)
        logger.info(f"Output directory: {run_output_dir}")

        # Create run-specific repo base dir to avoid conflicts in parallel runs
        repo_base_dir = f"{REPO_BASE_DIR}/run_{run_timestamp}_{model_name}"
        os.makedirs(repo_base_dir, exist_ok=True)
        logger.info(f"Repository base directory: {repo_base_dir}")

        # Create run-specific index store dir to avoid conflicts in parallel runs
        index_store_dir = f"{INDEX_STORE_DIR}/run_{run_timestamp}_{model_name}"
        os.makedirs(index_store_dir, exist_ok=True)
        logger.info(f"Index store directory: {index_store_dir}")

        result = run_single_instance(
            instance_id=instance_or_dataset,
            model_name=model_name,
            tree_search_config=tree_search_config,
            run_timestamp=run_timestamp,
            output_dir=run_output_dir,
            repo_base_dir=repo_base_dir,
            index_store_dir=index_store_dir
        )
        logger.info(f"Result: {result}")
        logger.info(f"Duration: {result['duration_seconds']}s")

        # Generate predictions.jsonl for SWE-bench harness (even for single instance)
        predictions_path = write_predictions_jsonl([result], model_name, run_output_dir)
        logger.info(f"SWE-bench predictions saved to: {predictions_path}")

        # Generate evaluation.json for moatless-streamlit
        evaluation_name = f"run_{run_timestamp}_{model_name}"
        start_time = datetime.fromisoformat(result["start_time"]) if result.get("start_time") else datetime.now(tz=timezone.utc)
        end_time = datetime.fromisoformat(result["end_time"]) if result.get("end_time") else datetime.now(tz=timezone.utc)
        eval_path = save_evaluation_json(
            results=[result],
            model_name=model_name,
            tree_search_config=tree_search_config,
            output_dir=run_output_dir,
            evaluation_name=evaluation_name,
            start_time=start_time,
            end_time=end_time,
        )
        logger.info(f"Moatless evaluation saved to: {eval_path}")

        run_id = f"moatless_{model_name}_{run_timestamp}"
        logger.info(f"To run SWE-bench harness:\npython -m swebench.harness.run_evaluation \\\n  --dataset_name princeton-nlp/SWE-bench_Lite \\\n  --predictions_path {predictions_path} \\\n  --max_workers 4 \\\n  --timeout 900 \\\n  --cache_level env \\\n  --clean True \\\n  --run_id {run_id} \\\n  --report_dir {run_output_dir}/logs")


if __name__ == "__main__":
    main()
