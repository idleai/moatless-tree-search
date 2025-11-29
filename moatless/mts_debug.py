import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from moatless.agent.code_agent import CodingAgent
from moatless.benchmark.swebench import create_repository, create_index
from moatless.benchmark.utils import get_moatless_instance
from moatless.completion.completion import CompletionModel, LLMResponseFormat
from moatless.discriminator import AgentDiscriminator
from moatless.feedback.reward_feedback import RewardFeedbackGenerator
from moatless.file_context import FileContext
from moatless.search_tree import SearchTree
from moatless.selector import BestFirstSelector
from moatless.schema import MessageHistoryType
from moatless.value_function.base import ValueFunction


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

    return parser.parse_args()


# Environment paths
INDEX_STORE_DIR = "/home/idlecs229/repos/moatless-tree-search/parts"
REPO_BASE_DIR = "/home/idlecs229/repos/moatless-tree-search/downloads"
OUTPUT_DIR = "/home/idlecs229/repos/moatless-tree-search/cover"

# MCTS components - enable for full MCTS behavior
USE_VALUE_FUNCTION = True
USE_DISCRIMINATOR = True
USE_FEEDBACK = False

# Message history type - matches eval configs
MESSAGE_HISTORY_TYPE = MessageHistoryType.MESSAGES  # or REACT, SUMMARY


def run_single_instance(instance_id: str, model_name: str, tree_search_config: dict, run_timestamp: str) -> dict:
    """Run MCTS on a single instance. Returns a result dict."""
    logger = logging.getLogger(f"mts.{instance_id}")
    result = {
        "instance_id": instance_id,
        "status": "unknown",
        "error": None,
        "final_node_id": None,
        "persist_path": None,
    }

    try:
        # Generate PERSIST_PATH with datetime, model name, and instance_id
        persist_path = f"{OUTPUT_DIR}/trajectory_{run_timestamp}_{model_name}_{instance_id}.json"
        result["persist_path"] = persist_path

        # Load SWE-bench instance
        instance = get_moatless_instance(instance_id)
        problem_statement = f"<task>\nSolve the following reported issue in the {instance['repo']} repository:\n\n{instance['problem_statement']}\n</task>"

        logger.info(f"Loaded instance: {instance_id}")
        logger.info(f"Repository: {instance['repo']}")

        # Create repository and index (matches evaluation_runner.py lines 377-380)
        repository = create_repository(instance, repo_base_dir=REPO_BASE_DIR)
        code_index = create_index(instance, repository=repository)

        logger.info(f"Repository created at: {repository.repo_dir}")

        # Create completion model (matches run_evaluation.py lines 431-442)
        completion_model = CompletionModel(
            model=model_name,
            response_format="tool_call"
        )

        logger.info(f"Model: {completion_model.model}, Response format: {completion_model.response_format}")
        selector = None
        value_function = None
        discriminator = None
        feedback_generator = None

        # Only use selector for multi-expansion MCTS
        if tree_search_config["max_expansions"] > 1:
            selector = BestFirstSelector()
            logger.info("Using BestFirstSelector for multi-expansion MCTS")

        if USE_VALUE_FUNCTION:
            value_function = ValueFunction(completion_model=completion_model, correction_award=0)
            logger.info("Value function enabled")

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

            # Print some stats
            total_usage = search_tree.total_usage()
            logger.info(f"Total usage: {total_usage}")

            # Show the tree structure
            from moatless.node import generate_ascii_tree
            logger.info(f"\nTree structure:\n{generate_ascii_tree(search_tree.root)}")

            result["status"] = "completed"
            result["final_node_id"] = node.node_id
        else:
            logger.warning("Search did not return a final node")
            result["status"] = "no_result"

    except Exception as e:
        logger.error(f"Error running instance {instance_id}: {e}", exc_info=True)
        result["status"] = "error"
        result["error"] = str(e)

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


def run_parallel_instances(instance_ids: list, model_name: str, tree_search_config: dict, max_parallel: int):
    """Run multiple instances in parallel using ThreadPoolExecutor."""
    logger = logging.getLogger("mts.parallel")
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    total = len(instance_ids)
    completed = 0
    results = []

    logger.info(f"Starting parallel run of {total} instances with max_parallel={max_parallel}")

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all tasks
        future_to_instance = {
            executor.submit(run_single_instance, instance_id, model_name, tree_search_config, run_timestamp): instance_id
            for instance_id in instance_ids
        }

        # Process completed futures
        for future in as_completed(future_to_instance):
            instance_id = future_to_instance[future]
            completed += 1

            try:
                result = future.result()
                results.append(result)
                logger.info(f"[{completed}/{total}] Instance {instance_id}: {result['status']}")
            except Exception as e:
                logger.error(f"[{completed}/{total}] Instance {instance_id} raised exception: {e}")
                results.append({
                    "instance_id": instance_id,
                    "status": "exception",
                    "error": str(e),
                })

    # Summary
    logger.info("=" * 60)
    logger.info("Run Summary:")
    logger.info(f"  Total instances: {total}")
    status_counts = {}
    for r in results:
        status = r["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    for status, count in sorted(status_counts.items()):
        logger.info(f"  {status}: {count}")
    logger.info("=" * 60)

    # Save summary to file
    summary_path = f"{OUTPUT_DIR}/run_summary_{run_timestamp}_{model_name}.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "run_timestamp": run_timestamp,
            "model_name": model_name,
            "tree_search_config": tree_search_config,
            "total_instances": total,
            "status_counts": status_counts,
            "results": results,
        }, f, indent=2)
    logger.info(f"Summary saved to: {summary_path}")

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

    # Check if input is a dataset file or a single instance ID
    if os.path.isfile(instance_or_dataset) and instance_or_dataset.endswith('.json'):
        # It's a dataset file - run multiple instances in parallel
        logger.info(f"Loading dataset from: {instance_or_dataset}")
        instance_ids = load_dataset(instance_or_dataset)
        logger.info(f"Found {len(instance_ids)} instances in dataset")

        run_parallel_instances(
            instance_ids=instance_ids,
            model_name=model_name,
            tree_search_config=tree_search_config,
            max_parallel=args.max_parallel
        )
    else:
        # It's a single instance ID
        logger.info(f"Running single instance: {instance_or_dataset}")
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = run_single_instance(
            instance_id=instance_or_dataset,
            model_name=model_name,
            tree_search_config=tree_search_config,
            run_timestamp=run_timestamp
        )
        logger.info(f"Result: {result}")


if __name__ == "__main__":
    main()
