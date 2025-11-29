import argparse
import logging
from datetime import datetime
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
        description="Run MCTS debug on a SWE-bench instance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "instance_id",
        type=str,
        help="SWE-bench instance ID (e.g., django__django-16379)"
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name to use (e.g., gh-gpt4o)"
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


# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse command-line arguments
args = parse_args()

# Instance and model from command line
INSTANCE_ID = args.instance_id
MODEL_NAME = args.model_name

# Environment paths
INDEX_STORE_DIR = "/home/idlecs229/repos/moatless-tree-search/parts"
REPO_BASE_DIR = "/home/idlecs229/repos/moatless-tree-search/downloads"

# Generate PERSIST_PATH with datetime, model name, and instance_id
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
PERSIST_PATH = f"/home/idlecs229/repos/moatless-tree-search/cover/trajectory_{timestamp}_{MODEL_NAME}_{INSTANCE_ID}.json"


# Tree search settings from command-line arguments
TREE_SEARCH_CONFIG = {
    "max_iterations": args.max_iterations,
    "max_expansions": args.max_expansions,
    "max_cost": args.max_cost,
    "max_depth": args.max_depth,
    "min_finished_nodes": args.min_finished_nodes,
    "max_finished_nodes": args.max_finished_nodes,
}

# MCTS components - enable for full MCTS behavior
USE_VALUE_FUNCTION = True
USE_DISCRIMINATOR = True
USE_FEEDBACK = False

# Message history type - matches eval configs
MESSAGE_HISTORY_TYPE = MessageHistoryType.MESSAGES  # or REACT, SUMMARY

# Load SWE-bench instance
instance = get_moatless_instance(INSTANCE_ID)
problem_statement = f"<task>\nSolve the following reported issue in the {instance['repo']} repository:\n\n{instance['problem_statement']}\n</task>"

logger.info(f"Loaded instance: {INSTANCE_ID}")
logger.info(f"Repository: {instance['repo']}")

# Create repository and index (matches evaluation_runner.py lines 377-380)
repository = create_repository(instance, repo_base_dir=REPO_BASE_DIR)
code_index = create_index(instance, repository=repository)

logger.info(f"Repository created at: {repository.repo_dir}")

# Create completion model (matches run_evaluation.py lines 431-442)
completion_model = CompletionModel(
    model=MODEL_NAME,
    response_format="tool_call"
)

logger.info(f"Model: {completion_model.model}, Response format: {completion_model.response_format}")
selector = None
value_function = None
discriminator = None
feedback_generator = None

# Only use selector for multi-expansion MCTS
if TREE_SEARCH_CONFIG["max_expansions"] > 1:
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
    max_iterations=TREE_SEARCH_CONFIG["max_iterations"],
    max_expansions=TREE_SEARCH_CONFIG["max_expansions"],
    max_cost=TREE_SEARCH_CONFIG["max_cost"],
    max_depth=TREE_SEARCH_CONFIG["max_depth"],
    min_finished_nodes=TREE_SEARCH_CONFIG["min_finished_nodes"],
    max_finished_nodes=TREE_SEARCH_CONFIG["max_finished_nodes"],
    persist_path=PERSIST_PATH,
    metadata={
        "instance_id": INSTANCE_ID,
        "debug": True,
    },
)

logger.info(f"Search tree created with max_iterations={TREE_SEARCH_CONFIG['max_iterations']}, max_expansions={TREE_SEARCH_CONFIG['max_expansions']}")

def tree_event_handler(event):
    """Simple event handler for debugging"""
    event_type = event.get("event_type", "unknown")
    logger.info(f"Tree event: {event_type}")

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
else:
    logger.warning("Search did not return a final node")
