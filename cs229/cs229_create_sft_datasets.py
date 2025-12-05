import argparse
import json
import logging
import os
import sys
import networkx as nx
import pandas as pd

module_path = os.path.abspath(os.path.join(".."))

# Insert the path at the beginning of sys.path
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from moatless.benchmark.report import analyse_file_context
from moatless.benchmark.utils import get_moatless_instance
from moatless.node import Node
from moatless.search_tree import SearchTree

# NOTE: note which scripts are directly based/derived from code from moatless repo

logger = logging.getLogger(__name__)


def decide_badge(node_info):
    """
    Decide which badge to show for a node based on its properties and the instance data.
    Returns a tuple of (symbol, color) or None if no badge should be shown.
    """
    if node_info.get("resolved") is not None:
        if node_info.get("resolved", False):
            return ("star", "gold")
        else:
            return ("x", "red")

    if node_info.get("error"):
        return ("circle", "red")

    if node_info.get("warning"):
        return ("circle", "yellow")

    if node_info.get("context_status") in ["found_spans"]:
        if node_info.get("patch_status") == "wrong_files":
            return ("circle", "yellow")

        return ("circle", "green")

    if (
        node_info.get("context_status") in ["found_files"]
        or node_info.get("patch_status") == "right_files"
    ):
        return ("circle", "yellow")

    return None


def build_graph(
    root_node: Node, eval_result: dict | None = None, instance: dict | None = None
):
    G = nx.DiGraph()

    def is_resolved(node_id):
        if not eval_result:
            return None

        if str(node_id) not in eval_result.get("node_results", {}):
            return None

        return eval_result["node_results"][str(node_id)].get("resolved", False)

    def add_node_to_graph(node: Node, parent_id: str | None = None):
        node_id = f"Node{node.node_id}"

        # Debug logging
        if parent_id:
            logger.info(f"Processing node {node_id} with parent {parent_id}")
        else:
            logger.info(f"Processing root node {node_id}")

        # Initialize node attributes
        node_attrs = {}

        # Sanitize node attributes to avoid Graphviz syntax errors
        if node.action:
            if node.action.name == "str_replace_editor":
                action_name = (
                    str(node.action.command).replace('"', '\\"').replace("\\", "\\\\")
                )
            else:
                action_name = (
                    str(node.action.name).replace('"', '\\"').replace("\\", "\\\\")
                )
        else:
            action_name = ""

        if instance:
            context_stats = analyse_file_context(instance, node.file_context)
        else:
            context_stats = None

        warning = ""
        error = ""
        if node.observation and node.observation.properties:
            if "test_results" in node.observation.properties:
                test_results = node.observation.properties["test_results"]
                failed_test_count = sum(
                    1 for test in test_results if test["status"] in ["FAILED", "ERROR"]
                )
                if failed_test_count > 0:
                    warning = f"{failed_test_count} failed tests"
            if "fail_reason" in node.observation.properties:
                error = (
                    str(node.observation.properties["fail_reason"])
                    .replace('"', '\\"')
                    .replace("\\", "\\\\")
                )

        if node.observation and node.observation.expect_correction:
            warning += f"\nExpected correction"

        resolved = is_resolved(node.node_id) if eval_result else None

        # Add feedback data to node attributes if it exists
        feedback_data = getattr(node, "feedback_data", None)
        if feedback_data:
            node_attrs.update(
                {
                    "feedback_analysis": (
                        str(feedback_data.analysis)
                        .replace('"', '\\"')
                        .replace("\\", "\\\\")
                        if feedback_data.analysis
                        else ""
                    ),
                    "feedback_text": (
                        str(feedback_data.feedback)
                        .replace('"', '\\"')
                        .replace("\\", "\\\\")
                        if feedback_data.feedback
                        else ""
                    ),
                    "feedback_suggested_node": feedback_data.suggested_node_id,
                }
            )

        # Only add the node if it doesn't exist
        if not G.has_node(node_id):
            # Add base attributes
            node_attrs.update(
                {
                    "name": action_name,
                    "type": "node",
                    "visits": node.visits or 1,
                    "duplicate": node.is_duplicate,
                    "avg_reward": node.value / node.visits if node.visits else 0,
                    "reward": node.reward.value if node.reward else 0,
                    "warning": warning.replace('"', '\\"').replace("\\", "\\\\"),
                    "error": error,
                    "resolved": resolved,
                    "context_status": context_stats.status if context_stats else None,
                    "patch_status": (
                        context_stats.patch_status if context_stats else None
                    ),
                    "explanation": (
                        str(node.reward.explanation)
                        .replace('"', '\\"')
                        .replace("\\", "\\\\")
                        if node.reward
                        else ""
                    ),
                }
            )

            # Remove None values to avoid Graphviz issues
            node_attrs = {k: v for k, v in node_attrs.items() if v is not None}

            G.add_node(node_id, **node_attrs)

        # Add edge from parent if provided
        if parent_id:
            if G.has_edge(parent_id, node_id):
                logger.warning(f"Duplicate edge detected: {parent_id} -> {node_id}")
            else:
                G.add_edge(parent_id, node_id)

        # Process children
        for child in node.children:
            add_node_to_graph(child, node_id)

    # Start from root with no parent
    add_node_to_graph(root_node)

    # Verify tree structure
    for node in G.nodes():
        in_edges = list(G.in_edges(node))
        if len(in_edges) > 1:
            logger.error(f"Node {node} has multiple parents: {in_edges}")

    G.graph["graph"] = {
        "rankdir": "TB",
        "ranksep": "1.5",
        "nodesep": "1.0",
        "splines": "ortho",  # Changed from polyline for simpler layout
    }

    return G


def get_node_badges(
    node_list,
    G,
):

    badge_list = []

    for node in node_list:
        badge = None
        node_info = G.nodes[node]

        if node_info.get("type") == "node":
            badge = decide_badge(node_info)
        else:
            if node_info.get("warnings"):
                badge = ("diamond", "red")
        badge_list.append(badge)
    return badge_list


def parse_nodes_for_value_functions(mydict: dict, training_objective: str = "bug_fixing"):
    """
    Extract completions for SFT training based on the training objective.

    Args:
        mydict: Node dictionary from trajectory.json
        training_objective: Either "bug_fixing" or "value_function"
            - "bug_fixing": Extract build_action completions (tool selection)
            - "value_function": Extract value_function completions (reward model evaluations)

    Returns:
        List of tuples: (node_name, completion_data)
    """
    if training_objective == "bug_fixing":
        if mydict["completions"].get("build_action", None) is not None:
            trace = mydict["completions"].get("build_action", None)
        else:
            trace = None
    elif training_objective == "value_function":
        if mydict["completions"].get("value_function", None) is not None:
            trace = mydict["completions"].get("value_function", None)
        else:
            trace = None
    else:
        raise ValueError(f"Invalid training_objective: {training_objective}. Must be 'bug_fixing' or 'value_function'")

    node_value_function_list = [("Node" + str(mydict["node_id"]), trace)]

    # do the same for each children
    for child in mydict["children"]:
        node_value_function_list += parse_nodes_for_value_functions(child, training_objective)
    return node_value_function_list


def compute_trajectory_reward(badge_list, discount=0.05):
    """
    The following badges are used to indicate the status of a node:
    | Badge | Shape | Color | Description |
    |-------|-------|-------|-------------|
    | ⭐ | Star | Green | Node is marked as resolved |
    | ❌ | X | Red | Invalid edits or failed tests |
    | 🟢 | Circle | Green | Correct code spans present in the context |
    | 🟡 | Circle | Yellow | Either:<br>• Found files but not spans<br>• Found spans but in wrong files<br>|

    """
    counter = 0
    cumulative_reward = 0
    for i, badge in enumerate(badge_list):
        if badge is None:
            cumulative_reward += 0
        elif badge == ("circle", "green"):
            cumulative_reward += 5 * (1 - discount) ** counter
        elif badge == ("circle", "yellow"):
            cumulative_reward += 2 * (1 + discount) ** counter
        elif badge == ("circle", "red"):
            cumulative_reward += -10 * (1 + discount) ** counter
        elif badge == ("x", "red"):
            cumulative_reward += -100 * (1 + discount) ** counter
        elif badge == ("star", "gold") and i != len(badge_list) - 1:
            cumulative_reward += 50 * (1 - discount) ** counter
        elif badge == ("star", "gold") and i == len(badge_list) - 1:
            cumulative_reward += 100 * (1 - discount) ** counter
        else:
            print(f"WARNING: Unknown badge encountered {badge}")
        counter += 1

    return cumulative_reward


def print_trajectory_trace(completion):
    printout = ""
    if completion["input"]:
        for input_idx, input_msg in enumerate(completion["input"]):
            content = (
                f"# Step {input_idx} by {input_msg['role']} in Solution Trajectory:\n"
            )
            try:
                if "content" in input_msg:
                    if isinstance(input_msg["content"], str):
                        content += input_msg["content"]
                    elif (
                        isinstance(input_msg["content"], list)
                        and input_msg["role"] == "user"
                    ):
                        content_list = [
                            c.get("content") or c.get("text")
                            for c in input_msg["content"]
                        ]

                        content += "\n\n".join(content_list)
                    else:
                        content += json.dumps(input_msg["content"], indent=2)

                    if "tool_calls" in input_msg:
                        content += str(input_msg["tool_calls"])
                else:
                    content += f"Message {input_idx + 1} by {input_msg['role']}:\n"
                    content += json.dumps(input_msg, indent=2)
            except Exception as e:
                logger.exception(f"Failed to parse {json.dumps(input_msg, indent=2)}")
            printout += content + "\n\n"
    else:
        print(f"Error: no input in completion")
    return printout


def get_value_function_sft_example(value_function_completion):
    """
    Extract SFT example for value function (reward model) training.
    This is completely different for bug-fixing agent training.

    The value_function completion structure:
    - input: List of messages including the response as the last message (duplicate)
    - response: The actual LLM response with evaluation

    For Tinker training format:
    - trajectory_conversation: input[:-1] + [response as assistant message]
    - trajectory_subid: Index of the assistant message we want to train on

    Args:
        value_function_completion: Dict with 'input' and 'response' keys

    Returns:
        List with single dict containing SFT training example
    """
    if not value_function_completion:
        return []

    # we inlclude all input messages except the last one which duplicates the response
    prompt_messages = value_function_completion['input'][:-1]

    response_content = value_function_completion['response']['choices'][0]['message']['content']

    full_conversation = prompt_messages + [
        {'role': 'assistant', 'content': response_content}
    ]


    assistant_count = sum(1 for msg in full_conversation if msg.get('role') == 'assistant')
    trajectory_subid = assistant_count - 1

    return [{
        # Tinker training required
        'trajectory_conversation': full_conversation,
        'trajectory_subid': trajectory_subid,
        # for reference/debugging
        'prompt': str(prompt_messages),
        'completion': response_content,
    }]


def get_bug_fixing_sft_examples(trajectory_conversation):
    """
    Extract SFT examples from a resolved trajectory for per-action training.

    Each example corresponds to one assistant action in the trajectory.
    We save the full conversation and the index of the assistant action
    so that the Tinker adapter can train on that specific action. 
    The PROMPT and completion are only saved for reference (or to be used with another implementation)
    since they are extracted directly from the conversation. 
    This is due to how Tinker format works for SFT.

    Args:
        trajectory_conversation: Full conversation history (list of message dicts)

    Returns:
        List of dicts with fields:
            - trajectory_conversation: Full conversation history (for Tinker)
            - trajectory_subid: Index of this assistant action among all assistant messages (for Tinker)
            - prompt: Flattened prompt string (for reference/debugging)
            - completion: Action name (for reference/debugging)
    """
    PROMPT = """
You are an autonomous AI assistant with superior programming skills. Your task is twofold: (1) review the in-progress work of a different AI agent who is trying to create a bug fix to solve a software issue provided to it by a user, (2) suggest the next action the AI assistant should take to get closer to a solution. The other AI agent is responsible for identifying and modifying the correct file(s) in response to the problem statement. At this stage, the agent is still working on the solution.

# Guidelines

1. **Analysis First**
   - Review all previous actions and their observations in the provided history
   - Understand what has been done and what information you have

2. **Document Your Thoughts**
   - ALWAYS write your reasoning in `<thoughts>` tags before providing your suggested action
   - Justify why you're choosing the next action
   - Describe what you expect the AI assistant will learn/achieve from performing your suggested action

# Available Actions

The following actions are available to you to suggest. Do not make up any other actions.

## **SemanticSearch

Use this when the AI agent doesn't know exact names or code but wants to find related functionality.

Perfect for:
- Finding functionality by description: query=\"code that handles password hashing\"
- Finding related test cases: query=\"tests for user registration\", category=\"test\"
- Finding implementations: query=\"database connection pooling\", category=\"implementation\"
- Finding patterns: query=\"error handling for API requests\"

This is the most flexible search when you:
- Don't know exact function/class names
- Want to find similar implementations
- Need to discover related code
- Want to explore how certain features are implemented

## **FindClass

Use this when the AI agent knows the exact name of a class it wants to find.

Perfect for:
- Finding class implementations: class_name=\"UserRepository\"
- Locating test classes: class_name=\"TestUserAuthentication\"
- Finding base classes: class_name=\"BaseController\"
- Finding classes in specific modules: class_name=\"Config\", file_pattern=\"src/config/*.py\"

## **FindFunction

Use this when the AI agent knows the exact name of a function or method it wants to find.

Perfect for:
- Finding test cases: function_name=\"test_user_login\"
- Locating specific implementations: function_name=\"process_payment\"
- Finding all methods with a name: function_name=\"validate\"
- Finding a specific class method: function_name=\"save\", class_name=\"UserRepository\"

## **FindCodeSnippet

Use this when the AI agent knows the exact code it wants to find.
     It will run the command: grep -n -r \"code_snippet\" \"file_pattern\"

Perfect for:
- Finding specific constant definitions: code_snippet=\"MAX_RETRIES = 3\"
- Finding decorator usage: code_snippet=\"@retry(max_attempts=3)\"
- Finding specific imports: code_snippet=\"from datetime import datetime\"
- Finding configuration patterns: code_snippet=\"DEBUG = os.getenv('DEBUG', False)\"

Note: The AI agent must know the exact code snippet. Use SemanticSearch if it only knows
what the code does but not its exact implementation.

## **ViewCode

View the code in a file or a specific code span.

## **StringReplace

Applies a change to a file by replacing text with exact string matching.

## **CreateFile

Create a new file with specified content.

Notes:
* Cannot be used if the specified path already exists
* Will create parent directories if they don't exist
* File content should include proper indentation and formatting

## **AppendString

Append text content to the end of a file.

## **RunTests

Run the specified unit tests on the codebase.

## **Finish

Indicate that the task is fully completed and verified with new or modified tests.

## **Reject

Reject the task and explain why.

# Response Format

You must respond with the following format:
- First, your reasoning in `<thoughts>` tags
- Second, your final selected action in `<answer>` tags. Your final selected action must be one of the following: SemanticSearch, FindClass, FindFunction, FindCodeSnippet, ViewCode, StringReplace, CreateFile

----------------------------------------------------------------------------------
----------------------------------------------------------------------------------

Previous Actions and Observations in Chat History with AI Assistant:

"""

    sft_examples = []
    assistant_action_count = 0

    for i, entry in enumerate(trajectory_conversation):
        if entry["role"] == "assistant":
            try:
                tool_call = entry["tool_calls"][0]["function"]["name"]
                context = PROMPT + str(trajectory_conversation[:i])

                sft_examples.append({
                    # Tinker training required
                    "trajectory_conversation": trajectory_conversation,
                    "trajectory_subid": assistant_action_count,
                    # only for reference/debugging or another different implementation
                    "prompt": context,
                    "completion": tool_call,
                })
                assistant_action_count += 1
            except:
                pass

    return sft_examples


def get_sft_examples_from_trajectory(trajectory_data, training_objective="bug_fixing"):
    """
    Extract SFT examples from trajectory data based on training objective.

    Args:
        trajectory_data: Either a conversation list (bug_fixing) or completion dict (value_function)
        training_objective: "bug_fixing" or "value_function"

    Returns:
        List of SFT training examples in Tinker format
    """
    if training_objective == "value_function":
        return get_value_function_sft_example(trajectory_data)
    elif training_objective == "bug_fixing":
        return get_bug_fixing_sft_examples(trajectory_data)
    else:
        raise ValueError(f"Invalid training_objective: {training_objective}")


def parse_trajectory_tree(
    selected_tree_path: str,
    training_objective: str = "bug_fixing",
):

    # initialize output
    output = []

    # get search tree
    search_tree = SearchTree.from_file(
        selected_tree_path,
    )
    # nodes = search_tree.root.get_all_nodes()
    # total_nodes = count_total_nodes(search_tree.root)

    # get eval result
    directory_path = os.path.dirname(selected_tree_path)
    eval_path = f"{directory_path}/eval_result.json"
    if os.path.exists(eval_path):
        with open(f"{directory_path}/eval_result.json", "r") as f:
            eval_result = json.load(f)

    # get instance/issue/training example
    if search_tree.metadata.get("instance_id"):
        instance = get_moatless_instance(search_tree.metadata["instance_id"])

    with open(selected_tree_path, "r") as f:
        traj = json.load(f)

    node_value_function_list = parse_nodes_for_value_functions(traj["root"], training_objective)
    node_list = [item[0] for item in node_value_function_list]
    value_function_map = {item[0]: item[1] for item in node_value_function_list}

    # Original graph visualization code
    G = build_graph(search_tree.root, eval_result, instance)

    # Get all nodes that have no neighbors -- these are terminal nodes
    terminal_nodes = []
    for node in G.nodes():
        if len([neighbor for neighbor in G.neighbors(node)]) == 0:
            terminal_nodes.append(node)

    print(f"Found {len(terminal_nodes)} trajectories.\n")

    # For each terminal node (i.e., each trajectory)
    for i, term_node in enumerate(terminal_nodes):

        # Get the shortest path from the root node ('Node0') to the terminal node
        shortest_path = nx.shortest_path(G, source="Node0", target=term_node)

        # Get the badge associated with each node on the trajectory
        badge_list = get_node_badges(shortest_path, G)

        # If it is a resolved trajectory
        if badge_list[-1] == ("star", "gold"):

            print(f"    Processing {term_node}'s RESOLVED trajectory.\n")

            if training_objective == "bug_fixing":
                # for bug-fixing we extract from terminal node full conversation
                # this contains all assistant actions with tool calls in the trajectory
                trajectory_data = value_function_map[term_node]["input"]
                examples = get_sft_examples_from_trajectory(trajectory_data, training_objective)

                for example in examples:
                    example["trajectory"] = str(shortest_path)
                    example["trajectory_badges"] = str(badge_list)
                    output.append(example)

            elif training_objective == "value_function":
                # for value_function we extract from ALL nodes in the resolved path
                # so each node has one evaluation to learn from
                for node_name in shortest_path:
                    if value_function_map.get(node_name):
                        trajectory_data = value_function_map[node_name]
                        examples = get_sft_examples_from_trajectory(trajectory_data, training_objective)

                        for example in examples:
                            example["trajectory"] = str(shortest_path)
                            example["trajectory_badges"] = str(badge_list)
                            example["node_name"] = node_name
                            output.append(example)

    return output


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create SFT datasets from MCTS trajectories")
    parser.add_argument(
        "--objective",
        type=str,
        choices=["bug_fixing", "value_function"],
        default="bug_fixing",
        help="Training objective: 'bug_fixing' for action agent or 'value_function' for reward model (default: bug_fixing)"
    )
    args = parser.parse_args()

    training_objective = args.objective

    dir = "./20251109_qwen3_coder_30b_a3b_instruct_0_7_exp_3_n_50_fmt_tool_call_hist_messages_8"
    output_dir = "./datasets"
    os.makedirs(output_dir, exist_ok=True)


    # filter to only directories that have trajectory.json, this prevents some issues for folders like 'prompt_logs' or 'logs'
    # that don't have a trajectory.json inside them
    all_items = os.listdir(dir)
    issues = [
        item
        for item in all_items
        if os.path.isdir(os.path.join(dir, item))
        and os.path.exists(os.path.join(dir, item, "trajectory.json"))
    ]

    # collect all rows in a list first, then create DataFrame once. Can end up being more efficient but is
    # primarly because of the deprecation warnings :)
    trajectory_rows = []

    for folder in issues:
        print(f"\nNow parsing {folder.split('/')[-1]}\n")
        output = parse_trajectory_tree(
            selected_tree_path=os.path.join(dir, folder, "trajectory.json"),
            training_objective=training_objective
        )
        output_id = folder.split("/")[-1]
        for example in output:
            example["trajectory_id"] = output_id
            # trajectory_subid is already set in get_sft_examples_from_trajectory()
            trajectory_rows.append(example)

    df = pd.DataFrame(trajectory_rows)

    first_sans_three_folders = [folder.split("/")[-1] for folder in issues[:-3]]
    last_three_folders = [folder.split("/")[-1] for folder in issues[-3:]]
    df_train = df[df.trajectory_id.isin(first_sans_three_folders)]
    df_test = df[df.trajectory_id.isin(last_three_folders)]

    # add objective suffix to filenames to differentiate datasets
    objective_suffix = f"_{training_objective}" if training_objective != "bug_fixing" else ""

    df.to_csv(os.path.join(output_dir, f"{dir.split('/')[-1]}_trajectories_sft{objective_suffix}.csv"))
    df_train.to_csv(
        os.path.join(output_dir, f"{dir.split('/')[-1]}_trajectories_sft{objective_suffix}_train.csv")
    )
    df_test.to_csv(
        os.path.join(output_dir, f"{dir.split('/')[-1]}_trajectories_sft{objective_suffix}_test.csv")
    )

    print(f"Training objective: {training_objective}")

    print(
        f"Length of df, df_train, df_test: {len(df)}, {len(df_train)}, {len(df_test)}"
    )
