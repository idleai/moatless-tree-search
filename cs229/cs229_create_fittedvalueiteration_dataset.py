import json
import logging
import os
import sys
import networkx as nx
import numpy as np
import pandas as pd
from copy import deepcopy
from sentence_transformers import SentenceTransformer

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

sentence_embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L12-v2"
)


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


def parse_nodes_for_value_functions_and_actions(mydict: dict):

    if mydict["completions"].get("value_function", None) is not None:
        trace = mydict["completions"].get("value_function", None)
    elif mydict["completions"].get("build_action", None) is not None:
        trace = mydict["completions"].get("build_action", None)
    else:
        trace = None

    action = mydict["action_steps"]  # [0]["action"]["action_args_class"].split(".")[2]

    try:
        action_message = mydict["assistant_message"]
    except:
        action_message = ""

    try:
        user_message = mydict["user_message"]
    except:
        user_message = ""

    node_value_function_action_list = [
        ("Node" + str(mydict["node_id"]), trace, action, action_message, user_message)
    ]

    # append to node_dict of mydict{'node_id':node_id,'response_id':response_id}
    for child in mydict["children"]:
        node_value_function_action_list += parse_nodes_for_value_functions_and_actions(
            child
        )
    return node_value_function_action_list


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


def get_sft_examples_from_trajectory(trajectory_conversation):

    sft_examples = []

    for i, entry in enumerate(trajectory_conversation):
        if entry["role"] == "assistant":
            try:
                tool_call = entry["tool_calls"][0]["function"]["name"]
                context = PROMPT + str(trajectory_conversation[:i])
                sft_examples.append({"prompt": context, "completion": tool_call})
            except:
                pass

    return sft_examples


def get_action_id(
    origin_node: str, action_steps: list
):  # origin_node, destination_node, graph,

    if len(action_steps) == 1:
        action = action_steps[0]["action"]["action_args_class"].split(".")[2]
    elif len(action_steps) > 1:
        print(f"Warning: multiple actions found for {origin_node}")
        # # get neighbors of origin node
        # neighbors = []
        # for nn in graph.neighbors(origin_node):
        #     neighbors.append(int(nn.strip('Node')))
        # neighbors = sorted(neighbors)
        action = ""
    elif len(action_steps) == 0:
        print(f"Warning: no action_steps found for {origin_node}")
        action = ""

    ACTION_TO_ID = {
        "semantic_search": 1,
        "view_code": 2,
        "find_class": 3,
        "find_code_snippet": 4,
        "create_file": 5,
        "find_function": 6,
        "run_tests": 7,
        "verified_finish": 8,
        "string_replace": 9,
    }
    if action not in ACTION_TO_ID.keys():
        print(f"Found unknown/none action: {action}")
    return ACTION_TO_ID.get(action, 0)


def get_badge_id(badge: tuple):
    BADGE_TO_ID = {
        ("star", "gold"): 1,
        ("x", "red"): 2,
        ("circle", "red"): 3,
        ("circle", "yellow"): 4,
        ("circle", "green"): 5,
        ("diamond", "red"): 6,
    }
    if badge not in BADGE_TO_ID.keys():
        print(f"Found unknown/none badge: {badge}")
    return BADGE_TO_ID.get(badge, 0)


def get_test_success_rates(node: str, prior_node: dict, eval_result: dict):
    if node.strip("Node") in eval_result["node_results"].keys():
        tests_status = eval_result["node_results"][node.strip("Node")]["tests_status"]
        if tests_status["status"] == "RESOLVED_FULL":
            p2p_test_success = 1
            f2p_test_success = 1
        else:
            try:
                num_p2p_success = len(tests_status["pass_to_pass"]["success"])
                num_p2p_failure = len(tests_status["pass_to_pass"]["failure"])
                p2p_test_success = num_p2p_success / max(
                    num_p2p_success + num_p2p_failure, 1
                )
            except:
                p2p_test_success = prior_node["p2p_test_success"]
            try:
                num_f2p_success = len(tests_status["fail_to_pass"]["success"])
                num_f2p_failure = len(tests_status["fail_to_pass"]["failure"])
                f2p_test_success = num_f2p_success / max(
                    num_f2p_success + num_f2p_failure, 1
                )
            except:
                f2p_test_success = prior_node["f2p_test_success"]
    else:
        p2p_test_success = prior_node["p2p_test_success"]
        f2p_test_success = prior_node["f2p_test_success"]

    return p2p_test_success, f2p_test_success


def create_example(origin_node: dict, action: int, destination_node: dict):
    # x dim: 1+1+1+1+1+1+384+384+1 = 775
    # y dim: 1+1+1+1+1+1+384+384 = 774 (-1 for no action)
    x_dim, y_dim = (775, 774)

    x = np.zeros((x_dim,))
    y = np.zeros((y_dim,))

    x[0] = origin_node["depth"]
    x[1] = origin_node["prior_action"]
    x[2] = origin_node["p2p_test_success"]
    x[3] = origin_node["f2p_test_success"]
    x[4] = origin_node["reward"]
    x[5] = origin_node["badge"]
    x[6 : 6 + 384] = origin_node["prompt_embedding"]
    x[6 + 384 : 6 + 384 * 2] = origin_node["prior_action_embedding"]
    x[6 + 384 * 2] = action

    y[0] = destination_node["depth"]
    y[1] = destination_node["prior_action"]
    y[2] = destination_node["p2p_test_success"]
    y[3] = destination_node["f2p_test_success"]
    y[4] = destination_node["reward"]
    y[5] = destination_node["badge"]
    y[6 : 6 + 384] = destination_node["prompt_embedding"]
    y[6 + 384 : 6 + 384 * 2] = destination_node["prior_action_embedding"]

    return x, y


def parse_trajectory_tree(
    selected_tree_path: str,
):

    # initialize output
    output_x = []
    output_y = []

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

    node_value_function_action_list = parse_nodes_for_value_functions_and_actions(
        traj["root"]
    )
    node_list = [item[0] for item in node_value_function_action_list]
    value_function_action_map = {
        item[0]: (item[1], item[2], item[3], item[4])
        for item in node_value_function_action_list
    }

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

        # Get embedding of the prompt/coding issue
        prompt = value_function_action_map["Node0"][3]
        prompt_embedding = sentence_embedding_model.encode(prompt)

        # For each node in the trajectory
        origin_node = {}
        destination_node = {}
        for j, node in enumerate(shortest_path[:-1]):

            # get origin_node state
            if j == 0:
                origin_node = {
                    "depth": j,
                    "prior_action": 0,
                    "p2p_test_success": 0,
                    "f2p_test_success": 0,
                    "reward": G.nodes[node].get("reward", 0),
                    "badge": get_badge_id(badge_list[j]),
                    "prompt_embedding": prompt_embedding,
                    "prior_action_embedding": np.zeros((384,)),
                }
            else:
                origin_node = deepcopy(destination_node)

            # get action from origin_node to destination_node
            action = get_action_id(
                shortest_path[j + 1], value_function_action_map[shortest_path[j + 1]][1]
            )

            # get test success at destination_node
            p2p_test_success, f2p_test_success = get_test_success_rates(
                shortest_path[j + 1], origin_node, eval_result
            )

            if value_function_action_map[shortest_path[j + 1]][2] is None:
                prior_action_embedding = np.zeros((384,))
            else:
                prior_action_embedding = sentence_embedding_model.encode(
                    value_function_action_map[shortest_path[j + 1]][2]
                )

            # get destination_node state
            destination_node = {
                "depth": j + 1,  # dim: 1
                "prior_action": action,  # dim: 1
                "p2p_test_success": p2p_test_success,  # dim: 1
                "f2p_test_success": f2p_test_success,  # dim: 1
                "reward": G.nodes[shortest_path[j + 1]].get("reward", 0),  # dim: 1
                "badge": get_badge_id(badge_list[j + 1]),  # dim: 1
                "prompt_embedding": prompt_embedding,  # dim: 384
                "prior_action_embedding": prior_action_embedding,  # dim: 384
            }

            x, y = create_example(origin_node, action, destination_node)
            if i == 0 and j == 0:
                output_x = x
                output_y = y
            else:
                output_x = np.vstack((output_x, x))
                output_y = np.vstack((output_y, y))

    return output_x, output_y


if __name__ == "__main__":

    ###########################################################
    ### EXTRACT TRAJECTORIES FROM PAST RUNS FOR FVI DATASET ###
    ###########################################################

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
    outputs_x, outputs_y = [], []

    for i, folder in enumerate(issues):

        print(f"\nNow parsing {folder.split('/')[-1]}\n")

        output_x, output_y = parse_trajectory_tree(
            selected_tree_path=os.path.join(dir, folder, "trajectory.json")
        )
        output_id = folder.split("/")[-1]

        if i == 0:
            outputs_x = output_x
            outputs_y = output_y
        else:
            outputs_x = np.vstack((outputs_x, output_x))
            outputs_y = np.vstack((outputs_y, output_y))

    with open(os.path.join(output_dir, "fvi_nn_x.npy"), "wb") as f:
        np.save(f, outputs_x)
    with open(os.path.join(output_dir, "fvi_nn_y.npy"), "wb") as f:
        np.save(f, outputs_y)

    print(f"dimensions of outputs_x, outputs_y: {outputs_x.shape}, {outputs_y.shape}")

    #####################################################################
