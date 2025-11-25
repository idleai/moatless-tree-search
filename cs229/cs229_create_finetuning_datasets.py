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


def parse_nodes_for_value_functions(mydict: dict):

    if mydict["completions"].get("value_function", None) is not None:
        trace = mydict["completions"].get("value_function", None)
    elif mydict["completions"].get("build_action", None) is not None:
        trace = mydict["completions"].get("build_action", None)
    else:
        trace = None

    node_value_function_list = [("Node" + str(mydict["node_id"]), trace)]

    # append to node_dict of mydict{'node_id':node_id,'response_id':response_id}
    for child in mydict["children"]:
        node_value_function_list += parse_nodes_for_value_functions(child)
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


def parse_trajectory_tree(
    selected_tree_path: str,
):

    # initialize output
    output = {}

    # get search tree
    search_tree = SearchTree.from_file(
        selected_tree_path,
        # repository=repository,
        # runtime=runtime,
        # code_index=code_index,
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

    node_value_function_list = parse_nodes_for_value_functions(traj["root"])
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

        print(f"    Processing {term_node}'s trajectory.")

        # Get the shortest path from the root node ('Node0') to the terminal node
        shortest_path = nx.shortest_path(G, source="Node0", target=term_node)

        # Get the badge associated with each node on the trajecotry
        badge_list = get_node_badges(shortest_path, G)

        # Collate the overall trajectory's set of actions
        trajectory_trace = print_trajectory_trace(value_function_map[term_node])
        trajectory_conversation = value_function_map[term_node]["input"]

        # Compute associated reward associated with the trajectory
        reward = compute_trajectory_reward(badge_list)

        # Resolved
        if badge_list[-1] == ("star", "gold"):
            resolved = True
        else:
            resolved = False

        # Save to output
        output[i] = {
            "trajectory": str(shortest_path),
            "trajectory_badges": str(badge_list),
            "trajectory_length": len(shortest_path),
            "trajectory_trace": trajectory_trace,
            "trajectory_conversation": trajectory_conversation,
            "trajectory_reward": reward,
            "trajectory_resolved": resolved,
        }

    return output


if __name__ == "__main__":
    df_columns = [
        "trajectory_id",
        "trajectory_subid",
        "trajectory",
        "trajectory_badges",
        "trajectory_length",
        "trajectory_trace",
        "trajectory_conversation",
        "trajectory_reward",
        "trajectory_resolved",
    ]

    df = pd.DataFrame(columns=df_columns)

    dir = "/Users/tbassman/Desktop/GitHub/External/cs229/project/20251109_qwen3_coder_30b_a3b_instruct_0_7_exp_3_n_50_fmt_tool_call_hist_messages_8"
    issues = os.listdir(dir)

    for folder in issues:
        print(f"\nNow parsing {folder.split('/')[-1]}\n")
        output = parse_trajectory_tree(
            selected_tree_path=os.path.join(dir, folder, "trajectory.json")
        )
        output_id = folder.split("/")[-1]
        for key, val in output.items():
            dict = {
                "trajectory_id": [output_id],
                "trajectory_subid": [key],
                "trajectory": [val["trajectory"]],
                "trajectory_badges": [val["trajectory_badges"]],
                "trajectory_length": [val["trajectory_length"]],
                "trajectory_trace": [val["trajectory_trace"]],
                "trajectory_conversation": [val["trajectory_conversation"]],
                "trajectory_reward": [val["trajectory_reward"]],
                "trajectory_resolved": [val["trajectory_resolved"]],
            }
            df = pd.concat([df, pd.DataFrame(dict)], ignore_index=True)

    first_sans_two_folders = [folder.split("/")[-1] for folder in issues[:-2]]
    last_two_folders = [folder.split("/")[-1] for folder in issues[-2:]]
    df_train = df[df.trajectory_id.isin(first_sans_two_folders)]
    df_test = df[df.trajectory_id.isin(last_two_folders)]
    df.to_csv(f"{dir.split('/')[-1]}_trajectories.csv")
    df_train.to_csv(f"{dir.split('/')[-1]}_trajectories_train.csv")
    df_test.to_csv(f"{dir.split('/')[-1]}_trajectories_test.csv")

    # Create preferencce dataset (e.g., for DPO, IPO, etc.)
    pref_df_columns = ["chosen", "rejected", "score_chosen", "score_rejected"]
    pref_df_train = pd.DataFrame(columns=pref_df_columns)
    pref_df_test = pd.DataFrame(columns=pref_df_columns)

    for folder in issues[:-2]:
        df_view = (
            df_train[df_train["trajectory_id"] == folder.split("/")[-1]]
            .sort_values(by="trajectory_reward", ascending=False)
            .reset_index(drop=True)
        )
        for i, i_row in df_view.iterrows():
            for j, j_row in df_view.iterrows():
                if i < j and i_row["trajectory_reward"] > j_row["trajectory_reward"]:
                    dict = {
                        "chosen": [i_row["trajectory_conversation"]],
                        "rejected": [j_row["trajectory_conversation"]],
                        "score_chosen": [i_row["trajectory_reward"]],
                        "score_rejected": [j_row["trajectory_reward"]],
                    }
                    pref_df_train = pd.concat(
                        [pref_df_train, pd.DataFrame(dict)], ignore_index=True
                    )
    pref_df_train.to_csv(
        f"{dir.split('/')[-1]}_trajectories_preference_train.csv", index=False
    )

    for folder in issues[-2:]:
        df_view = (
            df_test[df_test["trajectory_id"] == folder.split("/")[-1]]
            .sort_values(by="trajectory_reward", ascending=False)
            .reset_index(drop=True)
        )
        for i, i_row in df_view.iterrows():
            for j, j_row in df_view.iterrows():
                if i < j and i_row["trajectory_reward"] > j_row["trajectory_reward"]:
                    dict = {
                        "chosen": [i_row["trajectory_conversation"]],
                        "rejected": [j_row["trajectory_conversation"]],
                        "score_chosen": [i_row["trajectory_reward"]],
                        "score_rejected": [j_row["trajectory_reward"]],
                    }
                    pref_df_test = pd.concat(
                        [pref_df_test, pd.DataFrame(dict)], ignore_index=True
                    )
    pref_df_test.to_csv(
        f"{dir.split('/')[-1]}_trajectories_preference_test.csv", index=False
    )

    print(
        f"Length of df, df_train, df_test: {len(df)}, {len(df_train)}, {len(df_test)}"
    )
    print(
        f"Length of pref_df_train, pref_df_test: {len(pref_df_train)}, {len(pref_df_test)}"
    )
