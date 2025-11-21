"""Evaluation configuration settings"""

from datetime import datetime

# Default evaluation settings
DEFAULT_CONFIG = {
    # Model settings
    "api_key": None,
    "base_url": None,
    # Dataset settings
    "split": "lite_and_verified_solvable",
    "instance_ids": None,
    # Tree search settings
    "max_iterations": 20,
    "max_expansions": 1,
    "max_cost": 1.0,
    # Runner settings
    "num_workers": 10,
    # Evaluation settings
    "evaluation_name": None,
    "rerun_errors": False,
}

# Configuration for deepseek-chat with tool_call format
DEEPSEEK_TOOL_CALL_CONFIG = {
    **DEFAULT_CONFIG,
    "model": "deepseek/deepseek-chat",
    "response_format": "tool_call",
    "message_history": "messages",
    "thoughts_in_action": False,
}

# Configuration for deepseek-chat with tool_call format
DEEPSEEK_TOOL_CALL_SUMMARY_CONFIG = {
    **DEFAULT_CONFIG,
    "model": "deepseek/deepseek-chat",
    "response_format": "tool_call",
    "message_history": "summary",
    "thoughts_in_action": False,
}

# Configuration for deepseek-chat with react format
DEEPSEEK_REACT_CONFIG = {
    **DEFAULT_CONFIG,
    "model": "deepseek/deepseek-chat",
    "response_format": "react",
    "message_history": "react",
}

# Configuration for GPT-4o-mini with tool_call format
GPT4O_MINI_CONFIG = {
    **DEFAULT_CONFIG,
    "model": "azure/gpt-4o-mini",
    "response_format": "tool_call",
    "message_history": "messages",
    "thoughts_in_action": True,
}

# Configuration for GPT-4o with tool_call format
GPT4O_CONFIG = {
    **DEFAULT_CONFIG,
    "model": "azure/gpt-4o",
    "response_format": "tool_call",
    "message_history": "messages",
    "thoughts_in_action": True,
}

# Configuration for GPT-4o with tool_call format
CLAUDE_35_SONNET_CONFIG = {
    **DEFAULT_CONFIG,
    "model": "claude-3-5-sonnet-20241022",
    "response_format": "tool_call",
    "message_history": "messages",
    "thoughts_in_action": False,
    "split": "lite_and_verified_solvable",
}

QWEN3_30B_CONFIG = {
    **DEFAULT_CONFIG,
    "api_key": "noop",
    "model": "qwen3-coder-30b-a3b-instruct",
    "response_format": "tool_call",
    "message_history": "messages",
    "thoughts_in_action": False,
    "split": "easy",
    "max_iterations": 50,
    "timeout": 300000.0
}

QWEN3_480B_CONFIG = {
    **DEFAULT_CONFIG,
    "api_key": "noop",
    "model": "qwen3-coder-480b-a35b-instruct",
    "response_format": "tool_call",
    "message_history": "messages",
    "thoughts_in_action": False,
    "split": "easy",
}

# full MCTS configuration for Qwen3 30B model
QWEN3_30B_MCTS_CONFIG = {
    **DEFAULT_CONFIG,
    "api_key": "noop",
    "model": "qwen3-coder-30b-a3b-instruct",
    "response_format": "tool_call",
    "message_history": "messages", 
    "thoughts_in_action": False,
    "split": "medium",  # 20 balanced issues for testing
    "temperature": 0.7,
    # full MCTS settings
    "max_expansions": 3,
    "max_iterations": 50,
    "max_cost": 5.0,
    "min_finished_nodes": 2,
    "max_finished_nodes": 3,
    "use_value_function": True,
    "use_discriminator": True,
    "use_feedback": False,
    "timeout": 300000.0,
}

# full MCTS configuration for Qwen3 480B model  
QWEN3_480B_MCTS_CONFIG = {
    **DEFAULT_CONFIG,
    "api_key": "noop", 
    "model": "qwen3-coder-480b-a35b-instruct",
    "response_format": "tool_call",
    "message_history": "messages",
    "thoughts_in_action": False,
    "split": "lite_and_verified",  # more issues for bigger model
    # full MCTS settings
    "temperature": 0.7,
    "timeout": 300.0,
    "max_expansions": 4,
    "max_iterations": 100,
    "max_cost": 10.0,
    "min_finished_nodes": 2,
    "max_finished_nodes": 3,
    "use_value_function": True,
    "use_discriminator": True,
    "use_feedback": False,
}

# keep the original config pointing to 30B as default
QWEN3_CONFIG = QWEN3_30B_CONFIG

# Configuration for single instance runs
def get_single_instance_config(
    instance_id: str, base_config: dict = DEEPSEEK_TOOL_CALL_CONFIG
) -> dict:
    """Create a configuration for running a single instance"""
    return {
        **base_config,
        "instance_ids": [instance_id],
        "num_workers": 1,  # Override to 1 for single instance
        "evaluation_name": f"single_{instance_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }


# Example single instance configurations
DJANGO_17051_DEEPSEEK = get_single_instance_config(
    "django__django-17051", DEEPSEEK_REACT_CONFIG
)
DJANGO_17051_GPT4 = get_single_instance_config(
    "django__django-17051", GPT4O_MINI_CONFIG
)

# Active configuration - change this to switch between configs
ACTIVE_CONFIG = QWEN3_CONFIG  # Change this to run different configurations


def get_config() -> dict:
    """Get the active configuration"""
    return ACTIVE_CONFIG
