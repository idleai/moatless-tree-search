import json
import logging
from typing import List

import tenacity
from litellm.exceptions import (
    BadRequestError,
    NotFoundError,
    AuthenticationError,
    APIError,
)
from pydantic import BaseModel, ValidationError

from moatless.completion.completion import CompletionModel, CompletionResponse
from moatless.completion.model import Completion, StructuredOutput, Usage
from moatless.exceptions import CompletionRejectError, CompletionRuntimeError

logger = logging.getLogger(__name__)


def _is_truncated_json_error(error: Exception) -> bool:
    """Check if the error indicates a truncated JSON response."""
    error_str = str(error).lower()
    truncation_indicators = [
        "eof while parsing",
        "unterminated string",
        "unexpected end of json",
        "expecting value",
        "expecting property name",
        "expecting ':'",
        "expecting ','",
    ]
    return any(indicator in error_str for indicator in truncation_indicators)


class ToolCallCompletionModel(CompletionModel):
    def create_completion(
        self,
        messages: List[dict],
        system_prompt: str,
        response_model: List[type[StructuredOutput]] | type[StructuredOutput],
    ) -> CompletionResponse:
        tools = []

        if isinstance(response_model, list):
            tools.extend(
                [
                    r.openai_schema(thoughts_in_action=self.thoughts_in_action)
                    for r in response_model
                ]
            )
        elif response_model:
            tools.append(response_model.openai_schema())
        else:
            tools = None

        total_usage = Usage()
        retry_count = 0

        messages.insert(0, {"role": "system", "content": system_prompt})

        retries = tenacity.Retrying(
            retry=tenacity.retry_if_not_exception_type(
                (APIError, BadRequestError, NotFoundError, AuthenticationError)
            ),
            stop=tenacity.stop_after_attempt(3),
        )

        def _do_completion():
            nonlocal total_usage, retry_count
            llm_completion_response = None
            try:
                if self.thoughts_in_action:
                    tool_choice = "required"
                else:
                    tool_choice = "auto"

                llm_completion_response = self._litellm_base_completion(
                    messages=messages, tools=tools, tool_choice=tool_choice
                )

                if not llm_completion_response or not llm_completion_response.choices:
                    raise CompletionRuntimeError(
                        "No completion response or choices returned"
                    )

                total_usage += Usage.from_completion_response(
                    llm_completion_response, self.model
                )

                content = llm_completion_response.choices[0].message.content

                def get_response_model(tool_name: str):
                    if isinstance(response_model, list):
                        for r in response_model:
                            if getattr(r, "name", None) == tool_name:
                                return r
                    else:
                        return response_model

                response_objects = []
                invalid_function_names = []
                seen_arguments = set()
                flags = set()

                if llm_completion_response.choices[0].message.tool_calls:
                    for tool_call in llm_completion_response.choices[
                        0
                    ].message.tool_calls:
                        action = get_response_model(tool_call.function.name)

                        if not action:
                            logger.warning(
                                f"Invalid action name: {tool_call.function.name}"
                            )
                            invalid_function_names.append(tool_call.function.name)
                            continue

                        # Check for duplicate arguments
                        if tool_call.function.arguments in seen_arguments:
                            logger.warning(
                                f"Duplicate tool call arguments found for {tool_call.function.name}"
                            )
                            flags.add("duplicate_tool_call")
                            continue

                        seen_arguments.add(tool_call.function.arguments)

                        # Check if the arguments contain an error response from the LLM provider
                        try:
                            parsed_args = json.loads(tool_call.function.arguments)
                            if isinstance(parsed_args, dict) and "error" in parsed_args:
                                error_info = parsed_args["error"]
                                error_type = error_info.get("type", "unknown") if isinstance(error_info, dict) else str(error_info)
                                error_msg = error_info.get("message", str(error_info)) if isinstance(error_info, dict) else str(error_info)
                                raise CompletionRuntimeError(
                                    f"LLM returned an error response in tool call: type={error_type}, message={error_msg}"
                                )
                        except json.JSONDecodeError:
                            pass  # Will be handled by model_validate_json below

                        response_object = action.model_validate_json(
                            tool_call.function.arguments
                        )
                        response_objects.append(response_object)

                    if invalid_function_names:
                        available_actions = [getattr(r, "name", None) for r in response_model if getattr(r, "name", None) is not None]
                        raise ValueError(
                            f"Unknown functions {invalid_function_names}. Available functions: {available_actions}"
                        )

                if not content and not response_objects:
                    raise ValueError("No tool call or content in message.")

                completion = Completion.from_llm_completion(
                    input_messages=messages,
                    completion_response=llm_completion_response,
                    model=self.model,
                    retries=retry_count,
                    usage=total_usage,
                    flags=list(flags),
                )

                return CompletionResponse.create(
                    text=content, output=response_objects, completion=completion
                )

            except (ValidationError, ValueError, json.JSONDecodeError) as e:
                retry_count += 1
                error_str = str(e)

                # Detect truncated JSON responses
                if _is_truncated_json_error(e):
                    logger.warning(
                        f"Completion attempt {retry_count} failed due to truncated JSON response: {e}. "
                        f"This typically means the response exceeded max_tokens. Will retry."
                    )
                    error_message = (
                        "Your previous response was truncated and resulted in invalid JSON. "
                        "The response was cut off before completion, likely due to length limits. "
                        "Please provide a shorter, more concise response that fits within the token limit. "
                        "Focus on the essential information only."
                    )
                elif "Field required" in error_str:
                    # Missing required fields - provide more helpful guidance
                    logger.warning(
                        f"Completion attempt {retry_count} failed due to missing required fields: {e}. Will retry."
                    )
                    error_message = (
                        f"Your response is missing required fields. "
                        f"Make sure your tool call includes ALL required parameters as separate fields in the JSON. "
                        f"Error details: {e}"
                    )
                else:
                    logger.warning(
                        f"Completion attempt {retry_count} failed with error: {e}. Will retry."
                    )
                    error_message = f"The response was invalid. Fix the errors, exceptions found\n{e}"

                messages.append(
                    {
                        "role": "user",
                        "content": error_message,
                    }
                )
                raise CompletionRejectError(
                    message=str(e),
                    last_completion=llm_completion_response,
                    messages=messages,
                ) from e

        try:
            return retries(_do_completion)
        except tenacity.RetryError as e:
            raise e.reraise()
