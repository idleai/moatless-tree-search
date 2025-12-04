import json
import logging
import os
import time
from enum import Enum
from textwrap import dedent
from typing import Optional, Union, List, Any

import litellm
import tenacity
from litellm.exceptions import (
    BadRequestError,
    NotFoundError,
    AuthenticationError,
    APIError,
    RateLimitError,
)
from pydantic import BaseModel, Field, model_validator, ValidationError

from moatless.completion.model import Completion, StructuredOutput
from moatless.exceptions import CompletionRejectError, CompletionRuntimeError

logger = logging.getLogger(__name__)

# Rate limit backoff duration in seconds (10 minutes default, can be overridden)
RATE_LIMIT_BACKOFF_SECONDS = 600


def set_rate_limit_backoff(seconds: int) -> None:
    """Set the rate limit backoff duration in seconds."""
    global RATE_LIMIT_BACKOFF_SECONDS
    RATE_LIMIT_BACKOFF_SECONDS = seconds
    logger.info(f"Rate limit backoff set to {seconds} seconds ({seconds / 60:.1f} minutes)")

class LLMResponseFormat(str, Enum):
    TOOLS = "tool_call"
    JSON = "json"
    ANTHROPIC_TOOLS = "anthropic_tools"
    REACT = "react"


class CompletionResponse(BaseModel):
    """Container for completion responses that can include multiple structured outputs and text"""

    structured_outputs: List[StructuredOutput] = Field(default_factory=list)
    text_response: Optional[str] = Field(default=None)
    completion: Optional[Completion] = Field(default=None)

    @classmethod
    def create(
        cls,
        text: str | None = None,
        output: List[StructuredOutput] | StructuredOutput | None = None,
        completion: Completion | None = None,
    ) -> "CompletionResponse":
        if isinstance(output, StructuredOutput):
            outputs = [output]
        elif isinstance(output, list):
            outputs = output
        else:
            outputs = None

        return cls(
            text_response=text, structured_outputs=outputs, completion=completion
        )

    @property
    def structured_output(self) -> Optional[StructuredOutput]:
        """Get the first structured output"""
        if len(self.structured_outputs) > 1:
            ignored_outputs = [
                output.__class__.__name__ for output in self.structured_outputs[1:]
            ]
            logger.warning(
                f"Multiple structured outputs found in completion response, returning {self.structured_outputs[0].__class__.__name__} and ignoring: {ignored_outputs}"
            )
        return self.structured_outputs[0] if self.structured_outputs else None


class CompletionModel(BaseModel):
    model: str = Field(..., description="The model to use for completion")
    temperature: Optional[float] = Field(None, description="The temperature to use for completion")
    max_tokens: Optional[int] = Field(
        None, description="The maximum number of tokens to generate"
    )
    timeout: float = Field(
        120.0, description="The timeout in seconds for completion requests"
    )
    model_base_url: Optional[str] = Field(
        default=None,
        description="The base URL for the LiteLLM proxy API. Defaults to DEFAULT_LITELLM_BASE_URL if not set.",
    )
    model_api_key: Optional[str] = Field(
        default=None, description="The API key for the model", exclude=True
    )
    response_format: Optional[LLMResponseFormat] = Field(
        None, description="The response format expected from the LLM"
    )
    stop_words: Optional[list[str]] = Field(
        default=None, description="The stop words to use for completion"
    )
    metadata: Optional[dict] = Field(
        default=None, description="Additional metadata for the completion model"
    )
    thoughts_in_action: bool = Field(
        default=False,
        description="Whether to include thoughts in the action or in the message",
    )

    @property
    def api_base(self) -> str:
        """Get the API base URL, defaulting to the LiteLLM proxy endpoint."""
        return self.model_base_url or os.getenv("LITELLM_BASE_URL")

    @property
    def api_key(self) -> str:
        """Get the API key for authentication."""
        return self.model_api_key or os.getenv("LITELLM_API_KEY", "noop")

    def clone(self, **kwargs) -> "CompletionModel":
        """Create a copy of the completion model with optional parameter overrides.

        Args:
            **kwargs: Parameters to override in the cloned model

        Returns:
            A new CompletionModel instance with the specified overrides
        """
        model_data = self.model_dump()
        model_data.update(kwargs)
        return CompletionModel.model_validate(model_data)

    def create_completion(
        self,
        messages: List[dict],
        system_prompt: str,
        response_model: List[type[StructuredOutput]] | type[StructuredOutput],
    ) -> CompletionResponse:
        if not response_model:
            raise CompletionRuntimeError(f"Response model is required for completion")

        if isinstance(response_model, list) and len(response_model) > 1:
            avalabile_actions = [
                action for action in response_model
                if hasattr(action, "name") and getattr(action, "name", None) is not None
            ]
            if not avalabile_actions:
                raise CompletionRuntimeError(f"No actions found in {response_model}")

            class TakeAction(StructuredOutput):
                action: Union[tuple(response_model)] = Field(...)
                action_type: str = Field(
                    ..., description="The type of action being taken"
                )

                @model_validator(mode="before")
                def validate_action(cls, data: dict) -> dict:
                    if not isinstance(data, dict):
                        raise ValidationError("Expected dictionary input")

                    action_type = data.get("action_type")
                    if not action_type:
                        return data

                    # Find the correct action class based on action_type
                    action_class = next(
                        (
                            action
                            for action in avalabile_actions
                            if getattr(action, "name", None) == action_type
                        ),
                        None,
                    )
                    if not action_class:
                        action_names = [
                            getattr(action, "name", None)
                            for action in avalabile_actions
                            if getattr(action, "name", None) is not None
                        ]
                        raise ValidationError(
                            f"Unknown action type: {action_type}. Available actions: {', '.join(action_names)}"
                        )

                    # Validate the action data using the specific action class
                    action_data = data.get("action")
                    if not action_data:
                        raise ValidationError("Action data is required")

                    data["action"] = action_class.model_validate(action_data)
                    return data

            response_model = TakeAction

        system_prompt += dedent(f"""\n# Response format
        You must respond with only a JSON object that match the following json_schema:\n

        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

        Make sure to return an instance of the JSON, not the schema itself.""")

        messages.insert(0, {"role": "system", "content": system_prompt})

        retries = tenacity.Retrying(
            retry=tenacity.retry_if_not_exception_type(
                (APIError, BadRequestError, NotFoundError, AuthenticationError)
            ),
            stop=tenacity.stop_after_attempt(3),
        )

        def _do_completion():
            completion_response = None

            try:
                completion_response = self._litellm_base_completion(
                    messages=messages, response_format={"type": "json_object"}
                )

                if not completion_response or not completion_response.choices:
                    raise CompletionRuntimeError(
                        "No completion response or choices returned"
                    )

                if isinstance(
                    completion_response.choices[0].message.content, BaseModel
                ):
                    assistant_message = completion_response.choices[
                        0
                    ].message.content.model_dump()
                else:
                    assistant_message = completion_response.choices[0].message.content

                if not assistant_message:
                    raise CompletionRuntimeError("Empty response from model")

                # Check if the response is an error structure from the LLM provider
                try:
                    parsed = json.loads(assistant_message) if isinstance(assistant_message, str) else assistant_message
                    if isinstance(parsed, dict) and "error" in parsed:
                        error_info = parsed["error"]
                        error_type = error_info.get("type", "unknown") if isinstance(error_info, dict) else str(error_info)
                        error_msg = error_info.get("message", str(error_info)) if isinstance(error_info, dict) else str(error_info)
                        raise CompletionRuntimeError(
                            f"LLM returned an error response: type={error_type}, message={error_msg}"
                        )
                except json.JSONDecodeError:
                    pass  # Not valid JSON, will be handled by validation below

                messages.append({"role": "assistant", "content": assistant_message})

                response = response_model.model_validate_json(assistant_message)

                completion = Completion.from_llm_completion(
                    input_messages=messages,
                    completion_response=completion_response,
                    model=self.model,
                )
                if hasattr(response, "action"):
                    return CompletionResponse.create(
                        output=response.action, completion=completion
                    )

                return CompletionResponse.create(output=response, completion=completion)

            except (ValidationError, json.JSONDecodeError) as e:
                # Build a more helpful error message for missing fields
                error_str = str(e)
                if "Field required" in error_str:
                    # Extract the schema fields if possible
                    try:
                        schema = response_model.model_json_schema()
                        required_fields = schema.get("required", [])
                        properties = schema.get("properties", {})
                        field_descriptions = []
                        for field in required_fields:
                            field_info = properties.get(field, {})
                            field_type = field_info.get("type", "unknown")
                            field_descriptions.append(f"  - {field}: {field_type}")
                        fields_info = "\n".join(field_descriptions)
                        error_message = (
                            f"Your response is missing required fields. The JSON must have ALL of these fields:\n"
                            f"{fields_info}\n\n"
                            f"Your response was missing: {error_str}\n\n"
                            f"Please provide a valid JSON with ALL required fields as SEPARATE keys."
                        )
                    except Exception:
                        error_message = f"The response was invalid. Fix the errors, exceptions found\n{e}"
                else:
                    error_message = f"The response was invalid. Fix the errors, exceptions found\n{e}"

                logger.warning(
                    f"Completion attempt failed with error: {e}. Will retry."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": error_message,
                    }
                )
                raise CompletionRejectError(
                    message=str(e),
                    last_completion=completion_response,
                    messages=messages,
                ) from e
            except Exception as e:
                logger.exception(
                    f"Completion attempt failed with error: {e}. Will retry."
                )
                raise CompletionRuntimeError(
                    f"Failed to get completion response: {e}",
                )

        try:
            return retries(_do_completion)
        except tenacity.RetryError as e:
            raise e.reraise()

    def _litellm_base_completion(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        response_format: dict | None = None,
    ) -> Any:
        """Base method for making litellm completion calls with common parameters.

        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions for function calling
            tool_choice: Optional tool choice configuration
            response_format: Optional response format configuration

        Returns:
            The completion response from litellm
        """
        litellm.drop_params = True

        def _should_retry_rate_limit(exception: BaseException) -> bool:
            """Check if we should retry on rate limit with long backoff."""
            if isinstance(exception, RateLimitError):
                return True
            # Also check for 429 status code in APIError
            if isinstance(exception, APIError) and hasattr(exception, 'status_code'):
                return exception.status_code == 429
            return False

        def _handle_rate_limit_wait(retry_state: tenacity.RetryCallState) -> None:
            """Log and wait for rate limit backoff."""
            exception = retry_state.outcome.exception()
            logger.warning(
                f"Rate limit (429) encountered. Waiting {RATE_LIMIT_BACKOFF_SECONDS} seconds (10 minutes) before retry. "
                f"Error: {exception}"
            )
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS)

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(5),  # More attempts for rate limits
            wait=tenacity.wait_exponential(multiplier=3, max=60),
            retry=tenacity.retry_if_exception_type(Exception),
            reraise=True,
            before_sleep=lambda retry_state: (
                _handle_rate_limit_wait(retry_state)
                if _should_retry_rate_limit(retry_state.outcome.exception())
                else logger.warning(
                    f"Retrying litellm completion after error: {retry_state.outcome.exception()}"
                )
            ),
        )
        def _do_completion():
            # When using a LiteLLM proxy, prefix with openai/ so litellm routes to the proxy
            # The proxy then uses the model name to route to the correct backend
            model_name = self.model
            if "/" not in model_name:
                model_name = f"openai/{model_name}"

            return litellm.completion(
                model=model_name,
                api_base=self.api_base,
                api_key=self.api_key,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=messages,
                metadata=self.metadata or {},
                timeout=self.timeout,
                stop=self.stop_words,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
            )

        try:
            return _do_completion()
        except tenacity.RetryError as e:
            last_exception = e.last_attempt.exception()
            if isinstance(last_exception, litellm.APIError):
                logger.error(
                    "LiteLLM API Error: %s\nProvider: %s\nModel: %s\nStatus: %d\nDebug Info: %s\nRetries: %d/%d",
                    last_exception.message,
                    last_exception.llm_provider,
                    last_exception.model,
                    last_exception.status_code,
                    last_exception.litellm_debug_info,
                    last_exception.num_retries or 0,
                    last_exception.max_retries or 0,
                )
            else:
                logger.warning(
                    "LiteLLM completion failed after retries with error: %s",
                    str(last_exception),
                    exc_info=last_exception,
                )
            raise last_exception

    def model_dump(self, **kwargs):
        dump = super().model_dump(**kwargs)
        if "model_api_key" in dump:
            dump["model_api_key"] = None
        if "response_format" in dump:
            dump["response_format"] = dump["response_format"].value
        return dump

    @classmethod
    def model_validate(cls, obj):
        if isinstance(obj, dict) and "response_format" in obj:
            if "claude-3-5" in obj["model"]:
                from moatless.completion.anthropic import AnthtropicCompletionModel

                return AnthtropicCompletionModel(**obj)

            response_format = LLMResponseFormat(obj["response_format"])
            obj["response_format"] = response_format

            if response_format == LLMResponseFormat.TOOLS:
                from moatless.completion.tool_call import ToolCallCompletionModel

                return ToolCallCompletionModel(**obj)
            elif response_format == LLMResponseFormat.REACT:
                from moatless.completion.react import ReActCompletionModel

                return ReActCompletionModel(**obj)

        return cls(**obj)

    @model_validator(mode="after")
    def set_api_key(self) -> "CompletionModel":
        """
        Update the model with the API key from env vars if not already set.
        """
        if not self.model_api_key:
            self.model_api_key = os.getenv("LITELLM_API_KEY") or os.getenv("CUSTOM_LLM_API_KEY")

        return self
