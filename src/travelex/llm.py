"""The only module that knows about chat models."""
import os
from typing import TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from travelex.retry import call_with_retry

FAST_ROLES = {"food", "qa", "verify"}
# Provider packages whose with_structured_output(method="json_schema") we trust.
JSON_SCHEMA_PACKAGES = ("langchain_openai",)
MAX_ERROR_CHARS = 2000

T = TypeVar("T", bound=BaseModel)


class LLMConfigError(RuntimeError):
    pass


class StructuredOutputError(RuntimeError):
    pass


def model_name_for(role: str) -> str:
    names = [f"LLM_MODEL_{role.upper()}"]
    if role in FAST_ROLES:
        names.append("LLM_MODEL_FAST")
    names.append("LLM_MODEL")
    for name in names:
        value = os.getenv(name, "").strip()
        if not value:
            continue
        if "<" in value or ":" not in value:
            raise LLMConfigError(
                f'{name}="{value}" is not a model. Use "<provider>:<model>", e.g. "openai:gpt-4.1-mini".'
            )
        return value
    raise LLMConfigError("LLM_MODEL is not set. Copy .env.example to .env and fill it in.")


def get_llm(role: str, **kwargs) -> BaseChatModel:
    model = model_name_for(role)
    provider = model.split(":", 1)[0]
    try:
        return init_chat_model(model, **kwargs)
    except ImportError as e:
        package = "langchain-" + provider.replace("_", "-")
        raise LLMConfigError(f"Provider '{provider}' needs: pip install {package}") from e


def structured(llm: BaseChatModel, schema: type[T], messages: list[AnyMessage], max_repairs: int = 2) -> T:
    """Return a validated `schema` instance, re-prompting with the error up to `max_repairs` times."""
    use_json_schema = type(llm).__module__.startswith(JSON_SCHEMA_PACKAGES)
    convo = list(messages)
    for attempt in range(max_repairs + 1):
        parsed, error, raw = call_with_retry(lambda: _call_once(llm, schema, convo, use_json_schema))
        if error is None:
            return parsed
        if attempt < max_repairs:
            convo.append(HumanMessage(
                f"Your previous answer was invalid.\nAnswer: {raw[:MAX_ERROR_CHARS]}\n"
                f"Error: {error[:MAX_ERROR_CHARS]}\nReturn a corrected {schema.__name__}."
            ))
    raise StructuredOutputError(f"{schema.__name__} still invalid after {max_repairs} repairs: {error}")


def _call_once(llm, schema, messages, use_json_schema) -> tuple[BaseModel | None, str | None, str]:
    if use_json_schema:
        result = llm.with_structured_output(schema, method="json_schema", include_raw=True).invoke(messages)
        raw = str(result["raw"].content)
        if result["parsing_error"] is not None:
            return None, str(result["parsing_error"]), raw
        if result["parsed"] is None:
            return None, "The answer was empty or refused.", raw
        return result["parsed"], None, raw

    response = llm.bind_tools([schema], tool_choice="auto").invoke(messages)
    if not response.tool_calls:
        return None, f"You did not call the {schema.__name__} tool. Call it exactly once.", str(response.content)
    args = response.tool_calls[0]["args"]
    try:
        return schema.model_validate(args), None, str(args)
    except ValidationError as e:
        return None, str(e), str(args)