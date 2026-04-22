"""Kwargs for ``with_structured_output`` when the active model has strict schema rules."""

from __future__ import annotations

from pretend_compiler.state import PretendState


def hosted_structured_output_kwargs(state: PretendState) -> dict:
    """
    OpenAI (``--hosted``) defaults to JSON-schema structured outputs, which reject
    our Pydantic models (e.g. ``list[dict]`` for ``ir_ops``). Use tool/function
    calling instead to avoid UserWarning and match OpenAI's supported paths.
    """
    if state.get("hosted"):
        return {"method": "function_calling"}
    return {}
