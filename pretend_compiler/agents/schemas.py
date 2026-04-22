from __future__ import annotations

from pydantic import BaseModel, Field


class Diagnostic(BaseModel):
    line: int | None = None
    column: int | None = None
    message: str
    severity: str = Field(default="error")


class PrecompilerOut(BaseModel):
    validation_ok: bool
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    plan: str = ""


class CompilerOut(BaseModel):
    compile_ok: bool
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    ir_ops: list[dict] = Field(default_factory=list)
