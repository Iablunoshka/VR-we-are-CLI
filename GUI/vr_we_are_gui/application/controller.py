"""Coordinates the core contract and a replaceable process runner."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Protocol

from ..core import (
    CliTarget,
    ConversionConfig,
    ProcessEvent,
    ValidationResult,
    build_cli_command,
    validate_config,
)


class ProcessRunnerPort(Protocol):
    @property
    def is_active(self) -> bool: ...

    def start(self, command, *, working_directory=None, environment_overrides=None) -> str: ...

    def cancel(self, job_id: str) -> bool: ...

    def drain_events(self, maximum: int | None = None) -> tuple[ProcessEvent, ...]: ...


@dataclass(frozen=True, slots=True)
class PreparationResult:
    validation: ValidationResult
    command: tuple[str, ...] | None

    @property
    def is_ready(self) -> bool:
        return self.validation.is_valid and self.command is not None


@dataclass(frozen=True, slots=True)
class StartResult:
    preparation: PreparationResult
    job_id: str | None

    @property
    def started(self) -> bool:
        return self.job_id is not None


class ConversionController:
    """UI-independent conversion use cases."""

    def __init__(self, target: CliTarget, runner: ProcessRunnerPort) -> None:
        self.target = target
        self.runner = runner

    @property
    def is_active(self) -> bool:
        return self.runner.is_active

    def prepare(self, config: ConversionConfig) -> PreparationResult:
        validation = validate_config(config)
        command = build_cli_command(config, self.target) if validation.is_valid else None
        return PreparationResult(validation, command)

    def start(self, config: ConversionConfig) -> StartResult:
        preparation = self.prepare(config)
        if not preparation.is_ready:
            return StartResult(preparation, None)

        job_id = self.runner.start(
            preparation.command,
            working_directory=self.target.main_script.parent,
            environment_overrides={
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
            },
        )
        return StartResult(preparation, job_id)

    def cancel(self, job_id: str) -> bool:
        return self.runner.cancel(job_id)

    def drain_events(self, maximum: int = 250) -> tuple[ProcessEvent, ...]:
        return self.runner.drain_events(maximum)


def format_command(command: tuple[str, ...]) -> str:
    """Create a display-only command preview."""

    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
