"""Public core API."""

from .command import CliTarget, build_cli_command
from .config import (
    AutocastMode,
    Codec,
    ConversionConfig,
    DepthModel,
    HdrEncoder,
    InputType,
    Preset,
    VideoQuality,
)
from .jobs import (
    JobFinishedEvent,
    JobResult,
    JobState,
    JobStateEvent,
    JobStateMachine,
    OutputChannel,
    ProcessErrorEvent,
    ProcessEvent,
    ProcessOutputEvent,
)
from .validation import IssueLevel, ValidationIssue, ValidationResult, validate_config

__all__ = [
    "AutocastMode",
    "CliTarget",
    "Codec",
    "ConversionConfig",
    "DepthModel",
    "HdrEncoder",
    "InputType",
    "IssueLevel",
    "JobResult",
    "JobState",
    "JobStateEvent",
    "JobStateMachine",
    "JobFinishedEvent",
    "OutputChannel",
    "Preset",
    "ProcessErrorEvent",
    "ProcessEvent",
    "ProcessOutputEvent",
    "ValidationIssue",
    "ValidationResult",
    "VideoQuality",
    "build_cli_command",
    "validate_config",
]
