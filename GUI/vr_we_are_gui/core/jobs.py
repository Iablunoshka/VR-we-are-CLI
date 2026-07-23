"""Framework-neutral conversion job state and process events."""

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


class JobState(str, Enum):
    READY = "ready"
    STARTING = "starting"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OutputChannel(str, Enum):
    STDOUT = "stdout"
    STDERR = "stderr"


_ALLOWED_TRANSITIONS = {
    JobState.READY: {JobState.STARTING},
    JobState.STARTING: {JobState.RUNNING, JobState.CANCELLING, JobState.FAILED, JobState.CANCELLED},
    JobState.RUNNING: {JobState.CANCELLING, JobState.SUCCEEDED, JobState.FAILED},
    JobState.CANCELLING: {JobState.CANCELLED, JobState.FAILED},
    JobState.SUCCEEDED: set(),
    JobState.FAILED: set(),
    JobState.CANCELLED: set(),
}


class InvalidJobTransition(ValueError):
    pass


@dataclass(slots=True)
class JobStateMachine:
    state: JobState = JobState.READY

    def transition_to(self, new_state: JobState) -> None:
        if new_state not in _ALLOWED_TRANSITIONS[self.state]:
            raise InvalidJobTransition(f"Cannot transition from {self.state.value} to {new_state.value}.")
        self.state = new_state


@dataclass(frozen=True, slots=True)
class JobResult:
    state: JobState
    exit_code: int | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        terminal_states = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}
        if self.state not in terminal_states:
            raise ValueError("A job result must use a terminal state.")


@dataclass(frozen=True, slots=True)
class JobStateEvent:
    job_id: str
    state: JobState


@dataclass(frozen=True, slots=True)
class ProcessOutputEvent:
    job_id: str
    channel: OutputChannel
    text: str
    replace_line: bool = False


@dataclass(frozen=True, slots=True)
class ProcessErrorEvent:
    job_id: str
    message: str


@dataclass(frozen=True, slots=True)
class JobFinishedEvent:
    job_id: str
    result: JobResult


ProcessEvent: TypeAlias = JobStateEvent | ProcessOutputEvent | ProcessErrorEvent | JobFinishedEvent
