"""GUI-independent asynchronous subprocess execution."""

from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import uuid
import codecs
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..core.jobs import (
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


class RunnerBusyError(RuntimeError):
    pass


class SubprocessRunner:
    """Run one process at a time and publish neutral events."""

    def __init__(self, cancel_grace_seconds: float = 3.0) -> None:
        self.events: queue.Queue[ProcessEvent] = queue.Queue()
        self._cancel_grace_seconds = cancel_grace_seconds
        self._lock = threading.RLock()
        self._process: subprocess.Popen[bytes] | None = None
        self._worker: threading.Thread | None = None
        self._state_machine = JobStateMachine()
        self._job_id: str | None = None
        self._cancel_requested = threading.Event()

    @property
    def current_job_id(self) -> str | None:
        with self._lock:
            return self._job_id

    @property
    def state(self) -> JobState:
        with self._lock:
            return self._state_machine.state

    @property
    def is_active(self) -> bool:
        return self.state in {JobState.STARTING, JobState.RUNNING, JobState.CANCELLING}

    def start(
        self,
        command: Sequence[str],
        *,
        working_directory: Path | None = None,
        environment_overrides: Mapping[str, str] | None = None,
    ) -> str:
        if not command:
            raise ValueError("Command cannot be empty.")

        with self._lock:
            if self.is_active:
                raise RunnerBusyError("A process is already active.")

            job_id = uuid.uuid4().hex
            self._job_id = job_id
            self._process = None
            self._cancel_requested.clear()
            self._state_machine = JobStateMachine()
            self._transition(job_id, JobState.STARTING)

            self._worker = threading.Thread(
                target=self._run,
                args=(job_id, tuple(command), working_directory, environment_overrides),
                name=f"process-runner-{job_id[:8]}",
                daemon=True,
            )
            self._worker.start()

        return job_id

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            if job_id != self._job_id or not self.is_active:
                return False
            if self._state_machine.state is JobState.CANCELLING:
                return True

            self._cancel_requested.set()
            self._transition(job_id, JobState.CANCELLING)
            process = self._process

        if process is not None:
            threading.Thread(
                target=self._stop_process,
                args=(process,),
                name=f"process-cancel-{job_id[:8]}",
                daemon=True,
            ).start()
        return True

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the runner thread; GUI code should consume events instead."""

        with self._lock:
            worker = self._worker
        if worker is None:
            return True
        worker.join(timeout)
        return not worker.is_alive()

    def drain_events(self, maximum: int | None = None) -> tuple[ProcessEvent, ...]:
        drained: list[ProcessEvent] = []
        while maximum is None or len(drained) < maximum:
            try:
                drained.append(self.events.get_nowait())
            except queue.Empty:
                break
        return tuple(drained)

    def _run(
        self,
        job_id: str,
        command: tuple[str, ...],
        working_directory: Path | None,
        environment_overrides: Mapping[str, str] | None,
    ) -> None:
        environment = os.environ.copy()
        if environment_overrides:
            environment.update(environment_overrides)

        popen_options: dict[str, object] = {}
        if os.name == "nt":
            popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_options["start_new_session"] = True

        try:
            process = subprocess.Popen(
                command,
                cwd=str(working_directory) if working_directory else None,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                shell=False,
                **popen_options,
            )
        except Exception as exc:
            self._finish_start_failure(job_id, exc)
            return

        with self._lock:
            self._process = process
            cancelling = self._cancel_requested.is_set()
            if not cancelling:
                self._transition(job_id, JobState.RUNNING)

        readers = (
            threading.Thread(
                target=self._read_stream,
                args=(job_id, process.stdout, OutputChannel.STDOUT),
                name=f"stdout-{job_id[:8]}",
                daemon=True,
            ),
            threading.Thread(
                target=self._read_stream,
                args=(job_id, process.stderr, OutputChannel.STDERR),
                name=f"stderr-{job_id[:8]}",
                daemon=True,
            ),
        )
        for reader in readers:
            reader.start()

        if cancelling:
            self._stop_process(process)

        exit_code = process.wait()
        for reader in readers:
            reader.join()

        with self._lock:
            cancelled = self._cancel_requested.is_set()
            terminal_state = JobState.CANCELLED if cancelled else (
                JobState.SUCCEEDED if exit_code == 0 else JobState.FAILED
            )
            self._transition(job_id, terminal_state)
            self._process = None

        result = JobResult(terminal_state, exit_code)
        self.events.put(JobFinishedEvent(job_id, result))

    def _read_stream(self, job_id, stream, channel: OutputChannel) -> None:
        if stream is None:
            return
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        try:
            while chunk := stream.read(4096):
                pending += decoder.decode(chunk)
                pending = self._emit_output_records(job_id, channel, pending, final=False)

            pending += decoder.decode(b"", final=True)
            pending = self._emit_output_records(job_id, channel, pending, final=True)
            if pending:
                self.events.put(ProcessOutputEvent(job_id, channel, pending))
        except Exception as exc:
            self.events.put(ProcessErrorEvent(job_id, f"Failed to read {channel.value}: {exc}"))
        finally:
            stream.close()

    def _emit_output_records(
        self,
        job_id: str,
        channel: OutputChannel,
        text: str,
        *,
        final: bool,
    ) -> str:
        while True:
            carriage = text.find("\r")
            newline = text.find("\n")
            positions = tuple(position for position in (carriage, newline) if position >= 0)
            if not positions:
                return text

            position = min(positions)
            separator = text[position]
            if separator == "\r" and position + 1 == len(text) and not final:
                return text

            replace_line = separator == "\r"
            consumed = 1
            if separator == "\r" and position + 1 < len(text) and text[position + 1] == "\n":
                replace_line = False
                consumed = 2

            self.events.put(ProcessOutputEvent(job_id, channel, text[:position], replace_line))
            text = text[position + consumed :]

    def _finish_start_failure(self, job_id: str, exc: Exception) -> None:
        self.events.put(ProcessErrorEvent(job_id, f"Failed to start process: {exc}"))
        with self._lock:
            terminal_state = JobState.CANCELLED if self._cancel_requested.is_set() else JobState.FAILED
            self._transition(job_id, terminal_state)
        self.events.put(JobFinishedEvent(job_id, JobResult(terminal_state, None, str(exc))))

    def _transition(self, job_id: str, state: JobState) -> None:
        self._state_machine.transition_to(state)
        self.events.put(JobStateEvent(job_id, state))

    def _stop_process(self, process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return

        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_C_EVENT)
            else:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=self._cancel_grace_seconds)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass

        try:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            if process.poll() is None:
                process.kill()
