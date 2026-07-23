import sys
import time
from unittest import TestCase

from vr_we_are_gui.adapters import RunnerBusyError, SubprocessRunner
from vr_we_are_gui.core import (
    JobFinishedEvent,
    JobState,
    JobStateEvent,
    OutputChannel,
    ProcessOutputEvent,
)


class SubprocessRunnerTests(TestCase):
    def test_streams_both_channels_and_reports_success(self):
        runner = SubprocessRunner()
        command = (
            sys.executable,
            "-u",
            "-c",
            "import sys; print('normal'); print('problem', file=sys.stderr)",
        )

        job_id = runner.start(command)
        self.assertTrue(runner.wait(5))
        events = runner.drain_events()

        output = [event for event in events if isinstance(event, ProcessOutputEvent)]
        self.assertIn((OutputChannel.STDOUT, "normal"), {(event.channel, event.text) for event in output})
        self.assertIn((OutputChannel.STDERR, "problem"), {(event.channel, event.text) for event in output})
        finished = [event for event in events if isinstance(event, JobFinishedEvent)]
        self.assertEqual(finished[0].job_id, job_id)
        self.assertEqual(finished[0].result.state, JobState.SUCCEEDED)
        self.assertEqual(finished[0].result.exit_code, 0)

    def test_streams_utf8_special_characters(self):
        runner = SubprocessRunner()
        command = (sys.executable, "-u", "-c", "print('available — using GPU → encoder')")

        runner.start(
            command,
            environment_overrides={"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        self.assertTrue(runner.wait(5))
        lines = [event.text for event in runner.drain_events() if isinstance(event, ProcessOutputEvent)]

        self.assertIn("available — using GPU → encoder", lines)

    def test_preserves_carriage_return_as_line_replacement(self):
        runner = SubprocessRunner()
        code = (
            "import sys,time; "
            "sys.stderr.write('frame=1\\r'); sys.stderr.flush(); time.sleep(0.05); "
            "sys.stderr.write('frame=2\\r'); sys.stderr.flush(); time.sleep(0.05); "
            "sys.stderr.write('frame=3\\n'); sys.stderr.flush()"
        )

        runner.start((sys.executable, "-u", "-c", code))
        self.assertTrue(runner.wait(5))
        output = [
            event
            for event in runner.drain_events()
            if isinstance(event, ProcessOutputEvent) and event.channel is OutputChannel.STDERR
        ]

        self.assertEqual(
            [(event.text, event.replace_line) for event in output],
            [("frame=1", True), ("frame=2", True), ("frame=3", False)],
        )

    def test_reports_failure_to_start(self):
        runner = SubprocessRunner()

        runner.start(("executable-that-does-not-exist-vr-we-are",))
        self.assertTrue(runner.wait(5))
        events = runner.drain_events()

        finished = [event for event in events if isinstance(event, JobFinishedEvent)]
        self.assertEqual(finished[0].result.state, JobState.FAILED)
        self.assertIsNone(finished[0].result.exit_code)

    def test_cancels_running_process(self):
        runner = SubprocessRunner(cancel_grace_seconds=0.5)
        command = (sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(30)")

        job_id = runner.start(command)
        self._wait_for_state(runner, JobState.RUNNING)
        self.assertTrue(runner.cancel(job_id))
        self.assertTrue(runner.wait(5))
        events = runner.drain_events()

        finished = [event for event in events if isinstance(event, JobFinishedEvent)]
        self.assertEqual(finished[0].result.state, JobState.CANCELLED)

    def test_rejects_second_active_process(self):
        runner = SubprocessRunner(cancel_grace_seconds=0.5)
        command = (sys.executable, "-u", "-c", "import time; time.sleep(30)")
        job_id = runner.start(command)

        try:
            self._wait_for_state(runner, JobState.RUNNING)
            with self.assertRaises(RunnerBusyError):
                runner.start(command)
        finally:
            runner.cancel(job_id)
            runner.wait(5)

    def test_applies_environment_overrides(self):
        runner = SubprocessRunner()
        variable = "VR_WE_ARE_RUNNER_TEST"
        command = (sys.executable, "-u", "-c", f"import os; print(os.environ['{variable}'])")

        runner.start(command, environment_overrides={variable: "available"})
        self.assertTrue(runner.wait(5))
        events = runner.drain_events()

        lines = [event.text for event in events if isinstance(event, ProcessOutputEvent)]
        self.assertIn("available", lines)

    def _wait_for_state(self, runner: SubprocessRunner, expected: JobState) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            events = runner.drain_events()
            if any(
                isinstance(event, JobStateEvent) and event.state is expected
                for event in events
            ):
                return
            time.sleep(0.01)
        self.fail(f"Runner did not reach {expected.value}")
