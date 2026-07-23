"""Application-layer tests and shared runner fake."""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from vr_we_are_gui.application import ConversionController
from vr_we_are_gui.core import CliTarget, ConversionConfig


class FakeRunner:
    def __init__(self):
        self.is_active = False
        self.started = []
        self.cancelled = []
        self.events = []

    def start(self, command, *, working_directory=None, environment_overrides=None):
        self.started.append((command, working_directory, environment_overrides))
        self.is_active = True
        return "job-1"

    def cancel(self, job_id):
        self.cancelled.append(job_id)
        return True

    def drain_events(self, maximum=None):
        result = tuple(self.events[:maximum]) if maximum is not None else tuple(self.events)
        del self.events[: len(result)]
        return result


class ConversionControllerTests(TestCase):
    def setUp(self):
        self.runner = FakeRunner()
        self.target = CliTarget(Path(sys.executable), Path("project", "main.py"))
        self.controller = ConversionController(self.target, self.runner)

    def test_does_not_start_invalid_configuration(self):
        result = self.controller.start(ConversionConfig())

        self.assertFalse(result.started)
        self.assertFalse(result.preparation.validation.is_valid)
        self.assertEqual(self.runner.started, [])

    def test_starts_valid_configuration_in_cli_directory(self):
        with TemporaryDirectory() as directory:
            input_path = Path(directory, "input.mp4")
            input_path.touch()
            config = ConversionConfig(input_path, Path(directory, "output.mp4"))

            result = self.controller.start(config)

        self.assertTrue(result.started)
        command, working_directory, environment = self.runner.started[0]
        self.assertEqual(result.job_id, "job-1")
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(working_directory, Path("project"))
        self.assertEqual(
            environment,
            {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )

    def test_forwards_cancellation(self):
        self.assertTrue(self.controller.cancel("job-1"))
        self.assertEqual(self.runner.cancelled, ["job-1"])
