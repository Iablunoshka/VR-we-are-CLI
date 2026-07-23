from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from vr_we_are_gui.core import (
    CliTarget,
    Codec,
    ConversionConfig,
    InputType,
    JobResult,
    JobState,
    JobStateMachine,
    Preset,
    build_cli_command,
    validate_config,
)
from vr_we_are_gui.core.jobs import InvalidJobTransition


class CommandBuilderTests(TestCase):
    def setUp(self):
        self.target = CliTarget(Path("python"), Path("main.py"))

    def test_builds_minimal_video_command(self):
        config = ConversionConfig(Path("input.mp4"), Path("output.mp4"))

        command = build_cli_command(config, self.target)

        self.assertEqual(
            command,
            (
                "python",
                "-u",
                "main.py",
                "--input",
                "input.mp4",
                "--output",
                "output.mp4",
                "--input-type",
                "video",
            ),
        )

    def test_emits_preset_overrides_and_flags(self):
        config = ConversionConfig(
            Path("frames"),
            Path("out"),
            input_type=InputType.FOLDER,
            preset=Preset.BALANCE,
            codec=Codec.LIBX264,
            batch_size=7,
            debug=True,
            clean_output_pngs=True,
            symmetric=True,
        )

        command = build_cli_command(config, self.target)

        self.assertIn("--preset", command)
        self.assertIn("balance", command)
        self.assertIn("--batch-size", command)
        self.assertIn("7", command)
        self.assertIn("--debug", command)
        self.assertIn("--clean-output-pngs", command)
        self.assertIn("--symmetric", command)

    def test_distinguishes_unspecified_and_disabled_hdr(self):
        unspecified = ConversionConfig(Path("in.mp4"), Path("out.mp4"))
        disabled = ConversionConfig(Path("in.mp4"), Path("out.mp4"), hdr=False)

        self.assertNotIn("--hdr", build_cli_command(unspecified, self.target))
        self.assertNotIn("--no-hdr", build_cli_command(unspecified, self.target))
        self.assertIn("--no-hdr", build_cli_command(disabled, self.target))

    def test_rejects_command_with_missing_paths(self):
        with self.assertRaises(ValueError):
            build_cli_command(ConversionConfig(), self.target)


class ValidationTests(TestCase):
    def test_reports_missing_paths(self):
        result = validate_config(ConversionConfig())
        self.assertEqual({issue.field for issue in result.errors}, {"input_path", "output_path"})

    def test_accepts_existing_video_and_supported_output(self):
        with TemporaryDirectory() as directory:
            input_path = Path(directory, "input.mp4")
            input_path.touch()
            config = ConversionConfig(input_path, Path(directory, "output.mp4"))
            self.assertTrue(validate_config(config).is_valid)

    def test_rejects_video_worker_counts_other_than_one(self):
        with TemporaryDirectory() as directory:
            input_path = Path(directory, "input.mp4")
            input_path.touch()
            config = ConversionConfig(input_path, Path(directory, "output.mp4"), feeders=2, savers=2)
            codes = {issue.code for issue in validate_config(config).errors}
            self.assertIn("video_requires_one", codes)

    def test_folder_codec_is_warning(self):
        with TemporaryDirectory() as directory:
            config = ConversionConfig(
                Path(directory),
                Path(directory, "out"),
                input_type=InputType.FOLDER,
                codec=Codec.LIBX264,
            )
            result = validate_config(config)
            self.assertTrue(result.is_valid)
            self.assertEqual(result.warnings[0].code, "ignored")

    def test_rejects_i2i_preset_and_codec(self):
        with TemporaryDirectory() as directory:
            image = Path(directory, "image.png")
            image.touch()
            config = ConversionConfig(
                image,
                Path(directory, "output.png"),
                input_type=InputType.I2I,
                preset=Preset.MINIMUM,
                codec=Codec.LIBX264,
            )
            fields = {issue.field for issue in validate_config(config).errors}
            self.assertIn("preset", fields)
            self.assertIn("codec", fields)

    def test_rejects_hdr_with_h264(self):
        with TemporaryDirectory() as directory:
            input_path = Path(directory, "input.mp4")
            input_path.touch()
            config = ConversionConfig(
                input_path,
                Path(directory, "output.mp4"),
                hdr=True,
                codec=Codec.H264_NVENC,
            )
            codes = {issue.code for issue in validate_config(config).errors}
            self.assertIn("hdr_requires_hevc", codes)


class JobStateTests(TestCase):
    def test_successful_lifecycle(self):
        job = JobStateMachine()
        job.transition_to(JobState.STARTING)
        job.transition_to(JobState.RUNNING)
        job.transition_to(JobState.SUCCEEDED)
        self.assertEqual(job.state, JobState.SUCCEEDED)

    def test_rejects_invalid_transition(self):
        job = JobStateMachine()
        with self.assertRaises(InvalidJobTransition):
            job.transition_to(JobState.SUCCEEDED)

    def test_result_requires_terminal_state(self):
        with self.assertRaises(ValueError):
            JobResult(JobState.RUNNING)
