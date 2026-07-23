import os
import sys
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest import TestCase

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from vr_we_are_gui.application import ConversionController
from vr_we_are_gui.core import (
    CliTarget,
    Codec,
    InputType,
    JobFinishedEvent,
    JobResult,
    JobState,
    JobStateEvent,
    OutputChannel,
    Preset,
    ProcessOutputEvent,
)
from vr_we_are_gui.interfaces.pyside6 import MainWindow
from vr_we_are_gui.interfaces.pyside6.theme import configure_appearance

from test_application import FakeRunner


class PySideSmokeTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        target = CliTarget(Path(sys.executable), Path("main.py"))
        self.runner = FakeRunner()
        self.window = MainWindow(ConversionController(target, self.runner))

    def tearDown(self):
        self.runner.is_active = False
        self.window.event_timer.stop()
        self.window.completion_flash_timer.stop()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_builds_main_window(self):
        self.assertEqual(self.window.windowTitle(), "VR We Are (CLI)")
        self.assertEqual(self.window.mode_combo.count(), 3)
        self.assertFalse(self.window.cancel_button.isEnabled())
        self.assertEqual(self.window.cancel_button.size(), self.window.start_button.size())
        self.assertTrue(self.window.conversion_group.content.isHidden())
        self.assertTrue(self.window.video_group.content.isHidden())
        self.assertTrue(self.window.hdr_group.content.isHidden())
        self.assertTrue(self.window.advanced_section.content.isHidden())

    def test_collapsible_section_preserves_its_values(self):
        self.window.conversion_group.toggle_button.setChecked(True)
        self.window.depth_scale_spin.setValue(1.75)
        self.window.conversion_group.toggle_button.setChecked(False)
        self.window.conversion_group.toggle_button.setChecked(True)

        self.assertFalse(self.window.conversion_group.content.isHidden())
        self.assertEqual(self.window.depth_scale_spin.value(), 1.75)

    def test_folder_mode_maps_to_neutral_configuration(self):
        self.window.mode_combo.setCurrentIndex(1)
        self.window.input_edit.setText("frames")
        self.window.output_edit.setText("output")
        self.window.preset_combo.setCurrentIndex(2)
        self.window.clean_output_check.setChecked(True)

        config = self.window._collect_config()

        self.assertEqual(config.input_type, InputType.FOLDER)
        self.assertIs(config.input_type, InputType.FOLDER)
        self.assertEqual(config.preset, Preset.BALANCE)
        self.assertIs(config.preset, Preset.BALANCE)
        self.assertTrue(config.clean_output_pngs)
        self.assertIsNone(config.codec)

    def test_combo_values_are_normalized_to_enum_instances(self):
        self.window.codec_combo.setCurrentIndex(1)

        config = self.window._collect_config()

        self.assertIs(config.input_type, InputType.VIDEO)
        self.assertIs(config.codec, Codec.LIBX264)

    def test_i2i_disables_unsupported_controls(self):
        self.window.mode_combo.setCurrentIndex(2)

        self.assertFalse(self.window.preset_combo.isEnabled())
        self.assertFalse(self.window.video_group.isEnabled())
        self.assertFalse(self.window.hdr_group.isEnabled())
        self.assertFalse(self.window.advanced_section.isEnabled())

    def test_unchecked_advanced_group_does_not_apply_values(self):
        self.window.advanced_group.setChecked(True)
        self.window.batch_spin.setValue(8)
        self.window.advanced_group.setChecked(False)

        config = self.window._collect_config()

        self.assertIsNone(config.batch_size)

    def test_process_events_update_output_and_controls(self):
        self.window.current_job_id = "job-1"
        self.window._set_running(True)
        self.runner.events.extend(
            (
                JobStateEvent("job-1", JobState.RUNNING),
                ProcessOutputEvent("job-1", OutputChannel.STDOUT, "CLI line"),
                JobFinishedEvent("job-1", JobResult(JobState.SUCCEEDED, 0)),
            )
        )

        self.window._drain_process_events()

        self.assertIn("CLI line", self.window.log_output.toPlainText())
        self.assertEqual(self.window.status_label.text(), "Completed")
        self.assertTrue(self.window.start_button.isEnabled())
        self.assertFalse(self.window.cancel_button.isEnabled())

    def test_successful_job_flashes_green_then_returns_to_ready(self):
        self.window._finish_job(JobFinishedEvent("job-1", JobResult(JobState.SUCCEEDED, 0)))

        self.assertEqual(self.window.status_label.text(), "Completed")
        self.assertIn("#20a54a", self.window.status_label.styleSheet())

        for _ in range(self.window.COMPLETION_FLASH_STEPS - 1):
            self.window._advance_completion_flash()

        self.assertEqual(self.window.status_label.text(), "Ready")
        self.assertEqual(self.window.status_label.styleSheet(), "")
        self.assertFalse(self.window.completion_flash_timer.isActive())

    def test_carriage_return_output_replaces_previous_line(self):
        self.window.current_job_id = "job-1"
        self.runner.events.extend(
            (
                ProcessOutputEvent("job-1", OutputChannel.STDERR, "frame=1", True),
                ProcessOutputEvent("job-1", OutputChannel.STDERR, "frame=2", True),
                ProcessOutputEvent("job-1", OutputChannel.STDERR, "frame=3", False),
            )
        )

        self.window._drain_process_events()

        output = self.window.log_output.toPlainText()
        self.assertNotIn("frame=1", output)
        self.assertNotIn("frame=2", output)
        self.assertEqual(output, "frame=3")

    def test_long_status_updates_preserve_horizontal_scroll(self):
        self.window.log_output.resize(180, 120)
        self.window._append_process_output(
            ProcessOutputEvent("job-1", OutputChannel.STDERR, "frame=1 " + "x" * 300, True)
        )
        self.app.processEvents()
        horizontal = self.window.log_output.horizontalScrollBar()
        self.assertGreater(horizontal.maximum(), 0)
        horizontal.setValue(0)

        self.window._append_process_output(
            ProcessOutputEvent("job-1", OutputChannel.STDERR, "frame=2 " + "y" * 400, True)
        )
        self.app.processEvents()

        self.assertEqual(horizontal.value(), 0)

    def test_input_and_output_browser_directories_persist_separately(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_directory = root / "input-location"
            output_directory = root / "output-location"
            input_directory.mkdir()
            output_directory.mkdir()
            settings = QSettings(str(root / "settings.ini"), QSettings.Format.IniFormat)

            self.window._settings = settings
            self.window._save_browser_directory(self.window.INPUT_DIRECTORY_KEY, input_directory)
            self.window._save_browser_directory(self.window.OUTPUT_DIRECTORY_KEY, output_directory)
            restored = MainWindow(self.window.controller, settings)
            try:
                self.assertEqual(restored._input_browser_directory, input_directory)
                self.assertEqual(restored._output_browser_directory, output_directory)
            finally:
                restored.event_timer.stop()
                restored.close()
                restored.deleteLater()

    def test_widgets_follow_application_palette(self):
        configure_appearance(self.app)
        original = QPalette(self.app.palette())
        changed = QPalette(original)
        changed.setColor(QPalette.ColorRole.Base, QColor("#202020"))

        try:
            self.app.setPalette(changed)
            self.app.processEvents()
            actual = self.window.input_edit.palette().color(QPalette.ColorRole.Base)
            self.assertEqual(actual, QColor("#202020"))
        finally:
            self.app.setPalette(original)
