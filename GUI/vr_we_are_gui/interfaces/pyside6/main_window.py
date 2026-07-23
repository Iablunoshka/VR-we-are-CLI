"""Main PySide6 window."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtGui import QCloseEvent, QCursor, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...application import ConversionController, format_command
from ...core import (
    AutocastMode,
    Codec,
    ConversionConfig,
    DepthModel,
    HdrEncoder,
    InputType,
    JobFinishedEvent,
    JobState,
    JobStateEvent,
    Preset,
    ProcessErrorEvent,
    ProcessOutputEvent,
    VideoQuality,
)


class CollapsibleSection(QWidget):
    """A compact section whose content can be shown without changing values."""

    def __init__(self, title: str, content: QWidget) -> None:
        super().__init__()
        self.content = content
        self.toggle_button = QToolButton()
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        font = self.toggle_button.font()
        font.setBold(True)
        self.toggle_button.setFont(font)

        self.content.setVisible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content)
        self.toggle_button.toggled.connect(self._set_expanded)

    def _set_expanded(self, expanded: bool) -> None:
        arrow = Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        self.toggle_button.setArrowType(arrow)
        self.content.setVisible(expanded)


class MainWindow(QMainWindow):
    INPUT_DIRECTORY_KEY = "fileDialogs/inputDirectory"
    OUTPUT_DIRECTORY_KEY = "fileDialogs/outputDirectory"
    COMPLETION_FLASH_INTERVAL_MS = 500
    COMPLETION_FLASH_STEPS = 15

    def __init__(self, controller: ConversionController, settings: QSettings | None = None) -> None:
        super().__init__()
        self.controller = controller
        self._settings = settings or QSettings("VR We Are", "VR We Are")
        self.current_job_id: str | None = None
        self._close_when_finished = False
        self._transient_log_blocks: dict[object, int] = {}
        self._completion_flash_steps_remaining = 0
        self._completion_flash_on = False
        self._input_browser_directory = self._load_browser_directory(self.INPUT_DIRECTORY_KEY)
        self._output_browser_directory = self._load_browser_directory(self.OUTPUT_DIRECTORY_KEY)

        self.setWindowTitle("VR We Are (CLI)")
        self.resize(1180, 780)
        self.setMinimumSize(900, 620)
        self._build_ui()
        self._connect_signals()
        self._update_mode_controls()

        self.event_timer = QTimer(self)
        self.event_timer.setInterval(50)
        self.event_timer.timeout.connect(self._drain_process_events)
        self.event_timer.start()
        self.completion_flash_timer = QTimer(self)
        self.completion_flash_timer.setInterval(self.COMPLETION_FLASH_INTERVAL_MS)
        self.completion_flash_timer.timeout.connect(self._advance_completion_flash)

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        title = QLabel("VR We Are (CLI)")
        self._set_emphasis(title, 22)
        subtitle = QLabel("Stereoscopic SBS conversion")
        root.addWidget(title)
        root.addWidget(subtitle)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_settings_panel())
        splitter.addWidget(self._build_log_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([470, 690])
        root.addWidget(splitter, 1)
        root.addWidget(self._build_action_bar())
        self.setCentralWidget(central)

    def _build_settings_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)

        layout.addWidget(self._build_source_group())
        self.conversion_group = self._build_conversion_group()
        layout.addWidget(self.conversion_group)
        self.video_group = self._build_video_group()
        layout.addWidget(self.video_group)
        self.hdr_group = self._build_hdr_group()
        layout.addWidget(self.hdr_group)
        self.advanced_group = self._build_advanced_group()
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        self.advanced_section = CollapsibleSection("Advanced pipeline settings", self.advanced_group)
        layout.addWidget(self.advanced_section)
        layout.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _build_source_group(self) -> QGroupBox:
        group = QGroupBox("Source and processing")
        form = QFormLayout(group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Video", InputType.VIDEO)
        self.mode_combo.addItem("Image folder", InputType.FOLDER)
        self.mode_combo.addItem("Image to image", InputType.I2I)
        form.addRow("Mode", self.mode_combo)

        self.input_edit, self.input_button, input_row = self._path_row()
        self.input_edit.setPlaceholderText("Select an input video")
        form.addRow("Input", input_row)

        self.output_edit, self.output_button, output_row = self._path_row()
        self.output_edit.setPlaceholderText("Select an output video")
        form.addRow("Output", output_row)

        self.preset_combo = self._optional_combo(
            "No preset",
            (("Minimum", Preset.MINIMUM), ("Balance", Preset.BALANCE), ("Maximum quality", Preset.MAX_QUALITY)),
        )
        form.addRow("Preset", self.preset_combo)

        self.model_combo = self._optional_combo(
            "CLI default",
            (("Small", DepthModel.SMALL), ("Base", DepthModel.BASE), ("Large", DepthModel.LARGE)),
        )
        form.addRow("Depth model", self.model_combo)

        self.autocast_combo = self._optional_combo(
            "CLI default",
            (
                ("Auto", AutocastMode.AUTO),
                ("Disabled", AutocastMode.NONE),
                ("Float16", AutocastMode.FLOAT16),
                ("BFloat16", AutocastMode.BFLOAT16),
            ),
        )
        form.addRow("Autocast", self.autocast_combo)

        self.debug_check = QCheckBox("Include CLI debug output")
        form.addRow("", self.debug_check)
        self.clean_output_check = QCheckBox("Delete existing PNG output before starting")
        form.addRow("", self.clean_output_check)
        return group

    def _build_conversion_group(self) -> CollapsibleSection:
        content = QGroupBox()
        form = QFormLayout(content)

        self.depth_scale_spin = QDoubleSpinBox()
        self.depth_scale_spin.setRange(0.0, 10.0)
        self.depth_scale_spin.setDecimals(2)
        self.depth_scale_spin.setSingleStep(0.05)
        self.depth_scale_spin.setValue(1.0)
        form.addRow("Depth scale", self.depth_scale_spin)

        self.depth_offset_spin = QDoubleSpinBox()
        self.depth_offset_spin.setRange(-10.0, 10.0)
        self.depth_offset_spin.setDecimals(2)
        self.depth_offset_spin.setSingleStep(0.05)
        form.addRow("Depth offset", self.depth_offset_spin)

        self.blur_radius_spin = QSpinBox()
        self.blur_radius_spin.setRange(0, 999)
        self.blur_radius_spin.setValue(19)
        form.addRow("Blur radius", self.blur_radius_spin)

        self.switch_sides_check = QCheckBox("Swap left and right images")
        form.addRow("", self.switch_sides_check)
        self.symmetric_check = QCheckBox("Use symmetric rendering")
        form.addRow("", self.symmetric_check)
        return CollapsibleSection("Stereo conversion", content)

    def _build_video_group(self) -> CollapsibleSection:
        content = QGroupBox()
        form = QFormLayout(content)

        self.codec_combo = self._optional_combo(
            "Automatic",
            (
                ("H.264 CPU (libx264)", Codec.LIBX264),
                ("H.265 CPU (libx265)", Codec.LIBX265),
                ("H.264 NVIDIA", Codec.H264_NVENC),
                ("HEVC NVIDIA", Codec.HEVC_NVENC),
            ),
        )
        form.addRow("Codec", self.codec_combo)

        self.quality_combo = self._optional_combo(
            "CLI default",
            (("Low", VideoQuality.LOW), ("Medium", VideoQuality.MEDIUM), ("High", VideoQuality.HIGH)),
        )
        form.addRow("Quality", self.quality_combo)

        return CollapsibleSection("Video", content)

    def _build_hdr_group(self) -> CollapsibleSection:
        content = QGroupBox()
        form = QFormLayout(content)
        self.hdr_check = QCheckBox("Preserve 10-bit HDR")
        form.addRow("", self.hdr_check)
        self.hdr_encoder_combo = self._optional_combo(
            "Automatic",
            (("NVIDIA", HdrEncoder.NVENC), ("libx265", HdrEncoder.LIBX265)),
        )
        form.addRow("HDR encoder", self.hdr_encoder_combo)
        self.master_display_edit = QLineEdit()
        self.master_display_edit.setPlaceholderText("Optional mastering-display metadata")
        form.addRow("Master display", self.master_display_edit)
        self.max_cll_edit = QLineEdit()
        self.max_cll_edit.setPlaceholderText("For example: 1000,400")
        form.addRow("MaxCLL, MaxFALL", self.max_cll_edit)
        return CollapsibleSection("HDR", content)

    def _build_advanced_group(self) -> QGroupBox:
        group = QGroupBox("Apply advanced overrides")
        form = QFormLayout(group)

        self.batch_spin = self._optional_spin(256)
        self.infer_accum_spin = self._optional_spin(256)
        form.addRow("Batch size", self.batch_spin)
        form.addRow("Inference batches", self.infer_accum_spin)

        self.feeders_spin = self._optional_spin(64)
        self.preprocessors_spin = self._optional_spin(64)
        self.processors_spin = self._optional_spin(64)
        self.savers_spin = self._optional_spin(64)
        form.addRow("Feeders", self.feeders_spin)
        form.addRow("Preprocessors", self.preprocessors_spin)
        form.addRow("Processors", self.processors_spin)
        form.addRow("Savers", self.savers_spin)

        self.raw_queue_spin = self._optional_spin(1024)
        self.input_queue_spin = self._optional_spin(1024)
        self.process_queue_spin = self._optional_spin(1024)
        self.save_queue_spin = self._optional_spin(1024)
        form.addRow("Raw queue", self.raw_queue_spin)
        form.addRow("Input queue", self.input_queue_spin)
        form.addRow("Process queue", self.process_queue_spin)
        form.addRow("Save queue", self.save_queue_spin)
        return group

    def _build_log_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 0, 0)
        heading = QLabel("CLI output")
        self._set_emphasis(heading, 12)
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("CLI output will appear here.")
        self.log_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        clear_button = QPushButton("Clear output")
        clear_button.clicked.connect(self._clear_log)
        layout.addWidget(heading)
        layout.addWidget(self.log_output, 1)
        layout.addWidget(clear_button, 0, Qt.AlignmentFlag.AlignRight)
        return panel

    def _build_action_bar(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel("Ready")
        self._set_emphasis(self.status_label)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.start_button = QPushButton("Start conversion")
        self._set_emphasis(self.start_button)
        action_width = max(self.cancel_button.sizeHint().width(), self.start_button.sizeHint().width())
        action_height = max(34, self.cancel_button.sizeHint().height(), self.start_button.sizeHint().height())
        self.cancel_button.setFixedSize(action_width, action_height)
        self.start_button.setFixedSize(action_width, action_height)
        layout.addWidget(self.status_label, 1)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.start_button)
        return bar

    def _connect_signals(self) -> None:
        self.mode_combo.currentIndexChanged.connect(self._update_mode_controls)
        self.hdr_check.toggled.connect(self._update_hdr_controls)
        self.input_button.clicked.connect(self._browse_input)
        self.output_button.clicked.connect(self._browse_output)
        self.start_button.clicked.connect(self._start_conversion)
        self.cancel_button.clicked.connect(self._cancel_conversion)

    def _update_mode_controls(self) -> None:
        mode = self._selected_enum(self.mode_combo, InputType)
        is_video = mode == InputType.VIDEO
        is_folder = mode == InputType.FOLDER
        is_i2i = mode == InputType.I2I

        self.video_group.setEnabled(is_video)
        self.hdr_group.setEnabled(is_video)
        self.preset_combo.setEnabled(not is_i2i)
        self.clean_output_check.setEnabled(is_folder)
        if not is_folder:
            self.clean_output_check.setChecked(False)
        self.advanced_section.setEnabled(not is_i2i)

        if is_video:
            self.input_edit.setPlaceholderText("Select an input video")
            self.output_edit.setPlaceholderText("Select an output video")
        elif is_folder:
            self.input_edit.setPlaceholderText("Select a folder containing images")
            self.output_edit.setPlaceholderText("Select an output folder")
        else:
            self.input_edit.setPlaceholderText("Select an image or image folder")
            self.output_edit.setPlaceholderText("Select an output file or folder")
        self._update_hdr_controls()

    def _update_hdr_controls(self) -> None:
        enabled = self._selected_enum(self.mode_combo, InputType) is InputType.VIDEO and self.hdr_check.isChecked()
        self.hdr_encoder_combo.setEnabled(enabled)
        self.master_display_edit.setEnabled(enabled)
        self.max_cll_edit.setEnabled(enabled)

    def _browse_input(self) -> None:
        mode = self._selected_enum(self.mode_combo, InputType)
        initial = self._dialog_directory(self.input_edit, self._input_browser_directory)
        selected_directory = False
        if mode == InputType.VIDEO:
            path, _ = QFileDialog.getOpenFileName(self, "Select video", initial, "Video files (*.mp4 *.mkv *.avi *.mov);;All files (*)")
        elif mode == InputType.FOLDER:
            path = QFileDialog.getExistingDirectory(self, "Select image folder", initial)
            selected_directory = True
        else:
            menu = QMenu(self)
            image_action = menu.addAction("Select image")
            folder_action = menu.addAction("Select folder")
            selected = menu.exec(QCursor.pos())
            if selected == image_action:
                path, _ = QFileDialog.getOpenFileName(self, "Select image", initial, "Images (*.png *.jpg *.jpeg);;All files (*)")
            elif selected == folder_action:
                path = QFileDialog.getExistingDirectory(self, "Select image folder", initial)
                selected_directory = True
            else:
                path = ""
        if path:
            self.input_edit.setText(path)
            self._input_browser_directory = self._selected_directory(path, selected_directory)
            self._save_browser_directory(self.INPUT_DIRECTORY_KEY, self._input_browser_directory)

    def _browse_output(self) -> None:
        mode = self._selected_enum(self.mode_combo, InputType)
        initial = self._dialog_directory(self.output_edit, self._output_browser_directory)
        selected_directory = False
        if mode == InputType.VIDEO:
            path, _ = QFileDialog.getSaveFileName(self, "Select output video", initial, "MP4 video (*.mp4);;MKV video (*.mkv);;All files (*)")
        elif mode == InputType.FOLDER:
            path = QFileDialog.getExistingDirectory(self, "Select output folder", initial)
            selected_directory = True
        else:
            input_path = Path(self.input_edit.text().strip()) if self.input_edit.text().strip() else None
            if input_path and input_path.is_file():
                path, _ = QFileDialog.getSaveFileName(self, "Select output image", initial, "PNG image (*.png);;All files (*)")
            else:
                path = QFileDialog.getExistingDirectory(self, "Select output folder", initial)
                selected_directory = True
        if path:
            self.output_edit.setText(path)
            self._output_browser_directory = self._selected_directory(path, selected_directory)
            self._save_browser_directory(self.OUTPUT_DIRECTORY_KEY, self._output_browser_directory)

    def _start_conversion(self) -> None:
        self._stop_completion_flash()
        config = self._collect_config()
        preparation = self.controller.prepare(config)
        if not preparation.validation.is_valid:
            self._show_validation_errors(preparation.validation.errors)
            return
        if not self._confirm_output_actions(config):
            return

        for issue in preparation.validation.warnings:
            self._append_log(f"[warning] {issue.message}")
        self._append_log(f"[GUI] {format_command(preparation.command)}")

        try:
            result = self.controller.start(config)
        except Exception as exc:
            self._append_log(f"[GUI error] {exc}")
            QMessageBox.critical(self, "Unable to start", str(exc))
            return

        if not result.started:
            self._show_validation_errors(result.preparation.validation.errors)
            return
        self.current_job_id = result.job_id
        self._set_running(True)
        self.status_label.setText("Starting")

    def _cancel_conversion(self) -> None:
        if self.current_job_id and self.controller.cancel(self.current_job_id):
            self.status_label.setText("Cancelling")
            self.cancel_button.setEnabled(False)

    def _drain_process_events(self) -> None:
        for event in self.controller.drain_events():
            if self.current_job_id and event.job_id != self.current_job_id:
                continue
            if isinstance(event, ProcessOutputEvent):
                self._append_process_output(event)
            elif isinstance(event, ProcessErrorEvent):
                self._append_log(f"[GUI error] {event.message}")
            elif isinstance(event, JobStateEvent):
                self._apply_job_state(event.state)
            elif isinstance(event, JobFinishedEvent):
                self._finish_job(event)

    def _apply_job_state(self, state: JobState) -> None:
        labels = {
            JobState.STARTING: "Starting",
            JobState.RUNNING: "Running",
            JobState.CANCELLING: "Cancelling",
        }
        if state in labels:
            self.status_label.setText(labels[state])

    def _finish_job(self, event: JobFinishedEvent) -> None:
        result = event.result
        labels = {
            JobState.SUCCEEDED: "Completed",
            JobState.FAILED: f"Failed (exit code {result.exit_code})",
            JobState.CANCELLED: "Cancelled",
        }
        self.status_label.setText(labels[result.state])
        self._append_log(f"[GUI] {labels[result.state]}")
        self._transient_log_blocks.clear()
        self.current_job_id = None
        self._set_running(False)
        if result.state is JobState.SUCCEEDED:
            self._start_completion_flash()
        if self._close_when_finished:
            self._close_when_finished = False
            QTimer.singleShot(0, self.close)

    def _collect_config(self) -> ConversionConfig:
        mode = self._selected_enum(self.mode_combo, InputType)
        input_text = self.input_edit.text().strip()
        output_text = self.output_edit.text().strip()
        advanced = mode != InputType.I2I and self.advanced_group.isChecked()
        video = mode == InputType.VIDEO
        hdr = True if video and self.hdr_check.isChecked() else None

        return ConversionConfig(
            input_path=Path(input_text) if input_text else None,
            output_path=Path(output_text) if output_text else None,
            input_type=mode,
            preset=self._selected_enum(self.preset_combo, Preset) if mode is not InputType.I2I else None,
            batch_size=self._optional_value(self.batch_spin) if advanced else None,
            model=self._selected_enum(self.model_combo, DepthModel),
            codec=self._selected_enum(self.codec_combo, Codec) if video else None,
            quality=self._selected_enum(self.quality_combo, VideoQuality) if video else None,
            autocast=self._selected_enum(self.autocast_combo, AutocastMode),
            infer_accum_batches=self._optional_value(self.infer_accum_spin) if advanced else None,
            debug=self.debug_check.isChecked(),
            clean_output_pngs=mode == InputType.FOLDER and self.clean_output_check.isChecked(),
            in_queue=self._optional_value(self.input_queue_spin) if advanced else None,
            raw_queue=self._optional_value(self.raw_queue_spin) if advanced else None,
            save_queue=self._optional_value(self.save_queue_spin) if advanced else None,
            process_queue=self._optional_value(self.process_queue_spin) if advanced else None,
            feeders=self._optional_value(self.feeders_spin) if advanced else None,
            preprocessors=self._optional_value(self.preprocessors_spin) if advanced else None,
            processors=self._optional_value(self.processors_spin) if advanced else None,
            savers=self._optional_value(self.savers_spin) if advanced else None,
            depth_scale=self.depth_scale_spin.value(),
            depth_offset=self.depth_offset_spin.value(),
            switch_sides=self.switch_sides_check.isChecked(),
            symmetric=self.symmetric_check.isChecked(),
            blur_radius=self.blur_radius_spin.value(),
            hdr=hdr,
            hdr_encoder=self._selected_enum(self.hdr_encoder_combo, HdrEncoder) if hdr else None,
            master_display=self._text_or_none(self.master_display_edit) if hdr else None,
            max_cll=self._text_or_none(self.max_cll_edit) if hdr else None,
        )

    def _confirm_output_actions(self, config: ConversionConfig) -> bool:
        if config.clean_output_pngs:
            answer = QMessageBox.question(
                self,
                "Clean output folder?",
                "Existing PNG files in the output folder will be deleted before conversion. Continue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False

        output = config.output_path
        if output and output.is_file() and config.input_type in (InputType.VIDEO, InputType.I2I):
            answer = QMessageBox.question(self, "Replace output?", f"The output file already exists:\n{output}\n\nReplace it?")
            if answer != QMessageBox.StandardButton.Yes:
                return False
        return True

    def _show_validation_errors(self, issues) -> None:
        message = "\n".join(f"• {issue.message}" for issue in issues)
        QMessageBox.warning(self, "Check the settings", message)

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def _start_completion_flash(self) -> None:
        self._completion_flash_steps_remaining = self.COMPLETION_FLASH_STEPS
        self._completion_flash_on = False
        self._advance_completion_flash()
        self.completion_flash_timer.start()

    def _advance_completion_flash(self) -> None:
        self._completion_flash_on = not self._completion_flash_on
        self.status_label.setStyleSheet("color: #20a54a;" if self._completion_flash_on else "")
        self._completion_flash_steps_remaining -= 1
        if self._completion_flash_steps_remaining <= 0:
            self._stop_completion_flash()
            self.status_label.setText("Ready")

    def _stop_completion_flash(self) -> None:
        self.completion_flash_timer.stop()
        self._completion_flash_steps_remaining = 0
        self._completion_flash_on = False
        self.status_label.setStyleSheet("")

    def _append_log(self, text: str) -> None:
        scroll_state = self._capture_log_scroll()
        self.log_output.appendPlainText(text)
        self._restore_log_scroll(scroll_state)

    def _append_process_output(self, event: ProcessOutputEvent) -> None:
        scroll_state = self._capture_log_scroll()
        block_number = self._transient_log_blocks.get(event.channel)
        block = self.log_output.document().findBlockByNumber(block_number) if block_number is not None else None

        if block is not None and block.isValid():
            cursor = QTextCursor(block)
            cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertText(event.text)
        else:
            self.log_output.appendPlainText(event.text)
            block_number = self.log_output.document().lastBlock().blockNumber()

        if event.replace_line:
            self._transient_log_blocks[event.channel] = block_number
        else:
            self._transient_log_blocks.pop(event.channel, None)
        self._restore_log_scroll(scroll_state)

    def _clear_log(self) -> None:
        self.log_output.clear()
        self._transient_log_blocks.clear()

    def _capture_log_scroll(self) -> tuple[int, int, bool]:
        horizontal = self.log_output.horizontalScrollBar()
        vertical = self.log_output.verticalScrollBar()
        follows_tail = vertical.value() >= vertical.maximum() - 1
        return horizontal.value(), vertical.value(), follows_tail

    def _restore_log_scroll(self, state: tuple[int, int, bool]) -> None:
        horizontal_value, vertical_value, follows_tail = state
        horizontal = self.log_output.horizontalScrollBar()
        vertical = self.log_output.verticalScrollBar()
        horizontal.setValue(min(horizontal_value, horizontal.maximum()))
        vertical.setValue(vertical.maximum() if follows_tail else min(vertical_value, vertical.maximum()))

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self.controller.is_active:
            event.accept()
            return
        answer = QMessageBox.question(self, "Conversion is running", "Cancel the conversion and close the application?")
        if answer == QMessageBox.StandardButton.Yes:
            self._close_when_finished = True
            self._cancel_conversion()
        event.ignore()

    @staticmethod
    def _path_row():
        edit = QLineEdit()
        button = QPushButton("Browse…")
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return edit, button, row

    @staticmethod
    def _optional_combo(default_label, items) -> QComboBox:
        combo = QComboBox()
        combo.addItem(default_label, None)
        for label, value in items:
            combo.addItem(label, value)
        return combo

    @staticmethod
    def _selected_enum(combo: QComboBox, enum_type):
        value = combo.currentData()
        return enum_type(value) if value is not None else None

    @staticmethod
    def _optional_spin(maximum: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, maximum)
        spin.setSpecialValueText("CLI default")
        return spin

    @staticmethod
    def _optional_value(spin: QSpinBox) -> int | None:
        return spin.value() or None

    @staticmethod
    def _text_or_none(edit: QLineEdit) -> str | None:
        value = edit.text().strip()
        return value or None

    @staticmethod
    def _set_emphasis(widget: QWidget, point_size: int | None = None) -> None:
        font = widget.font()
        font.setBold(True)
        if point_size is not None:
            font.setPointSize(point_size)
        widget.setFont(font)

    @staticmethod
    def _dialog_directory(edit: QLineEdit, remembered: Path) -> str:
        value = edit.text().strip()
        if value:
            candidate = Path(value)
            if candidate.is_dir():
                return str(candidate)
            if candidate.parent.is_dir():
                return str(candidate.parent)
        return str(remembered)

    @staticmethod
    def _selected_directory(path: str, selected_directory: bool) -> Path:
        selected = Path(path)
        return selected if selected_directory else selected.parent

    def _load_browser_directory(self, key: str) -> Path:
        value = self._settings.value(key, "", type=str)
        if not value:
            return Path.home()
        saved = Path(value)
        return saved if saved.is_dir() else Path.home()

    def _save_browser_directory(self, key: str, directory: Path) -> None:
        self._settings.setValue(key, str(directory))
        self._settings.sync()
