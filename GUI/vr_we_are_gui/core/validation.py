"""Structured validation that does not mutate the filesystem."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .config import Codec, ConversionConfig, InputType


class IssueLevel(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    field: str
    code: str
    message: str
    level: IssueLevel = IssueLevel.ERROR


@dataclass(frozen=True, slots=True)
class ValidationResult:
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not any(issue.level is IssueLevel.ERROR for issue in self.issues)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.level is IssueLevel.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.level is IssueLevel.WARNING)


def validate_config(config: ConversionConfig) -> ValidationResult:
    """Validate user intent while leaving directory creation to an outer layer."""

    issues: list[ValidationIssue] = []

    def error(field: str, code: str, message: str) -> None:
        issues.append(ValidationIssue(field, code, message))

    def warning(field: str, code: str, message: str) -> None:
        issues.append(ValidationIssue(field, code, message, IssueLevel.WARNING))

    input_path = Path(config.input_path) if config.input_path is not None else None
    output_path = Path(config.output_path) if config.output_path is not None else None

    if input_path is None:
        error("input_path", "required", "Input path is required.")
    if output_path is None:
        error("output_path", "required", "Output path is required.")

    if config.clean_output_pngs and config.input_type is not InputType.FOLDER:
        error("clean_output_pngs", "folder_only", "Output cleanup is available only in folder mode.")
    if config.clean_output_pngs and input_path is not None and output_path is not None and _same_path(input_path, output_path):
        error("output_path", "cleanup_matches_input", "Cleanup cannot use the input folder as output.")

    if config.hdr and config.input_type is not InputType.VIDEO:
        error("hdr", "video_only", "HDR is available only in video mode.")
    if config.hdr and config.codec in (Codec.LIBX264, Codec.H264_NVENC):
        error("codec", "hdr_requires_hevc", "HDR requires an HEVC codec.")

    if input_path is None or output_path is None:
        return ValidationResult(tuple(issues))

    if config.input_type is InputType.VIDEO:
        _validate_video(config, input_path, output_path, error)
    elif config.input_type is InputType.FOLDER:
        _validate_folder(config, input_path, warning, error)
    elif config.input_type is InputType.I2I:
        _validate_i2i(config, input_path, output_path, error)
    else:
        error("input_type", "unknown", f"Unknown input type: {config.input_type}")

    for field, value in _positive_values(config):
        if value is not None and value < 1:
            error(field, "must_be_positive", f"{field.replace('_', ' ').title()} must be at least 1.")

    if config.blur_radius is not None and config.blur_radius < 0:
        error("blur_radius", "must_be_non_negative", "Blur radius cannot be negative.")

    return ValidationResult(tuple(issues))


def _validate_video(config, input_path, output_path, error) -> None:
    if not input_path.is_file():
        error("input_path", "video_file_required", "Video mode requires an existing input file.")
    if config.feeders is not None and config.feeders != 1:
        error("feeders", "video_requires_one", "Video mode requires exactly one feeder.")
    if config.savers is not None and config.savers != 1:
        error("savers", "video_requires_one", "Video mode requires exactly one saver.")
    if output_path.suffix.lower() not in (".mp4", ".mkv", ".avi", ".mov"):
        error("output_path", "video_extension", "Video output requires a supported video extension.")


def _validate_folder(config, input_path, warning, error) -> None:
    if not input_path.is_dir():
        error("input_path", "directory_required", "Folder mode requires an existing input directory.")
    if config.codec is not None:
        warning("codec", "ignored", "Codec is ignored in folder mode.")
    if config.quality is not None:
        error("quality", "video_only", "Video quality is available only in video mode.")


def _validate_i2i(config, input_path, output_path, error) -> None:
    if input_path.is_file():
        if output_path.is_dir():
            error("output_path", "file_required", "A single image requires a file output path.")
    elif input_path.is_dir():
        if output_path.exists() and not output_path.is_dir():
            error("output_path", "directory_required", "An image directory requires a directory output path.")
    else:
        error("input_path", "image_or_directory_required", "i2i input must be an existing file or directory.")

    if config.savers is not None and config.savers != 1:
        error("savers", "i2i_requires_one", "i2i mode requires exactly one saver.")
    if config.codec is not None:
        error("codec", "video_only", "Codec is available only in video mode.")
    if config.preset is not None:
        error("preset", "unsupported", "Presets are unavailable in i2i mode.")
    if config.quality is not None:
        error("quality", "video_only", "Video quality is available only in video mode.")
    if config.infer_accum_batches is not None:
        error("infer_accum_batches", "unsupported", "Inference accumulation is unavailable in i2i mode.")


def _positive_values(config: ConversionConfig):
    return (
        ("batch_size", config.batch_size),
        ("infer_accum_batches", config.infer_accum_batches),
        ("in_queue", config.in_queue),
        ("raw_queue", config.raw_queue),
        ("save_queue", config.save_queue),
        ("process_queue", config.process_queue),
        ("feeders", config.feeders),
        ("preprocessors", config.preprocessors),
        ("processors", config.processors),
        ("savers", config.savers),
    )


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left.absolute()) == str(right.absolute())
