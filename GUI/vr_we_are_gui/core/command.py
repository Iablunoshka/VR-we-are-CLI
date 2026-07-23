"""Safe construction of the existing CLI invocation."""

from dataclasses import dataclass
from pathlib import Path

from .config import ConversionConfig


@dataclass(frozen=True, slots=True)
class CliTarget:
    """Location of the Python runtime and CLI entry point."""

    python_executable: Path
    main_script: Path
    unbuffered: bool = True


def build_cli_command(config: ConversionConfig, target: CliTarget) -> tuple[str, ...]:
    """Build an immutable argument list without shell quoting."""

    if config.input_path is None or config.output_path is None:
        raise ValueError("Input and output paths are required to build a CLI command.")

    command = [str(target.python_executable)]
    if target.unbuffered:
        command.append("-u")
    command.extend(
        [
            str(target.main_script),
            "--input",
            str(config.input_path),
            "--output",
            str(config.output_path),
            "--input-type",
            str(config.input_type),
        ]
    )

    option_values = (
        ("--preset", config.preset),
        ("--batch-size", config.batch_size),
        ("--model", config.model),
        ("--codec", config.codec),
        ("--quality", config.quality),
        ("--autocast", config.autocast),
        ("--infer-accum-batches", config.infer_accum_batches),
        ("--in-queue", config.in_queue),
        ("--r-queue", config.raw_queue),
        ("--s-queue", config.save_queue),
        ("--p-queue", config.process_queue),
        ("--feeders", config.feeders),
        ("--preprocess", config.preprocessors),
        ("--processors", config.processors),
        ("--savers", config.savers),
        ("--depth-scale", config.depth_scale),
        ("--depth-offset", config.depth_offset),
        ("--blur-radius", config.blur_radius),
        ("--hdr-encoder", config.hdr_encoder),
        ("--master-display", config.master_display),
        ("--max-cll", config.max_cll),
    )
    for option, value in option_values:
        if value is not None:
            command.extend((option, str(value)))

    enabled_flags = (
        ("--debug", config.debug),
        ("--clean-output-pngs", config.clean_output_pngs),
        ("--switch-sides", config.switch_sides),
        ("--symmetric", config.symmetric),
    )
    command.extend(option for option, enabled in enabled_flags if enabled)

    if config.hdr is True:
        command.append("--hdr")
    elif config.hdr is False:
        command.append("--no-hdr")

    return tuple(command)
