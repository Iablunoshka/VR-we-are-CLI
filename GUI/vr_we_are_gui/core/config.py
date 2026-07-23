"""Configuration types shared by every interface adapter."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class StringEnum(str, Enum):
    """String-valued enum with stable CLI serialization."""

    def __str__(self) -> str:
        return self.value


class InputType(StringEnum):
    VIDEO = "video"
    FOLDER = "folder"
    I2I = "i2i"


class Preset(StringEnum):
    MINIMUM = "minimum"
    BALANCE = "balance"
    MAX_QUALITY = "max_quality"


class DepthModel(StringEnum):
    SMALL = "depth-anything/Depth-Anything-V2-Small-hf"
    BASE = "depth-anything/Depth-Anything-V2-Base-hf"
    LARGE = "depth-anything/Depth-Anything-V2-Large-hf"


class Codec(StringEnum):
    LIBX264 = "libx264"
    LIBX265 = "libx265"
    H264_NVENC = "h264_nvenc"
    HEVC_NVENC = "hevc_nvenc"


class VideoQuality(StringEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AutocastMode(StringEnum):
    AUTO = "auto"
    NONE = "none"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class HdrEncoder(StringEnum):
    AUTO = "auto"
    NVENC = "nvenc"
    LIBX265 = "libx265"


@dataclass(frozen=True, slots=True)
class ConversionConfig:
    """User intent before CLI defaults and presets are applied."""

    input_path: Path | None = None
    output_path: Path | None = None
    input_type: InputType = InputType.VIDEO
    preset: Preset | None = None
    batch_size: int | None = None
    model: DepthModel | None = None
    codec: Codec | None = None
    quality: VideoQuality | None = None
    autocast: AutocastMode | None = None
    infer_accum_batches: int | None = None
    debug: bool = False
    clean_output_pngs: bool = False
    in_queue: int | None = None
    raw_queue: int | None = None
    save_queue: int | None = None
    process_queue: int | None = None
    feeders: int | None = None
    preprocessors: int | None = None
    processors: int | None = None
    savers: int | None = None
    depth_scale: float | None = None
    depth_offset: float | None = None
    switch_sides: bool = False
    symmetric: bool = False
    blur_radius: int | None = None
    hdr: bool | None = None
    hdr_encoder: HdrEncoder | None = None
    master_display: str | None = None
    max_cll: str | None = None
