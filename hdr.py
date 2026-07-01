"""
hdr.py — ISOLATED 10-bit HDR support for the SBS (2D->3D) converter.
================================================================================

WHY THIS FILE EXISTS (read me first)
------------------------------------
The SBS converter is 8-bit SDR by default. An OPTIONAL true-10-bit-HDR path was
added so an HDR (PQ/BT.2020) source can be converted to a 3D side-by-side video
WITHOUT being tonemapped down to SDR.

To keep the existing 8-bit pipeline readable for people who already know the code,
**all** the HDR-specific logic lives in THIS module. The existing files
(main.py, pipeline_core.py, converter.py, sbsutils.py) contain only small,
clearly-marked one-line hooks tagged:  `# --- HDR (10-bit) ---`
Each hook just calls a function here. If you remove `--hdr`, none of this runs.

HOW THE HDR PATH WORKS (only active when --hdr is passed; video input only)
---------------------------------------------------------------------------
1. DECODE 16-bit: frames are decoded as `rgb48le` (uint16) instead of `rgb24`
   (uint8).  -> pipe_in_pix_fmt()
2. DEPTH uses an 8-bit PROXY: Depth-Anything was trained on normal sRGB images;
   a raw PQ frame looks far too dark to it and the predicted depth degrades.
   So the depth model receives a tonemapped 8-bit sRGB proxy (depth_proxy /
   pq_to_srgb8), while the WARP keeps the original 16-bit pixels. The depth map
   is pure geometry (pixel shift) -> the output COLOUR is unaffected by the proxy.
3. WARP 16-bit: converter.py runs the DIBR warp in uint16 instead of uint8
   (cv2.remap supports uint16). See the bit-depth profile below.
4. ENCODE 10-bit HEVC: output is libx265 `yuv420p10le` (or hevc_nvenc `p010le`),
   tagged BT.2020 + PQ, and — for libx265 only — carrying the source's HDR10
   static metadata (mastering display + MaxCLL) when available.
   -> select_codec(), encode_color_args()

Author: VR we are. MIT License.
"""

import numpy as np
import subprocess
import tempfile
import os


# =============================================================================
# Bit-depth profile
# =============================================================================
# The whole 8-bit vs 16-bit difference reduces to two numbers. converter.py
# mirrors these two lines inline (it must not import this module, because it is
# also loaded as a ComfyUI node); keep them in sync.

def pixel_max(hdr: bool) -> float:
    """Full-scale integer value: 65535 for 16-bit HDR, 255 for 8-bit SDR."""
    return 65535.0 if hdr else 255.0


def pixel_dtype(hdr: bool):
    """numpy integer dtype for the working frames: uint16 (HDR) or uint8 (SDR)."""
    return np.uint16 if hdr else np.uint8


def pipe_in_pix_fmt(hdr: bool) -> str:
    """ffmpeg raw-pipe INPUT pixel format for the encoder stage."""
    return "rgb48le" if hdr else "rgb24"


# =============================================================================
# Depth proxy: PQ (HDR10) -> 8-bit sRGB, ONLY for the depth model
# =============================================================================
# SMPTE ST 2084 (PQ) EOTF constants.
_PQ_M1 = 2610.0 / 16384.0
_PQ_M2 = 2523.0 / 4096.0 * 128.0
_PQ_C1 = 3424.0 / 4096.0
_PQ_C2 = 2413.0 / 4096.0 * 32.0
_PQ_C3 = 2392.0 / 4096.0 * 32.0


def pq_to_srgb8(rgb48: np.ndarray) -> np.ndarray:
    """
    Convert a 16-bit PQ/BT.2020 RGB frame to an 8-bit sRGB-ish image purely for
    feeding the depth model. This is a perceptual proxy, NOT a colour-accurate
    conversion (the warp uses the original HDR pixels for the visible output).

    Steps: normalise -> inverse-PQ (EOTF) to linear light -> Reinhard tone
    compression (so highlights don't blow out and mid-tones look "normal" to the
    model) -> sRGB-ish gamma -> 8-bit. The BT.2020->709 gamut rotation is
    intentionally skipped: depth estimation is robust to gamut, and skipping it
    keeps this cheap (runs per frame on the CPU preprocess threads).
    """
    # 1) normalise the PQ-encoded code values to [0,1]
    e = rgb48.astype(np.float32) / 65535.0
    np.clip(e, 0.0, 1.0, out=e)

    # 2) inverse-PQ (EOTF): PQ code value -> normalised linear light (1.0 = 10000 nits)
    ep = np.power(e, 1.0 / _PQ_M2)
    num = np.maximum(ep - _PQ_C1, 0.0)
    den = _PQ_C2 - _PQ_C3 * ep
    lin = np.power(num / np.maximum(den, 1e-6), 1.0 / _PQ_M1)  # linear, 0..1

    # 3) Reinhard tone compression to a viewable SDR range. The knee `k` sets the
    #    mid-grey: ~100 nits (lin ~= 0.01 of the 10000-nit range) maps near 0.5.
    k = 0.01
    ld = lin / (lin + k)

    # 4) sRGB-ish OETF (approx gamma 1/2.2) and quantise to 8-bit
    srgb = np.power(np.clip(ld, 0.0, 1.0), 1.0 / 2.2)
    return (srgb * 255.0 + 0.5).astype(np.uint8)


def depth_proxy(img: np.ndarray, hdr: bool) -> np.ndarray:
    """Return the image the depth model should see.

    SDR: the frame itself (unchanged). HDR: an 8-bit sRGB proxy of the 16-bit
    frame so depth quality matches the SDR case. The caller keeps the ORIGINAL
    frame for the warp.
    """
    return pq_to_srgb8(img) if hdr else img


# =============================================================================
# HDR-correct frame decode: BT.2020/PQ -> rgb48le
# =============================================================================
# BUG this works around: PyAV's `frame.to_ndarray(format="rgb48le")` builds a bare
# swscale context that IGNORES the source colorspace and assumes BT.709. On a BT.2020
# (HDR10) source the YUV->RGB matrix is then wrong and the colors are badly corrupted
# (the green channel blows out -> green speckles in the shadows). Verified: PyAV's
# direct conversion differs from ffmpeg by a mean green error of ~6500/65535.
#
# A libavfilter `format` filter DOES honour the frame's colorspace tags, so routing the
# decoded frame through a tiny graph (buffer -> format=rgb48le -> buffersink) gives a
# result that is BIT-EXACT to `ffmpeg -vf format=rgb48le` (mean error 0). SDR (rgb24 /
# BT.709) is unaffected and keeps the fast direct to_ndarray path - this is HDR only.

def make_hdr_rgb48_decoder(stream):
    """Build a per-stream converter `frame -> uint16 (H,W,3) RGB` that does the
    BT.2020/PQ -> rgb48le conversion with the CORRECT matrix via libavfilter.

    The graph is stateful, so create ONE per stream and call it per frame.
    """
    import av  # local import: only the HDR path needs this and av is already a dep
    graph = av.filter.Graph()
    buffer = graph.add_buffer(template=stream)
    fmt = graph.add("format", "rgb48le")
    sink = graph.add("buffersink")
    buffer.link_to(fmt)
    fmt.link_to(sink)
    graph.configure()

    def convert(frame):
        graph.push(frame)
        # `format` is a 1-in-1-out filter, so exactly one frame is available per push.
        return graph.pull().to_ndarray(format="rgb48le")

    return convert


# =============================================================================
# HDR encoder selection
# =============================================================================
def detect_hevc_nvenc_p010() -> bool:
    """True if ffmpeg can actually encode 10-bit HEVC on the GPU (hevc_nvenc,
    p010le). Requires a Pascal+ NVIDIA GPU + recent driver; probed with a tiny
    1-frame encode so we never pick a codec that fails at runtime."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        name = tmp.name
    try:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
               "-c:v", "hevc_nvenc", "-pix_fmt", "p010le", "-frames:v", "1", name]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           stdin=subprocess.DEVNULL, timeout=8)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.remove(name)
        except OSError:
            pass


def select_codec(preference: str, has_static_metadata: bool) -> str:
    """
    Pick the 10-bit HEVC encoder for the HDR output and log the reason.

    preference: "auto" (default) | "nvenc" | "libx265"
      - auto:    libx265 IF the source has HDR10 static metadata (mastering /
                 MaxCLL) so we can preserve it (only libx265 can WRITE it), else
                 hevc_nvenc for speed.
      - nvenc:   hevc_nvenc if available, else libx265. NOTE: hevc_nvenc cannot
                 write mastering/MaxCLL -> static metadata is dropped (a warning
                 is printed). PQ/BT.2020 (i.e. HDR) is still preserved.
      - libx265: always libx265 (full HDR10, but slow at 4K-SBS resolutions).
    """
    pref = (preference or "auto").lower()

    if pref == "libx265":
        print("[HDR] encoder: libx265 (forced via preference).")
        return "libx265"

    if pref == "nvenc":
        if detect_hevc_nvenc_p010():
            if has_static_metadata:
                print("[HDR] encoder: hevc_nvenc (forced). WARNING: mastering-display/"
                      "MaxCLL cannot be written by nvenc and will be DROPPED.")
            else:
                print("[HDR] encoder: hevc_nvenc (forced, fast).")
            return "hevc_nvenc"
        print("[HDR] encoder: libx265 (nvenc requested but 10-bit hevc_nvenc unavailable).")
        return "libx265"

    # auto
    if has_static_metadata:
        print("[HDR] encoder: libx265 (source has HDR10 static metadata -> preserve it).")
        return "libx265"
    if detect_hevc_nvenc_p010():
        print("[HDR] encoder: hevc_nvenc (no static metadata -> GPU speed).")
        return "hevc_nvenc"
    print("[HDR] encoder: libx265 (no nvenc available).")
    return "libx265"


# =============================================================================
# HDR encode output args (replaces the SDR `-pix_fmt yuv420p`)
# =============================================================================
def encode_color_args(codec: str, hdr: bool, master_display: str | None,
                      max_cll: str | None) -> list[str]:
    """
    Build the ffmpeg OUTPUT pixel-format + colour-signalling arguments.

    SDR (hdr=False): unchanged -> ["-pix_fmt", "yuv420p"].
    HDR (hdr=True):  10-bit pixel format + BT.2020/PQ signalling. For libx265,
                     the HDR10 static metadata (mastering display + MaxCLL) is
                     injected via -x265-params when the source provided it
                     (hevc_nvenc cannot carry it, so it is omitted there).

    master_display: x265/ffmpeg string `G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)` or None.
    max_cll:        `maxcll,maxfall` (e.g. "1000,400") or None.
    """
    if not hdr:
        return ["-pix_fmt", "yuv420p"]

    # essential HDR signalling (correct-by-construction for our PQ/BT.2020 output)
    color = ["-color_primaries", "bt2020", "-color_trc", "smpte2084",
             "-colorspace", "bt2020nc", "-color_range", "tv"]

    if codec == "hevc_nvenc":
        # GPU 10-bit. PROBLEM: when the input is a raw RGB pipe (our case), hevc_nvenc writes only the
        # matrix into the HEVC VUI and DROPS color_primaries + transfer (verified: they come out
        # "unknown"). We therefore FORCE the full VUI with the hevc_metadata bitstream filter. Codes per
        # H.265 Table E.x: colour_primaries=9 (bt2020), transfer_characteristics=16 (smpte2084 / PQ),
        # matrix_coefficients=9 (bt2020nc), video_full_range_flag=0 (tv/limited). nvenc still cannot
        # carry mastering-display/MaxCLL (use libx265 for those).
        return ["-pix_fmt", "p010le"] + color + [
            "-bsf:v",
            "hevc_metadata=colour_primaries=9:transfer_characteristics=16:"
            "matrix_coefficients=9:video_full_range_flag=0",
        ]

    # libx265 (CPU): full HDR10 incl. static metadata via -x265-params
    x265 = ["colorprim=bt2020", "transfer=smpte2084", "colormatrix=bt2020nc"]
    if master_display:
        x265 += [f"master-display={master_display}", "hdr-opt=1", "repeat-headers=1"]
    if max_cll:
        x265 += [f"max-cll={max_cll}"]
    return ["-pix_fmt", "yuv420p10le"] + color + ["-x265-params", ":".join(x265)]
