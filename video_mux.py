"""Streaming NV12 encoding with source presentation times and audio passthrough."""

from collections import deque
import codecs
from fractions import Fraction
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Lock, Thread
import time

import av
import PyNvVideoCodec as nvc
import torch


class FragmentedMP4Output:
    def __init__(self, source, output, width, height, codec, time_base, extradata):
        self.container = None
        self.process = None
        self.log = tempfile.TemporaryFile()
        self.log_lock = Lock()
        self.stderr_thread = None
        self.completed = False
        try:
            self.process = subprocess.Popen([
                "ffmpeg", "-hide_banner", "-loglevel", "warning", "-stats", "-y", "-copyts",
                "-f", "mp4", "-i", "pipe:0", "-i", source,
                "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1",
                "-c", "copy", "-fps_mode", "passthrough",
                "-avoid_negative_ts", "disabled", output,
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.stderr_thread = Thread(target=self._relay_stderr, daemon=True)
            self.stderr_thread.start()
            self.container = av.open(self.process.stdin, "w", format="mp4", options={
                "movflags": "empty_moov+default_base_moof+frag_keyframe",
                "frag_duration": "250000",
                "avoid_negative_ts": "disabled",
            })
            # A decoder template prevents PyAV from opening a second software encoder.
            with av.logging.Capture(), av.open(io.BytesIO(extradata), format=codec) as headers:
                self.stream = self.container.add_stream_from_template(headers.streams.video[0])
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"
            self.stream.time_base = time_base
            self.stream.codec_context.time_base = time_base
            self.stream.codec_context.extradata = extradata
            self.time_base = time_base
        except Exception:
            self.close()
            raise

    def _relay_stderr(self):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while data := self.process.stderr.read1(4096):
            with self.log_lock:
                self.log.seek(0, io.SEEK_END)
                self.log.write(data)
            try:
                sys.stderr.write(decoder.decode(data))
                sys.stderr.flush()
            except (OSError, UnicodeError):
                pass

    def _error(self):
        with self.log_lock:
            self.log.seek(0, io.SEEK_END)
            self.log.seek(max(0, self.log.tell() - 8192))
            return self.log.read(8192).decode("utf-8", errors="replace")

    def write_packet(self, data, pts, dts, duration, keyframe):
        packet = av.Packet(data)
        packet.stream = self.stream
        packet.time_base = self.time_base
        packet.pts, packet.dts, packet.duration = pts, dts, duration
        packet.is_keyframe = bool(keyframe)
        try:
            self.container.mux(packet)
        except Exception as exc:
            raise RuntimeError(f"Fragmented MP4 write failed: {self._error()}") from exc

    def finish(self):
        if self.completed:
            return
        try:
            self.container.close()
            self.container = None
            self.process.stdin.close()
            code = self.process.wait(timeout=60)
            self.stderr_thread.join()
        except Exception as exc:
            raise RuntimeError(f"FFmpeg finalization failed: {self._error()}") from exc
        if code:
            raise RuntimeError(f"FFmpeg mux failed ({code}): {self._error()}")
        self.completed = True

    def close(self):
        # Stop the reader first so aborting cannot block while flushing the pipe.
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=10)
        if self.stderr_thread is not None:
            self.stderr_thread.join()
        if self.process is not None:
            self.process.stderr.close()
        if self.container is not None:
            try:
                self.container.close()
            except Exception:
                pass
            self.container = None
        if self.process is not None and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        self.log.close()


class StreamingVideoMux:
    def __init__(self, source, output, width, height, fps, codec, gpu_id):
        self.encoder = None
        self.muxer = None
        self.demuxer = None
        self.finished = False
        self.frames = 0
        self.written = 0
        self.timings = {}
        self.packets = deque()
        self.previous_pts = None
        self.last_source_pts = None
        self.last_source_duration = 0
        self.source_eof = False
        self.encoded_bytes = 0
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error", "-show_streams", "-of", "json", source,
            ], capture_output=True, text=True, check=True, timeout=30)
            streams = json.loads(result.stdout)["streams"]
            video = next(s for s in streams if s["codec_type"] == "video"
                         and not s.get("disposition", {}).get("attached_pic"))
            self.source_end_pts = (
                int(video.get("start_pts", 0)) + int(video["duration_ts"])
                if video.get("duration_ts") is not None else None
            )
            self.demuxer = nvc.CreateDemuxer(filename=source)
            self.time_base = Fraction(self.demuxer.GetTimebaseNum(), self.demuxer.GetTimebaseDen())
            encoder_codec = {"h264_nvenc": "h264", "hevc_nvenc": "hevc"}[codec]
            # TODO: Evaluate B-frames/quality settings separately: DTS/reordering,
            # VFR playback, file size, quality, throughput and VRAM versus bf=0.
            self.encoder = nvc.CreateEncoder(
                width, height, "NV12", False, gpu_id=gpu_id,
                codec=encoder_codec, fps=str(fps), bf="0", preset="P1",
                rc="constqp", constqp="21", gop=str(max(1, round(fps * 2))),
                idrperiod=str(max(1, round(fps * 2))), extra_output_delay="8",
                split_encode_mode="NV_ENC_SPLIT_THREE_FORCED_MODE",
                source_av_format_context=int(self.demuxer.GetAVFormatInputContext()),
            )
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            self.muxer = FragmentedMP4Output(source, output, width, height,
                                            encoder_codec, self.time_base,
                                            self.encoder.GetSequenceParams())
        except Exception:
            self.close()
            raise

    def _scan_source_until(self, until=None):
        # FFmpeg copies audio; this scan only supplies the last video duration.
        if until is not None and self.last_source_pts is not None and self.last_source_pts * self.time_base >= until:
            return
        while not self.source_eof:
            packet = self.demuxer.DemuxNoSkipAudio()
            if not packet.bsl:
                self.source_eof = True
                break
            if not packet.is_video or packet.discardable:
                continue
            if self.last_source_pts is None or packet.pts >= self.last_source_pts:
                self.last_source_pts = int(packet.pts)
                self.last_source_duration = int(packet.duration)
            if until is not None and packet.pts * self.time_base >= until:
                break

    def _write_ready(self):
        while self.packets:
            packet = self.packets[0]
            index = int(packet["timestamp"])
            if index != self.written or index not in self.timings:
                raise RuntimeError("Encoder reordered packets despite bf=0")
            pts, duration = self.timings[index]
            if duration is None:
                break
            self._scan_source_until(pts * self.time_base)
            self.muxer.write_packet(packet["data"], pts, pts, duration,
                                    packet["picture_type"] in (2, 3))
            self.encoded_bytes += len(packet["data"])
            self.packets.popleft()
            del self.timings[index]
            self.written += 1

    def encode(self, frame, pts):
        if self.finished:
            raise RuntimeError("Encode called after finish")
        pts = int(pts)
        if pts == -(1 << 63) or (self.previous_pts is not None and pts <= self.previous_pts):
            raise ValueError("Streaming video requires valid, strictly increasing source PTS")
        if self.previous_pts is not None:
            self.timings[self.frames - 1][1] = pts - self.previous_pts
        self.timings[self.frames] = [pts, None]
        self.previous_pts = pts
        self.frames += 1
        # This binding replaces inputTimeStamp with its own zero-based frame index.
        self.packets.extend(self.encoder.Encode(frame))
        self._write_ready()

    def finish(self):
        if self.finished:
            return
        if not self.frames:
            raise ValueError("No video frames received")
        self.packets.extend(self.encoder.EndEncode())
        self._write_ready()
        self._scan_source_until()
        if self.last_source_duration <= 0 and self.source_end_pts is not None:
            self.last_source_duration = self.source_end_pts - self.previous_pts
        if self.last_source_pts != self.previous_pts or self.last_source_duration <= 0:
            raise ValueError("Cannot determine the source last-frame duration")
        self.timings[self.frames - 1][1] = self.last_source_duration
        self._write_ready()
        if self.written != self.frames or self.timings or self.packets:
            raise RuntimeError("Encoder flush lost video frames")
        self.muxer.finish()
        self.finished = True

    def close(self):
        # Source AVFormatContext must outlive both the muxer and encoder.
        if self.muxer is not None:
            self.muxer.close()
            self.muxer = None
        self.encoder = None
        self.demuxer = None


def nv12_encode_mux_worker(ready_queue, free_buffer_queue, video_path, output_path,
                           fps, codec, ctx, profile_gpu=False):
    torch.cuda.set_device(ctx.gpu_id)
    writer = StreamingVideoMux(video_path, output_path, ctx.W * 2, ctx.H,fps, codec, ctx.gpu_id)
    encode_ms = 0.0
    batches = 0
    try:
        while True:
            try:
                item = ready_queue.get()
            except EOFError:
                return
            if item is None:
                break
            batch, event, count, pts = item
            try:
                if len(pts) != count:
                    raise ValueError("NV12 batch timestamps do not match frame count")
                event.synchronize()
                started = time.perf_counter() if profile_gpu else 0.0
                for index in range(count):
                    writer.encode(batch.frame(index), pts[index])
                if profile_gpu:
                    encode_ms += (time.perf_counter() - started) * 1000.0
                batches += 1
            finally:
                try:
                    free_buffer_queue.put(batch)
                except EOFError:
                    pass
        writer.finish()
        ctx.result_dict["encoded_frames"] = writer.written
        if profile_gpu:
            print(f"NV12 encode/mux: {writer.written} frames, {batches} batches, "
                  f"{encode_ms:.2f} ms")
    finally:
        writer.close()
