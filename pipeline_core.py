from threading import Thread
from queue import Queue,Empty
from fractions import Fraction
from dataclasses import dataclass, field
import threading
import numpy as np
import os



import PyNvVideoCodec as nvc
import time, cv2, subprocess, av, torch
from sbsutils import force_exit , graceful_shutdown 
from depthestimator import DepthEstimator
from gpu_converter import RGBSBSConverter, NV12SBSConverter, NV12CudaBatch
# --- HDR (10-bit) --- all HDR-specific logic lives in the isolated hdr module; imported by
# function name so it never clashes with the `hdr` boolean flag threaded through the workers.
from hdr import depth_proxy, pixel_max, pixel_dtype, pipe_in_pix_fmt, encode_color_args, make_hdr_rgb48_decoder



@dataclass
class PipelineContext:
    """
    Central configuration and state container for the conversion pipeline.

    Holds:
        - All runtime parameters (queues, thread counts, image sizes)
        - All worker thread instances
        - Monitoring and debug references
        - Timing info for profiling

    Each static method below represents one stage in the pipeline.
    """
    
    # input parameters
    video_path: str
    input_type: str
    estimator: DepthEstimator
    SBSConverter: RGBSBSConverter | NV12SBSConverter
    output_path: str
    batch_size: int
    in_queue: int
    r_queue: int
    s_queue: int
    p_queue: int
    n_preprocess: int
    n_processors: int
    n_savers: int
    n_feeders: int
    model_name: str
    codec: str
    version: str
    autocast: str | None = None
    infer_accum_batches: int | None = None
    
    # calculated fields
    H: int = 0
    W: int = 0
    fps: float = 0.0

    # queues
    raw_queue: Queue = field(default=None)
    input_queue: Queue = field(default=None)
    save_queue: Queue = field(default=None)
    process_queue: Queue = field(default=None)

    # streams
    feeders: list[Thread] = field(default_factory=list)
    pre_workers: list[Thread] = field(default_factory=list)
    gpu_worker: Thread | None = None
    processors: list[Thread] = field(default_factory=list)
    savers: list[Thread] = field(default_factory=list)
    
    # converter settings
    depth_scale: float = 1.0
    depth_offset: float = 0.0
    switch_sides: bool = False
    symetric: bool = False
    blur_radius: int = 19
    

    # debugging/monitors
    debug: bool = False
    result_dict: dict = field(default_factory=dict)
    mem_mon: object | None = None
    q_mon: object | None = None

    # timings
    t_start: float = 0.0
    t_end: float = 0.0
    
    # etc
    fatal_error: bool = False
    video_quality: str = "medium"
    direct_nv12: bool = False

    # --- HDR (10-bit) --- optional true-10-bit HDR output (video input only). When hdr=False the
    hdr: bool = False
    hdr_encoder: str = "auto"
    master_display: str | None = None
    max_cll: str | None = None
    
    @staticmethod
    def create_nv12_encoder(width: int, height: int, fps: float, codec: str):
        """
        Create a GPU-input NV12 encoder using the verified PyNvVideoCodec contract.
        """
        codec_map = {
            "h264_nvenc": "h264",
            "hevc_nvenc": "hevc",
        }

        try:
            encoder_codec = codec_map[codec]
        except KeyError:
            raise ValueError(f"Direct NV12 encoding does not support codec: {codec}")
            
        return nvc.CreateEncoder(
            width,
            height,
            "NV12",
            False,
            gpu_id=0,
            codec=encoder_codec,
            fps=str(fps),
            bf="1",
            preset="P1",
            rc="constqp",
            constqp="22",
            gop=str(round(fps * 2)),
            idrperiod=str(round(fps * 2)),
            repeatspspps="1",
        )
                
    @staticmethod
    def nv12_encode_mux_worker_thread(
        ready_queue: Queue,
        free_buffer_queue: Queue,
        video_path: str,
        output_path: str,
        fps: float,
        codec: str,
        ctx,
    ):
        """
        Encode ready CUDA NV12 batches and mux them with the original audio.
        """
        bitstream_format = {
            "h264_nvenc": "h264",
            "hevc_nvenc": "hevc",
        }.get(codec)

        if bitstream_format is None:
            raise ValueError(f"Unsupported direct NV12 codec: {codec}")

        fps_q = Fraction(str(fps)).limit_denominator(1001)
        fps_text = f"{fps_q.numerator}/{fps_q.denominator}"
        time_base = f"{fps_q.denominator}/{fps_q.numerator}"

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "info",

            # 1) Encoded elementary video
            "-r", fps_text,
            "-f", bitstream_format,
            "-i", "pipe:0",

            # 2) Original audio
            "-i", video_path,

            "-map", "0:v:0",
            "-map", "1:a:0?",

            # 3) Assign timestamps without re-encoding
            "-c:v", "copy",
            "-bsf:v", f"setts=pts=N:dts=N:duration=1:time_base={time_base}",

            "-c:a", "copy",
            "-shortest",
            output_path,
        ]

        encoder = PipelineContext.create_nv12_encoder(
            width=ctx.W * 2,
            height=ctx.H,
            fps=fps,
            codec=codec,
        )

        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        stats = {
            "queue_wait_ms": 0.0,
            "event_wait_ms": 0.0,
            "encode_ms": 0.0,
            "pipe_write_ms": 0.0,
            "buffer_return_ms": 0.0,
            "frames": 0,
            "packets": 0,
            "bytes": 0,
            "batches": 0,
        }

        try:
            while True:
                t0 = time.perf_counter()

                try:
                    item = ready_queue.get()
                except EOFError:
                    break

                stats["queue_wait_ms"] += (time.perf_counter() - t0) * 1000.0

                if item is None:
                    break

                nv12_batch, ready_event, frame_count = item

                if not isinstance(nv12_batch, NV12CudaBatch):
                    raise TypeError("ready_queue returned an invalid NV12 buffer")

                try:
                    # 1) Wait for DIBR completion
                    t0 = time.perf_counter()
                    ready_event.synchronize()
                    stats["event_wait_ms"] += (time.perf_counter() - t0) * 1000.0

                    # 2) Measure Encode and pipe write separately
                    for frame_index in range(frame_count):
                        t0 = time.perf_counter()
                        packets = encoder.Encode(nv12_batch.frame(frame_index))
                        stats["encode_ms"] += (time.perf_counter() - t0) * 1000.0

                        stats["frames"] += 1

                        for packet in packets:
                            data = packet["data"]

                            stats["packets"] += 1
                            stats["bytes"] += len(data)

                            t0 = time.perf_counter()
                            proc.stdin.write(data)
                            stats["pipe_write_ms"] += (time.perf_counter() - t0) * 1000.0

                    stats["batches"] += 1

                finally:
                    t0 = time.perf_counter()

                    try:
                        free_buffer_queue.put(nv12_batch)
                    except EOFError:
                        pass

                    stats["buffer_return_ms"] += (time.perf_counter() - t0) * 1000.0

            # 4) Flush delayed encoder packets
            t0 = time.perf_counter()
            tail_packets = encoder.EndEncode()
            stats["encode_ms"] += (time.perf_counter() - t0) * 1000.0

            for packet in tail_packets:
                data = packet["data"]
                stats["packets"] += 1
                stats["bytes"] += len(data)

                t0 = time.perf_counter()
                proc.stdin.write(data)
                stats["pipe_write_ms"] += (time.perf_counter() - t0) * 1000.0
            frames = max(stats["frames"], 1)

            print("\n===== NV12 Encoder Worker Profile =====")
            print(f"Frames:              {stats['frames']}")
            print(f"Batches:             {stats['batches']}")
            print(f"Packets:             {stats['packets']}")
            print(f"Encoded bytes:       {stats['bytes']}")
            print(f"Ready queue wait:    {stats['queue_wait_ms']:.2f} ms")
            print(f"CUDA event wait:     {stats['event_wait_ms']:.2f} ms")
            print(f"Encode total:        {stats['encode_ms']:.2f} ms")
            print(f"Pipe write total:    {stats['pipe_write_ms']:.2f} ms")
            print(f"Buffer return:       {stats['buffer_return_ms']:.2f} ms")
            print(f"Event wait/frame:    {stats['event_wait_ms'] / frames:.3f} ms")
            print(f"Encode/frame:        {stats['encode_ms'] / frames:.3f} ms")
            print(f"Pipe write/frame:    {stats['pipe_write_ms'] / frames:.3f} ms")
            print("===============================\n")

        except Exception as exc:
            print(f"NV12 encode/mux worker failed - {exc}")
            ctx.fatal_error = True
            graceful_shutdown(ctx)

        finally:
            try:
                del encoder
            except Exception:
                pass

            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass

            try:
                return_code = proc.wait()
            except Exception:
                proc.kill()
                return

            if return_code != 0 and not ctx.fatal_error:
                print(f"FFmpeg mux failed with code {return_code}")
                ctx.fatal_error = True

    @staticmethod
    def video_worker_thread(save_queue: Queue,video_path, output_path: str, width: int, height: int, fps: float,codec: str,crf: int, cq: int,ctx):
        """
        Writes SBS frames from queue to video via FFmpeg, preserving audio if present.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner", 
            "-loglevel", "info",  # quiet removes all logs
            # Input video raw (pictures from pipe)
            "-f", "rawvideo",
            "-pix_fmt", pipe_in_pix_fmt(ctx.hdr),  # rgb48le (16-bit) when hdr, else rgb24
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-i", video_path,
            # mapping
            "-map", "0:v:0",
            "-map", "1:a:0?",
            # video
            "-c:v", codec,
            ] + (["-crf", str(crf)] if codec in ("libx264", "libx265") else ["-rc:v", "vbr", "-cq:v", str(cq), "-b:v", "0"]) + [
            ] + encode_color_args(codec, ctx.hdr, ctx.master_display, ctx.max_cll) + [
            "-c:a", "copy",
            output_path
        ]

        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        buffer = {}          # dict: index -> np.ndarray (image)
        next_index = 0       # the next frame index to be written
        finished = False

        try:
            while True:
                try:
                    item = save_queue.get()
                except EOFError:
                    return
                if item is None:
                    # Note: sentinel has arrived - there will be no more new batches.
                    finished = True
                else:
                    indices, sbs_images = item

                    for idx, image in zip(indices, sbs_images):
                        buffer[idx] = image 

                # We try to write all available frames in a row, starting from next_index
                while next_index in buffer:
                    try:
                        proc.stdin.write(buffer[next_index].tobytes())
                    except Exception as e:
                        print(f"FFmpeg pipe broken - {e}")
                        ctx.fatal_error = True
                        graceful_shutdown(ctx)
                        return
                        
                    del buffer[next_index]
                    next_index += 1

                if finished and not buffer:
                    break

        finally:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            # Wait for ffmpeg to finish flushing the encoder and writing the MP4 trailer.
            # A short timeout may truncate large/slow encodes (e.g. 4K HDR).
            try:
                proc.wait()
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    
    @staticmethod
    def save_worker_thread(save_queue: Queue, output_dir: str,input_type: str | None = None):
        """
        Saves SBS images from queue to disk (PNG). Works for folder and i2i.
        """
        if input_type == "folder":
            os.makedirs(output_dir, exist_ok=True)

        while True:
            try:
                item = save_queue.get()
            except EOFError:
                return
            if item is None:
                break

            names, sbs_images = item
            for name, image in zip(names, sbs_images):
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if input_type == "folder":
                    save_path = os.path.join(output_dir, f"{name}.png")
                else:  # i2i (single image)
                    save_path = output_dir  

                cv2.imwrite(save_path, image_bgr)
    
    @staticmethod
    def process_worker(process_queue: Queue, save_queue: Queue):
        """Wait for async D2H and forward completed SBS batches."""
        while True:
            try:
                item = process_queue.get()
            except EOFError:
                return

            if item is None:
                break

            keys, sbs_cpu, event = item
            event.synchronize()

            try:
                save_queue.put((keys, sbs_cpu.numpy()))
            except EOFError:
                return
                
    @staticmethod
    def gpu_worker_loop(
        estimator,
        processor,
        SBSConverter,
        queue: Queue,
        process_queue: Queue,
        save_queue: Queue,
        model_name,
        n_preprocess: int,
        H_orig: int,
        W_orig: int,
        n_processors: int,
        cudnn_benchmark: bool,
        input_type: str,
        autocast: str | None,
        infer_accum_batches: int,
        depth_scale: float,
        depth_offset: float,
        switch_sides: bool,
        symetric: bool,
        blur_radius: int,
        compiled_batch_size: int,
        fps: float,
        codec: str,
        profile_gpu: bool = True,
    ):
        """Run GPU preprocessing, depth inference and SBS conversion."""

        device = torch.device("cuda")
        direct_nv12 = input_type == "video" and isinstance(SBSConverter, NV12SBSConverter)
        copy_stream = None if direct_nv12 else torch.cuda.Stream()

        done_count = 0
        pending_items = []

        # process_queue becomes the free NV12 buffer pool in direct mode.
        if direct_nv12:
            buffer_count = 3

            if process_queue.maxsize and process_queue.maxsize < buffer_count:
                raise ValueError("process_queue capacity must be at least 3 for direct NV12")

            for _ in range(buffer_count):
                process_queue.put(
                    NV12CudaBatch(
                        SBSConverter.Bmax,
                        H_orig,
                        W_orig * 2,
                    )
                )

        profile = {
            "preprocess_ms": 0.0,
            "inference_ms": 0.0,
            "dibr_ms": 0.0,
            "buffer_wait_ms": 0.0,
            "batches": 0,
            "frames": 0,
        }

        stage_events = []
        dibr_events = []

        def send_rgb_result(keys, sbs_gpu):
            sbs_cpu = torch.empty(
                sbs_gpu.shape,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )

            compute_stream = torch.cuda.current_stream(sbs_gpu.device)

            with torch.cuda.stream(copy_stream):
                copy_stream.wait_stream(compute_stream)
                sbs_gpu.record_stream(copy_stream)
                sbs_cpu.copy_(sbs_gpu, non_blocking=True)

                event = torch.cuda.Event()
                event.record(copy_stream)

            try:
                process_queue.put((keys, sbs_cpu, event))
            except EOFError:
                return False

            return True

        def acquire_nv12_buffer():
            t0 = time.perf_counter()

            try:
                output = process_queue.get()
            except EOFError:
                return None

            profile["buffer_wait_ms"] += (time.perf_counter() - t0) * 1000.0

            if not isinstance(output, NV12CudaBatch):
                raise TypeError("free NV12 queue returned an invalid object")

            return output

        def send_nv12_result(
            nv12_batch: NV12CudaBatch,
            frame_count: int,
        ) -> bool:
            # The event is recorded after all Torch/CuPy work for this buffer.
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(device))

            try:
                save_queue.put((nv12_batch, ready_event, frame_count))
            except EOFError:
                return False

            return True

        def flush_pending():
            if not pending_items:
                return True

            chunk_sizes = [item[2].shape[0] for item in pending_items]
            batches = [item[2] for item in pending_items]

            if profile_gpu:
                pre_start = torch.cuda.Event(enable_timing=True)
                pre_end = torch.cuda.Event(enable_timing=True)
                infer_end = torch.cuda.Event(enable_timing=True)
                pre_start.record()

            # 1) Prepare the accumulated input batch
            if isinstance(batches[0], torch.Tensor):
                base_gpu = batches[0] if len(batches) == 1 else torch.cat(batches, dim=0)
            else:
                base_np = np.concatenate(batches, axis=0)

                if base_np.dtype != np.uint8:
                    raise TypeError("GPU preprocessing currently supports uint8 SDR only")

                base_gpu = torch.from_numpy(base_np).to(device, non_blocking=True)

            inputs = processor(
                images=base_gpu,
                return_tensors="pt",
                device=device,
                input_data_format="channels_last",
            )

            if profile_gpu:
                pre_end.record()

            # 2) Run depth inference
            depth_batch = estimator.predict_batch_tensor(
                inputs.pixel_values,
                cudnn_benchmark,
                compiled_batch_size * infer_accum_batches,
                target_size=(H_orig, W_orig),
                model_name=model_name,
                autocast=autocast,
            ).float()

            if profile_gpu:
                infer_end.record()
                stage_events.append(
                    (
                        pre_start,
                        pre_end,
                        infer_end,
                        base_gpu.shape[0],
                    )
                )

            # 3) Convert each original chunk
            start = 0

            for item, chunk_size in zip(pending_items, chunk_sizes):
                indices, names, _base = item
                end = start + chunk_size

                base_chunk = base_gpu[start:end]
                depth_chunk = depth_batch[start:end]
                start = end

                nv12_output = None

                if direct_nv12:
                    nv12_output = acquire_nv12_buffer()

                    if nv12_output is None:
                        return False

                    SBSConverter.set_output_buffer(nv12_output)

                if profile_gpu:
                    dibr_start = torch.cuda.Event(enable_timing=True)
                    dibr_end = torch.cuda.Event(enable_timing=True)
                    dibr_start.record()

                sbs_result = SBSConverter.process(
                    base_chunk,
                    depth_chunk,
                    depth_scale,
                    depth_offset,
                    switch_sides,
                    blur_radius,
                    symetric,
                )

                if profile_gpu:
                    dibr_end.record()
                    dibr_events.append((dibr_start, dibr_end, chunk_size))

                if direct_nv12:
                    if sbs_result is not nv12_output:
                        raise RuntimeError("NV12 converter returned the wrong output buffer")

                    if not send_nv12_result(sbs_result, chunk_size):
                        return False
                else:
                    keys = indices if input_type == "video" else names

                    if not send_rgb_result(keys, sbs_result):
                        return False

            profile["batches"] += 1
            profile["frames"] += base_gpu.shape[0]

            pending_items.clear()
            return True

        expected_done_count = 1 if input_type == "video" else n_preprocess

        # 4) Consume incoming batches
        while True:
            try:
                item = queue.get()
            except EOFError:
                return

            if item is None:
                done_count += 1

                if done_count == expected_done_count:
                    if not flush_pending():
                        return

                    break

                continue

            pending_items.append(item)

            if len(pending_items) >= infer_accum_batches:
                if not flush_pending():
                    return

        # 5) Collect profiling results
        if profile_gpu:
            torch.cuda.synchronize()

            for pre_start, pre_end, infer_end, frame_count in stage_events:
                profile["preprocess_ms"] += pre_start.elapsed_time(pre_end)
                profile["inference_ms"] += pre_end.elapsed_time(infer_end)

            for dibr_start, dibr_end, frame_count in dibr_events:
                profile["dibr_ms"] += dibr_start.elapsed_time(dibr_end)

            if profile["frames"]:
                frames = profile["frames"]

                print("\n===== GPU Stage Profile =====")
                print(f"Frames:              {frames}")
                print(f"GPU batches:         {profile['batches']}")
                print(f"Preprocess total:    {profile['preprocess_ms']:.2f} ms")
                print(f"Inference total:     {profile['inference_ms']:.2f} ms")
                print(f"DIBR total:          {profile['dibr_ms']:.2f} ms")
                print(f"NV12 buffer wait:    {profile['buffer_wait_ms']:.2f} ms")
                print(f"Preprocess/frame:    {profile['preprocess_ms'] / frames:.3f} ms")
                print(f"Inference/frame:     {profile['inference_ms'] / frames:.3f} ms")
                print(f"DIBR/frame:          {profile['dibr_ms'] / frames:.3f} ms")
                print(f"Buffer wait/frame:   {profile['buffer_wait_ms'] / frames:.3f} ms")

                compute_ms = (
                    profile["preprocess_ms"]
                    + profile["inference_ms"]
                    + profile["dibr_ms"]
                )

                print(f"Measured compute FPS: {frames * 1000 / compute_ms:.2f}")
                print("=============================\n")

        # Encoder worker receives its sentinel later through save_queue.
        if not direct_nv12:
            for _ in range(n_processors):
                try:
                    process_queue.put(None)
                except EOFError:
                    return
                    
            
    @staticmethod
    def preprocess_worker(raw_queue: Queue, batch_size: int, processor, device, input_queue: Queue, hdr=False):
        """Collect decoded frames into NumPy batches. GPU preprocessing is done in gpu_worker_loop."""

        batch_idx, batch_imgs, batch_names = [], [], []

        def flush_batch():
            if not batch_imgs:
                return True

            base_np = np.stack(batch_imgs)

            try:
                input_queue.put((list(batch_idx), list(batch_names), base_np))
            except EOFError:
                return False

            batch_idx.clear()
            batch_imgs.clear()
            batch_names.clear()
            return True

        while True:
            try:
                item = raw_queue.get()
            except EOFError:
                return

            if item is None:
                if not flush_batch():
                    return

                try:
                    input_queue.put(None)
                except EOFError:
                    return

                break

            if len(item) == 2:
                idx, img = item
                name = None
            elif len(item) == 3:
                idx, img, name = item
            else:
                raise ValueError("Unknown item format")

            batch_idx.append(idx)
            batch_imgs.append(img)
            batch_names.append(name)

            if len(batch_imgs) >= batch_size:
                if not flush_batch():
                    return


    @staticmethod
    def video_feeder(video_path, input_queue: Queue, batch_size: int, result_dict, max_frames: int | None = None, gpu_id: int = 0):
        """Decode video with NVDEC and send CUDA RGB batches directly to gpu_worker_loop."""

        decoder = nvc.SimpleDecoder(
            video_path,
            gpu_id=gpu_id,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.RGB,
        )

        total_frames = len(decoder)
        idx = 0

        while idx < total_frames:
            request_size = min(batch_size, total_frames - idx)

            if max_frames is not None:
                request_size = min(request_size, max_frames - idx)
                if request_size <= 0:
                    break

            frames = decoder.get_batch_frames(request_size)
            if not frames:
                break

            base_gpu = torch.stack([torch.from_dlpack(frame) for frame in frames])
            indices = list(range(idx, idx + len(frames)))

            try:
                input_queue.put((indices, [None] * len(indices), base_gpu))
            except EOFError:
                return

            idx += len(frames)

        result_dict["frames"] = idx

        try:
            input_queue.put(None)
        except EOFError:
            return
        
    @staticmethod
    def image_folder_feeder(folder_path, raw_queue, file_list,result_dict=None):
        """
        Reads images from assigned folder and pushes them into queue.
        """
        idx = 0
        for fname in file_list:
            path = os.path.join(folder_path, fname)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            name_stem, _ = os.path.splitext(fname)
            try:
                raw_queue.put((None, img, name_stem))
            except EOFError:
                return                
            idx += 1
            
        if result_dict:
            result_dict["frames"] = result_dict.get("frames", 0) + idx
        



    
