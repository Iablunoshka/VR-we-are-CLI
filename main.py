import os, sys

path = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to the path so we can import local modules
if path not in sys.path:
    sys.path.append(path)    
    
from threading import Thread
from queue import Queue, Empty, Full
from natsort import natsorted
import numpy as np
import time, cv2, signal, torch
from depthestimator import DepthEstimator 
from gpu_converter import DIBRCore, RGBSBSConverter, NV12SBSConverter
from pipeline_core import PipelineContext
from sbsutils import (
    clean_output_pngs,
    debug_report,
    detect_nvenc_support,
    force_exit,
    load_preset,
    merge_with_preset,
    validate_config,
)


class CloseableQueue(Queue):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.closed = False

    def close(self):
        with self.mutex:
            self.closed = True
            # let's wake up everyone who's waiting
            self.not_empty.notify_all()
            self.not_full.notify_all()

    def get(self, block=True, timeout=None):
        with self.not_empty:
            # waiting for something to appear
            while not self._qsize():
                # but if the queue is already closed, exit
                if self.closed:
                    raise EOFError("Queue closed")
                if not block:
                    raise Empty
                if timeout is None:
                    self.not_empty.wait()
                else:
                    endtime = time.time() + timeout
                    while not self._qsize():
                        remaining = endtime - time.time()
                        if remaining <= 0.0:
                            raise Empty
                        self.not_empty.wait(remaining)
                    break
            item = self._get()
            self.not_full.notify()
            return item

    def put(self, item, block=True, timeout=None):
        with self.not_full:
            if self.closed:
                raise EOFError("Queue closed")
            while self._qsize() >= self.maxsize > 0:
                if self.closed:
                    raise EOFError("Queue closed")
                if not block:
                    raise Full
                if timeout is None:
                    self.not_full.wait()
                else:
                    endtime = time.time() + timeout
                    while self._qsize() >= self.maxsize > 0:
                        remaining = endtime - time.time()
                        if remaining <= 0.0:
                            raise Full
                        self.not_full.wait(remaining)
                    break
            self._put(item)
            self.not_empty.notify()

def init_pipeline(
    version: str,
    video_path: str,
    estimator: DepthEstimator,
    output_path: str,
    *,
    batch_size: int = 5,
    in_queue: int = 16,
    r_queue: int = 16,
    s_queue: int = 16,
    p_queue: int = 16,
    n_preprocess: int = 2,
    n_processors: int = 8,
    n_savers: int = 1,
    n_feeders: int = 1,
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
    codec: str = "libx264",
    input_type: str = "video",
    debug: bool = False,
    depth_scale: float = 1.0,
    depth_offset: float = 0.0,
    crop_size: int = 0,
    switch_sides: bool = False,
    symetric: bool = False,
    blur_radius: int = 19,
    video_quality: str = "medium",
    autocast: str = None,
    infer_accum_batches: int = None,
    hdr: bool = False,
    hdr_encoder: str = "auto",
    master_display: str = None,
    max_cll: str = None
    ) -> PipelineContext:
        
    """
    Initialize the multistage conversion pipeline.
    """

    validate_config({
        "input_type": input_type,
        "video_path": video_path,
        "output_path": output_path,
        "batch_size": batch_size,
        "in_queue": in_queue,
        "r_queue": r_queue,
        "s_queue": s_queue,
        "p_queue": p_queue,
        "n_preprocess": n_preprocess,
        "n_processors": n_processors,
        "n_savers": n_savers,
        "n_feeders": n_feeders,
        "codec": codec if input_type == "video" else None,
        "video_quality": video_quality if input_type == "video" else None,
        "infer_accum_batches": infer_accum_batches if input_type != "i2i" else None,
        "crop_size": crop_size,
        "blur_radius": blur_radius,
        "hdr": hdr,
    })
        
    # --- Detect and prepare input source ---
    # Depending on input_type, determine dimensions, FPS, and I/O codec, crf
    frame_count = 0
    if input_type == "video":
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        detected_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if detected_frames and np.isfinite(detected_frames):
            frame_count = max(0, int(round(detected_frames)))
        if not fps or np.isnan(fps):
            fps = 30.0
            print("[warn] FPS autodetect failed - defaulting to 30")
        cap.release()
        if not ok:
            raise RuntimeError("Failed to read first frame")
        H, W = frame.shape[:2]
        cudnn_benchmark = True
       
        # Encoder selection
        if not codec or codec == "auto":
            if W*2 <= 4096 and H <= 4096 and detect_nvenc_support():
                codec = "h264_nvenc"
                print("NVENC available — using GPU encoder (h264_nvenc).")
            else:
                codec = "libx264"
                print("Using CPU encoder — libx264")
        elif codec == "h264_nvenc":
            if W*2 <= 4096 and H <= 4096 and detect_nvenc_support():
                pass
            else:
                codec = "hevc_nvenc"
                print("h264_nvenc not available — using encoder (hevc_nvenc).")

        # Definitions of quality
        if video_quality == "low":
            crf, cq = 30, 35
        elif video_quality == "medium":
            crf, cq = 26, 31
        elif video_quality == "high":  
            crf, cq = 23, 28
            
    elif input_type == "folder":
        files = natsorted([f for f in os.listdir(video_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        if not files:
            raise RuntimeError("No images found in folder")

        # test reading for sizes
        first = cv2.imread(os.path.join(video_path, files[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError("Failed to read first image")
        fps = 0
        codec = "png"
        H, W = first.shape[:2]
        cudnn_benchmark = True
    else:
        cudnn_benchmark = False
        first = cv2.imread(video_path, cv2.IMREAD_COLOR)
        if first is None:
            raise RuntimeError(f"Failed to read image {video_path}")
        fps = 0
        H, W = first.shape[:2]   
    
    direct_nv12 = (
        input_type == "video"
        and not hdr
        and codec in ("h264_nvenc", "hevc_nvenc")
    )

    validate_config({
        "input_type": input_type,
        "frame_width": W,
        "frame_height": H,
        "direct_nv12": direct_nv12,
    })

    estimator.load_model(model_name,cudnn_benchmark)
    processor = estimator.processor
    device = estimator.device
    gpu_id = torch.cuda.current_device() if device.type == "cuda" else 0
    
    if device.type == "cuda":
        device = torch.device("cuda", gpu_id)
        estimator.device = device

    autocast = estimator.resolve_autocast_mode(autocast)
    infer_accum_batches = max(1, int(infer_accum_batches or 1)) if estimator.device.type == "cuda" else 1

    core = DIBRCore(H, W, batch_size, device=device)

    if direct_nv12:
        SBSConverter = NV12SBSConverter(core)
    else:
        SBSConverter = RGBSBSConverter(core)
    
    # Create thread-safe queues to connect pipeline stages
    raw_q = CloseableQueue(maxsize=r_queue)  # feeders → preprocessors
    inp_q = CloseableQueue(maxsize=in_queue) # preprocessors → GPU inference
    proc_q = CloseableQueue(maxsize=p_queue) # GPU inference → SBS processors
    save_q = CloseableQueue(maxsize=s_queue) # processors → savers
    

    ctx = PipelineContext(
        version=version,
        video_path=video_path,
        estimator=estimator,
        SBSConverter=SBSConverter,
        output_path=output_path,
        batch_size=batch_size,
        in_queue=in_queue,
        r_queue=r_queue,
        s_queue=s_queue,
        p_queue=p_queue,
        n_preprocess=n_preprocess,
        n_processors=n_processors,
        n_savers=n_savers,
        model_name=model_name,
        codec=codec, debug=debug,
        H=H, W=W, fps=fps, frame_count=frame_count,
        raw_queue=raw_q,
        input_queue=inp_q,
        save_queue=save_q,
        process_queue=proc_q,
        depth_scale=depth_scale,
        depth_offset=depth_offset,
        crop_size=crop_size,
        switch_sides=switch_sides,
        symetric=symetric,
        blur_radius=blur_radius,
        input_type=input_type,
        n_feeders=n_feeders,
        video_quality=video_quality,
        autocast=autocast,
        infer_accum_batches=infer_accum_batches,
        gpu_id=gpu_id,
        direct_nv12=direct_nv12,
        hdr=hdr, master_display=master_display, max_cll=max_cll  
    )
    
    #if debug:
    #    print("\n[Pipeline Configuration]")
    #    for f in fields(ctx):
    #        name = f.name
    #        value = getattr(ctx, name)
    #        if isinstance(value, (list, dict)) or "queue" in name or "worker" in name:
    #            continue
    #        print(f"{name:>15}: {value}")
    
    max_frames = None

    def make_worker(target, *args):
        return Thread(target=PipelineContext.worker_entry, args=(ctx, target, *args))
        
    # --- Build and assign pipeline worker threads ---
    # Feeders -> Preprocessors -> GPU inference -> Processors -> Savers

    if input_type == "video":
        ctx.result_dict = {"frames": 0}
        ctx.feeders = [make_worker(PipelineContext.video_feeder, ctx.video_path, ctx.input_queue, ctx.batch_size, ctx.result_dict, max_frames, ctx.gpu_id)]
    elif input_type == "folder":
        ctx.result_dict = {"frames": 0} 
        chunks = np.array_split(files, n_feeders)
        for chunk in chunks:
            ctx.feeders.append(make_worker(
                PipelineContext.image_folder_feeder,
                video_path, raw_q, list(chunk), ctx.result_dict, ctx.result_lock))
    else:
        ctx.result_dict = {"frames": 1}
        ctx.feeders = [make_worker(
            PipelineContext.image_folder_feeder,
            os.path.dirname(video_path), raw_q, [os.path.basename(video_path)])]
    
    for _ in range(n_preprocess):
        ctx.pre_workers.append(make_worker(
            PipelineContext.preprocess_worker,
            raw_q, batch_size, estimator.processor, estimator.device, inp_q))
        
    ctx.gpu_worker = make_worker(
        PipelineContext.gpu_worker_loop,
        estimator, processor, SBSConverter, inp_q, proc_q, save_q,
        model_name, n_preprocess, H, W, n_processors,
        cudnn_benchmark, input_type, ctx.autocast,
        ctx.infer_accum_batches, ctx.depth_scale,
        ctx.depth_offset, ctx.crop_size, ctx.switch_sides,
        ctx.symetric, ctx.blur_radius, ctx.batch_size, fps, codec, ctx.gpu_id, ctx.debug,
    )
                            
    if direct_nv12:
        ctx.processors = []
        ctx.savers = [make_worker(PipelineContext.nv12_encode_worker,save_q,proc_q,video_path,output_path,fps,codec,ctx,)]
    else:
        ctx.processors = [make_worker(PipelineContext.process_worker, proc_q, save_q)for _ in range(n_processors)]

        if input_type == "video":
            ctx.savers = [make_worker(PipelineContext.video_worker_thread,save_q, video_path, output_path, W * 2, H, fps, codec, crf, cq, ctx)]
            
        elif input_type == "folder":
            ctx.savers = [make_worker(PipelineContext.save_worker_thread, save_q, output_path, input_type)for _ in range(n_savers)]
            
        else:
            ctx.savers = [make_worker(PipelineContext.save_worker_thread, save_q, output_path, input_type)]
        

    # --- Optional monitoring tools for debugging ---
    if debug:
        from monitor import MemoryMonitor, QueueMonitor
        ctx.mem_mon = MemoryMonitor(interval=0.1, include_children=True, gpu_id=ctx.gpu_id)
        ctx.q_mon = QueueMonitor(
            queues={"raw": raw_q, "input": inp_q, "process": proc_q, "save": save_q},
            interval=0.25
        )

    return ctx
    
    
def run_pipeline(ctx: PipelineContext):
    """
    Launch all pipeline threads and manage their lifecycle.
    """
    
    # monitors
    if ctx.debug:
        ctx.mem_mon.start()
        ctx.q_mon.start()

    ctx.t_start = time.perf_counter()

    # start
    for t in ctx.feeders: t.start()
    for t in ctx.pre_workers: t.start()
    ctx.gpu_worker.start()
    for t in ctx.processors: t.start()
    for t in ctx.savers: t.start()

    # join and the distribution of "poison pills"
    
    # Wait for feeders to finish producing frames
    for t in ctx.feeders:
        t.join()
        
    # Signal preprocess workers to stop (send poison pills)
    if not getattr(ctx, "fatal_error", False):
        for _ in range(ctx.n_preprocess):
            ctx.raw_queue.put(None)
            
    for t in ctx.pre_workers:
        t.join()
    
    # Wait for GPU inference and CPU processing to complete
    ctx.gpu_worker.join()
    for t in ctx.processors:
        t.join()
    
    # Signal saver threads to stop
    if not getattr(ctx, "fatal_error", False):
        for _ in range(ctx.n_savers):
            ctx.save_queue.put(None)
    for t in ctx.savers:
        t.join()

    ctx.t_end = time.perf_counter()
    print(f"Process time: {ctx.t_end - ctx.t_start:.4f} sec")

    if ctx.fatal_error == True:
        sys.exit(1)
 
# --- Command-line interface ---
if __name__ == "__main__":
    import argparse
    version = "1.1.6"
    parser = argparse.ArgumentParser(
        description="VR we are! CLI pipeline (video -> 3D SBS video, "
                    "folder -> batch of images, i2i -> single/multiple images one-by-one)."
    )
    parser.add_argument("--version","-v", action="version", version=f"VR We Are {version} (CLI)")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help=("Path to input.\n"
                              "  video: path to video file\n"
                              "  folder: path to folder with images (same resolution)\n"
                              "  i2i: single image file OR folder with mixed-resolution images"))
    parser.add_argument("--output", "-o", type=str, required=True,
                        help=("Path to output.\n"
                              "  video: output video file (e.g. out.mp4)\n"
                              "  folder: output directory for processed images\n"
                              "  i2i: if input is a single image -> output file; "
                              "if input is a folder -> output directory"))
    parser.add_argument("--batch-size", "-b", type=int, default=None,
                        help="Batch size for processing (video/folder modes only)")
    parser.add_argument("--model", "-m", type=str, default=None,
                    choices=DepthEstimator.AVAILABLE_MODELS,
                    help="Which depth model to use")
    parser.add_argument("--codec", "-c", type=str,
                    choices=["libx264", "libx265", "h264_nvenc", "hevc_nvenc"],
                    default=None,
                    help="Codec for output video (CPU: libx264/libx265, GPU: h264_nvenc/hevc_nvenc)")
    parser.add_argument("--quality", type=str,
            choices=["low", "medium", "high"],default=None,
            help="Output video quality (changes the values of -crf or -cq in ffmpeg)") 
    parser.add_argument("--autocast",type=str,choices=["auto", "none", "float16", "bfloat16"],default=None,
            help="AMP autocast mode: auto, none, float16, bfloat16")
    parser.add_argument("--input-type", type=str,
                        choices=["video", "folder", "i2i"], default="video",
                        help=("Processing mode:\n"
                              "  video = single video\n"
                              "  folder = batch same-resolution images\n"
                              "  i2i = images one-by-one (single or mixed-resolution folder)"))
    parser.add_argument("--infer-accum-batches", type=int, default=None,
                    help="Number of mini-batches merged into one GPU inference (uses more VRAM)")
    parser.add_argument("--debug", action="store_true",
                    help="Enable debug mode with memory/queue monitoring")
    parser.add_argument("--preset","-p", type=str, choices=["minimum", "balance", "max_quality"],
                        help="Use a predefined configuration preset (default: minimum)")
    parser.add_argument("--clean-output-pngs", action="store_true",
                    help="Folder mode only: delete existing PNG files in output folder before processing")

    # Queues
    parser.add_argument("--in-queue", type=int, default=None,
                        help="Max size of input queue (CPU -> GPU)")
    parser.add_argument("--r-queue", type=int, default=None,
                        help="Max size of raw queue (disk -> preprocess)")
    parser.add_argument("--s-queue", type=int, default=None,
                        help="Max size of save queue (process -> disk)")
    parser.add_argument("--p-queue", type=int, default=None,
                        help="Max size of process queue (GPU -> CPU)")

    # Streams 
    parser.add_argument("--feeders", type=int, default=None,
                        help="Number of feeder threads (video: must be 1)")
    parser.add_argument("--preprocess", "-pre", type=int, default=None,
                        help="Number of CPU preprocess threads")
    parser.add_argument("--processors", type=int, default=None,
                        help="Number of processing threads (depth -> SBS)")
    parser.add_argument("--savers", type=int, default=None,
                        help="Number of saver threads "
                             "(video/i2i: must be 1; folder: can be >1)")
                        
    # converter settings
    parser.add_argument("--depth-scale", type=float, default=None,
                        help="Scale factor for depth map (default=1.0)")
    parser.add_argument("--depth-offset", type=float, default=None,
                        help="Offset for depth map (default=0.0)")
    parser.add_argument("--crop-size", type=int, default=None,
                        help="Black crop width at the warped edge (default=0)")
    parser.add_argument("--switch-sides", action="store_true",default=None,
                        help="Swap left/right images in output (default=False)")
    parser.add_argument("--symmetric", dest="symetric", action="store_true",default=None,
                        help="Enable symmetric rendering (default=False)")
    parser.add_argument("--blur-radius", type=int, default=None,
                        help="Blur radius applied to depth map before shifting (default=19)")

    # Reserved for a future true 10-bit GPU path.
    parser.add_argument("--hdr",action=argparse.BooleanOptionalAction, default=None,
                        help="Reserved; the GPU pipeline currently rejects HDR input")
    parser.add_argument("--hdr-encoder", type=str, choices=["auto", "nvenc", "libx265"], default=None,
                        help="HDR HEVC encoder: auto (libx265 if HDR10 static metadata, else nvenc) | nvenc | libx265")
    parser.add_argument("--master-display", type=str, default=None,
                        help="HDR10 mastering-display string 'G(..)B(..)R(..)WP(..)L(..)' (libx265 only)")
    parser.add_argument("--max-cll", type=str, default=None,
                        help="HDR10 MaxCLL,MaxFALL e.g. '1000,400' (libx265 only)")

    args = parser.parse_args()
    
    # Force shutdown (Ctrl+C or SIGTERM)
    signal.signal(signal.SIGINT, force_exit)
    signal.signal(signal.SIGTERM, force_exit)
    
    preset_data = {}

    # --- i2i (image-to-image) mode ---
    # Processes a single image or folder of images individually (no batching).
    if args.input_type == "i2i":
        validate_config(args, parser)
        estimator = DepthEstimator()

        if os.path.isfile(args.input):
            # single image
            images = [args.input]
            if os.path.isdir(args.output):
                parser.error("For single image input (--input=file) you must provide --output as a file, not a folder.")
        else:
            # folder with pictures
            images = natsorted([os.path.join(args.input, f)
                                for f in os.listdir(args.input)
                                if f.lower().endswith((".png", ".jpg", ".jpeg"))])
            if not images:
                parser.error(f"No images found in input folder: {args.input}")
            os.makedirs(args.output, exist_ok=True)

        for img_path in images:
            if os.path.isfile(args.input):
                # input one image → output = file
                out_path = args.output
            else:
                # input folder → output = folder + filename
                out_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
                out_path = os.path.join(args.output, out_name)

            ctx = init_pipeline(
                version,
                video_path=img_path,
                estimator=estimator,
                output_path=out_path,
                batch_size=1,
                in_queue=1, r_queue=1, s_queue=1, p_queue=1,
                n_preprocess=1, n_processors=1, n_savers=1, n_feeders=1,
                model_name=args.model or "depth-anything/Depth-Anything-V2-Base-hf",
                codec="png",
                autocast=args.autocast,
                input_type=args.input_type,
                debug=args.debug,
                depth_scale=args.depth_scale if args.depth_scale is not None else 1.0,
                depth_offset=args.depth_offset if args.depth_offset is not None else 0.0,
                crop_size=args.crop_size if args.crop_size is not None else 0,
                switch_sides=bool(args.switch_sides),
                symetric=bool(args.symetric),
                blur_radius=args.blur_radius if args.blur_radius is not None else 19,
            )
            run_pipeline(ctx)
        debug_report(ctx)
    # --- video and folder modes ---
    else:
        preset_name = args.preset or "minimum"
        preset_variant = None

        if args.input_type == "video":
            # TODO: share one video probe with init_pipeline.
            cap = cv2.VideoCapture(args.input)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                parser.error(f"Failed to read first frame from {args.input}")

            height, width = frame.shape[:2]
            preset_variant = "1080p" if width * height <= 1920 * 1080 else "high_resolution"

        preset_data = load_preset(args.input_type, preset_name, preset_variant)
        preset_location = f"{args.input_type}.{preset_variant}" if preset_variant else args.input_type
        print(f"Loaded preset '{preset_name}' from '{preset_location}'")

        merged_params = merge_with_preset(args, preset_data, PipelineContext)
        validate_config({**merged_params, "clean_output_pngs": args.clean_output_pngs}, parser)
        estimator = DepthEstimator()

        if args.clean_output_pngs:
            clean_output_pngs(
                merged_params["output_path"],
                merged_params["video_path"]
            )

        ctx = init_pipeline(
            version,
            estimator=estimator,
            **merged_params
        )

        run_pipeline(ctx)
        estimator.print_depth_profile()
        debug_report(ctx)
        


