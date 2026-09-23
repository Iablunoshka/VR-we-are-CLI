import argparse
import os
import glob
import time
import numpy as np
import cv2
import torch
import contextlib
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthEstimator:
    """
    Depth estimation using the Depth-Anything-V2-Small model from Hugging Face.
    Supports batch processing.
    """
    
    AVAILABLE_MODELS = [
    "depth-anything/Depth-Anything-V2-Small-hf",
    "depth-anything/Depth-Anything-V2-Base-hf",
    "depth-anything/Depth-Anything-V2-Large-hf",
    ]
    
    def __init__(self):
        self.depth_profile = {
            "model_ms": 0.0,
            "resize_ms": 0.0,
            "normalize_ms": 0.0,
            "batches": 0,
            "frames": 0,
        }

        self.depth_profile_warmup = 3
        self.depth_profile_calls = 0
        
        
        self.device = torch.device("cpu")  # default CPU
        self.model_id = None
        
        self.cuda_available = False
        self.sm = 0.0
        
        self.bf16_supported = False
        self.processor = None
        self.model = None

        # auto check CUDA
        if self._cuda_ok():
            print("Using CUDA")
            self.device = torch.device("cuda")

            major, minor = torch.cuda.get_device_capability()
            self.sm = major + minor / 10
            self.bf16_supported = torch.cuda.is_bf16_supported()
            self.cuda_available = True

            print(f"CUDA capability: SM {self.sm:.1f}")
        else:
            torch.set_num_threads(os.cpu_count())
            print("Using CPU")

    def _cuda_ok(self) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            torch.zeros(1, device="cuda")
            return True
        except Exception as e:
            print("WARN CUDA not usable:", repr(e))
            return False
            
    def print_depth_profile(self):
        p = self.depth_profile
        frames = p["frames"]

        if frames == 0:
            #print("\nDepth profile: no measured frames")
            return

        total_ms = p["model_ms"] + p["resize_ms"] + p["normalize_ms"]

        print("\n===== Depth GPU Profile =====")
        print(f"Warm-up batches skipped: {self.depth_profile_warmup}")
        print(f"Measured batches:        {p['batches']}")
        print(f"Measured frames:         {frames}")
        print(f"Model total:             {p['model_ms']:.2f} ms")
        print(f"Resize total:            {p['resize_ms']:.2f} ms")
        print(f"Normalize total:         {p['normalize_ms']:.2f} ms")
        print(f"Model/frame:             {p['model_ms'] / frames:.4f} ms")
        print(f"Resize/frame:            {p['resize_ms'] / frames:.4f} ms")
        print(f"Normalize/frame:         {p['normalize_ms'] / frames:.4f} ms")
        print(f"Depth total/frame:       {total_ms / frames:.4f} ms")
        print(f"Depth-only FPS:          {frames * 1000 / total_ms:.2f}")
        print(f"Model share:             {p['model_ms'] / total_ms * 100:.2f}%")
        print(f"Resize share:            {p['resize_ms'] / total_ms * 100:.2f}%")
        print(f"Normalize share:         {p['normalize_ms'] / total_ms * 100:.2f}%")
        print("=============================\n")
            
    def resolve_autocast_mode(self, autocast: str | None) -> str | None:
        """
        Resolve CLI/preset autocast mode into runtime mode:
        None / "float16" / "bfloat16"
        """

        if autocast is None:
            autocast = "auto"

        if autocast == "none":
            print("AMP autocast: disabled")
            return None

        if self.device.type != "cuda" or not self.cuda_available:
            if autocast == "auto":
                print("AMP autocast: auto -> disabled (CUDA unavailable)")
            else:
                print(f"AMP autocast: {autocast} -> disabled (CUDA unavailable)")
            return None

        if autocast == "auto":
            if self.bf16_supported and self.sm >= 8.9:
                print(f"AMP autocast: auto -> bfloat16 (SM {self.sm:.1f})")
                return "bfloat16"

            if self.sm >= 8.0:
                print(f"AMP autocast: auto -> float16 (SM {self.sm:.1f})")
                return "float16"

            print(f"AMP autocast: auto -> disabled (SM {self.sm:.1f})")
            return None

        if autocast == "bfloat16":
            if self.bf16_supported:
                print("AMP autocast: bfloat16")
                return "bfloat16"

            if self.sm >= 8.0:
                print(f"AMP autocast: bfloat16 unsupported -> float16 (SM {self.sm:.1f})")
                return "float16"

            print(f"AMP autocast: bfloat16 unsupported -> disabled (SM {self.sm:.1f})")
            return None

        if autocast == "float16":
            if self.sm >= 8.0:
                print("AMP autocast: float16")
                return "float16"

            print(f"AMP autocast: float16 requested -> disabled (SM {self.sm:.1f})")
            return None

        raise ValueError(f"Unknown autocast mode: {autocast}")


    def load_model(self, model_id: str,cudnn_benchmark: bool):
        """
        Load model only if it's not already loaded or if different model requested.
        """
        if self.model_id != model_id:
            print(f"Loading model: {model_id}")
            self.model_id = model_id
            try:
                self.processor = AutoImageProcessor.from_pretrained(self.model_id, backend="torchvision")
            except TypeError:
                self.processor = AutoImageProcessor.from_pretrained(self.model_id)
                
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
            
            torch.backends.cudnn.benchmark = cudnn_benchmark
            #torch.backends.cuda.matmul.allow_tf32 = True
            #torch.set_float32_matmul_precision("high")
            
            if self.device.type == "cuda":
                self.model = (self.model.to(self.device).eval())
                major, minor = torch.cuda.get_device_capability(self.device)
                if major >= 8:
                    print("Compiling model with torch.compile...")
                    self.model = torch.compile(
                        self.model,
                        backend="inductor",
                        mode="default",
                        fullgraph=False,
                        dynamic=False,
                    )
                else:
                    print(f"torch.compile disabled on SM {major}.{minor}; using CUDA eager mode")
            else:
                self.model.eval()

        else:
            #print(f"Model '{model_id}' already loaded.")
            pass


    def predict_batch_tensor(self, pixel_values: torch.Tensor,cudnn_benchmark: bool,compiled_batch_size: int, target_size: tuple[int, int] = None, model_name: str = None,autocast: str | None = None, ) -> list[np.ndarray]:
        """
        Generate normalized depth maps for a batch.
        """
        # Make sure the model is loaded
        #if model_name is not None:
        #    self.load_model(model_name,cudnn_benchmark)
        #elif self.model is None:
        #    self.load_model(self.AVAILABLE_MODELS[0],cudnn_benchmark)

        B, _, H_in, W_in = pixel_values.shape
        profile = False

        # Amp autocast check
        if self.device.type == "cuda" and autocast is not None:
            if autocast == "bfloat16":
                ac_dtype = torch.bfloat16

            elif autocast == "float16":
                ac_dtype = torch.float16

            else:
                raise ValueError(f"Unknown autocast dtype: {autocast}")

            autocast_ctx = torch.amp.autocast(
                device_type="cuda",
                dtype=ac_dtype,
            )
        else:
            autocast_ctx = contextlib.nullcontext()
            
        # Inference
        real_batch_size = pixel_values.shape[0]

        if real_batch_size < compiled_batch_size:
            pad_count = compiled_batch_size - real_batch_size

            padding = pixel_values[-1:].expand(pad_count,*pixel_values.shape[1:],)

            pixel_values_for_model = torch.cat((pixel_values, padding),dim=0,)
        else:
            pixel_values_for_model = pixel_values

        if profile:
            model_start = torch.cuda.Event(enable_timing=True)
            model_end = torch.cuda.Event(enable_timing=True)
            resize_end = torch.cuda.Event(enable_timing=True)
            normalize_end = torch.cuda.Event(enable_timing=True)

            model_start.record()

        with torch.inference_mode(), autocast_ctx:
            preds = self.model(pixel_values_for_model).predicted_depth

        if profile:
            model_end.record()

        preds = preds[:real_batch_size]
        preds = preds.unsqueeze(1).float()

        preds_resized = torch.nn.functional.interpolate(
            preds,
            size=target_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

        if profile:
            resize_end.record()

        mins = preds_resized.amin(dim=(1, 2), keepdim=True)
        maxs = preds_resized.amax(dim=(1, 2), keepdim=True)
        normalized = (preds_resized - mins) / (maxs - mins).clamp_min_(1e-6)

        if profile:
            normalize_end.record()
            normalize_end.synchronize()

            model_ms = model_start.elapsed_time(model_end)
            resize_ms = model_end.elapsed_time(resize_end)
            normalize_ms = resize_end.elapsed_time(normalize_end)

            self.depth_profile_calls += 1

            if self.depth_profile_calls > self.depth_profile_warmup:
                self.depth_profile["model_ms"] += model_ms
                self.depth_profile["resize_ms"] += resize_ms
                self.depth_profile["normalize_ms"] += normalize_ms
                self.depth_profile["batches"] += 1
                self.depth_profile["frames"] += real_batch_size

        return normalized
    
    def predict_batch(self, images: list[np.ndarray],model_name,cudnn_benchmark,autocast: str | None = None) -> list[np.ndarray]:
        self.load_model(model_name,cudnn_benchmark)
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)
        return self.predict_batch_tensor(pixel_values,cudnn_benchmark,autocast=autocast)

    def predict_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Legacy single-image API, wraps predict_batch for convenience.
        """
        return self.predict_batch([image], model_name=self.model_id or self.AVAILABLE_MODELS[0], cudnn_benchmark=True)[0]

         
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=50, help="Number of images per batch")
    parser.add_argument("-i", type=str, default=".", help="Directory with input PNGs")
    parser.add_argument("-o", type=str, default="output", help="Directory to save depth maps")
    parser.add_argument("--model", "-m", type=str, default="depth-anything/Depth-Anything-V2-Small-hf", choices=DepthEstimator.AVAILABLE_MODELS, help="Which depth model to use")
    args = parser.parse_args()

    os.makedirs(args.o, exist_ok=True)
    start = time.perf_counter()

    image_paths = sorted(glob.glob(os.path.join(args.i, "*.png")))
    if not image_paths:
        print("[error] No PNG images found in input directory.")
        exit(1)

    estimator = DepthEstimator()

    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i + args.batch_size]
        batch_images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in batch_paths]

        depth_maps = estimator.predict_batch(batch_images, args.model, True)

        for path, depth in zip(batch_paths, depth_maps):
            filename = os.path.splitext(os.path.basename(path))[0]
            output_path = os.path.join(args.o, f"{filename}_depth.png")

            # Convert to 8-bit 
            depth_uint8 = (depth * 255).astype(np.uint8)

            cv2.imwrite(output_path, depth_uint8)

    end = time.perf_counter()
    print(f"Process time: {end - start:.4f} sec")
               
            
