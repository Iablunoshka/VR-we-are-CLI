import numpy as np
import time, torch
import cupy as cp
from cupy.cuda import texture, runtime
import torch.nn.functional as F


def resolve_cuda_device(device=None) -> torch.device:
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda":
        raise ValueError("GPU converter requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


RGB_REMAP_KERNEL_SRC = r'''
extern "C" __global__
void remap_tex4_u8_batch(const unsigned long long* __restrict__ tex_list,
                         const float* __restrict__ mapx,
                         unsigned char* __restrict__ out_u8, // packed (B, H, W*3)
                         int H, int W)
{
    int x = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    int y = (int)(blockDim.y * blockIdx.y + threadIdx.y);
    int b = (int)blockIdx.z;

    if (x >= W || y >= H) return;

    // idx in maps: (B, H*W)
    size_t HW  = (size_t)H * (size_t)W;
    size_t idx = (size_t)b * HW + (size_t)y * (size_t)W + (size_t)x;

    float sx = mapx[idx];

    float u = (sx + 0.5f) / (float)W;
    float v = ((float)y + 0.5f) / (float)H;

    cudaTextureObject_t tex = (cudaTextureObject_t)tex_list[b];
    float4 c = tex2D<float4>(tex, u, v);

    // out layout: (B, H, W*3)
    size_t row_stride = (size_t)W * 3;
    size_t o = ((size_t)b * (size_t)H + (size_t)y) * row_stride + (size_t)x * 3;

    float r = c.x * 255.0f + 0.5f;
    float g = c.y * 255.0f + 0.5f;
    float bb = c.z * 255.0f + 0.5f;

    r  = r  < 0.f ? 0.f : (r  > 255.f ? 255.f : r);
    g  = g  < 0.f ? 0.f : (g  > 255.f ? 255.f : g);
    bb = bb < 0.f ? 0.f : (bb > 255.f ? 255.f : bb);

    out_u8[o + 0] = (unsigned char)r;
    out_u8[o + 1] = (unsigned char)g;
    out_u8[o + 2] = (unsigned char)bb;
}
'''

NV12_KERNEL_SRC = r'''
struct RGBValue
{
    float r;
    float g;
    float b;
};

__device__ __forceinline__
unsigned char round_clip(float value, float min_value, float max_value)
{
    value = value < min_value ? min_value : value;
    value = value > max_value ? max_value : value;
    return (unsigned char)(value + 0.5f);
}

__device__ __forceinline__
RGBValue sample_rgb(cudaTextureObject_t tex, float sx, int y, int H, int W)
{
    float u = (sx + 0.5f) / (float)W;
    float v = ((float)y + 0.5f) / (float)H;
    float4 color = tex2D<float4>(tex, u, v);

    RGBValue rgb;

    // Match the current RGB kernel rounding before RGB -> NV12.
    rgb.r = (float)round_clip(color.x * 255.0f, 0.0f, 255.0f);
    rgb.g = (float)round_clip(color.y * 255.0f, 0.0f, 255.0f);
    rgb.b = (float)round_clip(color.z * 255.0f, 0.0f, 255.0f);

    return rgb;
}

__device__ __forceinline__
unsigned char rgb_to_y(RGBValue rgb)
{
    float value = 16.0f
        + 0.182586f * rgb.r
        + 0.614231f * rgb.g
        + 0.062007f * rgb.b;

    return round_clip(value, 16.0f, 235.0f);
}

__device__ __forceinline__
unsigned char rgb_to_u(RGBValue rgb)
{
    float value = 128.0f
        - 0.100644f * rgb.r
        - 0.338572f * rgb.g
        + 0.439216f * rgb.b;

    return round_clip(value, 16.0f, 240.0f);
}

__device__ __forceinline__
unsigned char rgb_to_v(RGBValue rgb)
{
    float value = 128.0f
        + 0.439216f * rgb.r
        - 0.398942f * rgb.g
        - 0.040274f * rgb.b;

    return round_clip(value, 16.0f, 240.0f);
}

__device__ __forceinline__
bool is_cropped(int x, int W, int crop)
{
    if (crop > 0) return x >= W - crop;
    if (crop < 0) return x < -crop;
    return false;
}

__device__ __forceinline__
RGBValue black_rgb()
{
    RGBValue rgb;
    rgb.r = 0.0f;
    rgb.g = 0.0f;
    rgb.b = 0.0f;
    return rgb;
}

__device__ __forceinline__
RGBValue load_rgb(
    const unsigned char* __restrict__ base_rgb,
    size_t pixel_index
)
{
    size_t offset = pixel_index * 3;

    RGBValue rgb;
    rgb.r = (float)base_rgb[offset + 0];
    rgb.g = (float)base_rgb[offset + 1];
    rgb.b = (float)base_rgb[offset + 2];

    return rgb;
}

extern "C" __global__
void warp_half_nv12_batch(
    const unsigned long long* __restrict__ tex_list,
    const float* __restrict__ mapx,
    unsigned char* __restrict__ output,
    int B,
    int H,
    int W,
    int dst_x,
    int crop
)
{
    int cx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    int cy = (int)(blockDim.y * blockIdx.y + threadIdx.y);
    int b = (int)blockIdx.z;

    if (b >= B || cx >= W / 2 || cy >= H / 2) return;

    // 1) Calculate the source 2x2 block
    int x0 = cx * 2;
    int x1 = x0 + 1;
    int y0 = cy * 2;
    int y1 = y0 + 1;

    size_t HW = (size_t)H * (size_t)W;
    size_t map_base = (size_t)b * HW;

    float sx00 = mapx[map_base + (size_t)y0 * W + x0];
    float sx01 = mapx[map_base + (size_t)y0 * W + x1];
    float sx10 = mapx[map_base + (size_t)y1 * W + x0];
    float sx11 = mapx[map_base + (size_t)y1 * W + x1];

    // 2) Sample four independently remapped RGB pixels
    cudaTextureObject_t tex = (cudaTextureObject_t)tex_list[b];

    RGBValue c00 = sample_rgb(tex, sx00, y0, H, W);
    RGBValue c01 = sample_rgb(tex, sx01, y0, H, W);
    RGBValue c10 = sample_rgb(tex, sx10, y1, H, W);
    RGBValue c11 = sample_rgb(tex, sx11, y1, H, W);

    // 3) Apply the local-half crop before RGB to NV12 conversion
    if (is_cropped(x0, W, crop)) {
        c00 = black_rgb();
        c10 = black_rgb();
    }

    if (is_cropped(x1, W, crop)) {
        c01 = black_rgb();
        c11 = black_rgb();
    }

    // 4) Locate the NV12 planes
    int out_W = W * 2;
    size_t frame_size = (size_t)H * out_W * 3 / 2;
    unsigned char* frame = output + (size_t)b * frame_size;
    unsigned char* y_plane = frame;
    unsigned char* uv_plane = frame + (size_t)H * out_W;

    int out_x0 = dst_x + x0;
    int out_x1 = dst_x + x1;

    // 5) Write four luma pixels
    y_plane[(size_t)y0 * out_W + out_x0] = rgb_to_y(c00);
    y_plane[(size_t)y0 * out_W + out_x1] = rgb_to_y(c01);
    y_plane[(size_t)y1 * out_W + out_x0] = rgb_to_y(c10);
    y_plane[(size_t)y1 * out_W + out_x1] = rgb_to_y(c11);

    // 6) Average the rounded RGB pixels and write one UV pair
    RGBValue average;

    average.r = (c00.r + c01.r + c10.r + c11.r) * 0.25f;
    average.g = (c00.g + c01.g + c10.g + c11.g) * 0.25f;
    average.b = (c00.b + c01.b + c10.b + c11.b) * 0.25f;

    size_t uv_index = (size_t)cy * out_W + out_x0;

    uv_plane[uv_index] = rgb_to_u(average);
    uv_plane[uv_index + 1] = rgb_to_v(average);
}

extern "C" __global__
void copy_half_nv12_batch(
    const unsigned char* __restrict__ base_rgb,
    unsigned char* __restrict__ output,
    int B,
    int H,
    int W,
    int dst_x,
    int crop
)
{
    int cx = (int)(blockDim.x * blockIdx.x + threadIdx.x);
    int cy = (int)(blockDim.y * blockIdx.y + threadIdx.y);
    int b = (int)blockIdx.z;

    if (b >= B || cx >= W / 2 || cy >= H / 2) return;

    // 1) Calculate the local 2x2 block
    int x0 = cx * 2;
    int x1 = x0 + 1;
    int y0 = cy * 2;
    int y1 = y0 + 1;

    size_t frame_pixels = (size_t)H * (size_t)W;
    size_t source_base = (size_t)b * frame_pixels;

    // 2) Read four source RGB pixels
    RGBValue c00 = load_rgb(base_rgb, source_base + (size_t)y0 * W + x0);
    RGBValue c01 = load_rgb(base_rgb, source_base + (size_t)y0 * W + x1);
    RGBValue c10 = load_rgb(base_rgb, source_base + (size_t)y1 * W + x0);
    RGBValue c11 = load_rgb(base_rgb, source_base + (size_t)y1 * W + x1);

    // 3) Apply crop in local-half coordinates
    if (is_cropped(x0, W, crop)) {
        c00 = black_rgb();
        c10 = black_rgb();
    }

    if (is_cropped(x1, W, crop)) {
        c01 = black_rgb();
        c11 = black_rgb();
    }

    // 4) Locate the destination NV12 planes
    int out_W = W * 2;
    size_t frame_size = (size_t)H * out_W * 3 / 2;

    unsigned char* frame = output + (size_t)b * frame_size;
    unsigned char* y_plane = frame;
    unsigned char* uv_plane = frame + (size_t)H * out_W;

    int out_x0 = dst_x + x0;
    int out_x1 = dst_x + x1;

    // 5) Write four luma pixels
    y_plane[(size_t)y0 * out_W + out_x0] = rgb_to_y(c00);
    y_plane[(size_t)y0 * out_W + out_x1] = rgb_to_y(c01);
    y_plane[(size_t)y1 * out_W + out_x0] = rgb_to_y(c10);
    y_plane[(size_t)y1 * out_W + out_x1] = rgb_to_y(c11);

    // 6) Average RGB and write one interleaved UV pair
    RGBValue average;
    average.r = (c00.r + c01.r + c10.r + c11.r) * 0.25f;
    average.g = (c00.g + c01.g + c10.g + c11.g) * 0.25f;
    average.b = (c00.b + c01.b + c10.b + c11.b) * 0.25f;

    size_t uv_index = (size_t)cy * out_W + out_x0;

    uv_plane[uv_index + 0] = rgb_to_u(average);
    uv_plane[uv_index + 1] = rgb_to_v(average);
}
'''


class RemapTextureSource:
    """
    Store shared CUDA texture resources and horizontal remap data.
    """

    def __init__(self, H: int, W: int, Bmax: int, device=None):
        self.H = int(H)
        self.W = int(W)
        self.Bmax = int(Bmax)
        self.device = resolve_cuda_device(device)

        # 1) Prepare horizontal remap buffers
        self.x = torch.arange(self.W, dtype=torch.float32, device=self.device)
        self.targets = self.x.view(1, 1, self.W).expand(self.Bmax, self.H, self.W).contiguous()

        # 2) Allocate texture handles
        with cp.cuda.Device(self.device.index):
            self.tex_handles_gpu = cp.empty((self.Bmax,), dtype=cp.uint64)

        # 3) Create RGBA CUDA texture arrays
        channel_desc = texture.ChannelFormatDescriptor(8, 8, 8, 8, runtime.cudaChannelFormatKindUnsigned)

        texture_desc = texture.TextureDescriptor(
            addressModes=(runtime.cudaAddressModeMirror, runtime.cudaAddressModeMirror),
            filterMode=runtime.cudaFilterModeLinear,
            readMode=runtime.cudaReadModeNormalizedFloat,
            normalizedCoords=1,
        )

        self.cu_arr_list = []
        self.tex_obj_list = []

        with cp.cuda.Device(self.device.index):
            for _ in range(self.Bmax):
                cu_arr = texture.CUDAarray(channel_desc, self.W, self.H)
                resource_desc = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuArr=cu_arr)
                tex_obj = texture.TextureObject(resource_desc, texture_desc)

                self.cu_arr_list.append(cu_arr)
                self.tex_obj_list.append(tex_obj)

        handles = np.array([self._tex_handle_u64(tex_obj) for tex_obj in self.tex_obj_list], dtype=np.uint64)
        with cp.cuda.Device(self.device.index):
            self.tex_handles_gpu.set(handles)

        self.mapx_view = None

    @staticmethod
    def _tex_handle_u64(tex_obj):
        """
        Return the CUDA texture handle as uint64.
        """
        if hasattr(tex_obj, "ptr"):
            return np.uint64(tex_obj.ptr)

        return np.uint64(int(tex_obj))

    def prepare_batch(self, rgba_batch: torch.Tensor, mapx_batch: torch.Tensor) -> int:
        """
        Upload RGBA source images and expose the horizontal remap map.
        """
        batch_size = int(rgba_batch.shape[0])
        HW = self.H * self.W

        if not 1 <= batch_size <= self.Bmax:
            raise ValueError(f"batch size must be 1..{self.Bmax}")

        # 1) Create zero-copy Torch to CuPy views
        rgba_cp = cp.from_dlpack(rgba_batch.contiguous())
        self.mapx_view = cp.from_dlpack(mapx_batch.contiguous()).reshape(batch_size, HW)

        # 2) Upload RGBA images to CUDA texture arrays
        for b in range(batch_size):
            self.cu_arr_list[b].copy_from(rgba_cp[b].reshape(self.H, self.W * 4))

        return batch_size
        
    def prepare_source(self, rgba_batch: torch.Tensor) -> int:
        """
        Upload RGBA source images to CUDA texture arrays.
        """
        batch_size = int(rgba_batch.shape[0])

        if not 1 <= batch_size <= self.Bmax:
            raise ValueError(f"batch size must be 1..{self.Bmax}")

        rgba_cp = cp.from_dlpack(rgba_batch.contiguous())

        for b in range(batch_size):
            self.cu_arr_list[b].copy_from(rgba_cp[b].reshape(self.H, self.W * 4))

        return batch_size

    def prepare_mapx(self, mapx_batch: torch.Tensor) -> cp.ndarray:
        """
        Expose a horizontal remap map as a zero-copy CuPy view.
        """
        batch_size = int(mapx_batch.shape[0])
        HW = self.H * self.W

        return cp.from_dlpack(mapx_batch.contiguous()).reshape(batch_size, HW)
        

class RemapTextureRGB:
    """
    Render horizontally remapped RGB batches from CUDA textures.
    """

    def __init__(self, H: int, W: int, Bmax: int = 8, block=(16, 16), debug=True, device=None):
        self.H = int(H)
        self.W = int(W)
        self.Bmax = int(Bmax)
        self.debug = debug
        self.device = resolve_cuda_device(device)

        # 1) Create shared texture source
        self.source = RemapTextureSource(self.H, self.W, self.Bmax, self.device)

        # 2) Configure the RGB kernel
        bx, by = int(block[0]), int(block[1])
        self.block = (bx, by, 1)
        self.grid_xy = ((self.W + bx - 1) // bx, (self.H + by - 1) // by)
        self.kernel = cp.RawKernel(RGB_REMAP_KERNEL_SRC, "remap_tex4_u8_batch")

        # 3) Allocate reusable RGB output
        with cp.cuda.Device(self.device.index):
            self.out_u8 = cp.empty((self.Bmax, self.H, self.W * 3), dtype=cp.uint8)
        
    @property
    def x(self) -> torch.Tensor:
        return self.source.x

    @property
    def targets(self) -> torch.Tensor:
        return self.source.targets


    def prepare_batch(self, rgba_batch: torch.Tensor, mapx_batch: torch.Tensor) -> int:
        """
        Prepare shared CUDA textures and horizontal remap data.
        """
        return self.source.prepare_batch(rgba_batch, mapx_batch)


    def run_batch(self, B: int) -> None:
        """
        Run the RGB remap kernel.
        """
        gx, gy = self.grid_xy
        grid = (gx, gy, int(B))

        self.kernel(grid, self.block, (self.source.tex_handles_gpu, self.source.mapx_view, self.out_u8, self.H, self.W))


    def get_u8_batch(self, B: int) -> torch.Tensor:
        """
        Return the reusable RGB output as a Torch CUDA tensor.
        """
        out_cp = self.out_u8[:int(B)].reshape(B, self.H, self.W, 3)
        return torch.from_dlpack(out_cp)


class NV12CudaFrameView:
    """
    Expose one NV12 CUDA frame to PyNvVideoCodec.
    """

    def __init__(self, y: cp.ndarray, uv: cp.ndarray):
        self.y = y
        self.uv = uv

    def cuda(self):
        return [self.y, self.uv]


class NV12CudaBatch:
    """
    Store a reusable contiguous batch of encoder-compatible NV12 frames.
    """

    def __init__(self, Bmax: int, H: int, W: int, device=None):
        if H % 2 or W % 2:
            raise ValueError("NV12 width and height must be even")

        self.Bmax = int(Bmax)
        self.H = int(H)
        self.W = int(W)
        self.device = resolve_cuda_device(device)

        # 1) Allocate contiguous NV12 storage
        with cp.cuda.Device(self.device.index):
            self.storage = cp.empty((self.Bmax, self.H * 3 // 2, self.W), dtype=cp.uint8)

        # 2) Create encoder-compatible plane views
        self.y = self.storage[:, :self.H].reshape(self.Bmax, self.H, self.W, 1)
        self.uv = self.storage[:, self.H:].reshape(self.Bmax, self.H // 2, self.W // 2, 2)

    def frame(self, index: int) -> NV12CudaFrameView:
        """
        Return one encoder-compatible NV12 frame view.
        """
        if not 0 <= index < self.Bmax:
            raise IndexError(f"frame index must be 0..{self.Bmax - 1}")

        return NV12CudaFrameView(self.y[index], self.uv[index])

    def frames(self, batch_size: int) -> list[NV12CudaFrameView]:
        """
        Return frame views for the active batch.
        """
        if not 1 <= batch_size <= self.Bmax:
            raise ValueError(f"batch size must be 1..{self.Bmax}")

        return [self.frame(index) for index in range(batch_size)]
        
def invert_map_1d_monotonic_torch(pixel_shifts: torch.Tensor,x: torch.Tensor,targets_buffer: torch.Tensor,) -> torch.Tensor:
    
    B, H, W = pixel_shifts.shape
    targets = targets_buffer[:B]

    u = x.view(1, 1, W) - pixel_shifts
    u_mono = torch.cummax(u, dim=-1).values

    j0 = torch.searchsorted(u_mono,targets,right=True,) - 1
    j0.clamp_(0, W - 1)
    j1 = (j0 + 1).clamp_(max=W - 1)

    u0 = torch.gather(u_mono, -1, j0)
    u1 = torch.gather(u_mono, -1, j1)

    x0 = j0.float()
    x1 = j1.float()
    du = u1 - u0

    xs = x0 + (targets - u0) / du
    xs = torch.where(du == 0, x1, xs)

    # No true inverse exists outside the destination interval covered by the
    # forward map.  Reflect the missing area back into the source without
    # touching the outermost texels: they can form a one-pixel seam at the
    # reflection fold.  With a one-pixel guard the folds are x=1 and x=W-2.
    edge_guard = 1
    u_min = u_mono[..., edge_guard:edge_guard + 1]
    u_max = u_mono[..., W - edge_guard - 1:W - edge_guard]

    xs = torch.where(targets < u_min,edge_guard + (u_min - targets),xs,)
    xs = torch.where(targets > u_max,(W - edge_guard - 1) - (targets - u_max),xs,)

    return xs
    
def prepare_remap_mapx_batch(
    pixel_shifts_in_batch: torch.Tensor,
    x: torch.Tensor,
    targets_buffer: torch.Tensor,
) -> torch.Tensor:
    """
    Build a batched horizontal CUDA remap map.
    """
    mapx_batch = invert_map_1d_monotonic_torch(pixel_shifts_in_batch, x, targets_buffer)
    mapx_batch.clamp_(0.0, pixel_shifts_in_batch.shape[-1] - 1)

    return mapx_batch

class RemapTextureNV12:
    """
    Render RGB source batches directly into reusable NV12 SBS storage.
    """

    def __init__(self, H: int, W: int, Bmax: int = 8, block=(16, 8), device=None):
        self.H = int(H)
        self.W = int(W)
        self.out_W = self.W * 2
        self.Bmax = int(Bmax)
        self.device = resolve_cuda_device(device)

        if self.H % 2 or self.W % 2:
            raise ValueError("NV12 rendering requires even source width and height")

        # 1) Create shared texture source
        self.source = RemapTextureSource(self.H, self.W, self.Bmax, self.device)

        # 2) Configure the 2x2 NV12 warp kernel
        bx, by = int(block[0]), int(block[1])
        self.block = (bx, by, 1)
        self.grid_xy = ((self.W // 2 + bx - 1) // bx, (self.H // 2 + by - 1) // by)
        self.warp_kernel = cp.RawKernel(NV12_KERNEL_SRC, "warp_half_nv12_batch")
        self.copy_kernel = cp.RawKernel(NV12_KERNEL_SRC, "copy_half_nv12_batch")

        # 3) Allocate reusable encoder-compatible NV12 output
        self.output = NV12CudaBatch(self.Bmax, self.H, self.out_W, self.device)
        self.batch_size = 0

    def build_mapx(self, pixel_shifts: torch.Tensor) -> torch.Tensor:
        """
        Build the same horizontal inverse-remap map as the RGB renderer.
        """
        return prepare_remap_mapx_batch(pixel_shifts, self.source.x, self.source.targets)

    def prepare_source(self, rgba_batch: torch.Tensor) -> int:
        """
        Upload the source batch once before rendering both SBS halves.
        """
        torch_stream = torch.cuda.current_stream(rgba_batch.device)

        with cp.cuda.ExternalStream(torch_stream.cuda_stream):
            self.batch_size = self.source.prepare_source(rgba_batch)

        return self.batch_size

    def render_warp_half(self, mapx_batch: torch.Tensor, dst_x: int, crop: int) -> None:
        """
        Render one horizontally warped SBS half into NV12 output.
        """

        # 1) Clamp crop to the local half width
        crop = max(-self.W, min(self.W, int(crop)))

        # 2) Run the kernel in the current Torch CUDA stream
        torch_stream = torch.cuda.current_stream(mapx_batch.device)

        with cp.cuda.ExternalStream(torch_stream.cuda_stream):
            mapx_view = self.source.prepare_mapx(mapx_batch)

            gx, gy = self.grid_xy
            grid = (gx, gy, self.batch_size)

            self.warp_kernel(grid,self.block,(self.source.tex_handles_gpu,mapx_view,self.output.storage,self.batch_size,self.H,self.W,int(dst_x),crop,),)

    def render_copy_half(self, base_batch: torch.Tensor, dst_x: int, crop: int = 0) -> None:
        """
        Render one unwarped SBS half directly into NV12 output.
        """

        # 1) Clamp crop to the local half width
        crop = max(-self.W, min(self.W, int(crop)))

        # 2) Launch the copy kernel in the current Torch CUDA stream
        torch_stream = torch.cuda.current_stream(base_batch.device)

        with cp.cuda.ExternalStream(torch_stream.cuda_stream):
            base_view = cp.from_dlpack(base_batch.contiguous())

            gx, gy = self.grid_xy
            grid = (gx, gy, self.batch_size)

            self.copy_kernel(grid,self.block,(base_view,self.output.storage,self.batch_size,self.H,self.W,int(dst_x),crop,),)

    def get_batch(self, batch_size: int) -> NV12CudaBatch:
        """
        Return the reusable NV12 output batch.
        """
        if batch_size != self.batch_size:
            raise ValueError(f"requested batch size {batch_size}, prepared {self.batch_size}")

        return self.output


def apply_subpixel_shift_texture_batch(
    rgba_batch: torch.Tensor,
    pixel_shifts_in_batch: torch.Tensor,
    remapper,
) -> torch.Tensor:
    """
    Apply batched horizontal subpixel remap entirely on CUDA.

    Returns:
        uint8 CUDA tensor with shape (B, H, W, 3).
    """
    assert rgba_batch.ndim == 4 and rgba_batch.shape[-1] == 4
    assert pixel_shifts_in_batch.ndim == 3

    B, H, W, _ = rgba_batch.shape
    assert pixel_shifts_in_batch.shape == (B, H, W)

    # 1) Build remap coordinates on GPU
    mapx_batch = prepare_remap_mapx_batch(pixel_shifts_in_batch, remapper.x, remapper.targets)

    # 2) Run CuPy texture operations in the current Torch CUDA stream
    torch_stream = torch.cuda.current_stream(rgba_batch.device).cuda_stream

    with cp.cuda.ExternalStream(torch_stream):
        remapper.prepare_batch(rgba_batch, mapx_batch)
        remapper.run_batch(B)
        out_batch = remapper.get_u8_batch(B)

    return out_batch
  
class DIBRCore:
    """
    Store shared DIBR parameters and prepare depth tensors.
    """

    def __init__(self, H: int, W: int, batch_size: int, device=None):
        self.H = int(H)
        self.W = int(W)
        self.Bmax = int(batch_size)
        self.device = resolve_cuda_device(device)

    def _validate_inputs(self, base_image: torch.Tensor, depth_image: torch.Tensor) -> int:
        """
        Validate CUDA RGB and depth batches.
        """
        if not isinstance(base_image, torch.Tensor) or base_image.dtype != torch.uint8:
            raise TypeError("base_image must be a torch.uint8 tensor")
        if not isinstance(depth_image, torch.Tensor) or depth_image.dtype != torch.float32:
            raise TypeError("depth_image must be a torch.float32 tensor")

        if not base_image.is_cuda or not depth_image.is_cuda:
            raise ValueError("base_image and depth_image must be on CUDA")
        if base_image.device != depth_image.device:
            raise ValueError("base_image and depth_image must be on the same device")
        if base_image.device != self.device:
            raise ValueError(f"inputs must be on {self.device}, got {base_image.device}")

        if base_image.shape[1:] != (self.H, self.W, 3):
            raise ValueError(f"base_image must have shape (B,{self.H},{self.W},3)")
        if depth_image.ndim not in (3, 4):
            raise ValueError("depth_image must have shape (B,H,W) or (B,H,W,C)")
        if depth_image.shape[1:3] != (self.H, self.W):
            raise ValueError(f"depth_image must have spatial shape ({self.H},{self.W})")
        if depth_image.shape[0] != base_image.shape[0]:
            raise ValueError("batch sizes must match")
        if depth_image.ndim == 4 and depth_image.shape[-1] not in (1, 3):
            raise ValueError("depth channels must be 1 or 3")

        batch_size = int(base_image.shape[0])

        if not 1 <= batch_size <= self.Bmax:
            raise ValueError(f"batch size must be 1..{self.Bmax}")

        return batch_size

    def _prepare_depth_batch(self, depth_batch: torch.Tensor, invert_depth: bool) -> torch.Tensor:
        """
        Convert normalized depth to the existing DIBR depth range.
        """
        if depth_batch.ndim == 4:
            depth_batch = depth_batch[..., 0]

        if invert_depth:
            depth_batch = 1.0 - depth_batch

        return depth_batch.mul(255.0).sub(128.0)

    def _blur_batch(self, shifts_batch: torch.Tensor, blur_radius: int) -> torch.Tensor:
        """
        Apply reflect-padded average blur to the shift field.
        """
        if blur_radius <= 0:
            return shifts_batch

        pad = blur_radius // 2
        padded = F.pad(shifts_batch[:, None], (pad, pad, pad, pad), mode="reflect")

        return F.avg_pool2d(padded, blur_radius, stride=1).squeeze(1)

class RGBSBSConverter:
    """
    Convert RGB and depth batches to uint8 RGB SBS images.
    """

    def __init__(self, core: DIBRCore):
        self.core = core
        self.H = core.H
        self.W = core.W
        self.Bmax = core.Bmax

        self._remapper = RemapTextureRGB(self.H, self.W, Bmax=self.Bmax, device=core.device)
        self._rgba_workspace = torch.empty((self.Bmax, self.H, self.W, 4), dtype=torch.uint8, device=core.device)
        self._rgba_workspace[..., 3].fill_(255)

        self._warmup_remapper()

    def _warmup_remapper(self) -> None:
        """
        Initialize CUDA texture resources before the first real batch.
        """
        device = self.core.device
        dummy_rgba = torch.zeros((1, self.H, self.W, 4), dtype=torch.uint8, device=device)
        dummy_mapx = torch.arange(self.W, dtype=torch.float32, device=device).view(1, 1, self.W).expand(1, self.H, self.W)
        torch_stream = torch.cuda.current_stream(device).cuda_stream

        with cp.cuda.ExternalStream(torch_stream):
            self._remapper.prepare_batch(dummy_rgba, dummy_mapx)
            self._remapper.run_batch(1)

        torch.cuda.synchronize(device)

    def _crop_blackout_and_swap(
        self,
        sbs_batch: torch.Tensor,
        crop_a: int,
        crop_b: int,
        symetric: bool,
        switch_sides: bool,
    ) -> torch.Tensor:
        """
        Apply crop blackout masks and optionally swap SBS sides.
        """
        out = sbs_batch

        # 1) Apply the left-half blackout
        if crop_a > 0:
            out[:, :, self.W - crop_a:self.W, :] = 0
        elif crop_a < 0:
            out[:, :, 0:-crop_a, :] = 0

        # 2) Apply the symmetric right-half blackout
        if symetric:
            if crop_b > 0:
                out[:, :, self.W:self.W + crop_b, :] = 0
            elif crop_b < 0:
                out[:, :, 2 * self.W + crop_b:2 * self.W, :] = 0

        # 3) Swap the left and right views
        if switch_sides:
            out = torch.cat((out[:, :, self.W:, :], out[:, :, :self.W, :]), dim=2)

        return out

    def process(
        self,
        base_image: torch.Tensor,
        depth_image: torch.Tensor,
        depth_scale: float,
        depth_offset: float,
        crop_size: int,
        switch_sides: bool,
        blur_radius: int,
        symetric: bool,
    ) -> torch.Tensor:
        """
        Convert uint8 RGB images and float32 depth maps to RGB SBS images.
        """

        # 1) Validate inputs
        batch_size = self.core._validate_inputs(base_image, depth_image)
        invert_depth = True
        height, width = self.H, self.W

        if blur_radius > 0 and blur_radius % 2 == 0:
            raise ValueError("blur_radius must be odd or <= 0")

        # 2) Prepare depth maps
        depth_batch = self.core._prepare_depth_batch(depth_image, invert_depth)

        # 3) Calculate shift parameters
        flip_offset = 0
        depth_scale_local = depth_scale * width * 50.0 / 1000000.0
        depth_offset_local = depth_offset * -8

        if symetric:
            depth_scale_local = depth_scale_local / 2.0
            depth_offset_local = depth_offset_local / 2.0

        if invert_depth:
            depth_offset_local = -depth_offset_local

        crop_size = max(-width, min(width, int(crop_size)))

        # 4) Build the main-view shift field
        pixel_shifts_main = depth_batch * depth_scale_local
        pixel_shifts_main += depth_offset_local
        pixel_shifts_main = self.core._blur_batch(pixel_shifts_main, blur_radius)

        # 5) Prepare RGB SBS and RGBA workspaces
        sbs_batch = torch.empty((batch_size, height, width * 2, 3), dtype=torch.uint8, device=base_image.device)
        sbs_batch[:, :, width:] = base_image

        rgba_batch = self._rgba_workspace[:batch_size]
        rgba_batch[..., :3].copy_(base_image)

        # 6) Render the main shifted view
        shifted_main = apply_subpixel_shift_texture_batch(rgba_batch, pixel_shifts_main, self._remapper)
        sbs_batch[:, :, flip_offset:flip_offset + width] = shifted_main

        # 7) Render the symmetric shifted view
        if symetric:
            symmetric_offset = width - flip_offset
            pixel_shifts_sym = depth_batch * (-depth_scale_local)
            pixel_shifts_sym += depth_offset_local
            pixel_shifts_sym = self.core._blur_batch(pixel_shifts_sym, blur_radius)

            shifted_sym = apply_subpixel_shift_texture_batch(rgba_batch, pixel_shifts_sym, self._remapper)
            sbs_batch[:, :, symmetric_offset:symmetric_offset + width] = shifted_sym

        # 8) Apply crop and layout options
        sbs_batch = self._crop_blackout_and_swap(sbs_batch, crop_size, crop_size, symetric, switch_sides)

        return sbs_batch


class NV12SBSConverter:
    """
    Convert RGB and depth batches directly to NV12 SBS frames.
    """

    def __init__(self, core: DIBRCore):
        self.core = core
        self.H = core.H
        self.W = core.W
        self.Bmax = core.Bmax

        self._remapper = RemapTextureNV12(self.H, self.W, Bmax=self.Bmax, device=core.device)
        self._rgba_workspace = torch.empty((self.Bmax, self.H, self.W, 4), dtype=torch.uint8, device=core.device)
        self._rgba_workspace[..., 3].fill_(255)
        
        
    def set_output_buffer(self, output: NV12CudaBatch) -> None:
        expected = (self.Bmax, self.H, self.W * 2)
        actual = (output.Bmax, output.H, output.W)

        if actual != expected:
            raise ValueError(f"NV12 output buffer must be {expected}, got {actual}")
        if output.device != self.core.device:
            raise ValueError(f"NV12 output buffer must be on {self.core.device}, got {output.device}")

        self._remapper.output = output

    def _build_mapx(self, pixel_shifts: torch.Tensor) -> torch.Tensor:
        """
        Build the horizontal inverse-remap map.
        """
        return self._remapper.build_mapx(pixel_shifts)

    def process(
        self,
        base_image: torch.Tensor,
        depth_image: torch.Tensor,
        depth_scale: float,
        depth_offset: float,
        crop_size: int,
        switch_sides: bool,
        blur_radius: int,
        symetric: bool,
    ) -> "NV12CudaBatch":
        """
        Convert uint8 RGB images and float32 depth maps to NV12 SBS frames.
        """

        # 1) Validate inputs
        batch_size = self.core._validate_inputs(base_image, depth_image)
        invert_depth = True
        width = self.W

        if blur_radius > 0 and blur_radius % 2 == 0:
            raise ValueError("blur_radius must be odd or <= 0")

        # 2) Prepare depth maps
        depth_batch = self.core._prepare_depth_batch(depth_image, invert_depth)

        # 3) Calculate shift parameters
        depth_scale_local = depth_scale * width * 50.0 / 1000000.0
        depth_offset_local = depth_offset * -8

        if symetric:
            depth_scale_local /= 2.0
            depth_offset_local /= 2.0

        if invert_depth:
            depth_offset_local = -depth_offset_local

        crop_size = max(-width, min(width, int(crop_size)))

        # 4) Build the main-view remap map
        pixel_shifts_main = depth_batch * depth_scale_local
        pixel_shifts_main += depth_offset_local
        pixel_shifts_main = self.core._blur_batch(pixel_shifts_main, blur_radius)
        mapx_main = self._build_mapx(pixel_shifts_main)

        # 5) Build the symmetric-view remap map
        mapx_sym = None

        if symetric:
            pixel_shifts_sym = depth_batch * (-depth_scale_local)
            pixel_shifts_sym += depth_offset_local
            pixel_shifts_sym = self.core._blur_batch(pixel_shifts_sym, blur_radius)
            mapx_sym = self._build_mapx(pixel_shifts_sym)

        # 6) Prepare the RGBA texture source
        rgba_batch = self._rgba_workspace[:batch_size]
        rgba_batch[..., :3].copy_(base_image)
        self._remapper.prepare_source(rgba_batch)

        # 7) Select output halves without copying or swapping
        main_offset = width if switch_sides else 0
        second_offset = 0 if switch_sides else width

        # 8) Render the main shifted view
        self._remapper.render_warp_half(mapx_main, dst_x=main_offset, crop=crop_size)

        # 9) Render the second view
        if symetric:
            self._remapper.render_warp_half(mapx_sym, dst_x=second_offset, crop=-crop_size)
        else:
            self._remapper.render_copy_half(base_image, dst_x=second_offset)

        # 10) Return encoder-compatible NV12 views
        return self._remapper.get_batch(batch_size)
