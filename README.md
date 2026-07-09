# VR we are! (CLI)

Fast, cross-platform **video → stereoscopic SBS** converter powered by **Depth Anything V2**.
Built for batch work, tuned for GPUs, and designed with a **multi-threaded** pipeline that keeps your CPU, GPU and I/O busy.



## ✨ Features

* **Video → SBS video**, **Folder with frames → SBS images**, **Image → Image (i2i)**
* **Presets** (minimum / balance / max_quality) for speed/quality/VRAM/RAM targets
* **CUDA + NVENC** support (auto-detects and switches when possible)
* **Pipeline parallelism** with queues: feeder → preprocess → GPU → convert → save
* Depth via **Depth Anything V2 (Small/Base/Large)**
* **Debugging**: Queue/Memory Monitor and Summary Report
* **10-bit HDR video path** with HEVC/NVENC/libx265 support
* Single-file **bootstrapper** (`setup_env.py`) to install PyTorch/FFmpeg-related dependencies



## 🔗 Related project (direct "relative")

This CLI grew out of the ComfyUI project:

* **VR we are! (ComfyUI nodes)** – stereoscopic nodes & docs
  GitHub: [https://github.com/FortunaCournot/comfyui_stereoscopic](https://github.com/FortunaCournot/comfyui_stereoscopic)

Huge thanks to **Fortuna** (author of nodes/docs) and the ComfyUI community.

## Installation

> Tested with **Python 3.12**.

1. **Install Python 3.12**
   [https://www.python.org/downloads/](https://www.python.org/downloads/)

2. **Get the sources**

   * Clone the repo: `git clone https://github.com/Iablunoshka/VR-we-are-CLI`
     or unzip a release to a folder.

3. **Create & activate venv**

Create venv
```bash
python -m venv venv
```
Activating venv *Windows*
```bash
venv\Scripts\activate
```
or *Linux*
```bash
source venv/bin/activate
```

4. **Run the bootstrapper**

```bash
python setup_env.py
```

This will:

* install packages from `requirements.txt`,
* detect **CUDA** and install a matching **PyTorch** wheel (falls back to CPU build),
* check **FFmpeg** availability (prints tips if missing).

> FFmpeg must be in `PATH`.
> Windows (Chocolatey): `choco install ffmpeg`
> Ubuntu/Debian: `sudo apt install ffmpeg`



## 🚀 Quick start

### Video → SBS video

```bash
python main.py -i input.mp4 -o output_sbs.mp4 --preset balance
```

### Folder of frames → SBS PNGs

```bash
python main.py -i ./frames -o ./out --input-type folder --preset balance
```

For repeated folder benchmarks/tests, existing PNG outputs can be removed automatically:

```bash
python main.py -i ./frames -o ./out --input-type folder --preset balance --clean-output-pngs
```

### Single image (i2i)

One image
```bash
python main.py -i image.png -o image_sbs.png --input-type i2i
```
Several images
```bash
python main.py -i ./Images -o ./out --input-type i2i
```

### Debug mode

To see detailed performance stats and system info, add `--debug`:
```bash
python main.py -i input.mp4 -o output_sbs.mp4 --preset balance --debug
```




## ⚡ Performance Presets (Full HD, Depth-Anything-V2)
Benchmarks measured on **AMD Ryzen 7 7700X + NVIDIA GeForce RTX 5090 + 32 GB DDR5**  
*(Windows 11, CUDA 13, Python 3.12)*

*AMP autocast: **bfloat16***

> **Note:**
> 
> Presets are designed for convenience and to provide stable performance baselines.
> This pipeline does not impose strict limits on RAM, VRAM, or other system resources - if something crashes due to lack of resources, it will usually not be prevented automatically.
>
> Therefore, it is highly recommended to use presets as a starting point (for fine-tuning via `--debug`), but do not assume you will get identical performance results, as many variables affect it - most importantly, the balance of your hardware and operating system.



###  Presets - Folder Mode (Full HD)

| Preset | Typical VRAM Usage | Model | Batch | Feeders | Preprocess | Processors | Savers | Queues (r/in/p/s) | RAM Usage (avg/max GB) | FPS | Notes |
|:-------|:-------------|:-------|:-------|:---------|:------------|:------------|:--------|:------------------|:------------------------|:------|:------|
| **Minimum** | 2.2 GB | Depth-Anything-V2-Small-hf | 4 | 1 | 1 | 4 | 4 | 8 | 3.4 / 3.6 | **36.5 FPS** | Optimized for low-VRAM/RAM system |
| **Balance** | 4.5 GB | Depth-Anything-V2-Base-hf | 5 | 2 | 3 | 8 | 5 | 16 | 6.4 / 7.1 | **40.7 FPS** | Best overall performance |
| **Max Quality** | 11.7 GB | Depth-Anything-V2-Large-hf | 8 | 2 | 2 | 8 | 5 | 16 | 5.7 / 6.5 | **40.0 FPS** | Highest depth accuracy, GPU-bound |

###  Presets - Video Mode (Full HD)
| Preset | Typical VRAM Usage | Model | Batch | Feeders | Preprocess | Processors | Savers | Queues (r/in/p/s) | RAM Usage (avg/max GB) | FPS | Notes |
|:-------|:-------------|:-------|:-------|:---------|:------------|:------------|:--------|:------------------|:------------------------|:------|:------|
| **Minimum** | 2.4 GB | Depth-Anything-V2-Small-hf | 3 | 1 | 1 | 4 | 1 | 6 | 3.5 / 3.8 | **36.7 FPS** | Optimized for low-VRAM/RAM system |
| **Balance** | 5.2 GB | Depth-Anything-V2-Base-hf | 5 | 1 | 2 | 8 | 1 | 16 | 5.5 / 6.0 | **39.5 FPS** | Best overall performance |
| **Max Quality** | 8.5 GB | Depth-Anything-V2-Large-hf | 5 | 1 | 2 | 8 | 1 | 16 | 5.3 / 5.7 | **37.0 FPS** | Highest depth accuracy, GPU-bound |

---

#### Linux Performance

- On **Linux** (tested on *Ubuntu 24.04 + CUDA 12*), performance is **significantly higher** compared to Windows.  
- In some cases, the save queue may fill faster - if RAM usage increases over time, raise the number of savers (`--savers +1…2`).  
- For best stability and throughput, **Linux is the recommended platform**.

#### Windows Performance

- After several hours of system uptime, performance may drop by **≈25 %**, likely due to Windows scheduler or GPU driver throttling (not related to the script).  
- To ensure consistent results, **reboot before long renders or benchmarking**.  
- The presets above were benchmarked on **Windows 11**, so FPS values reflect typical Windows performance.


## Project structure

```
├─ main.py               # CLI entrypoint (initialization, CLI-interface, launch)
├─ pipeline_core.py      # Worker threads (feeder → preprocess → GPU → process → save)
├─ depthestimator.py     # Depth Anything V2 (HF Transformers + PyTorch)
├─ converter.py          # SBS image conversion (Numba + OpenCV)
├─ sbsutils.py           # NVENC detect, presets merge, system info, debug report
├─ monitor.py            # Queue/Memory monitors + plots
├─ presets.json          # Ready presets (video/folder)
├─ test_cli.py           # Testing operational
├─ setup_env.py          # Bootstrap PyTorch/FFmpeg and requirements
├─ torch_detect.py       # Torch check (setup_env.py component)
├─ hdr.py                # Isolated 10-bit HDR support
├─ requirements.txt
└─ README.md
```

## Credits

* **Fortuna** - original ComfyUI nodes, docs, and project leadership
* Community testers and artists who shared feedback and examples

## Acknowledgements

This project uses the following open-source Python libraries:
`numpy`, `numba`, `opencv-python`, `natsort`, `psutil`, `py-cpuinfo`,
`matplotlib`, and `transformers`.

It also uses models from the project:
[Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2).

## 💬 Contact

* GitHub Issues: [https://github.com/Iablunoshka/VR-we-are-CLI/issues](https://github.com/Iablunoshka/VR-we-are-CLI/issues)
* Discord: [Activation Link](https://discord.gg/ZegT6Cc8FG)


