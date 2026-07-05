# Getting Started: QwenVL Docker on a New Machine

This guide walks you through setting up and running the [QwenVL Docker environment](Dockerfile) on a new machine that may have a different GPU than the one originally used.

---

## 1. Prerequisites

Make sure the following are installed on the new machine:

- **Docker** (≥ 20.10)
- **NVIDIA Container Toolkit** (`nvidia-docker2` or `nvidia-container-toolkit`)
- **NVIDIA GPU driver** (see CUDA compatibility below)

Verify your setup:
```bash
docker --version
nvidia-smi           # confirm GPU is visible and driver is loaded
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi  # confirm Docker can see GPUs
```

---

## 2. ⚠️ CUDA Compatibility Check (Critical for Different GPUs)

The Dockerfile is based on:
```
pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
```
This requires **CUDA 12.1** or a driver that supports it.

| GPU Architecture | Example GPUs | Min Driver Version |
|---|---|---|
| Ampere | A100, A6000, RTX 30xx | ≥ 525.x |
| Ada Lovelace | RTX 40xx, L40 | ≥ 525.x |
| Hopper | H100 | ≥ 525.x |
| Turing | RTX 20xx, T4 | ≥ 525.x |

> [!IMPORTANT]
> Run `nvidia-smi` and check the **CUDA Version** shown in the top-right corner. It must be **≥ 12.1**.
> If you're on an older driver/CUDA, you'll need to either update the driver or change the base image in the Dockerfile (e.g., use `cuda11.8` variant).

**If your driver supports a different CUDA version**, update line 1 of the [Dockerfile](Dockerfile):
```dockerfile
# Example for CUDA 11.8:
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
```
Available PyTorch base images: https://hub.docker.com/r/pytorch/pytorch/tags

---

## 3. Clone / Copy the Repo

If starting fresh on the new machine:
```bash
git clone <your-repo-url> /path/to/o3-d
```

Ensure the `docker/` directory is present at:
```
o3-d/src/vlm_qwenvl/docker/
├── Dockerfile
├── build_docker.sh
├── run_docker.sh
└── requirements.txt
```

---

## 4. Build the Docker Image

Navigate to the repo root and run:
```bash
cd /path/to/o3-d/src/vlm_qwenvl/docker
bash build_docker.sh
```

This builds with the default image name `base_images/pytorch:torch2.3`. The build script automatically passes your current user/group IDs so file permissions inside the container match your host user (`docker_user`).

**Custom image name or tag:**
```bash
bash build_docker.sh --image_name my_images/qwenvl --tag v1
```

> [!NOTE]
> The build installs all packages from [requirements.txt](requirements.txt), including `torch==2.3.1`, `transformers==4.46.1`, `xformers`, `bitsandbytes` (for int4 quantization), and `auto-gptq`. This may take **10–20 minutes** on first build.

---

## 5. Configure Paths in `run_docker.sh`

Before running, open [run_docker.sh](run_docker.sh) and update the two paths at the top (lines 12–13):

```bash
# ---- UPDATE THESE ----
CODE_FOLDER=/path/to/o3-d/          # absolute path to the repo root
DATA_FOLDER=/path/to/o3-d/data/     # absolute path to your data directory
```

The HuggingFace cache mount (line 120) defaults to `${HOME}/.cache/huggingface` — adjust this if your cache lives elsewhere.

---

## 6. Select Your GPU

The run script defaults to GPU device `0`. To use a different GPU:

```bash
bash run_docker.sh --gpu_device 1        # use GPU 1
bash run_docker.sh --gpu_device "0,1"    # use GPUs 0 and 1 (multi-GPU)
```

> [!TIP]
> Run `nvidia-smi` first to see which GPU indices are available and their memory. QwenVL-72B requires ~80GB VRAM; smaller variants (7B) fit in ~16GB with int4 quantization.

---

## 7. Run the Container

```bash
cd /path/to/o3-d/src/vlm_qwenvl/docker
bash run_docker.sh
```

You'll land in an interactive bash shell inside the container. The working directory inside the container is set to `o3-d/src/vlm_qwenvl/`.

**Other useful flags:**
```bash
bash run_docker.sh --gpu_device 0 --memory_limit 64g --container_name MyQwenRun
```

---

## 8. HuggingFace Model Download (First Run)

On first use, the model will be downloaded automatically via `transformers`. To pre-download on the host and avoid repeated downloads:

```bash
# On the host, before starting the container:
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --cache-dir ~/.cache/huggingface
```

The container maps your host's HF cache to `/home/docker_user/.hf_cache` (with `HF_HOME` set accordingly), so downloaded models persist across container restarts.

---

## 9. Troubleshooting

| Problem | Solution |
|---|---|
| `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]` | Install `nvidia-container-toolkit` and restart Docker: `sudo apt install nvidia-container-toolkit && sudo systemctl restart docker` |
| `CUDA error: no kernel image is available for execution on the device` | Your GPU arch isn't supported by the CUDA version in the base image. Change the base image to one compiled for your GPU. |
| HuggingFace mount fails (path doesn't exist) | Create the directory first: `mkdir -p ~/.cache/huggingface` |
| `bitsandbytes` errors with int4 quantization | Ensure your GPU supports int4 (Turing/Ampere+). For older GPUs, remove quantization args in your inference call. |
| `auto-gptq` install fails on newer GPU | Try pinning to a compatible version or replacing with `optimum` + `gptq` backend per your GPU. |

---

## Summary of Key Customizations per New Machine

| What to change | Where |
|---|---|
| Base CUDA version (if needed) | [Dockerfile](Dockerfile) line 1 |
| Code and data folder paths | [run_docker.sh](run_docker.sh) lines 12–13 |
| HuggingFace cache path | [run_docker.sh](run_docker.sh) line 120 |
| GPU device index | `--gpu_device` flag at runtime |
| Memory limit | `--memory_limit` flag at runtime |
