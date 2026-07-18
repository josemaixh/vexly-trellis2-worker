FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# --- Internal testing/prototyping build only ---
# nvdiffrec (a TRELLIS.2 dependency for PBR rendering) is licensed under the
# NVIDIA Source Code License-NC, which restricts use to non-commercial
# research/evaluation. Do NOT point production/customer-facing traffic at
# this image until that's resolved — see the TRELLIS.2 GitHub issue #22.

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    ATTN_BACKEND=flash-attn \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TORCH_CUDA_ARCH_LIST="8.0;9.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git wget ninja-build libjpeg-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# PyTorch 2.6.0 + CUDA 12.4 (matches TRELLIS.2's tested configuration)
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone TRELLIS.2 with submodules
WORKDIR /workspace
RUN git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
WORKDIR /workspace/TRELLIS.2

# Basic Python dependencies (mirrors setup.sh --basic)
RUN pip install --no-cache-dir \
    imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
    trimesh transformers gradio==6.0.1 tensorboard pandas lpips zstandard \
    kornia timm plyfile \
    && pip install --no-cache-dir utils3d

# flash-attention — compiled from source. MAX_JOBS caps parallel compile
# workers so this fits within GitHub Actions' free-runner RAM (~7GB).
# This step is slow (expect 30-60+ min) but only needs to happen once per
# image rebuild, well within GitHub Actions' free monthly minutes.
ENV MAX_JOBS=2
RUN pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

# Remaining components: nvdiffrast, nvdiffrec, cumesh, o-voxel, flexgemm
# (--new-env and --basic are skipped since we've handled those manually above)
# setup.sh only checks that an `nvidia-smi` command exists (not that a GPU is
# actually attached) before proceeding, so on a GPU-less build machine like a
# GitHub Actions runner we provide a harmless stand-in to pass that check.
RUN echo '#!/bin/bash' > /usr/local/bin/nvidia-smi && \
    chmod +x /usr/local/bin/nvidia-smi
RUN bash -c ". ./setup.sh --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm"

# RunPod worker SDK + HF download helper
RUN pip install --no-cache-dir runpod huggingface_hub

# Bake the TRELLIS.2-4B weights into the image so workers don't download on cold start
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/TRELLIS.2-4B')"

# RunPod handler
COPY handler.py /workspace/TRELLIS.2/handler.py

CMD ["python", "-u", "handler.py"]
