ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install the MMDetection3D required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install StreamPETR required packages
RUN pip install flash-attn==0.2.2 \
    && pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html \
    && pip install mmdet==2.28.2 \
    && pip install mmsegmentation==0.30.0

# Install MMDetection3D
RUN conda clean --all \
    && git clone https://github.com/exiawsh/StreamPETR \
    && cd StreamPETR \
    && git clone https://github.com/open-mmlab/mmdetection3d.git \
    && cd mmdetection3d \
    && git checkout v1.0.0rc6 \
    && pip install --no-cache-dir -e .

# Additional missing packages and error fixes
RUN pip install fvcore yapf==0.40.1
RUN pip install wandb
RUN git config --global --add safe.directory '*'

WORKDIR /workspace/StreamPETR/