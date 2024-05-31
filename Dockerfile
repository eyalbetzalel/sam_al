# Use the official PyTorch image as the base image
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# Set the working directory
WORKDIR /workspace

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Clone the GitHub repositories
COPY sam_al /workspace/sam_al

# Set the working directory to the cloned repository
WORKDIR /workspace/sam_al

# Install Python packages
RUN pip install --upgrade pip
RUN pip install \
    torchvision \
    matplotlib \
    numpy \
    opencv-python-headless \
    wandb \
    pandas

# Install Segment Anything
RUN pip install git+https://github.com/facebookresearch/segment-anything.git

# Optional dependencies for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format
RUN pip install opencv-python pycocotools matplotlib onnxruntime onnx

# Copy the SAM model file into the container (if needed)
COPY sam_vit_h_4b8939.pth /workspace/sam_al/path-to-your-model-directory/
