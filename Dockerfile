FROM ghcr.io/nerfstudio-project/nerfstudio:latest

RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

