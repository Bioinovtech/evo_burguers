FROM --platform=linux/amd64 nvcr.io/nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04
LABEL maintainer="BIOINOVTECH info@bioinovtech.com"
LABEL org.opencontainers.image.name="evo-burgers"
LABEL org.opencontainers.image.version="latest"
LABEL org.opencontainers.image.description="Evolutionary Burgers AI Application"
LABEL org.opencontainers.image.source="https://github.com/bioinovtech/evo-burgers"
LABEL org.opencontainers.image.licenses="MIT"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Mambaforge (includes mamba and conda) - using specific version for stability
RUN curl -L -o mambaforge.sh https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-25.3.0-3-Linux-x86_64.sh \
    && chmod +x mambaforge.sh \
    && bash mambaforge.sh -b -p /opt/mamba \
    && rm mambaforge.sh

# Add mamba to path
ENV PATH /opt/mamba/bin:$PATH

# Create non-root user
RUN useradd -m -s /bin/bash evoburgers
RUN chown -R evoburgers:evoburgers /opt/mamba

# Create working directory
WORKDIR /app

# Copy environment file
COPY --chown=evoburgers:evoburgers environment.yml .

# Create conda environment with cleanup
RUN mamba env create -f environment.yml --yes \
    && mamba clean --all --yes \
    && rm -rf /tmp/conda-pkgs \
    && rm -rf /tmp/*

# Activate conda environment
SHELL ["mamba", "run", "-n", "hamburguer", "/bin/bash", "-c"]

# Copy the rest of the application
COPY --chown=evoburgers:evoburgers . .

# Switch to non-root user
USER evoburgers

# Set the default command
ENTRYPOINT ["mamba", "run", "-n", "hamburguer", "python", "token_ev_ga.py"] 