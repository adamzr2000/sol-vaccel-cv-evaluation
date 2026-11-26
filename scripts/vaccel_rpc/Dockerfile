# Build environment for the patched vAccel RPC client/agent (scripts/vaccel_rpc/build.sh).
# Based on the SOL agent image so the vAccel 0.8.0 headers/library match the ones the binaries run with.
FROM torchvision-app:agent
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential pkg-config \
        libclang-dev clang git libcurl4-openssl-dev && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.88.0
ENV PATH=/root/.cargo/bin:$PATH
ENTRYPOINT []
