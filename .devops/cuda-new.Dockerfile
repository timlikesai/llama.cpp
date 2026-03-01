ARG UBUNTU_VERSION=24.04
# This needs to generally match the container host's environment.
ARG CUDA_VERSION=13.1.0
# Target the CUDA build image
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER} AS build

# CUDA architecture to build for (defaults to all supported archs)
ARG CUDA_DOCKER_ARCH=default

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && \
    apt-get install -y build-essential cmake python3 python3-pip git libssl-dev libgomp1

WORKDIR /app

COPY . .

# BuildKit cache mount: /app/build persists across builds for incremental compilation.
# Even when COPY invalidates, cmake only recompiles changed files (~seconds vs minutes).
RUN --mount=type=cache,target=/app/build \
    if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
        export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=OFF ${CMAKE_ARGS} -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . && \
    cmake --build build --config Release -j$(nproc) && \
    mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \; && \
    mkdir -p /app/full && \
    cp build/bin/* /app/full && \
    cp *.py /app/full && \
    cp -r gguf-py /app/full && \
    cp -r requirements /app/full && \
    cp requirements.txt /app/full && \
    cp .devops/tools.sh /app/full/tools.sh

## Base runtime deps — cached independently of build output.
## All runtime stages inherit from here so apt installs are never re-run
## just because source code changed.
FROM ${BASE_CUDA_RUN_CONTAINER} AS base-deps

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y libgomp1 curl

## Base image (deps + shared libs)
FROM base-deps AS base

COPY --from=build /app/lib/ /app

## Full deps — adds python/pip on top of base-deps, still independent of build output.
## requirements.txt is copied directly from context (not from build) so this layer
## only invalidates when requirements actually change, not on every code edit.
FROM base-deps AS full-deps

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y git python3 python3-pip python3-wheel

COPY requirements.txt /app/requirements.txt
COPY requirements/ /app/requirements/

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages --upgrade setuptools && \
    pip install --break-system-packages -r requirements.txt

### Full
FROM full-deps AS full

COPY --from=build /app/lib/ /app/
COPY --from=build /app/full /app

WORKDIR /app

ENTRYPOINT ["/app/tools.sh"]

### Light, CLI only
FROM base AS light

COPY --from=build /app/full/llama-cli /app/full/llama-completion /app

WORKDIR /app

ENTRYPOINT [ "/app/llama-cli" ]

### Server, Server only
FROM base AS server

ENV LLAMA_ARG_HOST=0.0.0.0

COPY --from=build /app/full/llama-server /app

WORKDIR /app

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
