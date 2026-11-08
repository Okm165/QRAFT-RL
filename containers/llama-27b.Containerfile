ARG BASE_IMAGE=docker.io/rocm/dev-ubuntu-24.04:7.2.4-complete
FROM ${BASE_IMAGE}

# llama.cpp ROCm build for gfx1100 (three 24 GiB RDNA3 cards).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp

RUN cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
      -DGGML_HIP=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_CURL=OFF \
      -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
      -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
    && cmake --build /opt/llama.cpp/build --config Release --target llama-server -j 16

ENV PATH=/opt/llama.cpp/build/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:/opt/rocm/lib

CMD ["llama-server", "--help"]
