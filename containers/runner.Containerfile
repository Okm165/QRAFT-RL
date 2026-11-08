ARG LLAMA_IMAGE=qraft-rl/llama-27b:local
ARG BASE_IMAGE=qraft-rl/model-base-local:local

FROM ${LLAMA_IMAGE} AS llama
FROM ${BASE_IMAGE}

WORKDIR /app
COPY telemetry.py ./telemetry.py
COPY model_family.py ./model_family.py
COPY preprocess ./preprocess
COPY rag ./rag
COPY raft ./raft
COPY sft ./sft
COPY cli ./cli
COPY llm/__init__.py ./llm/__init__.py
COPY llm/chat.py ./llm/chat.py
COPY llm/client.py ./llm/client.py
COPY llm/structured.py ./llm/structured.py
COPY llm/judge.py ./llm/judge.py
COPY llm/questions.py ./llm/questions.py
COPY eval ./eval
COPY tools/__init__.py ./tools/__init__.py
COPY tools/run_layout.py ./tools/run_layout.py
COPY tools/run_diagnostics.py ./tools/run_diagnostics.py
COPY tools/run_quality.py ./tools/run_quality.py
COPY tools/run_metrics.py ./tools/run_metrics.py
COPY tools/telemetry_report.py ./tools/telemetry_report.py
COPY tools/hardware.py ./tools/hardware.py
COPY tools/hardware_check.py ./tools/hardware_check.py
COPY tools/p2p.py ./tools/p2p.py
COPY tools/p2p_check.py ./tools/p2p_check.py
COPY tools/export_lora.py ./tools/export_lora.py

COPY --from=llama /opt/llama.cpp/convert_lora_to_gguf.py /opt/llama.cpp/convert_lora_to_gguf.py
COPY --from=llama /opt/llama.cpp/conversion /opt/llama.cpp/conversion
COPY --from=llama /opt/llama.cpp/gguf-py/gguf /opt/llama.cpp/gguf-py/gguf

ENTRYPOINT ["python3"]
