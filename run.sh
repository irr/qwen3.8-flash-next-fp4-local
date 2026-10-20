#!/usr/bin/env bash
# run.sh — Serve nvidia/Qwen3.8-Flash-Next-FP4-Local with YaRN 1M context
# on a single NVIDIA DGX Spark using eugr/spark-vllm-b12x
#
# Usage:
#   ./run.sh                 # native 262k
#   MAX_MODEL_LEN=1000000 ./run.sh   # 1M with YaRN (factor auto-computed = 4.0)
#   MAX_MODEL_LEN=524288 YARN_FACTOR=2.0 ./run.sh   # explicit YaRN factor
#
# Requirements: Docker + NVIDIA Container Toolkit, DGX Spark / GB10,
# model weights (~96 GB) in MODEL_DIR (default ~/models/Qwen3.8-Flash-Next-FP4-Local)

set -euo pipefail

# ---------- configuration (override via environment) ----------
MODEL_ID="${MODEL_ID:-nvidia/Qwen3.8-Flash-Next-FP4-Local}"    # name exposed by the API
MODEL_DIR="${MODEL_DIR:-${HOME}/models/Qwen3.8-Flash-Next-FP4-Local}"  # local weights on the host
IMAGE="${IMAGE:-eugr/spark-vllm-b12x:latest}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
#MAX_MODEL_LEN="${MAX_MODEL_LEN:-524288}"          # set to 1000000 for 1M YaRN
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"          # set to 1000000 for 1M YaRN
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"                 # 1 for true 1M / single-user latency; 4–8 for 262k
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MTP_TOKENS="${MTP_TOKENS:-4}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next}"
NATIVE_CTX=262144                                 # model's native context
YARN_FACTOR="${YARN_FACTOR:-}"                    # empty = ceil(MAX_MODEL_LEN / NATIVE_CTX)

# ---------- validate ----------
if [ ! -f "${MODEL_DIR}/config.json" ]; then
  echo "ERROR: no model found in MODEL_DIR=${MODEL_DIR} (missing config.json)." >&2
  exit 1
fi

# YaRN: automatically enable when requesting > native context
YARN_ENABLED=0
if [ "${MAX_MODEL_LEN}" -gt "${NATIVE_CTX}" ]; then
  YARN_ENABLED=1
  # Smallest whole-number factor that covers MAX_MODEL_LEN
  if [ -z "${YARN_FACTOR}" ]; then
    YARN_FACTOR=$(awk -v m="${MAX_MODEL_LEN}" -v n="${NATIVE_CTX}" \
      'BEGIN { f = m / n; c = int(f); if (c < f) c++; printf "%.1f", c }')
  fi
  YARN_MAX=$(awk -v f="${YARN_FACTOR}" -v n="${NATIVE_CTX}" 'BEGIN { printf "%d", f * n }')
  if [ "${MAX_MODEL_LEN}" -gt "${YARN_MAX}" ]; then
    echo "ERROR: MAX_MODEL_LEN=${MAX_MODEL_LEN} exceeds YaRN window ${YARN_MAX} (factor ${YARN_FACTOR})." >&2
    echo "       Increase YARN_FACTOR or unset it to auto-compute." >&2
    exit 1
  fi
fi

echo "=== Qwen3.8-Flash-Next on DGX Spark ==="
echo "Model:          ${MODEL_ID}"
echo "Model dir:      ${MODEL_DIR}"
echo "Image:          ${IMAGE}"
echo "Max context:    ${MAX_MODEL_LEN} (YaRN: $([ $YARN_ENABLED -eq 1 ] && echo "ON, factor ${YARN_FACTOR}" || echo OFF))"
echo "Max seqs:       ${MAX_NUM_SEQS}"
echo "GPU mem util:   ${GPU_MEMORY_UTILIZATION}"
echo "Port:           ${PORT}"
echo

# ---------- prepare host caches ----------
mkdir -p \
  "${HOME}/.cache/huggingface" \
  "${HOME}/.cache/vllm" \
  "${HOME}/.cache/flashinfer" \
  "${HOME}/.triton" \
  "${HOME}/.cache/b12x"

# ---------- pull image ----------
echo "Pulling container image (if needed)..."
docker pull "${IMAGE}"

# ---------- stop any previous instance ----------
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Stopping existing container ${CONTAINER_NAME}..."
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

# ---------- build YaRN override ----------
HF_OVERRIDES=""
if [ $YARN_ENABLED -eq 1 ]; then
  HF_OVERRIDES='--hf-overrides {"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":'"${YARN_FACTOR}"',"original_max_position_embeddings":'"${NATIVE_CTX}"'}}}'
  echo "YaRN factor ${YARN_FACTOR} enabled (${NATIVE_CTX} → ${YARN_MAX})"
fi

# ---------- speculative decoding (omit flag when MTP_TOKENS=0) ----------
SPEC_FLAG=""
if [ "${MTP_TOKENS}" -gt 0 ]; then
  SPEC_FLAG="--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}"
fi

# ---------- launch ----------
echo "Starting vLLM server..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_PLE_TABLE_MEMORY=disk \
  -e B12X_DISK_BACKEND=io_uring \
  -e CUTE_DSL_ARCH=sm_121a \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e B12X_POLICY_MODE=auto \
  -e VLLM_MXFP8_LM_HEAD=1 \
  -e VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE=w13 \
  -e VLLM_B12X_MOE_FP4_FORCE_A16=1 \
  -e B12X_W4A16_A4_PREFILL=1 \
  -v "${MODEL_DIR}:/models/model:ro" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -v "${HOME}/.cache/vllm:/root/.cache/vllm" \
  -v "${HOME}/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "${HOME}/.triton:/root/.triton" \
  -v "${HOME}/.cache/b12x:/root/.cache/b12x" \
  "${IMAGE}" \
  vllm serve /models/model \
    --served-model-name "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --mamba-cache-mode align \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --quantization modelopt_mixed \
    --block-size 16 \
    --load-format b12x \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    ${SPEC_FLAG} \
    --gdn-decode-kernel b12x \
    --linear-backend b12x \
    --moe-backend b12x \
    --no-enable-flashinfer-autotune \
    --mm-encoder-tp-mode data \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice \
    --compilation-config '{"pass_config":{"fuse_act_quant":true}}' \
    ${HF_OVERRIDES}

echo
echo "Container started: ${CONTAINER_NAME}"
echo "Follow logs:       docker logs -f ${CONTAINER_NAME}"
echo "API endpoint:      http://localhost:${PORT}/v1"
echo
echo "First launch may take 10–20+ minutes"
echo "(weight load + kernel autotune). Subsequent starts are much faster."
echo
echo "Test when ready:"
echo "  curl http://localhost:${PORT}/v1/models"
echo "  curl http://localhost:${PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${MODEL_ID}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":128}'"
