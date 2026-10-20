# Qwen3.8-Flash-Next FP4 — Local Serving on DGX Spark

Scripts to serve [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) on a single **NVIDIA DGX Spark (GB10, `sm_121a`)** with vLLM, using the [`eugr/spark-vllm-b12x`](https://hub.docker.com/r/eugr/spark-vllm-b12x) container image. The server exposes an **OpenAI-compatible API** with reasoning parsing, tool calling, MTP speculative decoding and optional **YaRN** context extension beyond the native 262,144 tokens.

## Contents

| File      | Purpose |
|-----------|---------|
| `run.sh`  | Pulls the image, prepares host caches, removes any previous container and launches `vllm serve` in a detached Docker container. Every setting can be overridden with an environment variable. |
| `view.sh` | Follows the container logs (`docker logs -f qwen38-flash-next`). |

## Requirements

- NVIDIA DGX Spark / GB10 (the kernels are built for `sm_121a`)
- Docker with the NVIDIA Container Toolkit (`--gpus all` must work)
- The model weights (about 96 GB) in `~/models/Qwen3.8-Flash-Next-FP4-Local`, or another directory set with `MODEL_DIR`

## Quick start

To fetch the weights the first time:

```bash
hf download nvidia/Qwen3.8-Flash-Next-FP4-Local --local-dir ~/models/Qwen3.8-Flash-Next-FP4-Local
```

Then:

```bash
./run.sh          # native 262k context
./view.sh         # follow logs until the server reports it is ready
```

### Defaults (plain `./run.sh`)

With no overrides, the script serves the model at its **native 262,144-token context with YaRN off**. YaRN only turns on when `MAX_MODEL_LEN` is above 262144, so no factor is computed and no `--hf-overrides` is passed.

| Setting | Default |
|---|---|
| Model | `nvidia/Qwen3.8-Flash-Next-FP4-Local` |
| Weights | `~/models/Qwen3.8-Flash-Next-FP4-Local` |
| Image | `eugr/spark-vllm-b12x:latest` |
| API | `http://0.0.0.0:8000/v1` |
| Context (`MAX_MODEL_LEN`) | 262,144 (native, YaRN off) |
| `MAX_NUM_SEQS` | 4 |
| `MAX_NUM_BATCHED_TOKENS` | 4096 |
| `GPU_MEMORY_UTILIZATION` | 0.80 |
| MTP speculative tokens | 4 |
| Container name | `qwen38-flash-next` |

If `MODEL_DIR` has no `config.json`, the script exits with an error.

Common variations:

```bash
MAX_MODEL_LEN=1000000 MAX_NUM_SEQS=1 ./run.sh   # ~1M context, YaRN factor 4.0
MAX_NUM_SEQS=1 ./run.sh          # single user: lowest per-request latency, best MTP gain
```

The **first launch** runs kernel compilation and autotuning, so it can take **10–20+ minutes**. Later starts reuse the host caches and are much faster. The engine-ready timeout is set to 1 hour (`VLLM_ENGINE_READY_TIMEOUT_S=3600`) so the slow first start doesn't fail.

When it is ready, test it:

```bash
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/Qwen3.8-Flash-Next-FP4-Local","messages":[{"role":"user","content":"Hello"}],"max_tokens":128}'
```

Any OpenAI-compatible client works with base URL `http://<host>:8000/v1`.

## Configuration

Set any of these in the environment before you run `./run.sh`:

| Variable                 | Default                                 | Description |
|--------------------------|-----------------------------------------|-------------|
| `MODEL_DIR`              | `~/models/Qwen3.8-Flash-Next-FP4-Local` | Host directory with the model weights. Mounted read-only at `/models/model`. |
| `MODEL_ID`               | `nvidia/Qwen3.8-Flash-Next-FP4-Local`   | Model name exposed by the API (`--served-model-name`). |
| `IMAGE`                  | `eugr/spark-vllm-b12x:latest`           | Container image. |
| `PORT`                   | `8000`                                  | API port. The container uses host networking. |
| `HOST`                   | `0.0.0.0`                               | Bind address. |
| `MAX_MODEL_LEN`          | `262144`                                | Maximum context length. Values above 262144 turn on YaRN automatically. |
| `MAX_NUM_SEQS`           | `4`                                     | Maximum concurrent sequences. Use 1 for very long contexts or single-user latency, 4–8 at 262k. |
| `MAX_NUM_BATCHED_TOKENS` | `4096`                                  | Chunked-prefill batch size. |
| `GPU_MEMORY_UTILIZATION` | `0.80`                                  | Fraction of unified memory vLLM may use. |
| `MTP_TOKENS`             | `4`                                     | Number of MTP speculative tokens. `0` turns speculative decoding off. |
| `YARN_FACTOR`            | *(auto)*                                | YaRN scaling factor. If empty, it is computed as `MAX_MODEL_LEN / 262144`, rounded up to a whole number. Only used when YaRN is on. |
| `CONTAINER_NAME`         | `qwen38-flash-next`                     | Docker container name. `view.sh` hardcodes this name. |

Examples:

```bash
# Higher concurrency at native context
MAX_NUM_SEQS=8 ./run.sh

# Extended context with YaRN (factor auto-computed: 2.0)
MAX_MODEL_LEN=524288 ./run.sh

# ~1M context (factor auto-computed: 4.0)
MAX_MODEL_LEN=1000000 MAX_NUM_SEQS=1 ./run.sh

# Explicit YaRN factor
MAX_MODEL_LEN=524288 YARN_FACTOR=2.5 ./run.sh

# No speculative decoding, different port
MTP_TOKENS=0 PORT=8080 ./run.sh
```

## Long context (YaRN)

The model's native context is **262,144** tokens. If `MAX_MODEL_LEN` is larger than that, `run.sh` passes this `--hf-overrides` to the model's `text_config.rope_parameters`:

```json
{
  "mrope_interleaved": true,
  "mrope_section": [11, 11, 10],
  "rope_type": "yarn",
  "rope_theta": 10000000,
  "partial_rotary_factor": 0.25,
  "factor": <YARN_FACTOR>,
  "original_max_position_embeddings": 262144
}
```

The effective window is `factor × 262,144`. If `YARN_FACTOR` is unset, the script picks the smallest whole-number factor that covers `MAX_MODEL_LEN`:

| `MAX_MODEL_LEN` | Auto factor | Effective window |
|-----------------|-------------|------------------|
| 524,288         | 2.0         | 524,288          |
| 1,000,000       | 4.0         | 1,048,576        |

If you set `YARN_FACTOR` yourself and `MAX_MODEL_LEN` is larger than the window it gives, the script exits with an error rather than starting a server that runs past its context window. `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` is set so vLLM accepts lengths above the model config.

> **Tip:** For very long contexts, keep `MAX_NUM_SEQS=1` and check that the FP8 KV cache fits within `GPU_MEMORY_UTILIZATION`.

## Launch details

### Docker runtime

- `--gpus all --ipc host --network host --shm-size=16g`
- `--ulimit memlock=-1 --ulimit stack=67108864`
- `--security-opt seccomp=unconfined --security-opt apparmor=unconfined` (needed for the `io_uring` disk backend)

### Persistent host caches

The model directory and each cache directory are mounted into the container, so later starts skip recompilation:

| Host path                   | Container path                | Contents |
|-----------------------------|-------------------------------|----------|
| `$MODEL_DIR`                | `/models/model` (read-only)   | Model weights |
| `~/.cache/huggingface`      | `/root/.cache/huggingface`    | Hugging Face cache (the hub runs offline, `HF_HUB_OFFLINE=1`) |
| `~/.cache/vllm`             | `/root/.cache/vllm`           | vLLM compile / AOT artifacts |
| `~/.cache/flashinfer`       | `/root/.cache/flashinfer`     | FlashInfer kernels |
| `~/.triton`                 | `/root/.triton`               | Triton kernel cache |
| `~/.cache/b12x`             | `/root/.cache/b12x`           | b12x kernel / policy cache |

### Environment variables set in the container

| Variable | Value | Purpose |
|----------|-------|---------|
| `HF_HUB_OFFLINE` | `1` | Load only from the local model directory, never from the Hub |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | Allow context lengths above the model config (for YaRN) |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `3600` | Give the slow first start up to 1 h |
| `VLLM_PLE_TABLE_MEMORY` | `disk` | Keep the PLE table on disk instead of in memory |
| `B12X_DISK_BACKEND` | `io_uring` | Async disk I/O backend for b12x |
| `CUTE_DSL_ARCH` | `sm_121a` | Target the GB10 GPU architecture |
| `SAFETENSORS_FAST_GPU` | `1` | Faster safetensors loading onto the GPU |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Worker process start method |
| `VLLM_SSM_CONV_STATE_LAYOUT` | `DS` | Layout for the SSM / linear-attention conv state |
| `VLLM_USE_AOT_COMPILE` / `VLLM_USE_MEGA_AOT_ARTIFACT` | `1` | Ahead-of-time compilation with cached artifacts |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | Use the V2 model runner |
| `B12X_POLICY_MODE` | `auto` | Automatic b12x kernel policy selection |
| `VLLM_MXFP8_LM_HEAD` | `1` | MXFP8 LM head |
| `VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE` | `w13` | MoE FP4 input-scale source |
| `VLLM_B12X_MOE_FP4_FORCE_A16` | `1` | Force 16-bit activations for FP4 MoE (W4A16) |
| `B12X_W4A16_A4_PREFILL` | `1` | 4-bit activation path during prefill |

### `vllm serve` flags

| Area | Flags |
|------|-------|
| Parallelism | `--tensor-parallel-size 1 --pipeline-parallel-size 1` (single GPU) |
| Precision | `--dtype bfloat16 --kv-cache-dtype fp8 --quantization modelopt_mixed` |
| Loading | `--load-format b12x` |
| Caching / scheduling | `--enable-prefix-caching --enable-chunked-prefill --mamba-cache-mode align --block-size 16` |
| Limits | `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`, `--gpu-memory-utilization` (from the configuration) |
| Speculative decoding | `--speculative-config {"method":"mtp","num_speculative_tokens":N}` (left out when `MTP_TOKENS=0`) |
| Kernels | `--gdn-decode-kernel b12x --linear-backend b12x --moe-backend b12x --no-enable-flashinfer-autotune` |
| Multimodal | `--mm-encoder-tp-mode data` |
| Reasoning & tools | `--reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice` |
| Compilation | `--compilation-config {"pass_config":{"fuse_act_quant":true}}` |
| Long context | `--hf-overrides …` (only when YaRN is on) |

## Operations

```bash
./view.sh                              # follow logs
docker stop qwen38-flash-next          # stop
docker rm -f qwen38-flash-next         # remove
./run.sh                               # restart (removes any old container first)
```

## Troubleshooting

- **`ERROR: no model found in MODEL_DIR`**: download the weights (see Quick start) or point `MODEL_DIR` at them.
- **`ERROR: MAX_MODEL_LEN=... exceeds YaRN window`**: raise `YARN_FACTOR`, or unset it so the script computes one.
- **Startup looks stuck**: the first run compiles kernels. Watch `./view.sh` and allow up to the 1 h ready timeout.
- **Out of memory / KV cache too small**: lower `MAX_MODEL_LEN` or `MAX_NUM_SEQS`, or adjust `GPU_MEMORY_UTILIZATION`.
- **Port already in use**: the container uses host networking, so pick another port with `PORT=...`.
- **Speculative decoding problems**: run with `MTP_TOKENS=0` to rule it out.
