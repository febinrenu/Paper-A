# Generation run notes (for the methods section)

All times are local (IST, UTC+05:30). Each row in `data/gen/<model>.jsonl` has a `ts` field, so every
change below can be checked against the rows.

## Local inference (Ollama 0.40.1, RTX 4060 Laptop 8 GB)

| Period | Ollama server settings | Models affected |
|---|---|---|
| until 2026-11-23 15:15 | defaults: 1 parallel slot, 4096-token context, f16 KV cache, flash attention off | CodeLlama-7B (all rows); DeepSeek-Coder-6.7B rows with `ts` < 15:15 |
| from 2026-11-23 15:18 | `OLLAMA_NUM_PARALLEL=3`, `OLLAMA_CONTEXT_LENGTH=3072` per slot, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0` | DeepSeek-Coder-6.7B rows with `ts` >= 15:18 and all later models |
| from 2026-11-23 15:29 | as above, plus request option `num_gpu=99`, which keeps all layers on the GPU (placement only) | DeepSeek-Coder-6.7B rows with `ts` >= 15:29 and all later models |

Why: throughput. With one slot the server handled the client's 4 concurrent requests one at a time.
Measured on DeepSeek-Coder-6.7B:

| Setting | Seconds per prompt |
|---|---|
| default (1 slot) | ~10 |
| 3 slots, 31 of 33 layers on GPU | ~7 |
| 3 slots, all layers on GPU | ~5 |

**Prompt length:** the longest code-step input is about 1,700 tokens and generation is capped at 1,024
tokens, so 3,072 tokens per slot does not truncate. This is checked by the absence of truncation warnings
in the server log.

**Robustness check for the paper:** compare DeepSeek-Coder-6.7B's hallucination share and claim rate
before and after the switch, on disjoint prompt subsets. The 2024 cohort comparison has to show the
change in settings does not move the estimates.

## Interruptions (resumed automatically; resumable per prompt)

| Time | Event |
|---|---|
| 2026-11-22 18:56 | Laptop slept and Ollama exited; restarted manually |
| 2026-11-23 05:22 | Windows Update restart; idle until 11:06 |
| 2026-11-23 12:11–12:16 | Two restarts; relaunched by the login script |
| 2026-11-23 12:32–13:42 | Modern Standby (lid closed) |
| 2026-11-23 15:15–15:18 | Ollama restarted three times to apply the settings above |

## 2024-era extension cohort: version checks (2026-11-24)

**`gemma:7b-instruct` points to Gemma 1.1** (April 2024, after the snapshot) and is therefore excluded.

`gemma:7b-instruct-q4_0` is used instead. The registry manifests show:

| Layer | `7b-instruct-q4_0` | `7b-instruct-v1.1-q4_0` |
|---|---|---|
| Weights | `456402914e83` | `ef311de6af9d` (explicitly 1.1) |
| License | `097a36493f71` | same |
| Template | `109037bec39c` | same |
| Params | `22a838ceb7fb` | `65bb16cf5983` |

Gemma 1 has only versions 1.0 and 1.1, so the `q4_0` weights are **Gemma 1.0** (released 2024-02-21). The bundled license text is dated 2024-02-21, which agrees.

| Tag | Same weights as | So it is |
|---|---|---|
| `qwen:7b-chat` | `qwen:7b-chat-v1.5-q4_0` | Qwen1.5, 2024-02-05 |
| `magicoder:7b` | `magicoder:7b-s-cl-q4_0` | Magicoder-S-CL |
| `zephyr:7b` | `zephyr:7b-beta-q4_0` | Zephyr beta |
| `llama2:7b-chat` | `llama2:7b-chat-q4_0` | Llama 2 chat |

## Hosted cohort (NVIDIA API, `integrate.api.nvidia.com/v1`)

- **Protocol:** identical prompts, seeds and sampling parameters. Reasoning is disabled via `chat_template_kwargs`, and each row records `served_model` and `reasoning_chars` (expected 0).
- **Rate limiting:** two API keys, rate-limited client side (36 requests/min per key). From 2026-11-23 15:18 the keys are pooled across models.
- **Excluded candidates:**
  - Kimi K3.1: top_p is fixed at 0.95.
  - Mistral Large 4 and Step-5: reasoning cannot be disabled.
  - GLM-5.3, Gemma-4-31B, Muse: timeouts or 504s on 2026-11-23.

## 2026-11-24/25: magicoder:7b and the context window

| Time | Event |
|---|---|
| 2026-11-24 22:08 – 2026-11-25 11:10 | magicoder:7b could not load: its Modelfile sets `num_ctx 16384`, which overrides the server's 3,072-token slot and needs a 13.7 GB KV cache. Every request failed at load time, so no rows were written. The 852 load-failure records were removed from `errors.jsonl`; a backup is in `data/derived/errors_before_magicoder_ctx_fix.jsonl`. |
| 2026-11-25 11:15 | magicoder:7b restarted with request option `num_ctx=4096` (placement and memory only). |
| from the next model on | request option `num_ctx=3072`, equal to the server setting used by every earlier model. |

The longest prompt plus the 1,024-token cap stays below 3,072 tokens, so neither value truncates input or output.

## 2026-11-24 23:50 – 2026-11-25 11:09: internet outage

Hosted generation, registry lookups and GitHub collection paused and resumed by themselves. 147 registry
lookups that failed during the outage were redone. From commit 9070c16, a lookup that fails on the network can
no longer be stored as "not registered".
