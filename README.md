<div align="center">

<img width="571" height="294" alt="d-ai-wp" src="https://github.com/user-attachments/assets/427d7a53-e1d6-457a-80b2-2be82f7150b6" />

</div>

# Intro

An interactive calculator for planning local AI model inference on unified-memory workstations. Built for machines like the HP Z2 Mini G1a (Strix Halo), NVIDIA DGX Spark, and Apple Mac M4 Max — compact desktop systems with 128 GB of shared CPU/GPU memory that can run frontier AI models locally.

<br>

## How it works

You select your hardware, pick a model, and the calculator shows you exactly what happens: how fast it generates tokens, how much context window you get, and how VRAM is divided between model weights and KV cache. Two optimization toggles — APEX-Quant (mixed-precision weight quantization) and TurboQuant (KV cache compression) — let you see the real impact of each technique independently. A context window slider lets you dial in your exact configuration, and every number updates in real time.

The goal is to answer the question: *"Can I run this model on my hardware, and what will the experience actually be like?"*

<br>

**[Try it live →](https://kahalewai.github.io/ai-workstation-planner/)**

<br>

## Features

**Three unified-memory systems** with real specs (bandwidth, VRAM limits, build flags):
- HP Z2 Mini G1a — Ryzen AI MAX+ PRO 395, Radeon 8060S, 128 GB LPDDR5X, 212 GB/s
- NVIDIA DGX Spark — GB10 Grace Blackwell, 128 GB unified, 273 GB/s
- Apple Mac M4 Max — 40-core GPU, 128 GB unified, 546 GB/s

**19 AI models** across four categories (recommended, new releases, frontier, won't fit), including MoE and dense architectures from Qwen, Gemma, GPT-OSS, Mistral, Meta, NVIDIA, MiniMax, DeepSeek, Kimi, and GLM.

**APEX-Quant toggle** — MoE-aware mixed-precision weight quantization ([mudler/apex-quant](https://github.com/mudler/apex-quant)). Restructures quantization precision across layers for better quality-per-bit — assigning higher precision to critical shared experts and edge layers while compressing redundant middle-layer routed experts. Toggle it on/off to see the impact on weight size and downstream VRAM. APEX I-Quality often uses *more* VRAM than standard quants (it invests bits where they matter most); only the aggressive APEX Mini tier is smaller. Only affects MoE models; dense models show "n/a."

**TurboQuant toggle** — KV cache compression ([TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)). Compresses the context memory 3–5x using turbo3/turbo4 quantization types. Toggle it on/off to see the impact on KV cache size and free VRAM. On high-VRAM workstations, context is often model-capped rather than VRAM-capped — TurboQuant frees VRAM without increasing context because the model's architectural limit is already reached. On tighter systems or with larger models, TurboQuant directly unlocks longer context. Includes sub-options for turbo3, turbo4, and asymmetric (K:q8_0 + V:turbo3) configurations.

**Context window slider** with logarithmic snap points (4K to 10M tokens). The slider's maximum is dynamically capped at the lower of the model's architectural limit and the VRAM-calculated ceiling — it's impossible to create an invalid configuration. Three visual zones show native range (full quality), extended range (YaRN/RoPE, quality may degrade), and a quality indicator updates as you drag.

**Three visualization tabs:**
- **Table** — all models with fit status, speed estimates, and context windows
- **VRAM breakdown** — horizontal stacked bar chart showing weight vs. KV cache vs. free memory
- **Speed vs. context** — bubble chart plotting token generation speed against max context, sized by parameter count

**First-principles calculations** — no lookup tables. Every number derives from:
```
tokens/sec = bandwidth / active_weight_read × 0.75 efficiency
KV/token   = 2 × attn_layers × kv_heads × head_dim × bytes_per_value
max_context = min(free_vram / kv_per_token, model_max)
```

**Light/dark theme** with a blue-toned light palette and the classic dark mode.

<br>

## Files

| File | Purpose |
|---|---|
| `index.html` | The complete application — works standalone with inline data |
| `models.js` | External model + system database — overrides inline data if present |

Both files together give you the full experience. The HTML works without `models.js` (it has a fallback copy of all data inline), but having `models.js` alongside it makes updating model data easier without touching the HTML.

<br>

## Hosting

**GitHub Pages (recommended):**
1. Fork this repo
2. Go to Settings → Pages → Source: Deploy from branch → `main` / `root`
3. Your planner is live at `https://yourusername.github.io/ai-workstation-planner/`

**Local:**
Just open `index.html` in a browser. No server needed. Everything runs client-side.

**Any static host:**
Upload `index.html` and `models.js` to the same directory. That's it.

<br>

## Adding or updating models

Edit `models.js`. Each model entry looks like:

```javascript
{
  id: "your_model_id",
  name: "Model Name (Size)",
  total_params_b: 35,          // Total parameters in billions
  active_params_b: 3,          // Active parameters per token (same as total for dense)
  is_moe: true,                // Mixture of Experts?
  n_layers: 64,                // Total transformer layers
  n_attn_layers: 16,           // Layers using full attention (for KV calculation)
  n_kv_heads: 8,               // Number of KV heads (GQA)
  head_dim: 128,               // Dimension per attention head
  native_ctx: 262144,          // Trained context window
  extended_ctx: 1010000,       // Max with extension method (same as native if none)
  ctx_extension: "YaRN",       // "YaRN" | "RoPE" | "unverified" | "none"
  quant_label: "Q4_K_M",       // Quantization method label
  weight_gb: 21.3,             // GGUF file size in GB
  weight_gb_apex: 19.7,       // APEX I-Quality estimated weight (for toggle comparison)
  apex_applied: false,         // Is the weight_gb value from APEX quantization?
  apex_available: true,        // Can APEX be applied? (MoE only)
  category: "recommended",     // "recommended" | "new" | "frontier" | "wontfit"
  release_date: "2025-09-01",
  license: "Apache 2.0",
  notes: "Your description here."
}
```

Adding a new system is similar — see the `SYSTEM_DB` array at the top of `models.js`.

<br>

## How the calculations work

**Token generation speed** is bandwidth-limited on unified-memory systems. The GPU must read the active model weights from shared memory for every token generated. For MoE models, only the active expert weights are read (plus ~30% overhead for routing/shared layers).

**KV cache** stores the attention state for every token in the context. Each token requires `2 × attention_layers × kv_heads × head_dim × bytes_per_value` of memory. TurboQuant compresses these bytes. Models with sliding window attention (like Gemma 4) only need full KV for a fraction of their layers, dramatically reducing cache size.

**APEX-Quant** assigns different quantization precision to different tensor types within MoE models — high precision (Q8_0) for always-active shared experts, moderate precision (Q6_K) for attention layers, and a layer-wise gradient for routed experts that gives edge layers higher precision than the redundant middle. This is a quality optimization, not purely a size optimization — APEX I-Quality often uses slightly more VRAM than uniform Q4_K_M but achieves measurably better accuracy. The APEX Mini tier trades quality for size, achieving smaller-than-Q2 weights while outperforming uniform quantization at that level.

**TurboQuant** applies random orthogonal rotation to KV vectors, then uses optimal scalar quantization near the Shannon limit. The rotation makes every coordinate follow a predictable distribution, enabling mathematically optimal compression. It's orthogonal to weight quantization — they stack. On 120+ GB unified-memory systems, most models are model-capped (the native context limit is reached before VRAM runs out), so TurboQuant's primary benefit is freeing VRAM rather than extending context. On smaller systems or with larger models, TurboQuant directly enables longer context windows.

<br>

## Data accuracy

Architecture details (layer counts, KV head configurations) are confirmed from model cards and HuggingFace configs where available, and estimated from typical architectures where not. Weight sizes are calculated from parameter counts at the listed quantization level and cross-referenced with community GGUF releases. The calculations are directionally accurate for planning purposes — exact numbers may vary by 10–20% from real-world measurements due to runtime overhead, driver differences, and model-specific optimizations.

Models using Multi-head Latent Attention (MLA) — including Mistral Small 4, Kimi K2.5, DeepSeek V3.2/V4, and GLM-5 — have compressed KV caches that this calculator does not yet model. The displayed KV usage for these models is conservative (overestimated).

<br>

## Built with

- Vanilla HTML, CSS, and JavaScript — no frameworks, no build step
- [Chart.js](https://www.chartjs.org/) for visualizations (loaded from CDN)
- [DM Sans](https://fonts.google.com/specimen/DM+Sans) + [JetBrains Mono](https://fonts.google.com/specimen/JetBrains+Mono) from Google Fonts
- Content Security Policy headers for script source restriction
- Data from model cards, HuggingFace configs, community benchmarks, and first-principles calculations

<br>

## Contributing

Pull requests welcome. The most useful contributions are:
- **Verified model architecture data** — exact layer counts, KV head configs, and head dimensions from model config.json files
- **New model entries** — especially models released after April 2026
- **New system entries** — other unified-memory AI workstations
- **MLA attention modeling** — the calculator overestimates KV cache for MLA models; a correction factor would improve accuracy
- **Bug fixes and UI improvements**

<br>

## License

Apache 2.0
