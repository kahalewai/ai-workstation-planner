<div align="center">

<img width="569" height="204" alt="daiwp1" src="https://github.com/user-attachments/assets/dd603f74-9136-42eb-b5d8-c2ef783e844f" />

</div>

# Intro

An interactive calculator for planning local AI model inference on modern AI workstations. Built for compact unified-memory machines like the HP Z2 Mini G1a (Strix Halo), NVIDIA DGX Spark, and Apple Mac M4 Max, alongside discrete-GPU builds ranging from a single RTX 3060 up to 8× datacenter accelerators. Whether you're stacking used RTX 3090s, speccing out an RTX PRO 6000 Blackwell workstation, or deciding between a Strix Halo box and a Mac Studio, the calculator shows you what actually runs.

<br>

## How it works

You select your hardware (and, for discrete GPUs, how many cards you're stacking), pick a model, and the calculator shows you exactly what happens: how fast it generates tokens, how much context window you get, and how VRAM is divided between model weights and KV cache. Two optimization toggles, APEX-Quant (mixed-precision weight quantization) and TurboQuant (KV cache compression), let you see the real impact of each technique independently. A context window slider lets you dial in your exact configuration, and every number updates in real time.

The goal is to answer the question: *"Can I run this model on my hardware, and what will the experience actually be like?"*

<br>

**[Try it live →](https://kahalewai.github.io/ai-workstation-planner/)**

<br>

## Features

**Sixteen systems across four tiers** with real specs (bandwidth, VRAM limits, build flags), organized into grouped dropdowns:
- **Unified memory** - HP Z2 Mini G1a (Strix Halo, 212 GB/s), NVIDIA DGX Spark (GB10, 273 GB/s), Apple Mac M4 Max 128 GB (546 GB/s)
- **Consumer GPU** - RTX 3060 12GB, 4060 Ti 16GB, 4070 Ti Super 16GB, 3090 24GB, 4090 24GB, 5090 32GB, AMD RX 7900 XTX 24GB
- **Professional GPU** - RTX A6000 48GB, RTX 6000 Ada 48GB, RTX PRO 6000 Blackwell 96GB
- **Datacenter GPU** - H100 SXM 80GB, H200 SXM 141GB, B200 SXM 192GB

**Multi-GPU stacking** - for discrete GPUs, a card-count selector (1, 2, 3, 4, 6, 8) lets you model stacked configurations. VRAM sums linearly, bandwidth stays per-card (llama.cpp's default pipeline-parallel layer-split). A 2× RTX 3090 build shows 48 GB VRAM at 936 GB/s per card; 8× H200 shows 1128 GB at 4800 GB/s per card.

**21 AI models** with dynamic fit grouping. Models are sorted on the fly into three categories based on your current VRAM and toggle settings: *Fits comfortably* (weights + native context fit with room), *Fits with tradeoffs* (weights load but usable context drops below the model's native max), and *Does not fit* (weights exceed VRAM, or usable context falls below 4K). Within each group, models are ordered newest-first, and recent releases (last 90 days) get a ✨ prefix. The lineup covers MoE and dense architectures from Qwen, Gemma, GPT-OSS, Arcee, Mistral, Meta, NVIDIA, MiniMax, DeepSeek, Kimi, and GLM, including April 2026 releases like Arcee Trinity-Large-Thinking and Gemma 4.

**APEX-Quant toggle** - MoE-aware mixed-precision weight quantization ([mudler/apex-quant](https://github.com/mudler/apex-quant)). Restructures quantization precision across layers for better quality-per-bit, assigning higher precision to critical shared experts and edge layers while compressing redundant middle-layer routed experts. Toggle it on/off to see the impact on weight size and downstream VRAM. APEX I-Quality often uses *more* VRAM than standard quants (it invests bits where they matter most); only the aggressive APEX Mini tier is smaller. Only affects MoE models; dense models show "n/a."

**TurboQuant toggle** - KV cache compression ([TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)). Compresses the context memory 3–5x using turbo3/turbo4 quantization types. Toggle it on/off to see the impact on KV cache size and free VRAM. On high-VRAM builds (128 GB unified boxes, 96 GB PRO 6000 Blackwell, stacked A6000s, datacenter nodes), context is often model-capped rather than VRAM-capped — TurboQuant frees VRAM without increasing context because the model's architectural limit is already reached. On 24 GB consumer cards or with larger-than-weights models, TurboQuant directly unlocks longer context windows. Includes sub-options for turbo3, turbo4, and asymmetric (K:q8_0 + V:turbo3) configurations.

**Context window slider** with logarithmic snap points (4K to 10M tokens). The slider's maximum is dynamically capped at the lower of the model's architectural limit and the VRAM-calculated ceiling, it's impossible to create an invalid configuration. Three visual zones show native range (full quality), extended range (YaRN/RoPE, quality may degrade), and a quality indicator updates as you drag.

**Three visualization tabs:**
- **Table** - all models with fit status, speed estimates, and context windows
- **VRAM breakdown** - horizontal stacked bar chart showing weight vs. KV cache vs. free memory
- **Speed vs. context** - bubble chart plotting token generation speed against max context, sized by parameter count

**First-principles calculations** - no lookup tables. Every number derives from:
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
  ctx_extension: "YaRN",       // "YaRN" | "RoPE" | "trained" | "unverified" | "none"
  quant_label: "Q4_K_M",       // Quantization method label
  weight_gb: 21.3,             // GGUF file size in GB
  weight_gb_apex: 19.7,       // APEX I-Quality estimated weight (for toggle comparison)
  apex_applied: false,         // Is the weight_gb value from APEX quantization?
  apex_available: true,        // Can APEX be applied? (MoE only)
  category: "recommended",     // Informational only — fit grouping is computed live from VRAM
  release_date: "2025-09-01",  // Used for newest-first sort and ✨ marker (last 90 days)
  license: "Apache 2.0",
  notes: "Your description here."
}
```

Use `ctx_extension: "trained"` for models fine-tuned to extended context (e.g., Llama 4 Maverick, Arcee Trinity-Large-Thinking) — native quality is preserved, unlike YaRN/RoPE which degrade beyond the training length.

Adding a new system is similar, see the `SYSTEM_DB` array at the top of `models.js`. System entries have two extra fields that control grouping and multi-card behavior:

```javascript
{
  // ... standard fields (id, name, chip, vram_default_gb, bandwidth_gbs, etc.) ...
  type: "gpu",                 // "unified" = fixed single config | "gpu" = stackable card
  group: "Consumer GPU",       // Dropdown optgroup label — free-form, but the priority
                               // order is Unified memory → Consumer GPU → Professional GPU
                               // → Datacenter GPU. Any unrecognized group is appended after.
}
```

For `type: "gpu"` systems, set `vram_default_gb` and `vram_max_gb` to the same per-card value — the UI multiplies by the selected card count at runtime.

<br>

## How the calculations work

**Token generation speed** is bandwidth-limited. On unified-memory systems, the GPU must read the active model weights from shared memory for every token generated; on multi-GPU pipeline-parallel builds (llama.cpp's default layer-split), each card reads its own slice of the weights in sequence, so the per-token time is still gated by a single card's bandwidth. For MoE models, only the active expert weights are read (plus ~30% overhead for routing/shared layers). This is why the card-count selector scales VRAM but not bandwidth — adding a second 3090 doubles your capacity to hold weights and KV cache, but token generation speed stays anchored to the 936 GB/s per-card bus. Tensor-parallel setups with NVLink can scale effective bandwidth further, but require specialized inference stacks and aren't modeled here.

**KV cache** stores the attention state for every token in the context. Each token requires `2 × attention_layers × kv_heads × head_dim × bytes_per_value` of memory. TurboQuant compresses these bytes. Models with sliding window attention (like Gemma 4) only need full KV for a fraction of their layers, dramatically reducing cache size.

**APEX-Quant** assigns different quantization precision to different tensor types within MoE models, high precision (Q8_0) for always-active shared experts, moderate precision (Q6_K) for attention layers, and a layer-wise gradient for routed experts that gives edge layers higher precision than the redundant middle. This is a quality optimization, not purely a size optimization, APEX I-Quality often uses slightly more VRAM than uniform Q4_K_M but achieves measurably better accuracy. The APEX Mini tier trades quality for size, achieving smaller-than-Q2 weights while outperforming uniform quantization at that level.

**TurboQuant** applies random orthogonal rotation to KV vectors, then uses optimal scalar quantization near the Shannon limit. The rotation makes every coordinate follow a predictable distribution, enabling mathematically optimal compression. It's orthogonal to weight quantization; they stack. On high-VRAM configurations, most models are model-capped (the native context limit is reached before VRAM runs out), so TurboQuant's primary benefit is freeing VRAM rather than extending context. On smaller systems or with larger models, TurboQuant directly enables longer context windows.

<br>

## Data accuracy

Architecture details (layer counts, KV head configurations) are confirmed from model cards and HuggingFace configs where available, and estimated from typical architectures where not. Weight sizes are calculated from parameter counts at the listed quantization level and cross-referenced with community GGUF releases. The calculations are directionally accurate for planning purposes, exact numbers may vary by 10–20% from real-world measurements due to runtime overhead, driver differences, and model-specific optimizations.

Models using Multi-head Latent Attention (MLA), including Mistral Small 4, Kimi K2.5, DeepSeek V3.2/V4, and GLM-5, have compressed KV caches that this calculator does not yet model. The displayed KV usage for these models is conservative (overestimated).

<br>

## Built with

- Vanilla HTML, CSS, and JavaScript - no frameworks, no build step
- [Chart.js](https://www.chartjs.org/) for visualizations (loaded from CDN)
- [DM Sans](https://fonts.google.com/specimen/DM+Sans) + [JetBrains Mono](https://fonts.google.com/specimen/JetBrains+Mono) from Google Fonts
- Content Security Policy headers for script source restriction
- Data from model cards, HuggingFace configs, community benchmarks, and first-principles calculations

<br>

## Contributing

Pull requests welcome. The most useful contributions are:
- **Verified model architecture data** - exact layer counts, KV head configs, and head dimensions from model config.json files
- **New model entries** - especially releases newer than what's currently in the DB
- **New system entries** - other AI workstations and GPU cards, especially AMD Instinct MI300 series and emerging Chinese accelerators
- **MLA attention modeling** - the calculator overestimates KV cache for MLA models; a correction factor would improve accuracy
- **Bug fixes and UI improvements**

<br>

## License

Apache 2.0
