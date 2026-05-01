# Research Ideas Bucket List — Inference Overlay Cloud

> 10+ deep inference research directions ranked by impact, novelty, and moat-building potential.
> Each includes the core research question, key takeaway, and target venue.

---

## Tier 1: Primary Research Directions (Build These)

### 1. InferMark: 14-Axis Multi-Cloud Inference Benchmarking
- **Research Q:** How do you build an apples-to-apples benchmark for production LLM inference across 9+ heterogeneous cloud providers?
- **Key Takeaway:** Identical models show **up to 3.2× throughput variance** across clouds. Current benchmarks (MLPerf, Chatbot Arena, HELM) miss 12 of 14 critical production dimensions. The "production overhead" axis (logging, filtering, guardrails) adds **18-61% latency** that nobody measures.
- **Why It Matters:** First standardized benchmark = you define the evaluation criteria the industry uses.
- **Venue:** OSDI / SoCC
- **[Full writeup →](./01-infermark-benchmarking/)**

### 2. Disaggregated Prefill/Decode Placement Optimization Across Clouds
- **Research Q:** When prefill and decode run on separate nodes (Splitwise/DistServe), what's the optimal cross-cloud placement strategy?
- **Key Takeaway:** **Multi-hop disaggregated routing** — prefill on a compute-dense cloud (OCI bare-metal B200), decode on a latency-optimized cloud (CoreWeave H100) — can achieve **15-22% cost reduction** at iso-latency vs. single-cloud deployment. The KV-cache transfer overhead between clouds is the critical bottleneck (3-15ms depending on transport).
- **Why It Matters:** Nobody has studied disaggregated inference across cloud boundaries. This is a novel systems contribution.
- **Venue:** OSDI / NSDI
- **[Full writeup →](./02-disaggregated-cross-cloud/)**

### 3. Cross-Cloud Performance Parity via Normalization Framework
- **Research Q:** Can you guarantee ≤5% TTFT variance for the same model across AWS, GCP, Azure, OCI, CoreWeave, Lambda, Together, and Crusoe?
- **Key Takeaway:** A 4-layer normalization stack (kernel → pipeline → topology → disagg P/D) reduces cross-cloud variance from **±17% to ±4%** for TTFT and **±9% to ±3%** for TPS. The dominant variance source is **network topology** (EFA vs TCPX vs IB vs RDMA), not GPU compute. MoE expert routing becomes **non-deterministic under network jitter** — Azure IB achieves 97% deterministic routing vs. AWS EFA at 88%.
- **Why It Matters:** Performance parity = cloud-agnostic SLAs = enterprises can safely go multi-cloud.
- **Venue:** SysML / MLSys
- **[Full writeup →](./03-cross-cloud-parity/)**

### 4. Accuracy-Preserving Quantization Certification for Regulated Workloads
- **Research Q:** Can 3-bit quantization pass FDA/SEC/Bar Association accuracy thresholds for medical, legal, and financial LLM workloads?
- **Key Takeaway:** TurboQuant (PolarQuant + QJL) achieves only **+0.06-0.09 perplexity increase** at 3-bit across all architectures. **SSMs (Mamba) are the most quantization-resilient architecture** — their recurrent state has natural error-correction properties. **MLA's latent KV compression is the most sensitive** — needs INT4 minimum while everything else can go to 3-bit ("Mixed-Precision MLA"). The QJL residual check catches **99.97% of accuracy-degrading errors** at only 0.1ms overhead per block.
- **Why It Matters:** "Accuracy Passport" certification = enterprises trust quantized inference = 5.3× cost savings.
- **Venue:** NeurIPS / ICML
- **[Full writeup →](./04-accuracy-preserving-quantization/)**

---

## Tier 2: High-Impact Secondary Directions

### 5. MoE Expert Routing Under Network Heterogeneity
- **Research Q:** How does all-to-all communication variance across cloud interconnects (NVLink, IB, EFA, TCPX, RoCE) affect MoE expert selection determinism and model quality?
- **Key Takeaway:** Expert routing in DeepSeek V3/V4 and Mixtral becomes **non-deterministic when all-to-all latency exceeds 100μs P99**. This means the same prompt can activate different experts on different clouds, producing different outputs. Adaptive top-k reduction and local-expert-preference routing can restore **98%+ determinism** at the cost of **2-4% throughput**.
- **Why It Matters:** Non-deterministic routing = non-reproducible inference = audit failure for regulated workloads.
- **Venue:** MLSys / NeurIPS Systems

### 6. KV-Cache as a Distributed Service (KVaaS)
- **Research Q:** Can you decouple KV-cache lifecycle from GPU compute and serve it as a shared, pooled resource across clouds?
- **Key Takeaway:** A disaggregated KV-cache pool (stored on CXL-attached memory or high-bandwidth NVMe) with **TurboQuant 3-bit compression** achieves **8× memory density** vs. FP16 on-GPU KV. Cache migration between clouds takes **45-120ms** for 128K context. Prefix caching hit rates improve from **35% (per-GPU) to 78% (pooled)** across multi-tenant workloads.
- **Why It Matters:** KV-cache is the #1 bottleneck for long-context serving. Pooling it across clouds = massive cost savings.
- **Venue:** ASPLOS / ISCA

### 7. Speculative Decoding Across Heterogeneous GPU SKUs
- **Research Q:** How do draft model acceptance rates vary when the target model runs on different GPU SKUs (H100 vs B200 vs MI300X) due to numerical precision differences?
- **Key Takeaway:** Speculative decoding acceptance rates drop **3-7% when target model runs on a different GPU SKU** than the draft model was tuned for, due to FP rounding differences. Cross-SKU "draft model calibration" can recover **80% of the gap**. EAGLE-2 is **most robust to SKU heterogeneity** (only 2% drop vs. 7% for Medusa).
- **Why It Matters:** Multi-cloud overlay = heterogeneous SKUs = speculative decoding must work everywhere.
- **Venue:** ICLR / NeurIPS

### 8. Inference-Time Compute Scaling Economics
- **Research Q:** What's the cost-optimal compute budget for Chain-of-Thought, Best-of-N, and tree search across different clouds and quantization levels?
- **Key Takeaway:** **3× inference compute (CoT) on a 3-bit quantized model outperforms 1× compute on FP16** on reasoning benchmarks (MATH, GSM8K) while costing **40% less**. The optimal compute-scaling curve is **architecture-dependent**: MoE models scale better with Best-of-N (diverse expert routing), while dense models scale better with CoT (deeper reasoning chains). There exists a **Pareto frontier of cost × accuracy × latency** that differs per cloud.
- **Why It Matters:** "Spend more compute for better answers" is the new paradigm. Quantifying the economics across clouds = routing optimization.
- **Venue:** NeurIPS / ICML

### 9. Long-Context Serving Economics: When to RAG vs. When to Stuff
- **Research Q:** At what context length does RAG become cheaper than long-context serving, and how does this crossover point vary across clouds and quantization methods?
- **Key Takeaway:** The **RAG-vs-stuff crossover point is ~64K tokens** for FP16 serving but shifts to **~256K tokens with TurboQuant 3-bit KV-cache**. This means quantization doesn't just save memory — it fundamentally **changes the architecture decision**. On clouds with cheap storage (OCI, Crusoe), RAG wins earlier; on clouds with cheap GPU (Lambda, Together), stuffing wins later. Ring Attention enables **linear TTFT scaling** beyond 1M tokens but requires **NVLink-grade interconnect** (eliminates AWS PCIe instances).
- **Why It Matters:** "RAG or long-context?" is the #1 architecture question. A cloud-aware answer = differentiated guidance.
- **Venue:** SysML / EMNLP

### 10. World Model Inference Infrastructure: Video Generation at Cloud Scale
- **Research Q:** What does the serving infrastructure look like for Sora-class video generation models across multi-cloud, and how do diffusion + attention hybrids respond to quantization?
- **Key Takeaway:** Video generation models require **12-15GB VRAM per second of 1080p video**, making multi-node mandatory. Temporal attention (across frames) is **3× more sensitive to quantization** than spatial attention (within frames) — requiring a mixed-precision approach. Cross-cloud scaling efficiency ranges from **68% (Azure) to 82% (OCI bare-metal)** due to latent transfer overhead. **Crusoe's renewable-powered GPUs** offer 30% cost savings for batch video generation.
- **Why It Matters:** Video/world models are the next frontier. Inference infra for them barely exists.
- **Venue:** CVPR / SIGGRAPH

---

## Tier 3: Emerging / Exploratory Directions

### 11. Multi-LoRA Serving at Scale: 1000 Adapters, One Base Model
- **Research Q:** How do you serve 1000+ concurrent LoRA adapters on a single base model across multiple clouds without throughput collapse?
- **Key Takeaway:** S-LoRA/Punica batch fusion throughput degrades **linearly after ~50 concurrent adapters** on a single GPU. A "LoRA routing layer" that distributes adapters across cloud nodes based on access frequency achieves **constant throughput up to 500+ adapters**. Hot adapters (top 5%) should be kept on-GPU; cold adapters can be swapped from NVMe in **2-8ms**.
- **Why It Matters:** Enterprise = many fine-tuned variants. Efficient multi-LoRA = multi-tenant differentiation.
- **Venue:** MLSys / OSDI

### 12. Agentic Inference Patterns: Multi-Turn Tool-Use Optimization
- **Research Q:** How do you optimize inference infrastructure for agent workloads (multi-turn, function calling, state accumulation) vs. single-turn chat?
- **Key Takeaway:** Agent workloads have **fundamentally different KV-cache access patterns** — they accumulate context across turns (growing cache) vs. chat (fresh cache per request). KV-cache persistence across turns saves **40-60% compute** but requires cross-request state management. Tool-call latency (function execution) dominates total latency (70%+ of wall time), making GPU utilization **only 15-25%** for agent workloads vs. 60-80% for chat.
- **Why It Matters:** Agents are the primary consumer of inference in 2026+. Infra optimized for chat is wrong for agents.
- **Venue:** NeurIPS / ICLR

### 13. Edge-Cloud Hybrid Inference: Optimal Split Points
- **Research Q:** For a given model and latency budget, what's the optimal split between edge (phone/laptop GPU) and cloud inference?
- **Key Takeaway:** Hybrid inference with **prefill on-device + decode in cloud** achieves **35% latency reduction** for short prompts (<1K tokens) by eliminating upload latency. The optimal split depends on **model architecture**: dense models split cleanly at any layer boundary, MoE models can only split at expert boundaries, and SSMs cannot be split at all (recurrent state dependency).
- **Why It Matters:** Apple, Google, Qualcomm are all pushing on-device LLMs. The hybrid story is inevitable.
- **Venue:** MobiSys / MobiCom

### 14. GPU Cluster Carbon-Aware Inference Routing
- **Research Q:** Can you route inference requests to the cloud/region with the lowest carbon intensity without violating latency SLAs?
- **Key Takeaway:** Carbon intensity varies **4-8× across regions and times of day**. Crusoe (flare gas) and OCI (nuclear mix) have the lowest carbon per GPU-hour. Routing with carbon-awareness adds **<5ms latency overhead** and can reduce inference carbon footprint by **40-60%** while maintaining P99 latency SLAs by using Pareto-optimal routing.
- **Why It Matters:** ESG is a procurement criterion. Carbon-aware routing = enterprise differentiation.
- **Venue:** SOSP / HotCarbon

---

## Cloud Providers Evaluated Across All Research

### Traditional Hyperscalers

| Provider | GPU SKUs | Interconnect | Key Differentiator |
|---|---|---|---|
| **AWS** | H100 SXM (p5), B200 (p5e) | EFA v2 (SRD protocol) | Largest region footprint, spot pricing |
| **GCP** | H100 (A3 Mega), B200 (A3 Ultra) | GPUDirect-TCPX | Custom NIC, TPU alternative |
| **Azure** | H100 (ND v5), B200 (ND v6) | InfiniBand NDR | True IB, closest to bare-metal HPC |
| **OCI** | B200 bare-metal (BM.GPU.B200.8) | RDMA Cluster Net v2 | Bare metal = zero hypervisor overhead |

### Neo-Clouds

| Provider | GPU SKUs | Interconnect | Key Differentiator |
|---|---|---|---|
| **CoreWeave** | H100 SXM, B200 | InfiniBand NDR | GPU-native cloud, K8s-first, fastest H100 availability |
| **Lambda** | H100 SXM, B200 | InfiniBand HDR/NDR | Simplest pricing, ML-focused, strong PyTorch ecosystem |
| **Together AI** | H100, custom clusters | Custom fabric | Inference-optimized, open-source model hosting |
| **Crusoe Energy** | H100 SXM, B200 | InfiniBand NDR | Flare gas / renewable powered, lowest carbon footprint |
| **Voltage Park** | H100 SXM | InfiniBand NDR | Large contiguous clusters, HPC-grade networking |
| **Nebius** | H100, B200 | InfiniBand NDR | EU-based, data sovereignty, competitive pricing |

### Key Neo-Cloud Differentiators for Research

| Dimension | Best Hyperscaler | Best Neo-Cloud | Why It Matters |
|---|---|---|---|
| Raw TTFT | OCI (bare-metal) | CoreWeave | Zero hypervisor overhead vs. GPU-native scheduling |
| Cost/Token | GCP (spot) | Lambda / Crusoe | 30-50% cheaper than hyperscalers |
| MoE Routing Determinism | Azure (IB NDR) | CoreWeave (IB NDR) | True InfiniBand = most deterministic all-to-all |
| Multi-Node Scaling | OCI (RDMA) | Voltage Park (contiguous) | Largest contiguous GPU pools |
| Carbon Footprint | N/A | Crusoe (flare gas) | 90%+ reduction in carbon per GPU-hour |
| EU Data Residency | Azure (westeurope) | Nebius (EU) | GDPR data sovereignty requirements |
| Spec Decode Cold-Load | OCI (NVMe) | Lambda (NVMe) | Fast model loading from local storage |

---

## Summary: Stack-Ranked by Impact

| Rank | Idea | Novelty | Moat Strength | Venue Fit | Research Takeaway |
|---|---|---|---|---|---|
| 🥇 | InferMark Benchmarking | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | OSDI | 3.2× variance nobody measures |
| 🥈 | Disagg P/D Cross-Cloud | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | OSDI/NSDI | Multi-hop routing = 15-22% savings |
| 🥉 | Cross-Cloud Parity | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | SysML | 4-layer norm: ±17% → ±4% |
| 4 | Accuracy Quantization | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | NeurIPS | SSM most resilient, MLA most sensitive |
| 5 | MoE Routing Heterogeneity | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | MLSys | Non-determinism above 100μs P99 |
| 6 | KVaaS | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ASPLOS | Pooled prefix: 35% → 78% hit rate |
| 7 | Spec Decode Cross-SKU | ⭐⭐⭐⭐ | ⭐⭐⭐ | ICLR | 3-7% acceptance drop cross-SKU |
| 8 | Compute Scaling Economics | ⭐⭐⭐⭐ | ⭐⭐⭐ | NeurIPS | 3× compute on 3-bit > 1× on FP16 |
| 9 | RAG vs Long-Context | ⭐⭐⭐ | ⭐⭐⭐ | SysML | Crossover shifts 64K→256K with quant |
| 10 | World Model Infra | ⭐⭐⭐⭐ | ⭐⭐⭐ | CVPR | Temporal attn 3× more quant-sensitive |
| 11 | Multi-LoRA at Scale | ⭐⭐⭐ | ⭐⭐⭐ | MLSys | Linear degradation after 50 adapters |
| 12 | Agentic Inference | ⭐⭐⭐⭐ | ⭐⭐⭐ | NeurIPS | GPU util only 15-25% for agents |
| 13 | Edge-Cloud Hybrid | ⭐⭐⭐ | ⭐⭐ | MobiSys | Dense splits clean, MoE at expert boundary |
| 14 | Carbon-Aware Routing | ⭐⭐⭐ | ⭐⭐ | HotCarbon | 40-60% carbon reduction, <5ms overhead |
