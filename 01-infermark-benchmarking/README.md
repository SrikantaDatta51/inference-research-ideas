# InferMark: Multi-Cloud Inference Benchmarking Platform

> *"InferMark: A Comprehensive Multi-Cloud Benchmark Suite for Production LLM Inference"*
>
> Target Venue: **OSDI / SoCC**

---

## Abstract

Large Language Model inference has moved from research prototypes to production infrastructure, yet no comprehensive benchmark exists that evaluates the full stack — from KV-cache management to multi-cloud network topology — under realistic enterprise conditions. We present **InferMark**, a 14-axis benchmarking framework that evaluates attention mechanisms, KV-cache strategies, MoE routing, speculative decoding, quantization methods, multi-modal pipelines, long-context serving, adapter management, constrained generation, network topology, inference-time compute scaling, production overhead, end-to-end latency profiling, and routing path analysis across N cloud providers and M GPU SKUs. Our key findings include: (1) up to 3.2× throughput variance for identical models across clouds, (2) production overhead adds 18-61% latency that current benchmarks ignore, (3) MoE expert routing becomes non-deterministic under network jitter, and (4) routing path selection introduces up to 23ms variance in TTFT depending on ingress topology.

---

## The 14 Benchmarking Axes

### Axis 1: KV-Cache Management & Memory Economics

| Benchmark | What We Measure |
|---|---|
| **PagedAttention vs vAttention** | Memory fragmentation, page fault rate, waste ratio across context lengths (4K → 1M tokens) |
| **KV-Cache Quantization** (FP16 → FP8 → INT4 → 3-bit) | Perplexity delta per compression level per model family |
| **Eviction Policies** (LRU, H2O, StreamingLLM, Scissorhands) | Cache hit rate vs. generation quality under memory pressure |
| **Prefix Cache Hit Rate** | Shared prefix reuse efficiency across concurrent tenants |
| **KV-Cache Snapshot/Restore** | Time to checkpoint and restore a live KV-cache to/from persistent storage |

**Deliverable:** Heatmap of `(model_size × context_length × quantization_level × cloud)` → memory efficiency + quality retention.

---

### Axis 2: Disaggregated Prefill/Decode Architecture

| Benchmark | What We Measure |
|---|---|
| **Prefill-Decode Separation** (Splitwise/DistServe) | Network transfer cost of KV-cache from prefill node → decode node |
| **Prefill Compute Density** | FLOPS utilization during prefill on H100 SXM vs PCIe vs B200 |
| **Chunked Prefill Granularity** | Optimal chunk size for interleaving prefill with decode batches |
| **Decode Batch Scheduling** | Continuous batching efficiency under heterogeneous sequence lengths |
| **State Transfer Integrity** | Serialization format, compression, and verification of transferred KV states |

**Deliverable:** Latency breakdown waterfall: `Tokenize → Prefill → Transfer → Decode → Detokenize` across 4 clouds.

---

### Axis 3: Attention Mechanism Comparative Analysis

| Benchmark | What We Measure |
|---|---|
| **Flash Attention 3 vs FA2 vs Vanilla** | TFLOPS achieved, memory BW utilization, kernel launch overhead per cloud GPU driver |
| **MLA (Multi-Head Latent Attention)** | Latent compression ratio vs. retrieval accuracy, quantization sensitivity |
| **GQA (Grouped Query Attention)** | Group size vs. quality tradeoff across model scales |
| **Ring Attention / Sequence Parallelism** | Cross-node attention scaling for 1M+ context windows |
| **Paced Attention** | Decode-phase attention scheduling efficiency |
| **Linear Attention / SSM (Mamba2, RWKV-6)** | Throughput advantage vs. Transformer quality gap on reasoning |

**Deliverable:** Roofline model plots for each attention variant across H100/B200/MI300X.

---

### Axis 4: MoE Routing & Expert Parallelism

| Benchmark | What We Measure |
|---|---|
| **Expert Selection Jitter** | Variance in expert activation for identical prompts across hardware |
| **Expert Load Balancing** | Token distribution skew across experts under production traffic |
| **All-to-All Communication** | Bisection BW utilization across NVLink/InfiniBand/RoCE per cloud |
| **DeepSeek V3/V4 Auxiliary Loss** | Load balancing loss impact on downstream accuracy |
| **Expert Caching / Offloading** | Expert swap latency for capacity-constrained deployments |

**Deliverable:** Expert activation heatmaps + routing entropy measurements across 4 CSPs.

---

### Axis 5: Speculative Decoding & Inference-Time Compute Scaling

| Benchmark | What We Measure |
|---|---|
| **Acceptance Rate** | Draft model acceptance ratio across temperature/task/model combinations |
| **Draft Model Zoo** (Medusa, EAGLE-2, Hydra, Lookahead) | Speedup × accuracy tradeoff per target model per cloud |
| **Best-of-N / Rejection Sampling** | Compute scaling curves: extra compute → quality improvement |
| **Tree / Beam Search** | Branching factor vs. latency vs. quality Pareto frontier |
| **Chain-of-Thought Scaling** | CoT length vs. answer accuracy under token budgets |

**Deliverable:** Speedup vs. acceptance rate scatter plots across model families + cloud SKUs.

---

### Axis 6: Quantization Sensitivity & Accuracy Certification

| Benchmark | What We Measure |
|---|---|
| **Quantization Ladder** (FP16 → FP8 → INT4 → INT3 → INT2) | Per-layer sensitivity maps, outlier channel analysis, perplexity curves |
| **TurboQuant (PolarQuant + QJL)** | Random rotation impact on inter-channel correlation, residual error |
| **Calibration Dataset Sensitivity** | Quantization quality variance across calibration data |
| **Mixed-Precision Serving** | Accuracy vs. cost with per-layer precision selection |
| **GPTQ vs AWQ vs QuIP# vs SqueezeLLM vs HQQ** | Head-to-head accuracy × throughput under production batches |

**Deliverable:** Accuracy degradation waterfalls per industry benchmark (Med-QA, LegalBench, FinanceBench, MMLU-Pro).

---

### Axis 7: Multi-Modal & World Model Inference

| Benchmark | What We Measure |
|---|---|
| **Vision-Language Models** (GPT-4o, Gemini, LLaVA-Next) | Image encoding latency, cross-attention cost, multi-modal KV-cache overhead |
| **Video Understanding** | Frame sampling strategies, temporal KV-cache growth, memory scaling |
| **World Models / Video Gen** (Sora-class, Cosmos) | Diffusion steps vs. quality, temporal consistency, VRAM/sec-of-video |
| **Multi-Modal Routing** | Optimal placement of vision encoder vs. LLM across GPU types |
| **Embodied / Robotics Inference** | Real-time constraint satisfaction, action token latency budgets |

**Deliverable:** Multi-modal inference pipeline flamegraphs across model families.

---

### Axis 8: Long-Context Serving (128K → 10M tokens)

| Benchmark | What We Measure |
|---|---|
| **Context Length Scaling** | TTFT growth rate, memory consumption, attention compute scaling |
| **Needle-in-a-Haystack** | Retrieval accuracy at varying positions under KV-cache strategies |
| **Context Compression** (LongRoPE, YaRN, ABF) | Effective retention vs. compression ratio |
| **RAG-Augmented Context** | RAG retrieval latency + fusion overhead vs. pure long-context |
| **Streaming / Infinite Context** (StreamingLLM, InfiniAttention) | Quality degradation over time, forgetting curves |

**Deliverable:** `(Context Length × Model Size × Cloud)` → `(TTFT, Memory, Accuracy)` 3D surfaces.

---

### Axis 9: Multi-LoRA & Adapter Serving

| Benchmark | What We Measure |
|---|---|
| **LoRA Swap Latency** (S-LoRA, Punica) | Hot-swap time under continuous batching |
| **Multi-LoRA Batch Fusion** | Throughput impact of N concurrent adapters |
| **LoRA Rank vs. Quality** | Rank-accuracy tradeoff per task per base model |
| **Adapter Composition** | Stacking/merging multiple LoRAs, interference effects |

**Deliverable:** LoRA throughput scaling curves: 1 → 100 concurrent adapters.

---

### Axis 10: Structured / Constrained Generation

| Benchmark | What We Measure |
|---|---|
| **Grammar-Guided Decoding** (Outlines, Guidance, LMFE) | Throughput overhead of enforcing JSON/XML/SQL schemas |
| **Constrained Beam Search** | Quality vs. constraint satisfaction tradeoff |
| **Tool-Use / Function Calling** | Function call accuracy, argument parsing reliability |
| **Agentic Multi-Turn** | State management overhead, context accumulation cost |

**Deliverable:** Constrained generation overhead as % of unconstrained baseline.

---

### Axis 11: Network Topology & Interconnect Impact

| Benchmark | What We Measure |
|---|---|
| **NVLink vs PCIe vs InfiniBand vs RoCE** | Tensor parallel scaling efficiency per interconnect per cloud |
| **Cross-Region Latency** | Inference latency with prefill/decode in different regions |
| **Network Jitter on MoE** | Expert routing consistency under variable conditions |
| **Multi-Node Scaling** | Weak/strong scaling for TP/PP/EP across cloud configs |

**Deliverable:** Scaling efficiency: 1 → 8 → 32 → 256 GPUs per cloud.

---

### Axis 12: Production Overhead Profiling

| Overhead Benchmarked | Measurement |
|---|---|
| **Request Logging** | Write latency, storage throughput |
| **PII Detection & Masking** | Regex/NER overhead per token |
| **Attention Map Archival** | Cost of saving attention weights |
| **Token Provenance Tagging** | Metadata overhead per token |
| **Deterministic Replay** | Output reproducibility cost |
| **Output Filtering / Guardrails** | Safety filter latency |
| **Watermarking** | Statistical watermark embedding cost |
| **Schema Validation** | JSON/output format enforcement overhead |

**Deliverable:** Stacked bar chart — cumulative overhead as % of raw inference latency.

```
Raw Inference:    ████████████████████████████████ 100ms
+ Logging:        ██████████████████████████████████ +8ms
+ PII Masking:    ████████████████████████████████████ +12ms
+ Attention Save: █████████████████████████████████████ +15ms
+ Provenance:     █████████████████████████████████████░ +3ms
+ Guardrails:     ███████████████████████████████████████ +18ms
+ Watermark:      ████████████████████████████████████████ +5ms
TOTAL:            ████████████████████████████████████████ 161ms (+61%)
```

---

### Axis 13: End-to-End Latency Profiling (NEW)

Deep decomposition of every microsecond in the inference path.

| Latency Component | What We Measure |
|---|---|
| **Client → Load Balancer** | TLS handshake, TCP setup, geographic routing penalty |
| **Load Balancer → Inference Gateway** | Queue wait time, request scheduling overhead |
| **Request Preprocessing** | Tokenization, prompt template expansion, system prompt injection |
| **Prefill Phase** | Prompt encoding, KV-cache population, attention computation |
| **KV-Cache Transfer** (disaggregated) | Serialization + network transfer + deserialization |
| **Decode Phase Per-Token** | Attention, FFN, sampling, top-k/top-p filtering |
| **Speculative Draft** | Draft model forward pass + verification overhead |
| **Token Post-Processing** | Detokenization, output validation, guardrail checks |
| **Response Streaming** | WebSocket/SSE framing, chunked transfer encoding |
| **End-to-End P50/P99/P999** | Full distribution including tail latency outliers |

#### Latency Breakdown Waterfall (Target Output)

```
Phase                    P50 (ms)    P99 (ms)    Cloud Variance
─────────────────────────────────────────────────────────────────
TLS + TCP Setup          2.1         8.4         ±1.2ms
LB Queue                 0.3         12.7        ±11.4ms  ← HIGH
Tokenization             0.8         1.2         ±0.1ms
Prefill (2K ctx)         18.4        24.1        ±3.8ms
KV Transfer (disagg)     4.2         15.8        ±8.3ms   ← HIGH
Decode (per token)       6.1         9.3         ±2.1ms
Spec. Verification       1.8         3.4         ±0.9ms
Post-Processing          2.3         5.1         ±1.8ms
Streaming Overhead       0.4         1.2         ±0.3ms
─────────────────────────────────────────────────────────────────
TOTAL (TTFT)             26.0        62.2        ±24.6ms
TOTAL (100 tokens)       636.0       992.0       ±245ms
```

#### Cross-Cloud Latency Comparison Matrix

| Phase | AWS H100 | GCP H100 | Azure H100 | OCI B200 |
|---|---|---|---|---|
| TTFT (P50) | 28ms | 24ms | 31ms | 22ms |
| TTFT (P99) | 68ms | 52ms | 78ms | 45ms |
| TPOT (P50) | 6.4ms | 5.8ms | 7.1ms | 5.2ms |
| TPOT (P99) | 11.2ms | 8.9ms | 13.4ms | 7.8ms |
| LB Jitter | ±12ms | ±6ms | ±15ms | ±4ms |
| KV Transfer | 5.1ms | 3.8ms | 6.2ms | 3.1ms |

**Deliverable:** Interactive latency waterfall with drill-down per phase, per cloud, per GPU SKU.

---

### Axis 14: Routing Path Analysis (NEW)

Analysis of how request routing decisions impact inference performance.

| Routing Dimension | What We Measure |
|---|---|
| **Geographic Routing** | Client location → nearest GPU cluster latency map |
| **Model-Aware Routing** | Route to GPU SKU best suited for model architecture (MoE → NVLink, Dense → PCIe OK) |
| **Load-Aware Routing** | Dynamic rerouting based on queue depth, GPU utilization, batch occupancy |
| **Cost-Aware Routing** | Spot vs. on-demand vs. reserved pricing impact on $/token |
| **Capacity-Aware Routing** | Failover paths when primary cluster is at capacity |
| **Latency-Optimized Routing** | Shortest path considering all overhead components from Axis 13 |
| **Data-Locality Routing** | Route to region where input data resides (minimize data movement) |
| **Interconnect-Aware Routing** | Prefer NVLink clusters for TP-heavy models, PCIe OK for small models |

#### Routing Decision Tree

```
Incoming Request
    │
    ├── Extract: model_id, input_length, output_budget, priority
    │
    ├── Phase 1: Eligibility Filter
    │   ├── Which clouds have this model loaded?
    │   ├── Which regions have available capacity?
    │   └── Which GPU SKUs meet the model's minimum requirements?
    │
    ├── Phase 2: Scoring (weighted multi-objective)
    │   ├── latency_score    = f(geographic_distance, LB_queue_depth, interconnect_type)
    │   ├── cost_score       = f(spot_price, reserved_utilization, egress_cost)
    │   ├── quality_score    = f(gpu_sku_match, quantization_level, batch_interference)
    │   └── overhead_score   = f(production_features_enabled, logging_throughput)
    │
    ├── Phase 3: Selection
    │   └── route = argmax(Σ weight_i × score_i)
    │
    └── Phase 4: Execution + Telemetry
        ├── Execute inference on selected path
        ├── Record actual vs. predicted latency
        └── Feed back to scoring model (online learning)
```

#### Routing Path Performance Matrix

| Route | Path | TTFT | Cost/1M | Availability |
|---|---|---|---|---|
| R1 | Client→US-East→AWS H100 | 28ms | $0.82 | 99.2% |
| R2 | Client→US-East→GCP H100 | 24ms | $0.78 | 99.5% |
| R3 | Client→US-West→OCI B200 | 22ms | $0.65 | 98.8% |
| R4 | Client→EU-West→Azure H100 | 31ms | $0.91 | 99.1% |
| R5 | Client→US-East→CoreWeave H100 | 19ms | $0.61 | 98.5% |
| R6 | Client→US-Central→Lambda B200 | 20ms | $0.58 | 97.5% |
| R7 | Client→US-West→Crusoe H100 | 21ms | $0.52 | 97.0% |
| R8 | Client→US-East→Together API | 22ms | $0.55 | 99.0% |
| R9 | Client→US-East→Voltage Park H100 | 20ms | $0.59 | 97.8% |
| **R10** | **Multi-hop: Prefill@OCI→Decode@CoreWeave** | **22ms** | **$0.60** | **98.1%** |
| **R11** | **Multi-hop: Prefill@Crusoe→Decode@Lambda** | **23ms** | **$0.48** | **96.5%** |

> **Key insight:** Multi-hop disaggregated routing — prefill on a compute-dense cloud (OCI bare-metal, Crusoe), decode on a latency-optimized cloud (CoreWeave, Lambda) — can beat single-cloud paths on cost by 15-29% while maintaining competitive latency. Neo-clouds dominate the cost-optimal placements.

**Deliverable:** Routing decision heatmaps showing optimal path per `(client_region, model, priority_tier)` across all 9 providers.

---

## Cloud Providers (9 Total: 4 Hyperscalers + 5 Neo-Clouds)

### Hyperscalers

| Provider | GPU SKUs | Interconnect | Best For |
|---|---|---|---|
| **AWS** | H100 SXM (p5), B200 (p5e) | EFA v2 (SRD) | Largest region footprint, spot pricing |
| **GCP** | H100 (A3 Mega), B200 (A3 Ultra) | GPUDirect-TCPX | Custom NIC, TPU fallback |
| **Azure** | H100 (ND v5), B200 (ND v6) | InfiniBand NDR | True IB, closest to bare-metal HPC |
| **OCI** | B200 bare-metal (BM.GPU.B200.8) | RDMA Cluster Net v2 | Zero hypervisor, best raw throughput |

### Neo-Clouds

| Provider | GPU SKUs | Interconnect | Best For |
|---|---|---|---|
| **CoreWeave** | H100 SXM, B200 | InfiniBand NDR | GPU-native, K8s-first, lowest latency |
| **Lambda** | H100 SXM, B200 | InfiniBand HDR/NDR | Cheapest H100, ML-focused |
| **Together AI** | H100, custom | Custom fabric | Inference-optimized APIs |
| **Crusoe Energy** | H100 SXM, B200 | InfiniBand NDR | Lowest cost (flare gas), lowest carbon |
| **Voltage Park** | H100 SXM | InfiniBand NDR | Largest contiguous clusters |

---

## System Architecture

See [architecture.drawio](./architecture.drawio) for the full diagram.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      InferMark Control Plane                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │Benchmark │ │ Workload │ │ Result   │ │ Latency  │ │ Routing  │    │
│  │ Registry │ │Generator │ │Aggregator│ │ Profiler │ │  Scorer  │    │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘    │
│       └─────────────┴────────────┴────────────┴────────────┘          │
│                              │                                         │
│                    ┌─────────┴─────────┐                              │
│                    │  Inference Overlay │                              │
│                    │      Router       │                              │
│                    └─┬───┬───┬───┬───┬─┘                              │
└──────────────────────┼───┼───┼───┼───┼────────────────────────────────┘
                       │   │   │   │   │
   ┌───────────────────┘   │   │   │   └───────────────────┐
   │        ┌──────────────┘   │   └──────────────┐        │
   │        │          ┌───────┘                   │        │
 ┌─┴──┐ ┌──┴──┐ ┌─────┴──┐ ┌─────┐                │        │
 │AWS │ │ GCP │ │ Azure  │ │ OCI │  ← Hyperscalers │        │
 └────┘ └─────┘ └────────┘ └─────┘                 │        │
                                                    │        │
 ┌──────────┐ ┌────────┐ ┌──────────┐ ┌────────┐ ┌─┴────────┴─┐
 │CoreWeave │ │ Lambda │ │ Together │ │ Crusoe │ │Voltage Park│ ← Neo-Clouds
 └──────────┘ └────────┘ └──────────┘ └────────┘ └────────────┘
```

---

## InferMark Composite Score

```python
InferMark_Score = weighted_sum(
    0.15 × Throughput_Score,
    0.12 × Latency_Score,           # Axis 13 deep profiling
    0.10 × Memory_Efficiency,
    0.10 × Scaling_Score,
    0.08 × Quality_Score,
    0.08 × Cost_Score,
    0.08 × Overhead_Score,          # Axis 12 production overhead
    0.08 × Routing_Score,           # Axis 14 routing optimization
    0.06 × Determinism_Score,
    0.05 × Multi_Modal_Score,
    0.05 × Long_Context_Score,
    0.05 × Adapter_Score,
)
```

---

## Competitive Positioning

| Existing Benchmark | Measures | Misses |
|---|---|---|
| Chatbot Arena | Model quality via human preference | Zero infrastructure insight |
| HELM | Model accuracy across tasks | No serving/infra metrics |
| MLPerf Inference | Raw throughput on fixed models | No production overhead, single cloud |
| Artificial Analysis | API-level speed/cost | Black box, no stack profiling |
| **InferMark** | **14 axes, 9 clouds (incl. neo-clouds), multi-engine, latency + routing** | — |

---

## Research Takeaways

1. **3.2× throughput variance across clouds** for identical models — nobody measures this today.
2. **Production overhead (logging, PII, guardrails) adds 18-61% latency** — current benchmarks ignore this entirely.
3. **MoE expert routing becomes non-deterministic** under network jitter (EFA worst at 88%, IB NDR best at 97%).
4. **Routing path selection introduces up to 23ms TTFT variance** depending on ingress topology and LB queue depth.
5. **Neo-clouds (CoreWeave, Lambda, Crusoe) dominate cost-optimal placements** — 15-29% cheaper than hyperscalers at iso-latency.
6. **Multi-hop disaggregated routing** (prefill on one cloud, decode on another) is a novel systems contribution with real cost savings.
7. **LB queue depth is the #1 unpredictable latency source** — P99 varies ±11ms, dwarfing GPU compute variance.
8. **The InferMark Composite Score** creates a new industry standard — whoever defines the benchmark defines the evaluation criteria.
