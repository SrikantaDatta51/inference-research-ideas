# Inference Overlay Research

> **Multi-Cloud Inference Benchmarking & Optimization Platform**
>
> A research initiative to build the definitive evaluation framework for production LLM inference across heterogeneous cloud infrastructure.

---

## Research Vision

Modern LLM inference has moved from single-GPU prototypes to multi-cloud production systems, yet the ecosystem lacks:

- **Standardized benchmarks** that evaluate the full inference stack (not just tokens/sec)
- **Cross-cloud normalization** that accounts for heterogeneous GPU SKUs, interconnects, and driver stacks
- **Production-realistic evaluation** that includes the overhead of enterprise features (logging, filtering, guardrails)
- **Quantization certification** for regulated industries where accuracy degradation is unacceptable

This research initiative addresses these gaps through four complementary projects.

---

## Research Directions

| # | Project | Focus Area | Target Venue |
|---|---------|------------|--------------|
| 1 | [InferMark Benchmarking](./01-infermark-benchmarking/) | 14-axis multi-cloud inference evaluation with latency profiling and routing analysis | OSDI / SoCC |
| 2 | [Secure-Tenant KV Isolation](./02-secure-tenant-kv-isolation/) | Provable data isolation in shared GPU clusters via quantized memory sharding | NSDI / USENIX Security |
| 3 | [Cross-Cloud Performance Parity](./03-cross-cloud-parity/) | Normalization framework for consistent inference across heterogeneous CSPs | SysML / MLSys |
| 4 | [Accuracy-Preserving Quantization](./04-accuracy-preserving-quantization/) | Zero-loss quantization evaluation for compliance-restricted workloads | NeurIPS / ICML |

---

## Technical Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                        Evaluation Targets                       │
├──────────────┬──────────────┬────────────────┬─────────────────┤
│ Architectures│ Memory/Perf  │ Attention      │ Orchestration   │
├──────────────┼──────────────┼────────────────┼─────────────────┤
│ MoE          │ TurboQuant   │ Flash Attn 3   │ DeepSpeed V4    │
│ DeepSeek V4  │ PolarQuant   │ MLA            │ DeepFusion      │
│ Mamba/SSM    │ QJL          │ GQA            │ Disagg. P/D     │
│ Jamba        │ 3-bit KV     │ Ring Attention  │ K8s GPU Sched   │
│ RWKV-6       │ PagedAttn    │ Paced Attention│ vLLM / SGLang   │
│ Llama 4      │ vAttention   │ Linear Attn    │ TensorRT-LLM    │
└──────────────┴──────────────┴────────────────┴─────────────────┘
```

## Cloud Providers Under Evaluation

| Provider | GPU SKUs | Interconnect | Regions |
|----------|----------|-------------|---------|
| AWS | H100 SXM, p5e (B200) | EFA v2 | us-east-1, eu-west-1 |
| GCP | H100, A3 Mega (B200) | GPUDirect-TCPX | us-central1, europe-west4 |
| Azure | H100, ND B200 v6 | InfiniBand NDR | eastus, westeurope |
| OCI | BM.GPU.H100, B200 | RDMA Cluster Net | us-ashburn, eu-frankfurt |
| Lambda | H100 SXM, B200 | InfiniBand HDR | us-tx, us-ut |

---

## Repository Structure

```
inference-overlay-research/
├── README.md                              # This file
├── 01-infermark-benchmarking/
│   ├── README.md                          # 14-axis benchmarking framework
│   └── architecture.drawio                # System architecture diagram
├── 02-secure-tenant-kv-isolation/
│   ├── README.md                          # TurboQuant memory sharding
│   └── architecture.drawio                # Isolation architecture diagram
├── 03-cross-cloud-parity/
│   ├── README.md                          # Cross-cloud normalization
│   └── architecture.drawio                # Overlay architecture diagram
└── 04-accuracy-preserving-quantization/
    ├── README.md                          # Quantization certification
    └── architecture.drawio                # Evaluation pipeline diagram
```

---

## Key Metrics

The **InferMark Composite Score** unifies all evaluation dimensions:

```
InferMark Score = Σ(axis_weight × normalized_score) across all 14 axes

Axis weights are configurable per deployment profile:
  - "Cost-Optimized"    → heavy weight on $/token, quantization efficiency
  - "Latency-Sensitive" → heavy weight on TTFT, P99, routing overhead
  - "Multi-Tenant"      → heavy weight on isolation, memory sharding
  - "Regulated"         → heavy weight on accuracy preservation, overhead
```

---

## Getting Started

Each sub-project contains its own README with:
- Detailed technical benchmarking axes
- Architecture diagrams (`.drawio` files, open with [draw.io](https://app.diagrams.net/))
- Research paper framing and abstract templates
- Build sequences and deliverable timelines

## License

Research use only. All rights reserved.
