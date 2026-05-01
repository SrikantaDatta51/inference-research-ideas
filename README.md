# Inference Overlay Research

> **Multi-Cloud Inference Benchmarking & Optimization Platform**
>
> A research initiative evaluating production LLM inference across 9 cloud providers — 4 hyperscalers + 5 neo-clouds.

---

## Research Vision

Modern LLM inference has moved from single-GPU prototypes to multi-cloud production systems, yet the ecosystem lacks standardized benchmarks, cross-cloud normalization, and quantization certification for regulated industries. This initiative addresses these gaps through four complementary research directions.

---

## Research Directions

| # | Project | Key Takeaway | Venue |
|---|---------|-------------|-------|
| 1 | [InferMark Benchmarking](./01-infermark-benchmarking/) | 3.2× throughput variance across clouds; 18-61% production overhead nobody measures | OSDI |
| 2 | [Disaggregated Cross-Cloud Inference](./02-disaggregated-cross-cloud/) | Multi-hop prefill/decode saves 15-29% cost; GQA/MLA are natural fit for cross-cloud disagg | OSDI/NSDI |
| 3 | [Cross-Cloud Performance Parity](./03-cross-cloud-parity/) | 4-layer normalization: ±17% → ±4% variance; network topology dominates, not GPU compute | SysML |
| 4 | [Accuracy-Preserving Quantization](./04-accuracy-preserving-quantization/) | SSMs most resilient to quantization; MLA latent KV most sensitive; QJL catches 99.97% errors | NeurIPS |
| — | [Full Bucket List (14 ideas)](./research-ideas-bucket-list.md) | Stack-ranked research ideas with takeaways, novelty, and venue fit | — |

---

## Cloud Providers Under Evaluation

### Hyperscalers

| Provider | GPU SKUs | Interconnect | Regions |
|----------|----------|-------------|---------|
| AWS | H100 SXM (p5), B200 (p5e) | EFA v2 (SRD) | us-east-1, eu-west-1 |
| GCP | H100 (A3 Mega), B200 (A3 Ultra) | GPUDirect-TCPX | us-central1, europe-west4 |
| Azure | H100 (ND v5), B200 (ND v6) | InfiniBand NDR | eastus, westeurope |
| OCI | B200 bare-metal (BM.GPU.B200.8) | RDMA Cluster Net v2 | us-ashburn, eu-frankfurt |

### Neo-Clouds

| Provider | GPU SKUs | Interconnect | Key Differentiator |
|----------|----------|-------------|-------------------|
| CoreWeave | H100 SXM, B200 | InfiniBand NDR | GPU-native, K8s-first, fastest availability |
| Lambda | H100 SXM, B200 | InfiniBand HDR/NDR | Simplest pricing, ML-focused |
| Together AI | H100, custom | Custom fabric | Inference-optimized, open-source hosting |
| Crusoe Energy | H100 SXM, B200 | InfiniBand NDR | Renewable/flare-gas powered, lowest carbon |
| Voltage Park | H100 SXM | InfiniBand NDR | Large contiguous clusters, HPC-grade |

---

## Repository Structure

```
inference-overlay-research/
├── README.md                                  # This file
├── research-ideas-bucket-list.md              # 14 ideas stack-ranked with takeaways
├── 01-infermark-benchmarking/
│   ├── README.md                              # 14-axis benchmarking framework
│   └── architecture.drawio                    # System architecture diagram
├── 02-disaggregated-cross-cloud/
│   ├── README.md                              # Cross-cloud prefill/decode optimization
│   └── architecture.drawio                    # Disaggregation flow diagram
├── 03-cross-cloud-parity/
│   ├── README.md                              # 4-layer normalization framework
│   └── architecture.drawio                    # Overlay architecture diagram
└── 04-accuracy-preserving-quantization/
    ├── README.md                              # TurboQuant certification framework
    └── architecture.drawio                    # Evaluation pipeline diagram
```

## License

Research use only. All rights reserved.
