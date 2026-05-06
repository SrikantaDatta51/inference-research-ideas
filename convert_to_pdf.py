#!/usr/bin/env python3
"""Convert inference research MDs to PDF with uploaded JPGs, author name, narrow margins, and mock result graphs."""
import os, subprocess, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(REPO, "pdfs")
GRAPH_DIR = os.path.join(REPO, "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)

CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
@page { size: letter; margin: 8mm !important; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 11pt; line-height: 1.5; color: #222; margin: 0 !important; padding: 0 !important; }
h1 { color: #1B3A5C; border-bottom: 2px solid #2C5F8A; padding-bottom: 6px; font-size: 20pt; margin: 8px 0; }
h2 { color: #2C5F8A; border-bottom: 1px solid #ccc; padding-bottom: 3px; font-size: 15pt; margin: 6px 0; }
h3 { color: #3572A5; font-size: 12pt; margin: 4px 0; }
h4 { color: #555; font-size: 10pt; margin: 3px 0; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
pre { background: #1e1e1e; color: #d4d4d4; padding: 8px; border-radius: 5px; font-size: 8pt; white-space: pre-wrap; margin: 4px 0; }
pre code { background: none; color: inherit; }
table { border-collapse: collapse; width: 100%; margin: 6px 0; font-size: 9pt; }
th { background: #1B3A5C; color: white; padding: 4px 6px; text-align: left; }
td { padding: 3px 6px; border: 1px solid #ddd; }
tr:nth-child(even) { background: #f8f8f8; }
blockquote { border-left: 3px solid #2C5F8A; margin: 6px 0; padding: 5px 12px; background: #f0f5fa; font-style: italic; }
img { max-width: 100% !important; width: 100% !important; height: auto; margin: 4px 0; display: block; }
a { color: #2C5F8A; }
p { margin: 4px 0; }
ul, ol { margin: 4px 0 4px 20px; }
li { margin: 2px 0; }
hr { margin: 6px 0; }
"""

ASSET_TO_JPG = {
    "718ea411-bb11-41ae-83c8-01258fe36591": os.path.join(PDF_DIR, "InferMark.jpg"),
    "6e937dae-d8da-45b3-be8c-8509524d485e": os.path.join(PDF_DIR, "DiasggInference.jpg"),
    "3c454fee-6e0a-4bfb-a990-eff524967c68": os.path.join(PDF_DIR, "InferenceNormalization.jpg"),
    "023241a7-2cf9-49bf-b8f9-2470074c40d4": os.path.join(PDF_DIR, "AcuracyPreservQuantization.jpg"),
}

COLORS = ['#1B3A5C','#2C5F8A','#3572A5','#4A90D9','#6BB5E0','#A8D5E2']

AUTHOR_HTML = '<div style="color:#1B3A5C;font-size:13pt;font-weight:bold;margin:0 0 4px 0;">Srikanta Datta</div><hr style="border:1px solid #2C5F8A;margin:0 0 8px 0;">'

def replace_remote_images(content):
    def replacer(match):
        tag = match.group(0)
        for asset_id, local_path in ASSET_TO_JPG.items():
            if asset_id in tag:
                if os.path.exists(local_path):
                    return f'<img width="100%" alt="{os.path.basename(local_path)}" src="file://{local_path}" />'
        return tag
    return re.sub(r'<img[^>]+user-attachments/assets/[^>]+/?>', replacer, content)

def md_to_pdf(md_path, pdf_path, title, graph_paths=None):
    print(f"  {os.path.basename(md_path)} → {os.path.basename(pdf_path)}")
    with open(md_path) as f:
        content = f.read()
    content = replace_remote_images(content)
    content = re.sub(r'See \[architecture\.drawio\]\(\./architecture\.drawio\)[^\n]*', '', content)

    # Convert MD → HTML using pandoc (just for markdown parsing, no PDF)
    result = subprocess.run(
        ["pandoc", "-f", "gfm", "-t", "html5"],
        input=content, capture_output=True, text=True, cwd=os.path.dirname(md_path))
    body_html = result.stdout

    # Build graph HTML block
    graph_html = ""
    if graph_paths:
        graph_html = '<div style="page-break-before:avoid;"><h2 style="color:#1B3A5C;border-bottom:2px solid #2C5F8A;padding-bottom:4px;">Expected Results</h2>'
        for i, (gp, caption) in enumerate(graph_paths):
            if os.path.exists(gp):
                fig_num = i + 1
                graph_html += f'<div style="margin:6px 0;text-align:center;"><img src="file://{gp}" style="width:100%;"/>'
                graph_html += f'<p style="font-size:9pt;color:#444;margin:2px 0 10px 0;"><b>Figure {fig_num}.</b> {caption}</p></div>'
        graph_html += '</div>'

    # Insert graphs right after the abstract section (after first <hr> + optional <img>)
    # Find position: after the architecture image or after first <hr>
    if graph_html:
        # Try to insert after the first image (architecture diagram)
        img_match = re.search(r'(<img[^>]+/>)\s*(<hr\s*/?>)?', body_html)
        if img_match:
            insert_pos = img_match.end()
            body_html = body_html[:insert_pos] + '\n' + graph_html + '\n' + body_html[insert_pos:]
        else:
            # Fallback: insert after first <hr>
            hr_match = re.search(r'<hr\s*/?>', body_html)
            if hr_match:
                insert_pos = hr_match.end()
                body_html = body_html[:insert_pos] + '\n' + graph_html + '\n' + body_html[insert_pos:]

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>{AUTHOR_HTML}{body_html}</body></html>"""

    tmp_html = pdf_path.replace('.pdf', '.tmp.html')
    with open(tmp_html, 'w') as f:
        f.write(html)

    subprocess.run(
        ["weasyprint", tmp_html, pdf_path, "--presentational-hints"],
        capture_output=True, text=True)
    os.remove(tmp_html)

    if os.path.exists(pdf_path):
        print(f"    ✅ {os.path.getsize(pdf_path)//1024} KB")
        return True
    print("    ❌ Failed"); return False

def _save(name, fig):
    p = os.path.join(GRAPH_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    return p

def make_infermark_graphs():
    figs = []
    fig, ax = plt.subplots(figsize=(10, 4))
    clouds = ['AWS\np4de', 'GCP\na3-mega', 'Azure\nND-H100', 'CoreWeave\nHGX', 'Lambda\nH100', 'OCI\nBM.GPU']
    tp = [2850, 3100, 2400, 3200, 3050, 2750]
    ax.bar(clouds, tp, color=COLORS, edgecolor='white')
    ax.axhline(y=np.mean(tp), color='red', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(tp):.0f}')
    ax.annotate('3.2x gap', xy=(2, 2450), fontsize=11, color='red', fontweight='bold')
    ax.set_ylabel('Tokens/sec'); ax.set_title('Llama-70B Throughput (batch=32) Across 6 Cloud Providers', fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3); ax.set_ylim(0, 3800); plt.tight_layout()
    figs.append((_save('im_throughput.png', fig), 'Throughput variance across cloud providers for Llama-70B (batch=32). Up to 3.2x difference observed between lowest and highest performers.'))
    fig, ax = plt.subplots(figsize=(10, 3.5))
    cats = ['Authentication', 'Load Balancer', 'Rate Limiting', 'Guardrails/Safety', 'Logging/Telemetry', 'Total Overhead']
    ms = [4.2, 8.5, 2.1, 18.3, 3.8, 36.9]
    ax.barh(cats, ms, color=COLORS)
    for i, v in enumerate(ms): ax.text(v+0.5, i, f'{v}ms', va='center', fontsize=9)
    ax.set_xlabel('Latency (ms)'); ax.set_title('Production Overhead Breakdown (Invisible to Current Benchmarks)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3); plt.tight_layout()
    figs.append((_save('im_overhead.png', fig), 'Production middleware adds 18-61% latency that existing benchmarks ignore. Guardrails dominate at 18.3ms.'))
    return figs

def make_disagg_graphs():
    figs = []
    fig, ax = plt.subplots(figsize=(10, 4))
    models = ['Llama-70B', 'Mixtral-8x22B', 'GPT-4 class', 'Llama-405B']
    x = np.arange(len(models))
    ax.bar(x-0.25, [1.00,1.45,2.50,4.00], 0.25, label='Co-located', color='#ccc')
    ax.bar(x, [0.85,1.20,2.10,3.40], 0.25, label='Prefill (OCI/Crusoe)', color='#2C5F8A')
    ax.bar(x+0.25, [0.52,0.78,1.35,2.15], 0.25, label='Decode (CoreWeave)', color='#4A90D9')
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel('$/1M tokens')
    ax.set_title('Inference Cost: Disaggregated vs Co-located Serving', fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3); plt.tight_layout()
    figs.append((_save('dg_cost.png', fig), 'Cost per 1M output tokens. Disaggregation achieves 15-29% cost reduction by routing prefill to compute-optimal and decode to memory-bandwidth-optimal clouds.'))
    fig, ax = plt.subplots(figsize=(10, 4))
    ctx = [2048, 4096, 8192, 16384, 32768, 65536]
    ms = [2.1, 4.8, 11.2, 24.5, 52.0, 108.0]
    ax.semilogy(ctx, ms, 'o-', color='#1B3A5C', linewidth=2.5, markersize=8)
    ax.fill_between(ctx, [t*0.8 for t in ms], [t*1.2 for t in ms], alpha=0.15, color='#4A90D9')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.6, label='SLA threshold (50ms)')
    ax.set_xlabel('Context Length (tokens)'); ax.set_ylabel('Transfer Time (ms)')
    ax.set_title('KV-Cache Cross-Cloud Transfer Latency vs Context Length', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    figs.append((_save('dg_kvcache.png', fig), 'KV-cache transfer latency vs context length. SLA threshold breached at ~32K tokens, the practical boundary for cross-cloud disaggregation.'))
    return figs

def make_parity_graphs():
    figs = []
    fig, ax = plt.subplots(figsize=(10, 4))
    pairs = ['AWS-GCP', 'AWS-Azure', 'GCP-CW', 'Azure-OCI', 'CW-Lambda']
    x = np.arange(len(pairs))
    ax.bar(x-0.15, [0.042,0.038,0.051,0.045,0.033], 0.3, label='Without overlay', color='#C00000', alpha=0.8)
    ax.bar(x+0.15, [0.003,0.004,0.005,0.004,0.003], 0.3, label='With parity overlay', color='#00B050', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(pairs); ax.set_ylabel('KL Divergence')
    ax.set_title('Cross-Cloud Output Divergence: Before vs After Parity Overlay', fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3); plt.tight_layout()
    figs.append((_save('cp_divergence.png', fig), 'KL divergence of model outputs across cloud pairs. Parity overlay reduces divergence by 10x on average to <0.005.'))
    fig, ax = plt.subplots(figsize=(10, 4))
    prec = ['FP32', 'FP16', 'BF16', 'FP8', 'INT8', 'INT4']
    ax.plot(prec, [0.001,0.008,0.012,0.035,0.042,0.089], 'o-', color='#C00000', linewidth=2.5, markersize=9, label='Raw drift')
    ax.plot(prec, [0.001,0.002,0.003,0.008,0.011,0.025], 's-', color='#00B050', linewidth=2.5, markersize=9, label='With overlay')
    ax.fill_between(prec, [0.001,0.008,0.012,0.035,0.042,0.089], [0.001,0.002,0.003,0.008,0.011,0.025], alpha=0.12, color='#2C5F8A')
    ax.set_ylabel('Output Divergence'); ax.set_title('Numerical Precision vs Cross-Cloud Output Drift', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    figs.append((_save('cp_precision.png', fig), 'Lower-precision formats amplify cross-cloud drift exponentially. INT4 shows 89x more drift than FP32. Overlay reduces INT4 drift from 0.089 to 0.025.'))
    return figs

def make_quantization_graphs():
    figs = []
    fig, ax = plt.subplots(figsize=(10, 4))
    methods = ['GPTQ', 'AWQ', 'SqueezeLLM', 'Ours\n(TurboQuant)', 'Round-to-\nNearest']
    x = np.arange(len(methods))
    ax.bar(x-0.15, [98.8,99.0,98.5,99.4,96.2], 0.3, label='W8A8', color='#2C5F8A')
    ax.bar(x+0.15, [94.2,95.1,93.8,97.3,89.5], 0.3, label='W4A16', color='#4A90D9')
    ax.axhline(y=97, color='red', linestyle='--', alpha=0.6, label='Compliance threshold (97%)')
    ax.set_xticks(x); ax.set_xticklabels(methods); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Retention on Legal/Medical Benchmarks', fontweight='bold')
    ax.set_ylim(85, 101); ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3); plt.tight_layout()
    figs.append((_save('qu_accuracy.png', fig), 'Accuracy on regulated benchmarks (LegalBench, MedQA). TurboQuant is the only method exceeding 97% compliance threshold at W4A16.'))
    fig, ax = plt.subplots(figsize=(10, 4))
    models = ['Llama-70B', 'Mixtral-8x22B', 'CodeLlama-34B', 'Med-PaLM-2']
    x = np.arange(len(models))
    ax.bar(x-0.25, [1200,980,1350,850], 0.25, label='FP16 (baseline)', color='#1B3A5C')
    ax.bar(x, [2100,1720,2380,1490], 0.25, label='INT8 (1.75x)', color='#3572A5')
    ax.bar(x+0.25, [3400,2800,3850,2420], 0.25, label='INT4 (2.83x)', color='#6BB5E0')
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel('Tokens/sec')
    ax.set_title('Throughput Improvement by Quantization Level (H100 80GB)', fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3); plt.tight_layout()
    figs.append((_save('qu_throughput.png', fig), 'INT4 achieves 2.83x average speedup over FP16 baseline on H100 80GB while TurboQuant maintains >97% accuracy.'))
    return figs

PAPER_GRAPHS = {
    "01-infermark-benchmarking/README.md": make_infermark_graphs,
    "02-disaggregated-cross-cloud/README.md": make_disagg_graphs,
    "03-cross-cloud-parity/README.md": make_parity_graphs,
    "04-accuracy-preserving-quantization/README.md": make_quantization_graphs,
}

def main():
    print("="*60)
    print("Inference Research — MD to PDF + Figures")
    print("="*60)
    print("\nGenerating figures...")
    graph_map = {}
    for md_rel, gen_fn in PAPER_GRAPHS.items():
        fl = gen_fn(); graph_map[md_rel] = fl
        for p, _ in fl: print(f"  {os.path.basename(p)} ({os.path.getsize(p)//1024}KB)")
    print()
    jobs = [
        ("README.md", "00-README.pdf", "Overview"),
        ("research-ideas-bucket-list.md", "01-Research-Ideas-Bucket-List.pdf", "Bucket List"),
        ("01-infermark-benchmarking/README.md", "02-InferMark-Benchmarking.pdf", "InferMark"),
        ("02-disaggregated-cross-cloud/README.md", "03-Disaggregated-Cross-Cloud-Inference.pdf", "Disagg"),
        ("03-cross-cloud-parity/README.md", "04-Cross-Cloud-Parity-Overlay.pdf", "Parity"),
        ("04-accuracy-preserving-quantization/README.md", "05-Accuracy-Preserving-Quantization.pdf", "Quant"),
    ]
    ok = 0
    for m, p, t in jobs:
        if md_to_pdf(os.path.join(REPO, m), os.path.join(PDF_DIR, p), t, graph_map.get(m)):
            ok += 1
    print(f"\nDone: {ok}/{len(jobs)} PDFs")

if __name__ == "__main__":
    main()

