#!/usr/bin/env python3
"""Convert inference research MDs to PDF using the actual JPG images from pdfs/ folder."""
import os, subprocess, re

REPO = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(REPO, "pdfs")

CSS = """<style>
body{font-family:'Segoe UI',Calibri,Arial,sans-serif;font-size:11pt;line-height:1.6;color:#222;max-width:900px;margin:0 auto;padding:20px 40px}
h1{color:#1B3A5C;border-bottom:2px solid #2C5F8A;padding-bottom:8px;font-size:22pt}
h2{color:#2C5F8A;border-bottom:1px solid #ccc;padding-bottom:4px;font-size:16pt}
h3{color:#3572A5;font-size:13pt}h4{color:#555;font-size:11pt}
code{background:#f4f4f4;padding:2px 5px;border-radius:3px;font-size:10pt}
pre{background:#1e1e1e;color:#d4d4d4;padding:12px;border-radius:6px;font-size:9pt;white-space:pre-wrap}
pre code{background:none;color:inherit}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:9.5pt}
th{background:#1B3A5C;color:white;padding:6px 8px;text-align:left}
td{padding:5px 8px;border:1px solid #ddd}tr:nth-child(even){background:#f8f8f8}
blockquote{border-left:4px solid #2C5F8A;margin:10px 0;padding:8px 16px;background:#f0f5fa;font-style:italic}
img{max-width:100%;height:auto;margin:10px 0}
a{color:#2C5F8A}
@page{size:letter;margin:0.75in}@page :first{margin-top:1.5in}
</style>"""

# Map GitHub asset IDs to local JPG filenames in pdfs/
ASSET_TO_JPG = {
    "718ea411-bb11-41ae-83c8-01258fe36591": os.path.join(PDF_DIR, "InferMark.jpg"),
    "6e937dae-d8da-45b3-be8c-8509524d485e": os.path.join(PDF_DIR, "DiasggInference.jpg"),
    "3c454fee-6e0a-4bfb-a990-eff524967c68": os.path.join(PDF_DIR, "InferenceNormalization.jpg"),
    "023241a7-2cf9-49bf-b8f9-2470074c40d4": os.path.join(PDF_DIR, "AcuracyPreservQuantization.jpg"),
}

def replace_remote_images(content):
    """Replace remote GitHub <img> tags with local JPG absolute paths."""
    def replacer(match):
        tag = match.group(0)
        for asset_id, local_path in ASSET_TO_JPG.items():
            if asset_id in tag:
                if os.path.exists(local_path):
                    return f'<img width="100%" alt="{os.path.basename(local_path)}" src="file://{local_path}" />'
        return tag
    return re.sub(r'<img[^>]+user-attachments/assets/[^>]+/?>', replacer, content)

def md_to_pdf(md_path, pdf_path, title):
    print(f"  {os.path.basename(md_path)} → {os.path.basename(pdf_path)}")
    with open(md_path) as f:
        content = f.read()
    content = replace_remote_images(content)
    content = re.sub(r'See \[architecture\.drawio\]\(\./architecture\.drawio\)[^\n]*', '', content)

    tmp = md_path + ".tmp.md"
    with open(tmp, 'w') as f:
        f.write(content)
    subprocess.run(
        ["pandoc", tmp, "-f", "gfm", "-t", "html5", "--pdf-engine=weasyprint",
         "--metadata", f"title={title}", "--include-in-header=/dev/stdin", "-o", pdf_path],
        input=CSS, capture_output=True, text=True, cwd=os.path.dirname(md_path))
    os.remove(tmp)
    if os.path.exists(pdf_path):
        print(f"    ✅ {os.path.getsize(pdf_path)//1024} KB")
        return True
    print("    ❌ Failed")
    return False

def main():
    print("="*60)
    print("Inference Research — MD→PDF with uploaded JPG images")
    print("="*60)
    for aid, path in ASSET_TO_JPG.items():
        status = f"✅ {os.path.getsize(path)//1024}KB" if os.path.exists(path) else "❌ missing"
        print(f"  {os.path.basename(path)}: {status}")
    print()

    jobs = [
        ("README.md", "00-README.pdf", "Inference Research Ideas — Overview"),
        ("research-ideas-bucket-list.md", "01-Research-Ideas-Bucket-List.pdf", "14 Research Ideas Bucket List"),
        ("01-infermark-benchmarking/README.md", "02-InferMark-Benchmarking.pdf", "InferMark: Cross-Cloud Inference Benchmarking"),
        ("02-disaggregated-cross-cloud/README.md", "03-Disaggregated-Cross-Cloud-Inference.pdf", "Disaggregated Cross-Cloud Inference"),
        ("03-cross-cloud-parity/README.md", "04-Cross-Cloud-Parity-Overlay.pdf", "Cross-Cloud Inference Parity Overlay"),
        ("04-accuracy-preserving-quantization/README.md", "05-Accuracy-Preserving-Quantization.pdf", "Accuracy-Preserving Quantization"),
    ]
    ok = sum(md_to_pdf(os.path.join(REPO, m), os.path.join(PDF_DIR, p), t) for m, p, t in jobs)
    print(f"\nDone: {ok}/{len(jobs)} PDFs with embedded JPG diagrams")

if __name__ == "__main__":
    main()
