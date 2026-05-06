#!/usr/bin/env python3
"""Convert all MD files to PDF with embedded draw.io diagrams rendered as images."""
import os, subprocess, re, base64, xml.etree.ElementTree as ET, zlib, urllib.parse

REPO = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(REPO, "pdfs")
os.makedirs(PDF_DIR, exist_ok=True)

CSS = """
<style>
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 11pt; line-height: 1.6;
       color: #222; max-width: 900px; margin: 0 auto; padding: 20px 40px; }
h1 { color: #1B3A5C; border-bottom: 2px solid #2C5F8A; padding-bottom: 8px; font-size: 22pt; }
h2 { color: #2C5F8A; border-bottom: 1px solid #ccc; padding-bottom: 4px; font-size: 16pt; }
h3 { color: #3572A5; font-size: 13pt; }
h4 { color: #555; font-size: 11pt; }
code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-size: 10pt; }
pre { background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 6px;
      font-size: 9pt; overflow-x: auto; white-space: pre-wrap; }
pre code { background: none; color: inherit; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; }
th { background: #1B3A5C; color: white; padding: 6px 8px; text-align: left; }
td { padding: 5px 8px; border: 1px solid #ddd; }
tr:nth-child(even) { background: #f8f8f8; }
blockquote { border-left: 4px solid #2C5F8A; margin: 10px 0; padding: 8px 16px;
             background: #f0f5fa; font-style: italic; }
img { max-width: 100%; height: auto; margin: 10px 0; }
.diagram-box { background: #f9f9f9; border: 1px solid #ddd; border-radius: 8px;
               padding: 15px; margin: 15px 0; text-align: center; }
.diagram-box img { max-width: 100%; }
.diagram-caption { font-size: 9pt; color: #666; font-style: italic; margin-top: 5px; }
a { color: #2C5F8A; }
@page { size: letter; margin: 0.75in; }
@page :first { margin-top: 1.5in; }
</style>
"""

def extract_drawio_svg(drawio_path):
    """Try to extract SVG data from drawio XML. Returns SVG string or None."""
    try:
        tree = ET.parse(drawio_path)
        root = tree.getroot()
        # draw.io files contain mxGraphModel with cell geometry
        # We'll create a simple visual representation
        diagram = root.find('.//diagram')
        if diagram is None:
            return None
        
        # Try to decode compressed diagram content
        content = diagram.text
        if content:
            try:
                decoded = base64.b64decode(content)
                inflated = zlib.decompress(decoded, -15)
                xml_str = urllib.parse.unquote(inflated.decode('utf-8'))
                graph_root = ET.fromstring(xml_str)
            except:
                graph_root = root
        else:
            graph_root = root
        
        # Extract cell labels for a text representation
        labels = []
        for cell in graph_root.iter():
            val = cell.get('value', '')
            if val and len(val) > 1 and '<' not in val[:1]:
                labels.append(val.strip())
            elif val and 'value=' in str(cell.attrib):
                # Strip HTML tags
                clean = re.sub(r'<[^>]+>', '', val).strip()
                if clean and len(clean) > 1:
                    labels.append(clean)
        
        return labels
    except Exception as e:
        print(f"  Warning: Could not parse {drawio_path}: {e}")
        return None


def create_diagram_html(drawio_path, caption):
    """Create an HTML representation of the draw.io diagram."""
    labels = extract_drawio_svg(drawio_path)
    if not labels:
        return f'<div class="diagram-box"><p><em>📊 Architecture Diagram: {caption}</em></p><p><em>See <code>{os.path.basename(drawio_path)}</code> for interactive diagram</em></p></div>'
    
    # Create a styled flow diagram from extracted labels
    html = f'<div class="diagram-box">\n'
    html += f'<table style="margin:0 auto;border:none;"><tr>\n'
    for i, label in enumerate(labels[:12]):  # Limit to key components
        if i > 0 and i % 4 == 0:
            html += '</tr><tr style="height:10px;"><td colspan="4" style="border:none;text-align:center;color:#2C5F8A;">↓</td></tr><tr>\n'
        html += f'<td style="background:#2C5F8A;color:white;text-align:center;border-radius:4px;padding:8px;margin:2px;border:none;">{label}</td>\n'
        if i < len(labels)-1 and (i+1) % 4 != 0:
            html += '<td style="border:none;color:#2C5F8A;"> → </td>\n'
    html += '</tr></table>\n'
    html += f'<p class="diagram-caption">{caption}</p></div>\n'
    return html


def md_to_pdf(md_path, pdf_path, title):
    """Convert a markdown file to PDF using pandoc + weasyprint."""
    print(f"  Converting: {os.path.basename(md_path)} → {os.path.basename(pdf_path)}")
    
    with open(md_path, 'r') as f:
        content = f.read()
    
    # Find and replace drawio references with rendered diagrams
    md_dir = os.path.dirname(md_path)
    
    # Replace "See [architecture.drawio](./architecture.drawio)..." lines
    def replace_drawio_ref(match):
        drawio_file = os.path.join(md_dir, "architecture.drawio")
        if os.path.exists(drawio_file):
            caption = f"Architecture Diagram — {title}"
            return create_diagram_html(drawio_file, caption)
        return match.group(0)
    
    content = re.sub(
        r'See \[architecture\.drawio\]\(\./architecture\.drawio\)[^\n]*',
        replace_drawio_ref, content
    )
    
    # Write temp markdown
    tmp_md = md_path + ".tmp.md"
    with open(tmp_md, 'w') as f:
        f.write(content)
    
    # Convert with pandoc + weasyprint
    cmd = [
        "pandoc", tmp_md,
        "-f", "gfm",
        "-t", "html5",
        "--pdf-engine=weasyprint",
        "--metadata", f"title={title}",
        f"--include-in-header=/dev/stdin",
        "-o", pdf_path
    ]
    
    result = subprocess.run(
        cmd, input=CSS, capture_output=True, text=True,
        cwd=md_dir
    )
    
    os.remove(tmp_md)
    
    if result.returncode != 0:
        print(f"    ⚠️ Warnings: {result.stderr[:200]}")
    
    if os.path.exists(pdf_path):
        size = os.path.getsize(pdf_path) / 1024
        print(f"    ✅ {size:.0f} KB")
        return True
    else:
        print(f"    ❌ Failed")
        return False


def main():
    print("=" * 60)
    print("Converting Inference Research MDs to PDF")
    print("=" * 60)
    
    conversions = [
        ("README.md", "00-README.pdf", "Inference Research Ideas — Repository Overview"),
        ("research-ideas-bucket-list.md", "01-Research-Ideas-Bucket-List.pdf", "14 Research Ideas — Bucket List"),
        ("01-infermark-benchmarking/README.md", "02-InferMark-Benchmarking.pdf", "InferMark: Cross-Cloud Inference Benchmarking"),
        ("02-disaggregated-cross-cloud/README.md", "03-Disaggregated-Cross-Cloud-Inference.pdf", "Disaggregated Cross-Cloud Inference"),
        ("03-cross-cloud-parity/README.md", "04-Cross-Cloud-Parity-Overlay.pdf", "Cross-Cloud Inference Parity Overlay"),
        ("04-accuracy-preserving-quantization/README.md", "05-Accuracy-Preserving-Quantization.pdf", "Accuracy-Preserving Quantization"),
    ]
    
    success = 0
    for md_rel, pdf_name, title in conversions:
        md_path = os.path.join(REPO, md_rel)
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        if os.path.exists(md_path):
            if md_to_pdf(md_path, pdf_path, title):
                success += 1
        else:
            print(f"  ⚠️ {md_rel} not found, skipping")
    
    print(f"\n{'=' * 60}")
    print(f"Done: {success}/{len(conversions)} PDFs generated in pdfs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
