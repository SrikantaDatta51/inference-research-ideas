#!/usr/bin/env python3
"""Convert inference research MDs to PDF. Downloads images from GitHub via git archive or renders draw.io as fallback."""
import os, subprocess, re, xml.etree.ElementTree as ET, base64, zlib, urllib.parse
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(REPO, "images")
PDF_DIR = os.path.join(REPO, "pdfs")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

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

# Map: github URL -> (local_image_path, drawio_source)
IMAGE_MAP = {
    "718ea411-bb11-41ae-83c8-01258fe36591": ("01-infermark.png", "01-infermark-benchmarking/architecture.drawio", "InferMark Benchmarking Architecture"),
    "6e937dae-d8da-45b3-be8c-8509524d485e": ("02-disagg.png", "02-disaggregated-cross-cloud/architecture.drawio", "Disaggregated Cross-Cloud Inference"),
    "3c454fee-6e0a-4bfb-a990-eff524967c68": ("03-parity.png", "03-cross-cloud-parity/architecture.drawio", "Cross-Cloud Parity Overlay"),
    "023241a7-2cf9-49bf-b8f9-2470074c40d4": ("04-quantization.png", "04-accuracy-preserving-quantization/architecture.drawio", "Accuracy-Preserving Quantization Pipeline"),
}

def parse_drawio_labels(drawio_path):
    """Extract meaningful labels from a draw.io XML file."""
    try:
        tree = ET.parse(drawio_path)
        root = tree.getroot()
        labels = []
        for diagram in root.iter('diagram'):
            content = diagram.text
            if content:
                try:
                    decoded = base64.b64decode(content)
                    inflated = zlib.decompress(decoded, -15)
                    xml_str = urllib.parse.unquote(inflated.decode('utf-8'))
                    sub_root = ET.fromstring(xml_str)
                except:
                    sub_root = root
            else:
                sub_root = root
            for cell in sub_root.iter():
                val = cell.get('value', '')
                if val:
                    clean = re.sub(r'<[^>]+>', ' ', val).strip()
                    clean = re.sub(r'\s+', ' ', clean)
                    if clean and len(clean) > 1 and clean not in labels:
                        labels.append(clean)
        return labels
    except:
        return []

def render_drawio_to_png(drawio_path, output_path, title):
    """Render draw.io components as a professional architecture diagram image."""
    labels = parse_drawio_labels(drawio_path)
    if not labels:
        labels = [title]

    # Create a professional-looking architecture diagram
    W, H = 1400, 800
    img = Image.new('RGB', (W, H), '#FFFFFF')
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_box = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font_title = ImageFont.load_default()
        font_box = font_title
        font_small = font_title

    # Header bar
    draw.rectangle([0, 0, W, 60], fill='#1B3A5C')
    draw.text((W//2 - len(title)*6, 18), title, fill='white', font=font_title, anchor=None)

    # Subtitle
    draw.text((20, 70), f"Architecture Components ({len(labels)} elements extracted from draw.io)", fill='#666', font=font_small)

    # Layout components in a grid
    cols = 4
    box_w, box_h = 300, 55
    margin_x, margin_y = 30, 15
    start_y = 100
    colors = ['#2C5F8A', '#3572A5', '#1B3A5C', '#4A90D9', '#2C5F8A', '#3572A5']

    for i, label in enumerate(labels[:24]):  # max 24 boxes
        row, col = divmod(i, cols)
        x = margin_x + col * (box_w + margin_x)
        y = start_y + row * (box_h + margin_y)
        if y + box_h > H - 40:
            break
        color = colors[i % len(colors)]
        # Rounded rectangle
        draw.rounded_rectangle([x, y, x+box_w, y+box_h], radius=8, fill=color)
        # Truncate label
        display = label[:40] + ('...' if len(label) > 40 else '')
        draw.text((x+10, y+18), display, fill='white', font=font_box)
        # Draw arrows between consecutive boxes
        if col < cols - 1 and i + 1 < len(labels):
            draw.text((x+box_w+5, y+20), "→", fill='#2C5F8A', font=font_box)

    # Footer
    draw.rectangle([0, H-30, W, H], fill='#f0f0f0')
    draw.text((20, H-22), f"Source: {os.path.basename(drawio_path)} | Generated for PDF rendering", fill='#999', font=font_small)

    img.save(output_path, 'PNG', quality=95)
    print(f"    📊 Rendered: {os.path.basename(output_path)} ({os.path.getsize(output_path)//1024}KB)")

def generate_images():
    """Generate PNG images from draw.io files."""
    print("Rendering draw.io diagrams to PNG...")
    for asset_id, (img_name, drawio_rel, title) in IMAGE_MAP.items():
        drawio_path = os.path.join(REPO, drawio_rel)
        img_path = os.path.join(IMG_DIR, img_name)
        if os.path.exists(drawio_path):
            render_drawio_to_png(drawio_path, img_path, title)
        else:
            print(f"    ⚠️ {drawio_rel} not found")

def replace_remote_images(content, md_dir):
    """Replace remote GitHub image URLs with local image paths."""
    def replacer(match):
        full_tag = match.group(0)
        for asset_id, (img_name, _, _) in IMAGE_MAP.items():
            if asset_id in full_tag:
                local_path = os.path.join(IMG_DIR, img_name)
                if os.path.exists(local_path):
                    return f'<img width="1400" alt="{img_name}" src="{local_path}" />'
        return full_tag
    return re.sub(r'<img[^>]+user-attachments/assets/[^>]+/>', replacer, content)

def md_to_pdf(md_path, pdf_path, title):
    """Convert MD to PDF with local images."""
    print(f"  Converting: {os.path.basename(md_path)} → {os.path.basename(pdf_path)}")
    with open(md_path, 'r') as f:
        content = f.read()

    md_dir = os.path.dirname(md_path)
    content = replace_remote_images(content, md_dir)

    # Also replace drawio link references
    content = re.sub(
        r'See \[architecture\.drawio\]\(\./architecture\.drawio\)[^\n]*',
        lambda m: '', content
    )

    tmp_md = md_path + ".tmp.md"
    with open(tmp_md, 'w') as f:
        f.write(content)

    result = subprocess.run(
        ["pandoc", tmp_md, "-f", "gfm", "-t", "html5",
         "--pdf-engine=weasyprint", "--metadata", f"title={title}",
         "--include-in-header=/dev/stdin", "-o", pdf_path],
        input=CSS, capture_output=True, text=True, cwd=md_dir
    )
    os.remove(tmp_md)

    if os.path.exists(pdf_path):
        print(f"    ✅ {os.path.getsize(pdf_path)//1024} KB")
        return True
    print(f"    ❌ Failed: {result.stderr[:200]}")
    return False

def main():
    print("="*60)
    print("Inference Research — MD to PDF with Embedded Images")
    print("="*60)

    generate_images()
    print()

    conversions = [
        ("README.md", "00-README.pdf", "Inference Research Ideas — Overview"),
        ("research-ideas-bucket-list.md", "01-Research-Ideas-Bucket-List.pdf", "14 Research Ideas Bucket List"),
        ("01-infermark-benchmarking/README.md", "02-InferMark-Benchmarking.pdf", "InferMark: Cross-Cloud Inference Benchmarking"),
        ("02-disaggregated-cross-cloud/README.md", "03-Disaggregated-Cross-Cloud-Inference.pdf", "Disaggregated Cross-Cloud Inference"),
        ("03-cross-cloud-parity/README.md", "04-Cross-Cloud-Parity-Overlay.pdf", "Cross-Cloud Inference Parity Overlay"),
        ("04-accuracy-preserving-quantization/README.md", "05-Accuracy-Preserving-Quantization.pdf", "Accuracy-Preserving Quantization"),
    ]
    ok = 0
    for md_rel, pdf_name, title in conversions:
        md_path = os.path.join(REPO, md_rel)
        if os.path.exists(md_path):
            if md_to_pdf(md_path, os.path.join(PDF_DIR, pdf_name), title):
                ok += 1
    print(f"\n{'='*60}\nDone: {ok}/{len(conversions)} PDFs\n{'='*60}")

if __name__ == "__main__":
    main()
