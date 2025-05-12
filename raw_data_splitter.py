import os
import subprocess
import os
from pdf2image import convert_from_path
from tqdm import tqdm



def pptx_to_pdf(pptx_path, output_dir):
    print(f"inside ppt to pdf, ppt path: {pptx_path}, output path: {output_dir}")

    """Convert PPTX to PDF using LibreOffice."""
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run([
        "libreoffice",
        "--headless",
        "--convert-to", "pdf",
        "--outdir", output_dir,
        pptx_path
    ], check=True)

    filename = os.path.splitext(os.path.basename(pptx_path))[0]
    return os.path.join(output_dir, f"{filename}.pdf")


def pdf_to_images(file_path, output_dir):
    print(f"inside pdf to image, file path: {file_path}, output path: {output_dir}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.ppt', '.pptx']:
        print ("converting ppt to pdf first")
        file_path = pptx_to_pdf(file_path, "raw_data")
        
    """Convert PDF pages to images using pdf2image."""
    filename = os.path.splitext(os.path.basename(file_path))[0]
    os.makedirs(os.path.join(output_dir,filename), exist_ok=True)
    images = convert_from_path(file_path)
    image_paths = []
    for i, img in tqdm(enumerate(images)):
        image_file = os.path.join(output_dir,filename, f"slide_{i+1}.png")
        img.save(image_file, "PNG")
        image_paths.append(image_file)

    return image_paths


def get_files_path(raw_file_path):
    print(f"raw file path: {raw_file_path}")
    files_path = pdf_to_images(raw_file_path, "all_doc_images")
    return files_path