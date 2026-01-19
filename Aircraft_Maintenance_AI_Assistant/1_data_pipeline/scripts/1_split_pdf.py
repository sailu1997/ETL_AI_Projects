#!/usr/bin/env python3
"""
Split multi-document PDF into individual files based on OCR footer detection.
Detects "Page 1 of N" patterns to identify document boundaries.
"""

import os
import argparse
from typing import List, Tuple
from tqdm import tqdm
from pdf2image import convert_from_path
import pytesseract
from PyPDF2 import PdfReader, PdfWriter
import re


def extract_document_ranges(input_pdf: str, chunk_size: int = 50) -> List[Tuple[int, int]]:
    """
    Parse PDF and identify document boundaries by detecting 'Page 1 of N' footers.
    
    Args:
        input_pdf: Path to input PDF file
        chunk_size: Number of pages to process at once
        
    Returns:
        List of (start_page, end_page) tuples for each document
    """
    reader = PdfReader(input_pdf)
    num_pages = len(reader.pages)
    doc_ranges = []
    start_page = None
    prev_footer = None

    for start in tqdm(range(0, num_pages, chunk_size), desc="Detecting document boundaries"):
        end = min(start + chunk_size, num_pages)
        images = convert_from_path(input_pdf, first_page=start + 1, last_page=end, dpi=150)

        for page_num, image in enumerate(images, start=start):
            # Extract footer region
            width, height = image.size
            footer_region = (0, height - 500, width, height)
            footer_image = image.crop(footer_region)

            # OCR the footer
            text = pytesseract.image_to_string(footer_image).strip()
            footer_match = re.search(r"(Page 1 of \d+)|(Page 1of\d+)|(Page 1 of [a-zA-Z])", text)

            if footer_match:
                current_footer = footer_match.group(0)
                if prev_footer and start_page is not None:
                    # End previous document
                    doc_ranges.append((start_page, page_num - 1))
                
                # Start new document
                start_page = page_num
                prev_footer = current_footer

    # Add final document
    if start_page is not None:
        doc_ranges.append((start_page, num_pages - 1))

    return doc_ranges


def split_pdf_by_ranges(input_pdf: str, output_folder: str, ranges: List[Tuple[int, int]], prefix: str = "doc") -> None:
    """
    Split PDF into separate files based on page ranges.
    
    Args:
        input_pdf: Path to input PDF
        output_folder: Output directory
        ranges: List of (start, end) page ranges
        prefix: Filename prefix for output files
    """
    reader = PdfReader(input_pdf)
    os.makedirs(output_folder, exist_ok=True)

    for idx, (start, end) in enumerate(tqdm(ranges, desc="Splitting PDF")):
        writer = PdfWriter()
        for i in range(start, end + 1):
            writer.add_page(reader.pages[i])
        
        output_path = os.path.join(output_folder, f"{prefix}_{idx}.pdf")
        with open(output_path, "wb") as output_file:
            writer.write(output_file)
        
        print(f"Document {idx}: pages {start+1}-{end+1} → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Split multi-document PDF by detecting page 1 footers")
    parser.add_argument("--input", required=True, help="Input PDF file path")
    parser.add_argument("--output", required=True, help="Output directory for split PDFs")
    parser.add_argument("--prefix", default="MM", help="Filename prefix (default: MM)")
    parser.add_argument("--chunk-size", type=int, default=50, help="Pages to process at once (default: 50)")
    
    args = parser.parse_args()
    
    print(f"Processing: {args.input}")
    ranges = extract_document_ranges(args.input, args.chunk_size)
    print(f"Found {len(ranges)} documents")
    
    split_pdf_by_ranges(args.input, args.output, ranges, args.prefix)
    print(f"✓ Split complete. Output: {args.output}")


if __name__ == "__main__":
    main()
