#!/usr/bin/env python3
"""
Extract structured data from maintenance document PDFs using Azure OpenAI Vision.
Supports: Maintenance Memos (MM), MEL, CDL, FTD.
"""

import os
import argparse
import base64
import ast
import pandas as pd
from pdf2image import convert_from_path
from openai import AzureOpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class DocumentExtractor:
    """Extract structured data from maintenance documents using Vision AI."""
    
    def __init__(self, doc_type: str = "MM"):
        self.doc_type = doc_type
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        )
        self.deployment = os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT", "gpt-4-vision")
        self.prompt = self._load_prompt()
    
    def _load_prompt(self) -> str:
        """Load extraction prompt for document type."""
        prompt_file = Path(__file__).parent.parent / "prompts" / f"{self.doc_type.lower()}_extraction_prompt.txt"
        
        if prompt_file.exists():
            with open(prompt_file) as f:
                return f.read()
        
        # Default MM prompt
        return """Your task is to extract specific sections from a series of images that represent pages from a Maintenance Memo for an Aircraft.

The expected output is a dictionary, where each key represents a section.

Here are the steps to follow:

1. Iterate through each image provided.
2. Use the following <scratchpad> to extract sections from each image:

<scratchpad>
For the current image:
- Look for Memo No on the top right corner of first page and extract it into "Memo No" key value pair
- Extract "AIRCRAFT MODEL" and the text beside as key value pair
- Extract "AIRCRAFT EFFECTIVITY" and the text beside it as key value pair
- Extract "SUBJECT" and the text below it as key value pair
- Extract "DISCUSSION" and the text below it as key value pair
- Extract "ACTION" and the text below it as key value pair
- Extract "COMPLIANCE" and any text below until next section starts as key value pair
- Extract "REASON FOR REVISION", "REFERENCE" and the text corresponding to those sections as key value pairs
</scratchpad>

Once all images have been processed, format the complete data into a dictionary with keys for each section, returning them as a dictionary.

<output>
[
{Dictionary}
]
</output>

Note: 1. Do not modify the text present in the images
2. If the text is spanning across images, make sure extract everything in the same order as it was present including the subsections and sections
3. Do not add any extra text outside of the Dictionary list
"""
    
    def extract_from_pdf(self, pdf_path: str) -> list:
        """Extract data from a single PDF file."""
        pages = convert_from_path(pdf_path, dpi=250)
        
        # Build message content with images
        message_content = [{"type": "text", "text": self.prompt}]
        
        for count, page in enumerate(pages):
            # Convert image to base64
            image_path = f"/tmp/page_{count}.jpg"
            page.save(image_path, 'JPEG')
            
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            message_content.append({
                "type": "text",
                "text": f"Here's Image {count + 1}:"
            })
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            })
            
            # Clean up temp file
            os.remove(image_path)
        
        # Call Vision API
        messages = [
            {"role": "system", "content": "Extract the text from the images and maintain the original order."},
            {"role": "user", "content": message_content}
        ]
        
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=messages,
            temperature=0.0
        )
        
        # Parse response
        content = response.choices[0].message.content
        cleaned = content.strip().strip('<output>').strip('</output>').strip()
        start_idx = cleaned.find('[')
        end_idx = cleaned.rfind(']') + 1
        json_string = cleaned[start_idx:end_idx]
        
        data_list = ast.literal_eval(json_string)
        
        # Add filename to each record
        for item in data_list:
            item['file_name'] = os.path.basename(pdf_path)
        
        return data_list
    
    def extract_from_folder(self, folder_path: str, max_workers: int = 4) -> pd.DataFrame:
        """Extract data from all PDFs in a folder."""
        pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pdf')]
        
        results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.extract_from_pdf, pdf): pdf for pdf in pdf_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting documents"):
                try:
                    result = future.result()
                    results.extend(result)
                except Exception as e:
                    error_file = futures[future]
                    errors.append(error_file)
                    print(f"✗ Error processing {error_file}: {e}")
        
        if errors:
            print(f"\nFailed files ({len(errors)}):")
            for err in errors:
                print(f"  - {err}")
        
        return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Extract structured data from maintenance PDFs")
    parser.add_argument("--input", required=True, help="Input folder containing split PDFs")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--doc-type", default="MM", choices=["MM", "MEL", "CDL", "FTD"],
                        help="Document type (default: MM)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    
    args = parser.parse_args()
    
    print(f"Extracting {args.doc_type} documents from: {args.input}")
    
    extractor = DocumentExtractor(doc_type=args.doc_type)
    df = extractor.extract_from_folder(args.input, max_workers=args.workers)
    
    # Save to CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    
    print(f"✓ Extracted {len(df)} records → {args.output}")


if __name__ == "__main__":
    main()
