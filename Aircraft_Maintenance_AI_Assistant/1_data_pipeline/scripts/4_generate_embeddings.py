#!/usr/bin/env python3
"""
Generate vector embeddings and create FAISS index from processed CSV data.
"""

import argparse
import os
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()


def create_embeddings(csv_file: str, output_dir: str, text_column: str = "text", chunk_size: int = 2100, chunk_overlap: int = 300):
    """
    Create FAISS vector store from CSV data.
    
    Args:
        csv_file: Input CSV file path
        output_dir: Output directory for FAISS index
        text_column: Column name containing text to embed
        chunk_size: Maximum chunk size for text splitting
        chunk_overlap: Overlap between chunks
    """
    print(f"Loading data from {csv_file}")
    df = pd.read_csv(csv_file)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV. Available: {list(df.columns)}")
    
    print(f"Loaded {len(df)} records")
    
    # Initialize embeddings model
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", "text-embedding-ada-002"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
    print("✓ Initialized Azure OpenAI Embeddings")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Create documents with metadata
    documents = []
    for idx, row in df.iterrows():
        text = str(row[text_column])
        metadata = {k: v for k, v in row.items() if k != text_column}
        metadata['row_id'] = idx
        
        chunks = text_splitter.split_text(text)
        for chunk_idx, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={**metadata, 'chunk_id': chunk_idx}
            )
            documents.append(doc)
    
    print(f"Split into {len(documents)} chunks")
    
    # Create FAISS vector store
    print("Generating embeddings and building FAISS index...")
    faiss_store = FAISS.from_documents(documents, embeddings)
    
    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    faiss_store.save_local(output_dir)
    
    print(f"✓ FAISS index saved to {output_dir}")
    print(f"  - {len(documents)} vectors")
    print(f"  - {len(df)} original documents")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings and create FAISS index")
    parser.add_argument("--input", required=True, help="Input CSV file with processed data")
    parser.add_argument("--output", required=True, help="Output directory for FAISS index")
    parser.add_argument("--text-column", default="text", help="Column name with text to embed (default: text)")
    parser.add_argument("--chunk-size", type=int, default=2100, help="Chunk size (default: 2100)")
    parser.add_argument("--chunk-overlap", type=int, default=300, help="Chunk overlap (default: 300)")
    
    args = parser.parse_args()
    
    create_embeddings(
        csv_file=args.input,
        output_dir=args.output,
        text_column=args.text_column,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )


if __name__ == "__main__":
    main()
