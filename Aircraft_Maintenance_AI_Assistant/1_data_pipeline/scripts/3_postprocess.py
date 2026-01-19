#!/usr/bin/env python3
"""
Post-process extracted data by combining fields into a formatted text column for embeddings.
"""

import argparse
import pandas as pd
import os


def format_mm_data(df: pd.DataFrame) -> pd.DataFrame:
    """Format Maintenance Memo data."""
    def format_row(row):
        fields = []
        field_names = [
            'Memo No', 'AIRCRAFT MODEL', 'AIRCRAFT EFFECTIVITY', 'DATE RAISED',
            'SUBJECT', 'DISCUSSION', 'ACTION', 'COMPLIANCE', 'REASON FOR REVISION', 'REFERENCE'
        ]
        
        for field in field_names:
            if field in row and pd.notna(row[field]) and str(row[field]).strip():
                fields.append(f"{field}: {row[field]}")
        
        return "\n".join(fields)
    
    df['text'] = df.apply(format_row, axis=1)
    # Keep only file_name and text columns
    return df[['file_name', 'text']]


def format_mel_data(df: pd.DataFrame) -> pd.DataFrame:
    """Format MEL data."""
    def format_row(row):
        fields = []
        field_names = [
            'Main-item-title', 'sub-item-title', 'sub-sub-item-title', 'sub-sub-sub-item-title',
            'Dispatch alternative', 'Interval', 'Installed', 'Required', 'Procedure', 'text'
        ]
        
        for field in field_names:
            if field in row and pd.notna(row[field]) and str(row[field]).strip():
                fields.append(f"{field}: {row[field]}")
        
        return "\n".join(fields)
    
    df['data'] = df.apply(format_row, axis=1)
    return df[['file_name', 'data']].rename(columns={'data': 'text'})


def format_cdl_data(df: pd.DataFrame) -> pd.DataFrame:
    """Format CDL data."""
    text_fields = []
    if 'CDL-title' in df.columns:
        text_fields.append("CDL-title: " + df['CDL-title'].fillna(''))
    if 'text' in df.columns:
        text_fields.append("text: " + df['text'].fillna(''))
    if 'Missing items' in df.columns:
        text_fields.append("Missing items: " + df['Missing items'].fillna(''))
    if 'Number Installed' in df.columns:
        text_fields.append("Number Installed: " + df['Number Installed'].astype(str))
    if 'Takeoff&Landing' in df.columns:
        text_fields.append("Takeoff&Landing: " + df['Takeoff&Landing'].fillna(''))
    if 'EnrouteClimb' in df.columns:
        text_fields.append("EnrouteClimb: " + df['EnrouteClimb'].fillna(''))
    if 'penalty' in df.columns:
        text_fields.append(df['penalty'].fillna(''))
    
    df['text'] = '\n'.join([field for field in text_fields if not field.isna().all()])
    return df[['file_name', 'text']]


def format_ftd_data(df: pd.DataFrame) -> pd.DataFrame:
    """Format FTD data."""
    columns_to_keep = ['ATA', 'file_name']
    columns_to_combine = [col for col in df.columns if col not in columns_to_keep]
    
    def combine_columns(row):
        combined = ' '.join([
            f"{col}:{str(row[col]).strip()}"
            for col in columns_to_combine
            if pd.notna(row[col]) and str(row[col]).strip() != ''
        ])
        return combined
    
    df['text'] = df.apply(combine_columns, axis=1)
    return df[columns_to_keep + ['text']]


def postprocess(input_csv: str, output_csv: str, doc_type: str):
    """Post-process extracted data based on document type."""
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} records from {input_csv}")
    
    if doc_type == "MM":
        df_formatted = format_mm_data(df)
    elif doc_type == "MEL":
        df_formatted = format_mel_data(df)
    elif doc_type == "CDL":
        df_formatted = format_cdl_data(df)
    elif doc_type == "FTD":
        df_formatted = format_ftd_data(df)
    else:
        raise ValueError(f"Unknown document type: {doc_type}")
    
    # Remove empty text rows
    df_formatted = df_formatted[df_formatted['text'].str.strip() != '']
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_formatted.to_csv(output_csv, index=False)
    print(f"✓ Saved {len(df_formatted)} formatted records → {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Post-process extracted data for embeddings")
    parser.add_argument("--input", required=True, help="Input CSV file from extraction")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--doc-type", required=True, choices=["MM", "MEL", "CDL", "FTD"],
                        help="Document type")
    
    args = parser.parse_args()
    postprocess(args.input, args.output, args.doc_type)


if __name__ == "__main__":
    main()
